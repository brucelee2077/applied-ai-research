"""SnapKV-style KV cache eviction simulator (JAX).

Reimplements the SnapKV paper's ``snap_kv`` pipeline almost verbatim:

    1. Take the queries from the last ``window_size`` positions of the
       prompt (the "observation window").
    2. Compute their attention weights against every prefix key — this
       reuses attention that a real forward pass already computes during
       prefill, so it is "free" (no extra model call).
    3. Sum those weights over the observation-window queries to get one
       importance "vote" per prefix position, per head.
    4. Run a 1D max-pool over the vote scores so that high votes spread
       to their neighbors, encouraging clusters (contiguous spans) of
       tokens to survive together instead of isolated spikes.
    5. Keep only the top-k highest-scoring prefix positions.
    6. Always keep the observation window itself, unmodified.

Reference: Li et al., "SnapKV: LLM Knows What You Are Looking for Before
Generation" (arXiv:2404.14469), Section 4, Algorithm / Listing 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pool1d(vote: jnp.ndarray, kernel_size: int, stride: int = 1) -> jnp.ndarray:
    """1D max-pool over the last axis of ``vote``, same-length output.

    Args:
        vote: array of shape [..., L] — one score per prefix position.
        kernel_size: pooling window width. kernel_size=1 is a no-op
            (returns ``vote`` unchanged), matching "no pooling" in the
            paper's ablation.
        stride: pooling stride (paper always uses 1 — "same" padding
            keeps the output length equal to the input length).

    Returns:
        Array of the same shape as ``vote``, each entry replaced by the
        max of its kernel_size-wide neighborhood (edge-padded so the
        window never runs off the sequence).
    """
    if kernel_size <= 1:
        return vote

    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    # Edge padding (repeat the boundary value) avoids artificially
    # dragging down the vote of positions near the start/end of the prefix.
    padded = jnp.pad(
        vote,
        [(0, 0)] * (vote.ndim - 1) + [(pad_left, pad_right)],
        mode="edge",
    )
    length = vote.shape[-1]
    # Gather every kernel_size-wide window and take its max.
    windows = jnp.stack(
        [padded[..., i : i + length] for i in range(kernel_size)], axis=-1
    )
    return jnp.max(windows, axis=-1)


def snap_kv(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int,
    attention_mask: jnp.ndarray | None = None,
):
    """Run the SnapKV eviction pipeline once, at the end of prefill.

    Args:
        query_states: [batch, num_heads, q_len, head_dim] queries for the
            full prompt (the output of prefill's attention projections).
        key_states: [batch, num_heads, q_len, head_dim] keys for the full
            prompt.
        value_states: [batch, num_heads, q_len, head_dim] values for the
            full prompt.
        window_size: size of the observation window (L_obs). The last
            ``window_size`` prompt tokens are always kept unmodified.
        max_capacity_prompt: total number of KV positions to keep per
            head after eviction (must be >= window_size).
        kernel_size: 1D max-pool kernel width applied to the vote scores
            before top-k selection. kernel_size=1 disables pooling.
        attention_mask: optional additive mask broadcastable to
            [batch, num_heads, window_size, q_len] (e.g. causal mask).
            If ``None``, no mask is applied (matches the paper's
            simplified pseudocode, since the observation window sits at
            the end of the prompt and only attends to positions that are
            already causally visible for a full-prompt prefill pass).

    Returns:
        (key_states, value_states) with sequence length either unchanged
        (if q_len < max_capacity_prompt — the guard clause fires and no
        eviction happens) or exactly ``max_capacity_prompt`` per head.
    """
    if max_capacity_prompt < window_size:
        raise ValueError("max_capacity_prompt must be >= window_size")

    batch, num_heads, q_len, head_dim = query_states.shape

    # Guard clause: nothing to evict if the prompt already fits the budget.
    if q_len < max_capacity_prompt:
        return key_states, value_states

    obs_q = query_states[..., -window_size:, :]
    scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=query_states.dtype))
    logits = jnp.einsum("bhwd,bhld->bhwl", obs_q, key_states) * scale
    if attention_mask is not None:
        logits = logits + attention_mask
    attn_weights = jax.nn.softmax(logits, axis=-1)
    # attn_weights: [batch, num_heads, window_size, q_len]

    # Drop the observation window's own columns — we only vote on the prefix.
    prefix_weights = attn_weights[..., -window_size:, :-window_size]
    vote = prefix_weights.sum(axis=-2)  # [batch, num_heads, L_prefix]

    pool_vote = pool1d(vote, kernel_size=kernel_size, stride=1)

    k = max_capacity_prompt - window_size
    _, indices = jax.lax.top_k(pool_vote, k)  # [batch, num_heads, k]

    # Gather the corresponding K/V vectors for the selected prefix positions.
    k_prefix = key_states[..., :-window_size, :]
    v_prefix = value_states[..., :-window_size, :]
    k_past = jnp.take_along_axis(k_prefix, indices[..., None], axis=2)
    v_past = jnp.take_along_axis(v_prefix, indices[..., None], axis=2)

    k_obs = key_states[..., -window_size:, :]
    v_obs = value_states[..., -window_size:, :]

    new_key_states = jnp.concatenate([k_past, k_obs], axis=2)
    new_value_states = jnp.concatenate([v_past, v_obs], axis=2)
    return new_key_states, new_value_states


def selected_prefix_indices(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int,
) -> jnp.ndarray:
    """Return just the top-k prefix indices SnapKV would keep (for tests).

    Same voting → pooling → top-k logic as ``snap_kv``, but stops short
    of gathering K/V, so tests can directly compare *which* positions
    were selected under different ``kernel_size`` settings.
    """
    obs_q = query_states[..., -window_size:, :]
    head_dim = query_states.shape[-1]
    scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=query_states.dtype))
    logits = jnp.einsum("bhwd,bhld->bhwl", obs_q, key_states) * scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    prefix_weights = attn_weights[..., -window_size:, :-window_size]
    vote = prefix_weights.sum(axis=-2)
    pool_vote = pool1d(vote, kernel_size=kernel_size, stride=1)
    k = max_capacity_prompt - window_size
    _, indices = jax.lax.top_k(pool_vote, k)
    return indices


def votes_to_indices(
    vote: jnp.ndarray, k: int, kernel_size: int
) -> jnp.ndarray:
    """Pool a raw vote signal and return its top-k indices.

    A lower-level helper used by the unit tests to directly probe the
    pooling → top-k stage on synthetic vote signals (bypassing attention
    entirely), so the "spiky vs. clustered" ablation isolates exactly the
    pooling variable described in the walkthrough.
    """
    pooled = pool1d(vote, kernel_size=kernel_size, stride=1)
    _, indices = jax.lax.top_k(pooled, k)
    return indices
