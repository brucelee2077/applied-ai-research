/* =========================================================================
   app.js — Mock Interview System frontend (vanilla JS, no framework).
   Screen flow: picker -> topics -> config -> chat -> scorecard (+ history).
   Phase 2 wires the offline flow (answer -> reveal model answers -> self-grade).
   The live AI path (SSE streaming + AI grading) slots into submitAnswer/probe.
   ========================================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const on = (el, ev, fn) => el && el.addEventListener(ev, fn);
  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };

  // Make a non-<button> element keyboard-operable (Enter/Space), like a button.
  const makeButtonlike = (node, handler) => {
    node.setAttribute("role", "button");
    node.setAttribute("tabindex", "0");
    on(node, "click", handler);
    on(node, "keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        handler(ev);
      }
    });
    return node;
  };

  const VIZ = window.__INTERVIEW_VIZ || null;

  // Static metadata for all four tracks (counts merged from the API).
  const TRACK_META = {
    ml_system_design: { emoji: "🏗", name: "ML System Design", desc: "Whiteboard end-to-end ML products: search, recommendations, moderation, generation." },
    fundamentals: { emoji: "📐", name: "ML / AI Fundamentals", desc: "Concept deep-dives graded on the 4-level rubric: transformers, attention, RL, fine-tuning." },
    frontier_lab: { emoji: "⚡", name: "Frontier Lab (24-Week Track)", desc: "Staff-level systems depth from the 24-week curriculum: JAX/parallelism, scaling laws, inference/MoE, kernels/quantization, ADRS, and derive-it-live systems math." },
    frontier_research: { emoji: "🔬", name: "Frontier Research", desc: "RLHF, alignment, decoding, inference systems — research-engineer depth." },
    behavioral: { emoji: "🤝", name: "Behavioral & Leadership", desc: "Scope, conflict, ambiguity, failure, exec-level impact framing." },
  };
  const TRACK_ORDER = ["ml_system_design", "fundamentals", "frontier_lab", "frontier_research", "behavioral"];
  const LEVELS = [
    ["no_hire", "No Hire", "lc-no"],
    ["weak_hire", "Weak Hire", "lc-weak"],
    ["hire", "Hire", "lc-hire"],
    ["strong_hire", "Strong Hire", "lc-strong"],
  ];
  const BAND_CLASS = { no_hire: "v-no", weak_hire: "v-weak", hire: "v-hire", strong_hire: "v-strong" };

  // ---------- app state ----------
  const S = {
    tracksByName: {},     // track -> api summary
    liveAvailable: false,
    sel: { track: null, module: null, topic: null, mode: "drill", questionId: null },
    session: null,        // {session_id, mode, live, total_questions}
    current: null,        // current QuestionView
    selfLevel: null,      // chosen self-grade level
    timer: null,
    lastScorecard: null,
  };

  // ---------- api ----------
  async function getJSON(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }
  async function postJSON(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  // ---------- screen router ----------
  function show(screenId) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
    const target = $(screenId);
    if (target) {
      target.classList.remove("hidden");
      // Move focus to the new screen so keyboard/AT users aren't stranded on the
      // old one (each .screen has tabindex="-1" in index.html). enterChat then
      // re-focuses the textarea, which is the desired target for the chat screen.
      if (typeof target.focus === "function") target.focus({ preventScroll: true });
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ---------- top bar progress ----------
  async function refreshProgress() {
    try {
      const p = await getJSON("/api/progress");
      $("lvl-num").textContent = p.level;
      $("lvl-name").textContent = p.level_name;
      $("token-chip").textContent = "🪙 " + p.tokens;
      const pct = p.xp_needed_for_next > 0 ? Math.min(100, (p.xp_into_level / p.xp_needed_for_next) * 100) : 100;
      $("xp-fill").style.width = pct + "%";
      $("xp-label").textContent = p.xp + " XP · " + p.percentile + "th pct";
    } catch (e) {
      /* progress is non-critical */
    }
  }

  async function initHealth() {
    try {
      const h = await getJSON("/api/health");
      S.liveAvailable = !!h.live_available;
      $("offline-banner").classList.toggle("hidden", S.liveAvailable);
    } catch (e) {
      $("offline-banner").classList.remove("hidden");
    }
  }

  // ========================================================================
  // SCREEN 1: track picker
  // ========================================================================
  async function loadPicker() {
    show("screen-picker");
    let apiTracks = [];
    try {
      apiTracks = await getJSON("/api/tracks");
    } catch (e) {
      $("track-grid").innerHTML = "";
      $("track-grid").appendChild(el("div", "error", "Could not load tracks: " + e.message));
      return;
    }
    S.tracksByName = {};
    apiTracks.forEach((t) => (S.tracksByName[t.track] = t));

    const grid = $("track-grid");
    grid.innerHTML = "";
    TRACK_ORDER.forEach((tk) => {
      const meta = TRACK_META[tk];
      const api = S.tracksByName[tk];
      const count = api ? api.question_count : 0;
      const card = el("div", "track-card" + (count ? "" : " disabled"));
      card.appendChild(el("div", "tname", meta.emoji + "  " + meta.name));
      card.appendChild(el("div", "tdesc", meta.desc));
      card.appendChild(el("div", count ? "tcount" : "tchip", count ? count + " questions" : "Coming soon"));
      if (count) makeButtonlike(card, () => openTrack(tk));
      grid.appendChild(card);
    });
  }

  // ========================================================================
  // SCREEN 1b: topic picker
  // ========================================================================
  function openTrack(track) {
    S.sel.track = track;
    const t = S.tracksByName[track];
    const meta = TRACK_META[track];
    $("topic-title").textContent = meta.emoji + "  " + meta.name;
    $("topic-sub").textContent = "Pick a topic to interview on.";
    const cr = $("topic-crumbs");
    cr.innerHTML = "";
    const home = el("button", null, "Tracks");
    on(home, "click", loadPicker);
    cr.appendChild(home);
    cr.appendChild(el("span", "sep", "/"));
    cr.appendChild(el("span", null, meta.name));

    const list = $("topic-list");
    list.innerHTML = "";
    (t.modules || []).forEach((m, i) => {
      const row = el("div", "topic-row");
      row.appendChild(el("div", "tnum", String(i + 1)));
      const info = el("div", "tinfo");
      info.appendChild(el("div", "name", m.topic));
      info.appendChild(el("div", "meta", m.question_count + " questions"));
      row.appendChild(info);
      row.appendChild(el("div", "arrow", "→"));
      makeButtonlike(row, () => openTopic(m));
      list.appendChild(row);
    });
    show("screen-topics");
  }

  // ========================================================================
  // SCREEN 2: config
  // ========================================================================
  function openTopic(module) {
    S.sel.module = module.module_id;
    S.sel.topic = module.topic;
    S.sel.mode = "drill";
    const meta = TRACK_META[S.sel.track];

    const cr = $("config-crumbs");
    cr.innerHTML = "";
    const home = el("button", null, "Tracks");
    on(home, "click", loadPicker);
    const back = el("button", null, meta.name);
    on(back, "click", () => openTrack(S.sel.track));
    cr.appendChild(home);
    cr.appendChild(el("span", "sep", "/"));
    cr.appendChild(back);
    cr.appendChild(el("span", "sep", "/"));
    cr.appendChild(el("span", null, module.topic));

    $("config-title").textContent = "Set up: " + module.topic;

    // mode toggle
    document.querySelectorAll(".mode-opt").forEach((opt) => {
      opt.classList.toggle("sel", opt.dataset.mode === "drill");
      opt.onclick = () => {
        S.sel.mode = opt.dataset.mode;
        document.querySelectorAll(".mode-opt").forEach((o) => o.classList.toggle("sel", o === opt));
        $("drill-question-field").classList.toggle("hidden", S.sel.mode !== "drill");
      };
    });
    $("drill-question-field").classList.remove("hidden");

    // drill question dropdown
    const sel = $("drill-question-select");
    sel.innerHTML = "";
    const auto = el("option", null, "First question (recommended)");
    auto.value = "";
    sel.appendChild(auto);
    (module.questions || []).forEach((q) => {
      const o = el("option", null, (q.section_title || q.prompt_preview).slice(0, 80));
      o.value = q.id;
      sel.appendChild(o);
    });

    $("config-error").classList.add("hidden");
    show("screen-config");
  }

  async function startSession() {
    const btn = $("start-btn");
    btn.disabled = true;
    try {
      const body = { track: S.sel.track, module_id: S.sel.module, mode: S.sel.mode };
      if (S.sel.mode === "drill") {
        const qid = $("drill-question-select").value;
        if (qid) body.question_id = qid;
      }
      const resp = await postJSON("/api/session/start", body);
      S.session = resp;
      S.current = resp.question;
      enterChat();
    } catch (e) {
      const err = $("config-error");
      err.textContent = "Could not start: " + e.message;
      err.classList.remove("hidden");
    } finally {
      btn.disabled = false;
    }
  }

  // ========================================================================
  // SCREEN 3: chat
  // ========================================================================
  function enterChat() {
    $("chat-topic").textContent = S.sel.topic;
    $("chat-mode").textContent = S.session && S.session.live ? "🟢 Live AI" : "📄 Offline reveal";
    $("transcript").innerHTML = "";
    $("reveal-wrap").classList.add("hidden");
    $("answer-box").classList.remove("hidden");
    $("answer-text").value = "";
    updateProg();
    askCurrentQuestion();
    startTimer();
    show("screen-chat");
    $("answer-text").focus();
  }

  function updateProg() {
    const d = S.current && S.current.difficulty;
    const diff = d ? "  ·  Difficulty " + Math.min(5, d) + "/5" : "";
    $("chat-prog").textContent =
      (S.session.mode === "drill" ? "Drill" : "Full mock") +
      " · Q" + (S.current.index + 1) + " of " + S.current.total + diff;
  }

  function askCurrentQuestion() {
    // Live mode types the question out; offline shows it instantly.
    if (S.session && S.session.live) typeBubble(S.current.prompt);
    else addBubble("interviewer", S.current.prompt);
  }

  function addBubble(role, text) {
    const b = el("div", "bubble " + role);
    b.appendChild(el("div", "who", role === "interviewer" ? "Interviewer" : "You"));
    b.appendChild(el("div", "btext", text));
    $("transcript").appendChild(b);
    b.scrollIntoView({ behavior: "smooth", block: "end" });
    return b;
  }

  // A temporary "interviewer is thinking" bubble shown while a live turn is in
  // flight. Reuses the .dots animation from styles.css. Fixed markup — no user
  // data — so innerHTML is safe here.
  function addThinking() {
    const b = el("div", "bubble interviewer thinking");
    b.appendChild(el("div", "who", "Interviewer"));
    const body = el("div", "btext");
    body.innerHTML = 'Thinking <span class="dots"><span></span><span></span><span></span></span>';
    b.appendChild(body);
    $("transcript").appendChild(b);
    b.scrollIntoView({ behavior: "smooth", block: "end" });
    return b;
  }

  // Typewriter effect for live interviewer turns (gives a "streaming" feel
  // without SSE — robust and fully testable).
  function typeBubble(text, done) {
    const b = el("div", "bubble interviewer");
    b.appendChild(el("div", "who", "Interviewer"));
    const body = el("div", "btext");
    const cursor = el("span", "cursor");
    b.appendChild(body);
    b.appendChild(cursor);
    $("transcript").appendChild(b);
    let i = 0;
    const step = Math.max(1, Math.round(text.length / 220)); // ~finish in ~ a few seconds
    const iv = setInterval(() => {
      i += step;
      body.textContent = text.slice(0, i);
      b.scrollIntoView({ behavior: "smooth", block: "end" });
      if (i >= text.length) {
        clearInterval(iv);
        body.textContent = text;
        cursor.remove();
        if (done) done();
      }
    }, 16);
    return b;
  }

  async function submitAnswer() {
    const text = $("answer-text").value.trim();
    if (!text) {
      $("answer-text").focus();
      return;
    }
    addBubble("candidate", text);
    $("answer-text").value = "";
    $("submit-answer").disabled = true;
    // In live mode the /answer call is a real LLM round-trip; show a "thinking"
    // indicator so the transcript never looks frozen. Removed on every branch.
    const thinking = S.session && S.session.live ? addThinking() : null;
    try {
      const resp = await postJSON(`/api/session/${S.session.session_id}/answer`, { text });
      if (thinking) thinking.remove();
      if (resp.action === "probe" && resp.probe) {
        typeBubble(resp.probe);
        $("submit-answer").disabled = false;
        $("answer-text").focus();
      } else if (resp.action === "grade_ready") {
        await gradeCurrent();
      } else {
        // reveal_ready: offline mode, or live degraded to offline this question.
        await showReveal();
      }
    } catch (e) {
      if (thinking) thinking.remove();
      // Restore the answer so the user can retry without retyping.
      $("answer-text").value = text;
      addBubble("interviewer", "Couldn't reach the interviewer (" + e.message + "). Your answer is restored — try again.");
      $("submit-answer").disabled = false;
      $("answer-text").focus();
    }
  }

  // ---------- live AI grading + feedback ----------
  async function gradeCurrent() {
    $("answer-box").classList.add("hidden");
    const wrap = $("reveal-wrap");
    wrap.classList.remove("hidden");
    wrap.innerHTML = "";
    const loading = el("div", "rsub");
    loading.innerHTML = 'Grading your answer <span class="dots"><span></span><span></span><span></span></span>';
    wrap.appendChild(loading);
    try {
      const resp = await postJSON(`/api/session/${S.session.session_id}/grade`, {});
      if (resp.degraded || !resp.grade) {
        await showReveal(); // live grading unreachable -> self-grade fallback
        return;
      }
      renderFeedback(resp.grade, resp.has_next, resp.next_question);
    } catch (e) {
      wrap.innerHTML = "";
      wrap.appendChild(el("div", "error", "Grading failed: " + e.message));
      // Never dead-end: give recovery paths back into the flow.
      const row = el("div", "btn-row");
      const retry = el("button", "btn", "Retry grading");
      on(retry, "click", gradeCurrent);
      const selfg = el("button", "btn ghost", "Self-grade instead");
      on(selfg, "click", showReveal);
      row.appendChild(retry);
      row.appendChild(selfg);
      wrap.appendChild(row);
    }
  }

  function renderFeedback(grade, hasNext, nextQ) {
    const wrap = $("reveal-wrap");
    wrap.innerHTML = "";
    const cls = { no_hire: "lc-no", weak_hire: "lc-weak", hire: "lc-hire", strong_hire: "lc-strong" }[grade.level];
    const label = { no_hire: "No Hire", weak_hire: "Weak Hire", hire: "Hire", strong_hire: "Strong Hire" }[grade.level];

    const card = el("div", "level-card " + cls + " sel");
    const head = el("div", "lhead");
    head.appendChild(el("span", "ltag", label));
    head.appendChild(el("span", null, Math.round(grade.score_pct * 100) + "%"));
    card.appendChild(head);
    if (grade.strengths && grade.strengths.length) {
      card.appendChild(el("div", "field-label", "Strengths"));
      card.appendChild(el("div", "lbody", "• " + grade.strengths.join("\n• ")));
    }
    if (grade.gaps && grade.gaps.length) {
      card.appendChild(el("div", "field-label", "Gaps"));
      card.appendChild(el("div", "lbody", "• " + grade.gaps.join("\n• ")));
    }
    if (grade.what_would_elevate) {
      card.appendChild(el("div", "field-label", "To level up"));
      card.appendChild(el("div", "lbody", grade.what_would_elevate));
    }
    wrap.appendChild(el("div", "rtitle", "Interviewer feedback"));
    wrap.appendChild(card);

    const next = el("button", "btn", hasNext ? "Next question →" : "Finish & see scorecard →");
    on(next, "click", () => {
      wrap.classList.add("hidden");
      if (hasNext && nextQ) {
        S.current = nextQ;
        updateProg();
        $("answer-box").classList.remove("hidden");
        $("answer-text").value = "";
        askCurrentQuestion();
        $("answer-text").focus();
      } else {
        endSession();
      }
    });
    wrap.appendChild(next);
    wrap.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- offline reveal + self-grade ----------
  async function showReveal() {
    let data;
    try {
      data = await getJSON(`/api/session/${S.session.session_id}/reveal`);
    } catch (e) {
      // Don't strand the user: surface the error, offer Retry, restore composer.
      const wrap = $("reveal-wrap");
      wrap.innerHTML = "";
      wrap.appendChild(el("div", "error", "Couldn't load the model answers: " + e.message));
      const retry = el("button", "btn", "Retry");
      on(retry, "click", showReveal);
      wrap.appendChild(retry);
      wrap.classList.remove("hidden");
      $("answer-box").classList.remove("hidden");
      $("submit-answer").disabled = false;
      return;
    }
    $("answer-box").classList.add("hidden");
    S.selfLevel = null;
    const wrap = $("reveal-wrap");
    wrap.innerHTML = "";
    wrap.appendChild(el("div", "rtitle", "Model answers by level"));
    wrap.appendChild(el("div", "rsub", "Read the calibrated answers, then mark where your answer landed."));

    LEVELS.forEach(([key, label, cls]) => {
      const lv = data.rubric[key];
      if (!lv) return;
      const card = el("div", "level-card " + cls);
      const head = el("div", "lhead");
      head.appendChild(el("span", "ltag", label));
      const radio = el("input");
      radio.type = "radio";
      radio.name = "self-level";
      radio.value = key;
      radio.setAttribute("aria-label", label);
      head.appendChild(radio);
      card.appendChild(head);
      card.appendChild(el("div", "lbody", lv.answer));
      const pick = () => {
        radio.checked = true;
        S.selfLevel = key;
        document.querySelectorAll(".level-card").forEach((c) => c.classList.toggle("sel", c === card));
        $("self-grade-submit").disabled = false;
      };
      on(card, "click", pick);
      on(radio, "change", pick);   // keyboard: focus radio + Space selects
      wrap.appendChild(card);
    });

    wrap.appendChild(el("div", "self-grade-q", "Be honest — where did your answer land?"));
    const submit = el("button", "btn", "Submit self-assessment →");
    submit.id = "self-grade-submit";
    submit.disabled = true;
    on(submit, "click", submitSelfGrade);
    wrap.appendChild(submit);
    wrap.classList.remove("hidden");
    wrap.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function submitSelfGrade() {
    if (!S.selfLevel) return;
    $("self-grade-submit").disabled = true;
    try {
      const resp = await postJSON(`/api/session/${S.session.session_id}/self_grade`, { level: S.selfLevel });
      $("reveal-wrap").classList.add("hidden");
      if (resp.has_next && resp.next_question) {
        S.current = resp.next_question;
        updateProg();
        $("answer-box").classList.remove("hidden");
        $("answer-text").value = "";
        askCurrentQuestion();
        $("answer-text").focus();
      } else {
        await endSession();
      }
    } catch (e) {
      $("reveal-wrap").appendChild(el("div", "error", e.message));
    }
  }

  // ---------- timer ----------
  function startTimer() {
    stopTimer();
    const start = Date.now();
    // Soft, whole-session pacing target (not a hard limit) shown beside elapsed.
    const target = { drill: 10 * 60, full_mock: 45 * 60 }[S.session && S.session.mode] || 0;
    const fmt = (x) => String(Math.floor(x / 60)).padStart(2, "0") + ":" + String(x % 60).padStart(2, "0");
    const tick = () => {
      const s = Math.floor((Date.now() - start) / 1000);
      const t = $("chat-timer");
      t.textContent = target ? fmt(s) + " / " + fmt(target) : fmt(s);
      t.classList.toggle("over", target > 0 && s > target);
    };
    tick();
    S.timer = setInterval(tick, 1000);
  }
  function stopTimer() {
    if (S.timer) clearInterval(S.timer);
    S.timer = null;
  }

  // ========================================================================
  // SCREEN 4: scorecard
  // ========================================================================
  async function endSession() {
    stopTimer();
    let sc;
    try {
      sc = await postJSON(`/api/session/${S.session.session_id}/end`, {});
    } catch (e) {
      addBubble("interviewer", "(could not finish: " + e.message + ")");
      return;
    }
    S.lastScorecard = sc;
    renderScorecard(sc);
    refreshProgress();
  }

  function renderScorecard(sc) {
    // verdict
    const host = $("verdict-host");
    host.innerHTML = "";
    const v = el("div", "verdict " + (BAND_CLASS[sc.overall_level] || "v-weak"));
    v.appendChild(el("div", "band", sc.overall_band));
    v.appendChild(el("div", "score", Math.round(sc.overall_score_pct * 100) + "% · " + (sc.passed ? "Passed" : "Keep practicing")));
    v.appendChild(el("div", "meta", sc.topic + " · " + (sc.mode === "drill" ? "Quick drill" : "Full mock") + " · " + sc.elapsed_minutes + " min"));
    host.appendChild(v);

    // per-question bar
    if (VIZ) VIZ.renderBar($("bar-svg"), sc.per_question);
    else renderBarFallback(sc.per_question);

    // radar (only if criteria coverage present)
    const radarCard = $("sc-radar-card");
    let drew = false;
    if (VIZ && sc.criteria_coverage && Object.keys(sc.criteria_coverage).length >= 3) {
      drew = VIZ.renderRadar($("radar-svg"), sc.criteria_coverage);
    }
    radarCard.classList.toggle("hidden", !drew);

    // rewards
    renderReward(sc.xp);

    $("review-host").innerHTML = "";
    show("screen-scorecard");
  }

  function renderBarFallback(per) {
    const svg = $("bar-svg");
    svg.outerHTML = "<div id='bar-svg'>" + (per || []).map((p) =>
      `<div style="margin:4px 0;font-size:.85rem">${p.section_title}: <b>${Math.round(p.score_pct * 100)}%</b></div>`).join("") + "</div>";
  }

  function renderReward(xp) {
    const host = $("reward-host");
    host.innerHTML = "";
    const gain = el("div", "xp-gain", "+" + xp.xp_awarded + " XP");
    host.appendChild(gain);
    host.appendChild(el("div", null, "🪙 +" + xp.tokens_awarded + " tokens"));
    if (xp.level_up) {
      host.appendChild(el("div", "levelup", "⬆ Level up! You're now " + xp.new_level_name));
    }
    if (xp.new_badges && xp.new_badges.length) {
      const row = el("div", "badge-row");
      xp.new_badges.forEach((b) => {
        const badge = el("div", "badge");
        badge.appendChild(el("span", null, b.emoji));
        badge.appendChild(el("span", null, b.name));
        badge.title = b.desc;
        row.appendChild(badge);
      });
      host.appendChild(el("div", "field-label", "New badges"));
      host.appendChild(row);
    }
    if (xp.newly_unlocked_modules && xp.newly_unlocked_modules.length) {
      host.appendChild(el("div", "unlocked", "🔓 Unlocked: " + xp.newly_unlocked_modules.join(", ")));
    }
  }

  function renderReview(record) {
    const host = $("review-host");
    host.innerHTML = "";
    host.appendChild(el("h3", null, "Transcript"));
    (record.questions || record.per_question || []).forEach((q) => {
      const block = el("div", "q-block");
      block.appendChild(el("div", "qp", q.prompt || q.section_title));
      (q.transcript || []).forEach((t) => {
        block.appendChild(el("div", "turn " + t.role, (t.role === "candidate" ? "You: " : "Interviewer: ") + t.text));
      });
      if (q.grade) {
        block.appendChild(el("div", "turn interviewer", "Graded: " + q.grade.level + " (" + Math.round(q.grade.score_pct * 100) + "%)"));
      }
      host.appendChild(block);
    });
    host.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ========================================================================
  // SCREEN 5: history
  // ========================================================================
  async function loadHistory() {
    let idx = [];
    try {
      idx = await getJSON("/api/history");
    } catch (e) {
      /* ignore */
    }
    const list = $("history-list");
    list.innerHTML = "";
    if (!idx.length) {
      list.appendChild(el("div", "empty", "No interviews yet. Go practice!"));
    }
    idx.forEach((e) => {
      const row = el("div", "topic-row");
      const info = el("div", "tinfo");
      info.appendChild(el("div", "name", (e.topic || e.track) + " — " + (e.overall_band || "")));
      info.appendChild(el("div", "meta", (e.mode || "") + " · " + Math.round((e.overall_score_pct || 0) * 100) + "% · " + (e.ended_at || "").slice(0, 16).replace("T", " ")));
      row.appendChild(info);
      row.appendChild(el("div", "arrow", "→"));
      makeButtonlike(row, async () => {
        try {
          const detail = await getJSON("/api/history/" + e.session_id);
          S.lastScorecard = detail;
          renderScorecard(detail);
          renderReview(detail);
        } catch (err) {
          list.appendChild(el("div", "error", "Couldn't open that interview: " + err.message));
        }
      });
      list.appendChild(row);
    });
    show("screen-history");
  }

  // ========================================================================
  // wiring
  // ========================================================================
  function init() {
    on($("brand-home"), "click", () => { stopTimer(); loadPicker(); });
    on($("start-btn"), "click", startSession);
    on($("config-back"), "click", () => openTrack(S.sel.track));
    on($("submit-answer"), "click", submitAnswer);
    on($("answer-text"), "keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submitAnswer();
    });
    on($("quit-session"), "click", () => {
      if (confirm("Quit this interview? Progress on ungraded questions is lost.")) {
        stopTimer();
        loadPicker();
      }
    });
    on($("sc-review"), "click", async () => {
      const sc = S.lastScorecard;
      if (!sc) return;
      // A fresh scorecard carries per_question but NOT the transcript; the full
      // record (with every turn) is written to history on /end, so fetch that.
      if (!sc.questions && sc.session_id) {
        try {
          renderReview(await getJSON("/api/history/" + sc.session_id));
          return;
        } catch (e) {
          /* fall back to the lean per-question view below */
        }
      }
      renderReview(sc);
    });
    on($("sc-new"), "click", loadPicker);
    on($("sc-history"), "click", loadHistory);
    on($("history-back"), "click", loadPicker);

    // Make the session-type options keyboard-operable (their click handler is
    // (re)assigned per topic in openTopic; a keydown that calls .click() runs it).
    document.querySelectorAll(".mode-opt").forEach((opt) => {
      opt.setAttribute("role", "button");
      opt.setAttribute("tabindex", "0");
      on(opt, "keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          opt.click();
        }
      });
    });

    initHealth();
    refreshProgress();
    loadPicker();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
