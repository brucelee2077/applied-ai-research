> **What this file covers**
> - 🎯 Why approximate nearest neighbor (ANN) search is necessary: the curse of dimensionality
> - 🧮 HNSW construction and search: the full algorithm with complexity derivation
> - 🧮 IVF clustering: nprobe analysis and the recall-latency trade-off
> - 🧮 Product Quantization: compression math, distortion bounds, memory savings
> - ⚠️ 4 failure modes: dimensionality curse, index staleness, metadata filter interaction, embedding version drift
> - 📊 Complexity analysis for Flat, IVF, HNSW, PQ — time, memory, build time
> - 💡 Indexing algorithm comparison and vector database selection
> - 🏭 Production: sharding, replication, index rebuild, embedding version management
> - Staff/Principal Q&A with all four hiring levels shown (5 questions)

---

# Vector Databases — Interview Deep-Dive

This file assumes you have read [vector-databases.md](./vector-databases.md) and understand the intuition for distance metrics, the four indexing algorithms (Flat, IVF, HNSW, PQ), and the popular vector database landscape. Everything here is for Staff/Principal depth.

---

## 🧮 Distance Metrics: The Math

### Cosine Similarity

```
🧮 Cosine similarity:

    cos(a, b) = (a · b) / (‖a‖ × ‖b‖) = Σ aᵢbᵢ / (√(Σ aᵢ²) × √(Σ bᵢ²))

    Range: [-1, 1]
    1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite
```

When vectors are L2-normalized (‖a‖ = 1), cosine similarity reduces to the dot product: cos(a, b) = a · b. Most vector databases normalize embeddings at insert time, so search is just a dot product — the fastest similarity operation.

### Euclidean Distance (L2)

```
🧮 L2 distance:

    d(a, b) = √(Σ (aᵢ - bᵢ)²)

    Range: [0, ∞)
    0 = identical, larger = more different
```

For L2-normalized vectors, minimizing L2 distance is equivalent to maximizing cosine similarity: d²(a,b) = 2 - 2·cos(a,b). So the choice between cosine and L2 does not matter when vectors are normalized.

### When the Choice Matters

The choice only matters when vectors are **not** normalized. If your embeddings encode magnitude as meaningful information (e.g., popularity-weighted embeddings where more popular items have larger norms), use dot product or L2 instead of cosine. Cosine discards magnitude by design.

---

## 🧮 Flat Index (Brute Force)

Compare the query to every stored vector. This is the baseline — 100% recall, maximum latency.

```
🧮 Flat search:

    For query q and database of D vectors, each of dimension d:
      scores[i] = dot(q, database[i])   for i = 1..D
      return top-k(scores)

    Time:  O(D × d)
    Memory: O(D × d × sizeof(float))
```

**Real numbers:** D = 1M vectors, d = 768, float32:
- Memory: 1M × 768 × 4 bytes = 3.07 GB
- Time: 1M × 768 multiply-adds per query = ~768M FLOPs
- On CPU: ~50ms per query. On GPU: ~2ms per query.

Flat search is the ground truth. Use it for evaluation (to know what the "correct" top-k is) and for datasets under 10K vectors.

---

## 🧮 IVF (Inverted File Index)

Partition vectors into clusters using k-means. At query time, find the closest cluster centroids, then search only within those clusters.

### Construction

```
🧮 IVF construction:

    1. Run k-means on all D vectors to produce C cluster centroids
       Time: O(D × C × d × n_iterations)
       Typical: C = √D (e.g., 1000 clusters for 1M vectors), n_iterations = 20

    2. Assign each vector to its nearest centroid
       Time: O(D × C × d)

    3. Store vectors grouped by cluster (inverted lists)
```

### Search

```
🧮 IVF search:

    1. Compare query to all C centroids → O(C × d)
    2. Select top nprobe centroids
    3. Search all vectors in those nprobe clusters → O(nprobe × D/C × d)

    Total: O(C × d + nprobe × D/C × d)
```

### nprobe Analysis

nprobe is the number of clusters searched. It controls the recall-latency trade-off:

| nprobe | Clusters searched | Recall@10 (typical) | Relative latency |
|--------|------------------|--------------------|--------------------|
| 1 | 1 of 1000 | 40-60% | 1× |
| 5 | 5 of 1000 | 70-85% | 5× |
| 10 | 10 of 1000 | 85-93% | 10× |
| 50 | 50 of 1000 | 95-98% | 50× |
| 1000 | All (= brute force) | 100% | 1000× |

🎯 **Key insight:** nprobe = C recovers brute force search exactly. The recall curve is concave — going from nprobe=1 to nprobe=10 gains 30+ percentage points of recall, but going from 50 to 100 gains only 1-2 points.

**When IVF fails:** If a relevant vector sits near the boundary between two clusters, it may be assigned to the "wrong" cluster and missed when nprobe is low. The failure rate increases when clusters are imbalanced (some clusters have 10× more vectors than others).

---

## 🧮 HNSW (Hierarchical Navigable Small World)

HNSW is the dominant ANN algorithm in production. It builds a multi-layer graph where similar vectors are connected, and search navigates the graph from coarse to fine.

### Construction

```
🧮 HNSW construction:

    Parameters:
      M     = max connections per node per layer (typically 16-64)
      M_max = max connections at layer 0 (typically 2M)
      ef_construction = search width during construction (typically 200)

    For each new vector v:
      1. Assign v to layer l = floor(-log(random()) × m_L)
         where m_L = 1/ln(M)
         → Most vectors go to layer 0. Higher layers have exponentially fewer nodes.

      2. Starting from the entry point at the top layer:
         - At each layer above l: greedy search to find the closest node
         - At each layer from l down to 0: search with ef_construction width
           to find M nearest neighbors, connect v to them

    Time per insertion: O(M × ef_construction × log(D))
    Total build time: O(D × M × ef_construction × log(D))
```

### Search

```
🧮 HNSW search:

    Parameters:
      ef_search = search width (controls recall-speed trade-off)

    1. Start at entry point (top layer)
    2. At each layer above 0: greedy search → find closest node
    3. At layer 0: beam search with width ef_search
       → Visit ef_search candidates, expanding to their neighbors
    4. Return top-k from the ef_search candidates

    Time: O(ef_search × M × log(D) × d)
    Memory: O(D × (d × sizeof(float) + M × sizeof(int)))
```

### Why HNSW Works

The hierarchical structure creates a "highway system." The top layer has few nodes far apart — like an interstate highway that gets you to the right region quickly. Each lower layer has more nodes — like local roads that get you to the exact destination. The search starts on the highway and progressively zooms in.

**M controls the graph density.** Higher M means more connections per node. More connections → better recall (more paths to the right answer) but more memory and slower search (more neighbors to check at each step).

**ef_search controls the beam width.** Higher ef_search means exploring more candidates at layer 0. This increases recall but takes longer. ef_search = k is the minimum (only keep exactly k candidates). ef_search = 500 gives near-perfect recall but is 10× slower.

| ef_search | Recall@10 | QPS (queries/sec) |
|-----------|-----------|-------------------|
| 10 | 85% | 10,000 |
| 50 | 95% | 3,000 |
| 100 | 98% | 1,500 |
| 500 | 99.5% | 300 |

---

## 🧮 Product Quantization (PQ)

PQ compresses vectors to reduce memory and speed up distance computation.

### The Algorithm

```
🧮 Product Quantization:

    1. Split each d-dimensional vector into m sub-vectors of dimension d/m
       Example: d=768, m=48 → 48 sub-vectors of 16 dimensions each

    2. For each sub-space, run k-means to learn k=256 centroids
       (256 = 2⁸, so each centroid ID fits in 1 byte)

    3. Replace each sub-vector with the ID of its nearest centroid
       → Each vector compressed from d × 4 bytes to m × 1 byte

    Compression ratio: (d × 4) / m = (768 × 4) / 48 = 64×
```

### Distance Computation with PQ

```
🧮 Approximate distance using PQ codes:

    d(q, x_pq) ≈ Σⱼ ‖qⱼ - centroid[j][code[j]]‖²

    Where:
      qⱼ      = j-th sub-vector of the query (not quantized)
      code[j]  = PQ code for j-th sub-space of vector x
      centroid[j][code[j]] = the centroid that code[j] points to

    Precompute a distance table: for each sub-space j, compute distances
    from qⱼ to all 256 centroids. Table size: m × 256 × sizeof(float).
    Then each distance lookup is just a table read + addition.

    Time per distance: O(m) table lookups (instead of O(d) multiplications)
```

### Memory Savings

| Configuration | Memory per vector | 1M vectors |
|--------------|------------------|------------|
| float32, d=768 | 3,072 bytes | 3.07 GB |
| PQ, m=48, k=256 | 48 bytes | 48 MB |
| PQ, m=96, k=256 | 96 bytes | 96 MB |
| PQ, m=24, k=256 | 24 bytes | 24 MB |

**Distortion:** PQ introduces approximation error. Each sub-vector is snapped to its nearest centroid, losing information. The distortion is bounded by the k-means quantization error in each sub-space. Finer sub-spaces (larger m, larger k) reduce distortion but increase memory.

**Typical recall impact:** PQ alone drops recall@10 by 5-15% compared to exact search. Combined with IVF (IVF-PQ), the total recall loss is 10-20%. This is acceptable when memory constraints are binding (billions of vectors).

---

## ⚠️ Failure Modes

### 1. Curse of Dimensionality

In high dimensions (d > 100), the ratio of the nearest neighbor distance to the average distance converges to 1. All vectors become approximately equidistant. This means the "nearest" neighbor is barely closer than a random vector, making the search problem fundamentally harder.

**Impact:** ANN algorithms like HNSW and IVF maintain good recall because they exploit the structure of real embeddings (which lie on a low-dimensional manifold within the high-dimensional space). But if your embeddings are poorly trained and genuinely fill the high-dimensional space uniformly, ANN recall will degrade.

**Detection:** Compute the ratio: (distance to 10th neighbor) / (distance to 1st neighbor). If this ratio is > 0.95, your effective dimensionality is very high and ANN will struggle.

### 2. Index Staleness

When documents are added or updated, the vector index becomes stale. For IVF, new vectors must be assigned to existing clusters — the clusters do not adapt. After enough updates, cluster centroids no longer represent their members well, and recall degrades.

**Detection:** Monitor recall@10 over time using a fixed eval set. A 5% drop signals index staleness.

**Fix:** Periodic index rebuild. For IVF, rebuild clusters every time the corpus grows by 20-30%. For HNSW, the graph structure is more resilient to insertions — it supports online insertion natively — but deletions leave "holes" in the graph that degrade search quality over time.

### 3. Metadata Filter + ANN Interaction

A common production pattern: retrieve the top 100 vectors, then filter by metadata (date, category, source). If only 5 of the 100 pass the filter, you effectively have recall@5 from the vector search, which may be very low.

**The problem:** ANN search and metadata filtering are separate operations. The ANN search does not know about the filter, so it returns the globally nearest vectors, not the nearest vectors that satisfy the filter.

**Fix:** Pre-filtering (apply metadata filter before ANN search) or integrated filtering (vector databases like Qdrant and Weaviate support filtered HNSW search that prunes during graph traversal). Pre-filtering is simpler but can be slow if the filter is highly selective (only 1% of vectors pass).

### 4. Embedding Version Drift

When you upgrade your embedding model, the new embeddings are in a different vector space. Old embeddings and new embeddings are incompatible — cosine similarity between them is meaningless. You must re-embed the entire corpus.

**Impact:** A model upgrade on a 10M-vector corpus requires 10M forward passes through the new model, which takes hours to days.

**Fix:** Blue-green deployment for embedding upgrades. Build the new index in parallel while the old one serves traffic. Swap atomically when the new index is ready. Never mix embeddings from different models in the same index.

---

## 📊 Complexity Summary

| Algorithm | Build Time | Query Time | Memory | Recall@10 |
|-----------|-----------|------------|--------|-----------|
| Flat | O(D × d) (just store) | O(D × d) | O(D × d) | 100% |
| IVF | O(D × C × d × iters) | O(C×d + nprobe×D/C×d) | O(D × d + C × d) | 85-98% |
| HNSW | O(D × M × ef × log D) | O(ef × M × log D × d) | O(D × (d + M)) | 95-99% |
| PQ | O(D × m × k × iters) | O(D × m) (with IVF: much less) | O(D × m) | 85-95% |
| IVF-PQ | O(D × C × d × iters + D × m × k × iters) | O(C×d + nprobe×D/C×m) | O(D × m + C × d) | 80-93% |

Real numbers for 1M vectors, d=768:

| Algorithm | Build Time | Query Latency | Memory |
|-----------|-----------|---------------|--------|
| Flat | ~10s | ~50ms (CPU) | 3.07 GB |
| IVF (C=1000, nprobe=10) | ~60s | ~5ms | 3.1 GB |
| HNSW (M=16, ef=100) | ~300s | ~1ms | 3.5 GB |
| PQ (m=48) | ~120s | ~15ms | 48 MB |
| IVF-PQ (C=1000, m=48) | ~180s | ~2ms | 52 MB |

---

## 💡 Design Trade-offs: Choosing an Indexing Algorithm

| Scenario | Best Algorithm | Why |
|----------|---------------|-----|
| < 10K vectors | Flat | Fast enough, 100% recall |
| 10K-1M vectors, latency matters | HNSW | Best recall-latency Pareto front |
| 10K-1M vectors, memory limited | IVF-PQ | 60× compression, acceptable recall |
| 1M-100M vectors, general purpose | HNSW | Sub-millisecond queries, high recall |
| 100M+ vectors, memory constrained | IVF-PQ | Only feasible option at this scale |
| Need exact recall (evaluation) | Flat | Ground truth baseline |

### Choosing a Vector Database

| Need | Recommendation | Why |
|------|---------------|-----|
| Learning / prototyping | FAISS or ChromaDB | Zero infrastructure, runs locally |
| Production, managed | Pinecone | No ops burden, scales automatically |
| Production, self-hosted | Qdrant | Strong filtering, Rust performance |
| Production, hybrid search | Weaviate | Native BM25 + vector search |
| Already using PostgreSQL | pgvector | No new infrastructure |
| Billions of vectors | Milvus | Distributed architecture, sharding |

---

## 🏭 Production Considerations

### Sharding and Replication

At scale (>10M vectors), a single node cannot hold the full index in memory. Sharding splits the index across multiple nodes, each holding a subset of vectors. Query routing broadcasts the query to all shards and merges results.

**Sharding strategy:** Hash-based (vector ID mod N) for uniform distribution, or attribute-based (shard by document category) for filtered search efficiency.

**Replication:** Each shard has 2-3 replicas for fault tolerance and read throughput. Writes go to the primary; reads are distributed across replicas.

### Index Rebuild Strategy

HNSW supports online insertions but not deletions. Deleted vectors are tombstoned and excluded from results, but their graph edges remain, wasting memory and slowing traversal. After 10-20% of vectors are deleted, rebuild the index.

IVF clusters drift as the data distribution changes. Rebuild clusters when corpus grows by 20-30% or when recall monitoring shows degradation.

**Schedule:** Weekly index rebuilds during low-traffic hours for actively changing corpora. Monthly for stable corpora.

### Embedding Version Management

Track which embedding model version produced each vector. Never mix model versions in the same index. During upgrades:

1. Build a new index with the new model in parallel (shadow mode)
2. Run both old and new indexes simultaneously, comparing results
3. When the new index passes quality checks, swap traffic
4. Decommission the old index

Store the model version as index metadata. If you need to roll back, you can re-activate the old index immediately.

---

## Staff/Principal Interview Depth

---

**Q1: Explain how HNSW works. Walk through the construction and search algorithms, and analyze the complexity.**

---
**No Hire**
*Interviewee:* "HNSW is a graph-based algorithm that connects similar vectors. You search by hopping between connected nodes until you find the closest one."
*Interviewer:* The candidate knows it is graph-based but cannot explain the hierarchical structure, the construction process, or the search algorithm. "Hopping between nodes" could describe any graph search. No complexity analysis.
*Criteria — Met:* Graph-based awareness. *Missing:* Hierarchical structure, construction algorithm, search algorithm, complexity analysis.

**Weak Hire**
*Interviewee:* "HNSW has multiple layers. The top layer has few nodes and provides long-range connections. Lower layers have more nodes for fine-grained search. During search, you start at the top and greedily navigate to the closest node at each layer, then drop down. The bottom layer has all nodes, and you do a more thorough search there. It is O(log N) per query."
*Interviewer:* Correct high-level description. The candidate understands the hierarchical structure and the top-down search strategy. What is missing: how nodes are assigned to layers, how edges are created during construction, what M and ef parameters control, and the full complexity expression (O(log N) is incomplete — it hides the dependence on M, ef, and d).
*Criteria — Met:* Hierarchical structure, top-down search, approximate complexity. *Missing:* Layer assignment, edge construction, parameter roles, full complexity.

**Hire**
*Interviewee:* "Construction: each new vector is assigned to a random layer using an exponential distribution — l = floor(-ln(random()) × m_L) where m_L = 1/ln(M). Most vectors go to layer 0; higher layers have exponentially fewer nodes. When inserting vector v at layer l, the algorithm searches from the entry point downward. At layers above l, it does a greedy search to find the closest node. At layers l down to 0, it searches with a wider beam (ef_construction) to find M nearest neighbors and connects v to them bidirectionally. Search: start at the entry point, greedy search at each layer above 0, then beam search at layer 0 with width ef_search. The beam keeps track of the ef_search best candidates seen so far. Complexity: each layer has roughly D^(1-l/L_max) nodes. The search visits O(ef × M) nodes at layer 0 and O(M) nodes at each upper layer. With O(log D) layers, total search is O(ef_search × M × log D × d). Memory: O(D × (d × sizeof(float) + M × sizeof(int))) — storing both the vectors and the graph edges."
*Interviewer:* Strong. The candidate walks through both construction and search with parameter names, gives the layer assignment formula, and provides the correct complexity expressions. What would push to Strong Hire: explaining why the exponential layer assignment creates the right density at each level, discussing the recall-latency Pareto frontier as ef_search varies, and comparing to skip lists.
*Criteria — Met:* Layer assignment formula, construction algorithm, search algorithm, complexity with all parameters, memory analysis. *Missing:* Why exponential distribution, Pareto frontier, skip list connection.

**Strong Hire**
*Interviewee:* "HNSW is based on the navigable small world graph model, combined with a hierarchical structure inspired by skip lists. The key insight from skip lists: by maintaining multiple levels with geometrically decreasing node counts, you can search in O(log N) by starting coarse and refining. HNSW applies this to similarity search in vector space. Construction: layer assignment uses l = floor(-ln(uniform(0,1)) / ln(M)). This creates an exponential distribution — layer 0 has all D nodes, layer 1 has D/M nodes, layer 2 has D/M² nodes. The parameter M controls both the branching factor and the probability of assignment to higher layers. When inserting vector v at layer l: at layers above l, greedy search finds the single nearest neighbor. At layer l and below, a beam search with width ef_construction finds up to M candidates, and v is connected to the M nearest. If a node already has M_max connections at layer 0, the algorithm prunes the farthest connection to keep the graph sparse. This pruning is critical — without it, popular nodes accumulate too many edges and search degrades because the beam wastes time checking a popular hub's many neighbors. Search: entry point at the top layer. At each layer above 0, greedy search to find the nearest node. At layer 0, beam search with width ef_search: maintain a priority queue of candidates and a set of visited nodes. At each step, expand the closest unvisited candidate by checking its M neighbors. Continue until the priority queue's best candidate is farther than the ef_search-th candidate. Return top-k from the ef_search candidates. The Pareto frontier between recall and QPS is controlled by ef_search: at ef_search = k, you get minimum recall but maximum speed. As ef_search increases, recall approaches 100% but QPS drops linearly. The practical operating point for most production systems is ef_search ≈ 4-8× k, which gives 95-98% recall. Memory: O(D × d × 4) for vectors + O(D × M × 4 × 2) for bidirectional edges ≈ D × (4d + 8M) bytes. For d=768, M=16: 3200 bytes per vector, so 1M vectors ≈ 3.2GB. The edges add only 128 bytes per vector — negligible compared to the vectors themselves. Build time is the real cost: O(D × ef_construction × M × log D × d). For D=1M, ef_construction=200, M=16, this is a few minutes on GPU. For 100M vectors, build time is hours — which is why online insertion (adding one vector at a time) is important for production systems that grow continuously."
*Interviewer:* Comprehensive staff-level answer. The candidate connects HNSW to skip lists, derives the exponential layer distribution from the parameter M, explains the pruning mechanism and why it matters, gives the full Pareto frontier analysis, calculates memory with real numbers, and addresses the build-time concern for large-scale production. The pruning discussion is the detail that separates someone who has read the paper from someone who has only read a tutorial.
*Criteria — Met:* Skip list connection, exponential distribution derivation, construction with pruning, search with priority queue, Pareto frontier, memory calculation, build time scaling, production online insertion.

---

**Q2: What is Product Quantization and how does it trade accuracy for memory? Derive the compression ratio and distortion bound.**

---
**No Hire**
*Interviewee:* "Product Quantization compresses vectors to save memory. It makes vectors smaller so you can store more of them."
*Interviewer:* Correct at the surface but no mechanism explained. The candidate cannot describe how PQ works, what the compression ratio is, or what accuracy loss occurs.
*Criteria — Met:* Purpose awareness. *Missing:* Algorithm, compression mechanism, distortion, trade-off analysis.

**Weak Hire**
*Interviewee:* "PQ splits each vector into sub-vectors and replaces each sub-vector with the ID of its nearest centroid from a learned codebook. Each ID is 1 byte (256 centroids), so a 768-dimensional vector becomes 48 bytes if split into 48 sub-vectors. That is a 64× compression."
*Interviewer:* Correct compression mechanism and ratio calculation. The candidate understands the algorithm at a high level. What is missing: how distances are computed on PQ codes, the distortion analysis, and when the accuracy loss becomes unacceptable.
*Criteria — Met:* Algorithm description, compression ratio. *Missing:* Distance computation, distortion bound, accuracy trade-off.

**Hire**
*Interviewee:* "PQ works in three steps. Step 1: split each d-dimensional vector into m sub-vectors of d/m dimensions. Step 2: for each sub-space, run k-means with k=256 centroids (so each centroid ID fits in 1 byte). Step 3: replace each sub-vector with the 1-byte centroid ID. Compression ratio: original is d × 4 bytes (float32), compressed is m bytes. For d=768, m=48: compression = 3072/48 = 64×. Distance computation: you do not decompress. For a query q, precompute a distance table: for each sub-space j, compute distances from qⱼ to all 256 centroids. Table size: m × 256 = 12,288 entries. Then for each stored vector, look up m table entries and sum them. This is O(m) per distance instead of O(d) — a d/m speedup. The distortion: each sub-vector is approximated by its nearest centroid, introducing quantization error. The total distortion is the sum of quantization errors across sub-spaces. More sub-spaces (larger m) means smaller sub-vectors, which means each sub-space has lower quantization error. But larger m also means more bytes per vector. The typical recall impact: PQ alone drops recall@10 by 5-15% compared to exact search."
*Interviewer:* Strong. The candidate walks through the algorithm, derives the compression ratio, explains the distance table optimization, and gives recall numbers. What would push to Strong Hire: deriving the distortion bound formally, discussing the interaction between m and k, and analyzing when PQ becomes the only feasible option.
*Criteria — Met:* Full algorithm, compression ratio derivation, distance table, recall impact. *Missing:* Formal distortion bound, m vs k trade-off, scale threshold analysis.

**Strong Hire**
*Interviewee:* "PQ splits each d-dimensional vector into m sub-vectors of d' = d/m dimensions. Within each sub-space, a codebook of k centroids is learned via k-means. Each sub-vector is replaced by its nearest centroid index (log₂(k) bits). The total code length is m × log₂(k) bits = m bytes when k=256. Compression ratio: d × 32 bits / (m × 8 bits) = 4d/m. For d=768, m=48: ratio = 64×. For d=768, m=96: ratio = 32×. The distortion bound comes from quantization theory. The mean squared error of PQ is the sum of k-means quantization errors across sub-spaces: E[‖x - x̂‖²] = Σⱼ E[‖xⱼ - cⱼ(xⱼ)‖²], where cⱼ is the nearest centroid mapping in sub-space j. The k-means error in each sub-space depends on the sub-space dimension d' and the number of centroids k. For uniformly distributed data in d' dimensions with k centroids, the expected quantization error scales as O(k^(-2/d')). This means that lower sub-space dimensions (larger m, more splits) gives better quantization per sub-space, but you have more sub-spaces. The total error Σⱼ O(k^(-2/d')) = m × O(k^(-2/(d/m))) = m × O(k^(-2m/d)). The trade-off: increasing m (more sub-spaces) decreases error per sub-space but increases the code length. The sweet spot depends on the memory budget. For distance computation: precompute an asymmetric distance table (ADC). For query q, compute ‖qⱼ - cⱼ[i]‖² for all j ∈ [1,m] and i ∈ [1,k]. Table size: m × k entries. Then ‖q - x̂‖² ≈ Σⱼ table[j][code_x[j]], which takes m lookups and additions. This is called 'asymmetric' because q is not quantized — only the database vectors are. Asymmetric distance computation (ADC) is strictly more accurate than symmetric (SDC, where both q and x are quantized) at the cost of computing the distance table per query. When PQ is the only option: at billions of vectors. 1B vectors × 768 dims × 4 bytes = 3 TB. No single machine can hold this in memory. With PQ at m=48, the same dataset is 48 GB — fits on a single GPU. This is why every billion-scale vector search system (Spotify, Meta, Google) uses PQ or a variant."
*Interviewer:* Exceptional. The candidate derives compression and distortion from first principles, explains the ADC/SDC distinction, connects to k-means quantization theory, and gives real-world scale examples. The distortion scaling analysis (k^(-2m/d)) shows mathematical maturity. The "when PQ is the only option" framing with TB-scale numbers demonstrates production awareness.
*Criteria — Met:* Algorithm, compression derivation, distortion bound, ADC vs SDC, distortion scaling formula, m vs k trade-off, billion-scale motivation.

---

**Q3: How do metadata filters interact with approximate nearest neighbor search? What goes wrong, and how do production systems solve it?**

---
**No Hire**
*Interviewee:* "You can add filters to vector search to only return results that match certain criteria, like a date range or category."
*Interviewer:* The candidate knows filtering exists but does not understand the architectural challenge. The question is about what goes wrong, not what filtering is.
*Criteria — Met:* Awareness of filtering. *Missing:* The interaction problem, failure mode, solutions.

**Weak Hire**
*Interviewee:* "If you filter after retrieving the top-k results, you might filter out most of them and end up with very few results. For example, retrieving top-100 then filtering by category might leave only 3 results, which means your effective recall is based on those 3, not 100."
*Interviewer:* Correct identification of the post-filtering problem. The candidate understands the recall degradation. What is missing: pre-filtering challenges, integrated filtering approaches, and when each strategy works.
*Criteria — Met:* Post-filtering problem. *Missing:* Pre-filtering challenges, integrated filtering, strategy selection.

**Hire**
*Interviewee:* "There are three approaches, each with trade-offs. Post-filtering: run ANN search, then filter results. Fast but recall degrades when the filter is selective — if only 1% of vectors match the filter, you need to retrieve top-10,000 to get 100 matching results, which is slow and defeats the purpose of ANN. Pre-filtering: apply the filter first to get the set of matching vector IDs, then search only within that set. Accurate but requires either a separate inverted index for metadata or scanning all vectors for the filter condition. For highly selective filters on large datasets, the set of matching IDs can be small enough that brute-force search within the set is fast. Integrated filtering: some vector databases (Qdrant, Weaviate) support filtering during HNSW traversal. The search algorithm skips nodes that do not match the filter during graph navigation. This is the best approach but requires database-level support and can degrade graph connectivity (if most neighbors are filtered out, the search has fewer paths to follow). The choice depends on filter selectivity: high selectivity (< 5% pass) → pre-filter then brute-force. Medium selectivity → integrated filtering. Low selectivity (> 50% pass) → post-filter."
*Interviewer:* Strong three-approach answer with clear trade-offs and a decision rule. What would push to Strong Hire: discussing how HNSW graph structure affects integrated filtering, and the engineering complexity of maintaining metadata indexes alongside vector indexes.
*Criteria — Met:* Three approaches with trade-offs, selectivity-based decision rule, database examples. *Missing:* HNSW graph interaction, metadata index maintenance.

**Strong Hire**
*Interviewee:* "The core problem: ANN search and metadata filtering have conflicting objectives. ANN searches in vector space — it finds the geometrically nearest points. Metadata filtering searches in attribute space — it finds points matching a predicate. There is no guarantee that the geometrically nearest vectors satisfy the predicate. This creates three strategies, each failing in a different regime. Post-filtering: retrieve top-N from ANN, filter by metadata. Fails when filter selectivity is high (< 5% match): you need N = k / selectivity to get k results, which for 1% selectivity means N = 10,000 for k=100. At N=10,000, HNSW search is no longer O(log D) — it degrades toward brute force. Pre-filtering: build a bitmap of matching vector IDs, then search only within the bitmap. Fails when the matching set is large: if 50% of vectors match, you are doing brute-force on half the database with no ANN acceleration. Integrated filtering: modify the HNSW search to skip non-matching nodes during traversal. This is the best general approach but has a subtle failure mode. HNSW's graph connectivity relies on the assumption that all nodes participate in routing. When you filter out 90% of nodes, the remaining 10% may not form a well-connected subgraph. The search can get stuck in a local minimum because all connecting paths pass through filtered-out nodes. Qdrant's solution: during construction, build additional edges between nodes that share common metadata values, creating 'filtered subgraphs' within the HNSW graph. Weaviate's solution: maintain per-filter HNSW indexes for common filter combinations, falling back to post-filtering for rare combinations. The engineering complexity: you need a metadata index (inverted index on attribute values) that stays synchronized with the vector index. Every insert, update, and delete must update both atomically. This is why integrated filtering is typically only available in purpose-built vector databases, not in libraries like FAISS."
*Interviewer:* Staff-level answer. The candidate frames the problem as a conflict between geometric and attribute search, explains all three strategies with their specific failure regimes, identifies the subtle graph connectivity problem with integrated filtering, and names real database solutions (Qdrant's filtered subgraphs, Weaviate's per-filter indexes). The atomicity requirement for metadata synchronization is a production engineering insight.
*Criteria — Met:* Problem framing, three strategies with failure regimes, HNSW graph connectivity issue, real database solutions, metadata synchronization requirement.

---

**Q4: You need to serve a vector search system with 100M vectors, sub-10ms latency at P99, and 99.9% uptime. Design the architecture.**

---
**No Hire**
*Interviewee:* "I would use Pinecone because it is managed and handles scaling automatically."
*Interviewer:* This is a vendor selection, not an architecture design. The candidate has not engaged with the technical challenges: memory requirements, sharding, replication, or the latency constraint.
*Criteria — Met:* Awareness of a managed solution. *Missing:* Architecture design, capacity planning, latency analysis.

**Weak Hire**
*Interviewee:* "100M vectors at 768 dimensions = ~300GB of raw vectors. This does not fit on a single machine's RAM. I would shard across multiple nodes — maybe 10 nodes each holding 10M vectors. Each node runs HNSW. For 99.9% uptime, I would replicate each shard 2-3 times. Queries are broadcast to all shards, and results are merged."
*Interviewer:* Correct capacity analysis and the right architectural components. What is missing: the latency analysis (can 10-shard broadcast + merge fit in 10ms?), the choice between HNSW and IVF-PQ for this scale, and the tail latency concern with fan-out.
*Criteria — Met:* Capacity analysis, sharding, replication, broadcast+merge. *Missing:* Latency analysis, algorithm selection, tail latency.

**Hire**
*Interviewee:* "Let me work through the requirements. Memory: 100M × 768 × 4 bytes = 307 GB for raw vectors. HNSW edges add ~10%, so ~340 GB total. This requires sharding. Latency: sub-10ms at P99. A single HNSW query on 10M vectors takes ~1ms. With 10 shards of 10M each, the query fans out to all 10 shards in parallel. P99 latency = max(P99 of each shard) + network overhead + merge time. Each shard at P99: ~2ms. Network round-trip: ~1ms within a datacenter. Merge 10 result lists: <1ms. Total: ~4ms — within budget. But fan-out to 10 shards means P99 is dominated by the slowest shard. To keep P99 < 10ms, each shard's P99.9 must be < 8ms, which requires careful GC tuning and dedicated resources. Replication: 3 replicas per shard. Read traffic distributed across replicas. If one replica fails, the other two serve traffic. With 10 shards × 3 replicas = 30 nodes. Each node: 34 GB RAM for vectors + overhead = 48-64 GB RAM. Alternatively: use IVF-PQ to compress vectors. 100M × 48 bytes = 4.8 GB — fits on a single machine. Query latency: ~5ms with nprobe=10. Recall drops to 85-90%, but the infrastructure is drastically simpler. The decision between HNSW-sharded and IVF-PQ depends on whether the application can tolerate 10-15% recall loss."
*Interviewer:* Strong. The candidate does explicit capacity planning, latency math, and presents the HNSW vs IVF-PQ trade-off as an architectural decision. What would push to Strong Hire: discussing read/write separation, index rebuild strategy, monitoring, and the hot-standby failover design for 99.9% uptime.
*Criteria — Met:* Capacity planning, latency math with fan-out, tail latency awareness, replication, HNSW vs IVF-PQ trade-off. *Missing:* Read/write separation, index rebuild, monitoring, failover design.

**Strong Hire**
*Interviewee:* "Requirements: 100M vectors, P99 < 10ms, 99.9% uptime (~8.7 hours downtime/year). Architecture: HNSW on sharded nodes with read/write separation. Capacity: 100M × 768 × 4 = 307 GB vectors. HNSW with M=16 adds 100M × 16 × 2 × 4 = 12.8 GB edges. Total: ~320 GB. Shard into 8 shards of 12.5M vectors each. Each shard: ~40 GB — fits in RAM on a 64 GB node. Replicate each shard 3× for both fault tolerance and read throughput. Total: 24 nodes. Latency analysis: HNSW on 12.5M vectors with ef=50: ~0.8ms per shard. Fan-out to 8 shards in parallel. P99 = max(P99 across 8 shards) + network + merge. With 8 shards, the effective P99 is roughly the P99.9 of a single shard (extreme value theory). Single-shard P99.9 ≈ 2ms (typical for HNSW). Network RTT within datacenter: 0.5ms. Merge 8 lists: 0.1ms. Total: ~2.6ms P99 — well within the 10ms budget. This leaves headroom for re-ranking if needed. Write path: writes go to a single primary replica per shard. The primary applies the write, then replicates to followers asynchronously. For HNSW, insertions are online — no rebuild needed. Deletes are tombstoned and cleaned up during compaction. Index rebuild strategy: HNSW graph quality degrades after ~20% of vectors are tombstoned (dead edges, wasted traversal). Schedule weekly compaction during off-peak hours: build a fresh graph from live vectors, swap atomically. Each shard rebuilds independently so the system is never fully down. 99.9% uptime design: each shard has 3 replicas across 3 availability zones. If one AZ goes down, 2 replicas remain. A load balancer health-checks each replica and routes around failures within seconds. Planned maintenance (index rebuild, model upgrades) is done one shard at a time with at least 2 replicas always serving. Monitoring: track P50/P95/P99 latency per shard, recall@10 against a weekly eval set, QPS per node, memory utilization, and tombstone ratio. Alert on: P99 > 5ms (still within budget but trending), recall drop > 3%, tombstone ratio > 15%. Alternative: if the team wants simpler infrastructure and can tolerate 10% recall loss, IVF-PQ compresses to 100M × 48 bytes = 4.8 GB — fits on a single beefy node (no sharding). Add 2 read replicas for fault tolerance. This is 3 nodes instead of 24, which is a massive operational simplification."
*Interviewer:* This is a systems architecture answer at the level expected of a staff engineer owning a production search service. The capacity planning, latency math with extreme value theory for fan-out, read/write separation, compaction strategy, AZ-aware replication, and monitoring with specific alert thresholds all demonstrate production ownership. Presenting the IVF-PQ alternative with the explicit recall/infrastructure trade-off shows architectural maturity — the candidate does not just design one system, they design the decision between two systems.
*Criteria — Met:* Full capacity planning, fan-out latency with extreme value theory, read/write separation, online insertion + compaction, AZ-aware replication, monitoring with alert thresholds, IVF-PQ alternative with trade-off analysis.

---

**Q5: Your vector database uses HNSW with M=16 and ef_search=100. Users report that recall has degraded over the past 3 months, but you have not changed the index parameters. What could be happening?**

---
**No Hire**
*Interviewee:* "Maybe the data changed and the index needs to be rebuilt."
*Interviewer:* Directionally correct but too vague to be actionable. The candidate does not distinguish between possible causes of degradation or propose diagnostic steps.
*Criteria — Met:* Awareness that data changes affect index quality. *Missing:* Specific degradation mechanisms, diagnostic approach, multiple hypotheses.

**Weak Hire**
*Interviewee:* "A few things could cause recall degradation without parameter changes. First, if many vectors have been deleted and tombstoned, the graph has dead edges that waste search steps. Second, if the data distribution has shifted — new documents are about different topics than the original corpus — the graph structure built on the old distribution does not serve the new data well. I would check the tombstone ratio and the age distribution of vectors."
*Interviewer:* Two valid hypotheses with reasonable diagnostics. The candidate understands both tombstoning and distribution shift. What is missing: embedding model version drift, query distribution shift (the vectors did not change but the queries did), and a structured diagnostic approach.
*Criteria — Met:* Tombstone hypothesis, distribution shift hypothesis, basic diagnostics. *Missing:* Embedding version, query distribution shift, structured debugging.

**Hire**
*Interviewee:* "I would investigate four hypotheses. Hypothesis 1: Tombstone accumulation. Deleted vectors leave dead edges in the graph. After 3 months of deletions, the tombstone ratio might be 15-30%. Dead edges cause the search to explore nodes that no longer exist, wasting the ef_search budget. Diagnostic: check the tombstone ratio. Fix: compact the index (rebuild without tombstones). Hypothesis 2: Data distribution shift. New vectors added over 3 months may be from a different distribution than the original data. The HNSW graph was built to connect vectors from the old distribution — new vectors may have poor connectivity. Diagnostic: compare the distribution of pairwise distances in the original vs new vectors. Fix: rebuild the index with all current vectors. Hypothesis 3: Embedding model degradation. If the embedding model is served via an API, the provider may have updated the model without notice. New embeddings may be in a different vector space. Diagnostic: re-embed a sample of old documents and compare cosine similarity between old and new embeddings. If they differ significantly, the model changed. Fix: re-embed the entire corpus with the current model. Hypothesis 4: Query distribution shift. The underlying recall of the index has not changed, but users are asking different types of questions that the index serves poorly. Diagnostic: compute recall@10 using the original eval set. If recall on the eval set is still high, the problem is query drift, not index degradation."
*Interviewer:* Strong four-hypothesis framework with diagnostics for each. What would push to Strong Hire: discussing the interaction between hypotheses (multiple can be true simultaneously), and proposing a monitoring system that would have caught this earlier.
*Criteria — Met:* Four hypotheses with diagnostics and fixes, structured debugging approach. *Missing:* Interaction between hypotheses, monitoring recommendations.

**Strong Hire**
*Interviewee:* "I would approach this systematically. First, separate 'has the index degraded?' from 'has the problem changed?' Run the original eval set (the one used when the index was first built). If recall on the eval set is still high, the index is fine — the problem is query distribution shift or user perception. If recall has dropped, the index itself has degraded. Assuming the index has degraded, I would investigate three mechanisms in order of likelihood. Mechanism 1: Tombstone accumulation. HNSW does not remove edges when vectors are deleted. After 3 months of CRUD operations, the tombstone ratio might be 15-30%. Each tombstoned node contributes M dead edges that the search explores but gets nothing from. With M=16 and 20% tombstones, roughly 20% of each search step is wasted, which is equivalent to reducing ef_search from 100 to ~80. Diagnostic: query the tombstone ratio directly (most vector DBs expose this metric). Fix: online compaction or full rebuild. Mechanism 2: Insert distribution shift. New vectors added over 3 months may cluster differently from the original data. HNSW builds edges based on the state of the graph at insertion time — early vectors get well-connected because the graph is small, late vectors may get suboptimal connections because the graph only found mediocre neighbors. If the new vectors represent a new topic cluster, they may be poorly connected to the rest of the graph. Diagnostic: measure recall@10 separately for vectors inserted in month 1 vs month 3. If month-3 recall is lower, insertion order effects are the cause. Fix: rebuild the index (all vectors get equal treatment). Mechanism 3: Embedding version drift. If embeddings are computed via an API (OpenAI, Cohere), the provider may have silently updated the model. Old and new embeddings are then in different vector spaces — cosine similarity between them is meaningless. Diagnostic: pick 10 documents, compute embeddings now and compare to the stored embeddings. If cosine similarity between old and new embeddings of the same document is < 0.95, the model changed. Fix: re-embed the entire corpus with the current model version. Going forward, I would add three monitoring checks: (1) weekly recall@10 on a fixed eval set, (2) tombstone ratio alert at 15%, (3) embedding consistency check (monthly re-embed 100 documents and compare to stored vectors). These three checks would have caught this degradation in week 2, not month 3."
*Interviewer:* Exceptional. The candidate starts by separating index degradation from problem shift — the correct first diagnostic step. The three mechanisms are ordered by likelihood and each has a specific diagnostic test. The insertion-order effect (month 1 vs month 3 recall) is a subtle HNSW-specific insight. The proactive monitoring recommendations close the loop: this is not just debugging, it is designing the system to prevent recurrence.
*Criteria — Met:* Structured diagnostic approach, three mechanisms with specific tests, insertion-order effect, embedding version drift detection, proactive monitoring with specific thresholds.

---

## Key Takeaways

🎯 1. **HNSW is dominant because it achieves 95-99% recall at sub-millisecond latency** — the hierarchical structure creates O(log D) search that exploits the manifold structure of real embeddings
   2. **IVF recall is controlled by nprobe** — the curve is concave, with nprobe=10 capturing most of the benefit
🎯 3. **Product Quantization gives 64× compression** at 5-15% recall cost — the only feasible option at billion-scale
⚠️ 4. **Metadata filtering and ANN search conflict** — post-filtering degrades recall, pre-filtering loses ANN speed, integrated filtering requires database support
   5. **HNSW does not handle deletions well** — tombstoned nodes waste search budget; compact when tombstone ratio exceeds 15%
🎯 6. **Never mix embedding model versions in the same index** — old and new embeddings are in different vector spaces
   7. **For L2-normalized vectors, cosine similarity = dot product** — most vector databases normalize at insert time
   8. **Sharding becomes necessary at ~50M+ vectors** — fan-out latency with extreme value theory determines the shard count
   9. **The distance table trick in PQ reduces per-vector comparison from O(d) to O(m)** — this is what makes PQ fast, not just small

---

[← Back to Vector Databases (Layer 1)](./vector-databases.md) | [Back to RAG module](./README.md)
