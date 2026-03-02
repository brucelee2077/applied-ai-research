"""Quiz questions for Module 02: Visual Search"""

QUESTIONS = [
    {
        "concept_id": "vs_contrastive_loss",
        "module": "02-visual-search",
        "question": "Why does visual search use contrastive loss (NT-Xent/triplet) rather than cross-entropy classification loss?",
        "choices": [
            "A. Cross-entropy requires discrete class labels; visual search needs continuous similarity over an open-world set",
            "B. NT-Xent is faster to compute than cross-entropy",
            "C. Cross-entropy cannot handle images as input",
            "D. NT-Xent produces smaller embedding vectors"
        ],
        "correct": "A",
        "hint": "What happens when a user searches for a product category never seen during training?",
        "explanation": "Classification requires a fixed closed set of classes. Visual similarity is open-world — two images of the same new product should be similar even if that product was never seen during training. Contrastive loss trains the embedding space so similar images cluster together regardless of class.",
        "difficulty": 3,
        "tags": ["loss_functions", "contrastive_learning"]
    },
    {
        "concept_id": "vs_ann_tradeoff",
        "module": "02-visual-search",
        "question": "You have 200B product images. Why is exact nearest neighbor search infeasible at query time?",
        "choices": [
            "A. It requires too much RAM to store the index",
            "B. O(N × D) time complexity makes query latency unacceptable — 200B × 256 dims = ~51 trillion ops per query",
            "C. Exact search produces lower quality results than ANN",
            "D. Exact search cannot handle float32 embeddings"
        ],
        "correct": "B",
        "hint": "Focus on query-time computation cost, not storage.",
        "explanation": "Exact search computes distance between the query and every one of the 200B embeddings. At 256 dimensions, this is ~51 trillion multiply-add operations per query — impossibly slow for a 100ms latency SLA. ANN methods (HNSW, IVF-PQ) trade a small recall drop for 1000x+ speedup.",
        "difficulty": 2,
        "tags": ["ann_search", "scale"]
    },
    {
        "concept_id": "vs_pq_compression",
        "module": "02-visual-search",
        "question": "Using PQ32x8 compression on 200B × 256-dim float32 embeddings, what is the approximate compressed index size?",
        "choices": [
            "A. 200 TB",
            "B. 50 TB",
            "C. 6 TB",
            "D. 1 TB"
        ],
        "correct": "C",
        "hint": "PQ32x8: 32 sub-quantizers × 8 bits = 32 bytes/vector. Original: 256 × 4 = 1024 bytes/vector.",
        "explanation": "Float32: 256 dims × 4 bytes = 1024 bytes/vector. PQ32x8: 32 bytes/vector (32× compression). 200B × 32 bytes = 6.4 TB. This fits on ~12 servers with 512 GB RAM, making the index servable.",
        "difficulty": 4,
        "tags": ["product_quantization", "scale", "memory"]
    },
    {
        "concept_id": "vs_two_tower",
        "module": "02-visual-search",
        "question": "In a visual search two-tower model, why are the query encoder and catalog encoder kept separate (not weight-shared)?",
        "choices": [
            "A. Two towers are faster to train",
            "B. The query (a single user photo) and catalog items (professional product images with different distributions) benefit from separate representation spaces",
            "C. Weight sharing causes gradient vanishing",
            "D. There is no reason — weight sharing is equally good"
        ],
        "correct": "B",
        "hint": "Think about what a user's phone photo looks like vs. a professional product photo on a white background.",
        "explanation": "User query photos: blurry, varied lighting, real-world context. Catalog items: clean backgrounds, consistent lighting, professional angles. These are different data distributions. Separate encoders allow each tower to specialize. Weight sharing would force both to share the same feature extraction, hurting both.",
        "difficulty": 3,
        "tags": ["two_tower", "architecture"]
    },
    {
        "concept_id": "vs_hard_negative_mining",
        "module": "02-visual-search",
        "question": "Why is hard negative mining critical for contrastive learning in visual search?",
        "choices": [
            "A. Hard negatives make training faster",
            "B. Random negatives are too easy (clearly different objects), giving near-zero gradient. Hard negatives (visually similar but semantically different) force the model to learn fine-grained discrimination",
            "C. Hard negatives reduce overfitting",
            "D. Hard negatives are required for the NT-Xent loss formula"
        ],
        "correct": "B",
        "hint": "What happens to the gradient when the model correctly identifies a random negative with 99.9% confidence?",
        "explanation": "With random negatives, the loss is near zero for most pairs (a shoe vs. a laptop is easy to distinguish). The gradient signal is dominated by these trivial pairs and the model stops improving. Hard negatives (a red Nike shoe vs. a red Adidas shoe) provide gradient signal that forces learning subtle discriminative features.",
        "difficulty": 4,
        "tags": ["contrastive_learning", "training", "hard_negative_mining"]
    },
    {
        "concept_id": "vs_embedding_freshness",
        "module": "02-visual-search",
        "question": "Your catalog has 200B items and 10M new products are added daily. How do you keep the ANN index fresh without full daily rebuilds?",
        "choices": [
            "A. Rebuild the entire HNSW index daily",
            "B. Use a dual-index strategy: a large static index for existing items + a small real-time index for new items, merged periodically",
            "C. Only index products older than 30 days",
            "D. Use exact search for new items only"
        ],
        "correct": "B",
        "hint": "A full rebuild of a 200B-item index takes days. What can you do for the small set of new items?",
        "explanation": "Full daily rebuilds of a 200B-item ANN index are infeasible (takes days, requires 6+ TB of memory). The delta approach: maintain the large static index and a small real-time index (only new/updated items). Query both in parallel, merge results. Periodically (weekly) rebuild the full index to consolidate.",
        "difficulty": 4,
        "tags": ["index_maintenance", "serving", "freshness"]
    },
    {
        "concept_id": "vs_recall_at_k",
        "module": "02-visual-search",
        "question": "What does Recall@10 = 0.65 mean for a visual search system?",
        "choices": [
            "A. 65% of the top-10 results are relevant",
            "B. For 65% of queries, at least one ground-truth similar item appears in the top-10 results",
            "C. The model correctly ranks 6.5 out of 10 items",
            "D. The precision of the model is 65%"
        ],
        "correct": "B",
        "hint": "Recall@K is a query-level metric, not an item-level metric.",
        "explanation": "Recall@K is computed per query: does at least one relevant item appear in the top K retrieved results? Recall@10 = 0.65 means 65% of test queries have at least one ground-truth match in the top 10 returned items. It measures whether the system is useful at all, not how well it ranks within the top K.",
        "difficulty": 3,
        "tags": ["metrics", "recall", "evaluation"]
    },
    {
        "concept_id": "vs_hnsw_vs_ivf",
        "module": "02-visual-search",
        "question": "When would you choose IVF-PQ over HNSW for your ANN index?",
        "choices": [
            "A. When you need higher recall",
            "B. When memory is the primary constraint — IVF-PQ with product quantization has much lower memory footprint than HNSW",
            "C. When query latency must be < 1ms",
            "D. When the embedding dimension is very high (> 1024)"
        ],
        "correct": "B",
        "hint": "HNSW stores full floating-point vectors. What does IVF-PQ store instead?",
        "explanation": "HNSW stores full float32 vectors + graph structure, requiring 1024+ bytes/vector for 256-dim embeddings. IVF-PQ uses product quantization to compress vectors to 32 bytes each, enabling 200B items in ~6 TB vs. ~200 TB for HNSW. Trade-off: IVF-PQ has lower recall. Use HNSW for smaller, high-precision indexes; IVF-PQ for massive-scale catalogs.",
        "difficulty": 4,
        "tags": ["hnsw", "ivf_pq", "ann_search", "memory"]
    },
    {
        "concept_id": "vs_clip_finetuning",
        "module": "02-visual-search",
        "question": "You start with CLIP embeddings for visual search. When does fine-tuning CLIP on your own data help most?",
        "choices": [
            "A. When your domain (e.g., fashion, home decor) has a different visual vocabulary than CLIP's training distribution",
            "B. CLIP fine-tuning always helps",
            "C. When your catalog is smaller than 1M items",
            "D. When you don't have labeled data"
        ],
        "correct": "A",
        "hint": "CLIP was trained on internet image-text pairs. How does a fashion catalog differ?",
        "explanation": "CLIP's generalist training may not capture domain-specific fine-grained distinctions (e.g., subtle texture differences between fabric types, neckline styles in fashion). Fine-tuning with domain-specific positive/negative pairs teaches it to embed these distinctions. Generic domains (dogs vs. cats) benefit less — CLIP already handles those well.",
        "difficulty": 3,
        "tags": ["clip", "fine_tuning", "embeddings"]
    },
    {
        "concept_id": "vs_adversarial_robustness",
        "module": "02-visual-search",
        "question": "An attacker adds imperceptible pixel noise to a product image, causing it to rank at the top of unrelated searches. What is this attack and how do you defend?",
        "choices": [
            "A. SQL injection — add input validation",
            "B. Adversarial perturbation against the embedding model — defend with adversarial training, input smoothing, and ensemble of models with different augmentation policies",
            "C. Data poisoning — retrain the model",
            "D. Model inversion — encrypt the embeddings"
        ],
        "correct": "B",
        "hint": "Small changes to pixel values that are invisible to humans but drastically change model outputs.",
        "explanation": "Adversarial perturbations exploit the continuous nature of embedding spaces. Defenses: (1) adversarial training (add perturbed images to training), (2) input smoothing/denoising before embedding, (3) ensemble of models trained with different augmentations (perturbations optimized against one model don't transfer well to others).",
        "difficulty": 5,
        "tags": ["adversarial_robustness", "security", "edge_cases"]
    },
]
