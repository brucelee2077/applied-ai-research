"""Quiz questions for Module 04: YouTube Video Search"""

QUESTIONS = [
    {
        "concept_id": "yvs_multimodal_embeddings",
        "module": "04-youtube-video-search",
        "question": "YouTube video search uses multimodal signals. Which combination provides the best search relevance?",
        "choices": [
            "A. Title text only",
            "B. Video frame embeddings + audio transcript + title/description text, fused into a joint embedding space",
            "C. Thumbnail image only",
            "D. View count + engagement signals"
        ],
        "correct": "B",
        "hint": "A video about 'how to cook pasta' has visual, audio, and text signals. Which one alone is most reliable?",
        "explanation": "Title text is noisy (clickbait). Thumbnails alone miss content. Audio transcripts capture the actual spoken content. Video frames capture visual content. Fusing all three in a joint embedding space (similar to CLIP's image-text alignment) gives the most robust representation of what a video is actually about.",
        "difficulty": 3,
        "tags": ["multimodal", "embeddings", "video_search"]
    },
    {
        "concept_id": "yvs_query_understanding",
        "module": "04-youtube-video-search",
        "question": "A user types 'how to' into YouTube search. How should the system interpret this ambiguous query?",
        "choices": [
            "A. Return an error asking for more specifics",
            "B. Use contextual signals: user watch history, location, trending topics, and session context to disambiguate and personalize the autocomplete and initial results",
            "C. Return the most popular 'how to' videos globally",
            "D. Return only the most recently uploaded 'how to' videos"
        ],
        "correct": "B",
        "hint": "Two users typing the same query might have very different intent based on their history.",
        "explanation": "Query understanding must fuse: (1) query text, (2) user context (watch history, subscriptions, demographics), (3) temporal signals (trending topics right now), (4) session context (what else they searched today). A fitness enthusiast typing 'how to' likely wants 'how to do a deadlift'; a cooking channel subscriber likely wants 'how to make pasta'.",
        "difficulty": 3,
        "tags": ["query_understanding", "personalization", "search"]
    },
    {
        "concept_id": "yvs_two_stage_retrieval",
        "module": "04-youtube-video-search",
        "question": "YouTube has 800M videos. What is the standard two-stage approach to return results in < 200ms?",
        "choices": [
            "A. Search all 800M videos with a heavy ranking model",
            "B. Stage 1: Recall — lightweight model retrieves ~1000 candidates from 800M. Stage 2: Ranking — heavy model re-ranks 1000 candidates to top 20.",
            "C. Use a single end-to-end transformer",
            "D. Cache the top-1000 results for all queries"
        ],
        "correct": "B",
        "hint": "A heavy model scoring 800M items per query at even 1µs/item would take 800 seconds.",
        "explanation": "Two-stage is the standard pattern for web-scale retrieval: Stage 1 (recall) uses a fast ANN index over pre-computed video embeddings to retrieve 100-10000 candidates in < 50ms. Stage 2 (ranking) applies a computationally expensive model (with many features including user context, fresh signals) to re-rank the small candidate set in < 100ms.",
        "difficulty": 2,
        "tags": ["retrieval", "ranking", "two_stage", "scale"]
    },
    {
        "concept_id": "yvs_watch_time_vs_clicks",
        "module": "04-youtube-video-search",
        "question": "YouTube optimized for CTR (clicks) in 2012 and saw engagement drop. What metric did they shift to and why?",
        "choices": [
            "A. Number of videos uploaded",
            "B. Watch time — CTR optimizes for clickbait thumbnails, watch time optimizes for actual user satisfaction and content quality",
            "C. Revenue per click",
            "D. Number of comments per video"
        ],
        "correct": "B",
        "hint": "A video with a misleading thumbnail gets many clicks but users immediately leave. What metric captures this?",
        "explanation": "CTR incentivizes thumbnails/titles that entice clicks regardless of content quality. YouTube found CTR optimization led to a race to the bottom in content quality. Watch time (and later, viewer satisfaction surveys) better proxies for whether users actually found the content valuable. This shift is a canonical example of Goodhart's Law: when a measure becomes a target, it ceases to be a good measure.",
        "difficulty": 2,
        "tags": ["metrics", "goodharts_law", "engagement"]
    },
    {
        "concept_id": "yvs_sparse_dense_retrieval",
        "module": "04-youtube-video-search",
        "question": "When does sparse retrieval (BM25/TF-IDF) outperform dense retrieval (embeddings) for video search?",
        "choices": [
            "A. Never — dense is always better",
            "B. For exact keyword matches — a rare product name (e.g., 'iPhone 15 Pro Max review') is better handled by BM25 which rewards exact term matches; dense models may miss rare tokens not well-represented in the embedding space",
            "C. When the query is longer than 10 words",
            "D. When the corpus is larger than 1B documents"
        ],
        "correct": "B",
        "hint": "What happens when a user searches for a very specific model number or person's name?",
        "explanation": "Dense retrieval excels at semantic similarity (paraphrase matching, intent understanding). Sparse retrieval excels at exact term matching — rare named entities, product SKUs, specific version numbers. Best-practice production systems hybrid both: dense retrieval for semantic queries + sparse retrieval for exact matches + learned fusion (e.g., RRF or learned rank fusion).",
        "difficulty": 4,
        "tags": ["sparse_retrieval", "dense_retrieval", "hybrid_search"]
    },
    {
        "concept_id": "yvs_position_bias",
        "module": "04-youtube-video-search",
        "question": "Your training data shows videos ranked #1 get 10x more clicks than videos ranked #5. How do you correct for position bias in your ranking model?",
        "choices": [
            "A. Ignore it — more clicks means the model is working",
            "B. Propensity score weighting, randomization experiments (randomly demote rank-1 to lower positions to observe counterfactual CTR), or use a bias correction model that separates relevance from position propensity",
            "C. Remove rank #5 videos from training data",
            "D. Normalize CTR by position"
        ],
        "correct": "B",
        "hint": "Is rank-1 video getting more clicks because it's more relevant, or because users always click first results?",
        "explanation": "Position bias means clicks are confounded with position — a video at rank 1 gets clicks regardless of quality. Solutions: (1) propensity scores: weight each training example by the inverse probability of being shown, (2) randomized controlled experiments: randomly swap positions to observe counterfactual, (3) twin-tower models with a separate position tower that models the propensity.",
        "difficulty": 5,
        "tags": ["position_bias", "debiasing", "counterfactual"]
    },
    {
        "concept_id": "yvs_realtime_freshness",
        "module": "04-youtube-video-search",
        "question": "A breaking news event happens. How does YouTube surface new relevant videos (uploaded minutes ago) with no engagement history?",
        "choices": [
            "A. Only surface videos with 1000+ views",
            "B. Use content-based signals (title, transcript, thumbnail) for zero-shot retrieval of new videos, boosted by freshness signals. Engagement signals are added as they accumulate.",
            "C. Manual curation by editors",
            "D. Wait 24 hours for engagement to accumulate"
        ],
        "correct": "B",
        "hint": "A new video has no click history. What signals DO you have immediately?",
        "explanation": "Cold-start for new videos: immediately available are title, description, transcript (auto-generated), thumbnail, channel metadata, upload time. These content signals allow relevance estimation before any engagement. Freshness boosting (higher score for recent uploads on trending queries) ensures new content surfaces. As engagement accumulates (minutes to hours), it's incorporated into re-ranking.",
        "difficulty": 3,
        "tags": ["cold_start", "freshness", "real_time"]
    },
    {
        "concept_id": "yvs_safe_search",
        "module": "04-youtube-video-search",
        "question": "How does YouTube's search system handle queries that could return harmful content?",
        "choices": [
            "A. Remove all such videos from the platform",
            "B. Query understanding classifies intent and applies content policy rules: harmful queries return vetted/authoritative sources, SafeSearch filters control content visibility by user preference and age",
            "C. Return results in a random order for sensitive queries",
            "D. Block all queries containing flagged keywords"
        ],
        "correct": "B",
        "hint": "Think about the difference between a blocked query and a query that redirects to authoritative sources.",
        "explanation": "Nuanced content moderation for search: (1) Query intent classification (informational vs. harmful intent for the same query), (2) for dangerous how-to queries: redirect to expert/authoritative sources, (3) age-restricted content: only visible to verified adult accounts, (4) SafeSearch filtering as user-controlled layer. Hard keyword blocking has poor precision and hurts legitimate queries.",
        "difficulty": 3,
        "tags": ["content_moderation", "safe_search", "query_understanding"]
    },
    {
        "concept_id": "yvs_internationalization",
        "module": "04-youtube-video-search",
        "question": "A user in Thailand searches in Thai for English-language content. How does the search system handle cross-lingual retrieval?",
        "choices": [
            "A. Only return Thai-language videos",
            "B. Cross-lingual embeddings (multilingual models like mBERT or LaBSE) map queries and documents into a shared multilingual space, enabling semantic matching across languages",
            "C. Machine translate the query to English first, then search",
            "D. Show global trending videos"
        ],
        "correct": "B",
        "hint": "Translation introduces errors and latency. What if the model could match semantics directly across languages?",
        "explanation": "Cross-lingual embeddings (LaBSE, mUSE, multilingual E5) are trained on parallel multilingual corpora and map semantically equivalent text in different languages to nearby points in a shared vector space. A Thai query and an English video about the same topic will have similar embeddings, enabling direct cross-lingual retrieval without translation errors.",
        "difficulty": 4,
        "tags": ["multilingual", "cross_lingual", "embeddings"]
    },
    {
        "concept_id": "yvs_evaluation_framework",
        "module": "04-youtube-video-search",
        "question": "What is the hierarchy of evaluation metrics for YouTube search quality?",
        "choices": [
            "A. Only A/B test CTR",
            "B. Offline (NDCG on human-labeled relevance judgments) → Online proxy (CTR, watch time per search session) → Business metric (DAU, time spent, ad revenue)",
            "C. Only use human rater scores",
            "D. View count is sufficient"
        ],
        "correct": "B",
        "hint": "Think about cost, speed, and what each metric actually measures.",
        "explanation": "Evaluation hierarchy: (1) Offline metrics (NDCG, MRR) on human-rated query-video relevance pairs — fast, cheap, no user risk. (2) Online A/B proxy metrics (CTR, avg watch time per search, zero-result rate) — real user behavior, hours to days. (3) Business metrics (DAU, overall time spent, subscription rate) — weeks to months. Each layer validates the previous and catches metrics gaming.",
        "difficulty": 3,
        "tags": ["evaluation", "metrics_hierarchy", "ndcg"]
    },
]
