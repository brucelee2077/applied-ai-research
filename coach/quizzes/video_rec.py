"""Quiz questions for Module 06: Video Recommendation"""

QUESTIONS = [
    {
        "concept_id": "vrec_two_tower_training",
        "module": "06-video-recommendation",
        "question": "In YouTube's two-tower recommendation model, what do the user tower and item tower each encode?",
        "choices": [
            "A. User tower: demographics; Item tower: video metadata only",
            "B. User tower: user history, demographics, context (time, device); Item tower: video content features, engagement statistics, metadata",
            "C. User tower: clicks; Item tower: video length",
            "D. Both towers encode the same features"
        ],
        "correct": "B",
        "hint": "Each tower must encode everything relevant to its side of the interaction.",
        "explanation": "User tower inputs: watch history sequence (encoded via bag-of-words or sequence model), search history, age, gender, device, time-of-day, geographic location. Item tower inputs: video title/description embeddings, audio/visual features, upload time, channel features, engagement rates. The towers produce embeddings whose dot product predicts interaction probability.",
        "difficulty": 2,
        "tags": ["two_tower", "architecture", "features"]
    },
    {
        "concept_id": "vrec_exploration_exploitation",
        "module": "06-video-recommendation",
        "question": "Pure exploitation (always recommend most likely to be clicked) leads to what failure mode?",
        "choices": [
            "A. The system works perfectly",
            "B. Filter bubble / echo chamber: users only see content similar to what they've seen before, never discovering new interests, leading to long-term engagement decline",
            "C. The system becomes too slow",
            "D. The model overfits to recent data"
        ],
        "correct": "B",
        "hint": "If you only recommend what users have watched before, what happens to their content diversity over time?",
        "explanation": "Pure exploitation creates a positive feedback loop where narrow interests are reinforced: if you watched one cooking video, you only get cooking videos → you only watch cooking videos → you only get cooking videos. Long-term this reduces engagement as users feel 'stuck in a loop'. Solution: deliberate exploration (epsilon-greedy, UCB, Thompson sampling, or learned exploration policies) to occasionally surface diverse content.",
        "difficulty": 3,
        "tags": ["exploration_exploitation", "filter_bubble", "diversity"]
    },
    {
        "concept_id": "vrec_implicit_feedback",
        "module": "06-video-recommendation",
        "question": "What is the problem with using 'watch' as the positive signal and 'not watch' as the negative signal?",
        "choices": [
            "A. There are too many positive signals",
            "B. Non-watch is noisy: users may not watch a video because they didn't see it (position bias), not because they dislike it — this conflates 'not shown' with 'not liked'",
            "C. Watch data is not available",
            "D. Binary labels are sufficient for recommendation"
        ],
        "correct": "B",
        "hint": "If a video was never shown to a user, is not watching it a negative signal?",
        "explanation": "Implicit feedback is noisy: absence of interaction could mean (1) user doesn't like it, (2) user never saw it, (3) user saw it but was distracted. Treating all non-watches as negatives conflates these signals. Solutions: use only items the user was exposed to (shown in feed) for negatives, use propensity scoring to account for position bias, or weight negatives by exposure probability.",
        "difficulty": 4,
        "tags": ["implicit_feedback", "negative_sampling", "bias"]
    },
    {
        "concept_id": "vrec_watch_time_modeling",
        "module": "06-video-recommendation",
        "question": "YouTube predicts expected watch time, not just click probability. Why?",
        "choices": [
            "A. Watch time is easier to predict",
            "B. Click probability can be gamed by clickbait; watch time is harder to fake and better correlates with user satisfaction and advertiser value",
            "C. Watch time prediction requires less data",
            "D. Clicks are not logged"
        ],
        "correct": "B",
        "hint": "A 30-second trailer gets lots of clicks but low watch time. Does that mean users are satisfied?",
        "explanation": "CTR optimization incentivizes clickbait. Watch time captures whether users actually valued the content. YouTube's 2012 paper 'Deep Neural Networks for YouTube Recommendations' explicitly describes the shift from CTR to weighted logistic regression on watch time. Watch time also correlates with ad revenue (more ads shown per watch).",
        "difficulty": 2,
        "tags": ["watch_time", "metrics", "optimization_target"]
    },
    {
        "concept_id": "vrec_real_time_features",
        "module": "06-video-recommendation",
        "question": "What features must be computed in real-time (< 100ms) vs. precomputed offline for a recommendation system?",
        "choices": [
            "A. All features should be real-time for maximum freshness",
            "B. Real-time: current session context (last 5 videos watched this session, current time), trending signals. Offline: user long-term embeddings, video content embeddings, engagement statistics.",
            "C. All features should be precomputed for low latency",
            "D. Only user demographics are needed at serving time"
        ],
        "correct": "B",
        "hint": "What changes in the last 5 minutes that a precomputed feature wouldn't capture?",
        "explanation": "Feature freshness tradeoff: user long-term embeddings change slowly (precompute nightly, read from feature store at serving). Video content embeddings are static (precompute once at upload). But session context (what the user just watched, their current mood/intent) changes by the minute and must be computed in real-time. Trend signals (spikes in video views in last 1 hour) also require near-real-time computation.",
        "difficulty": 3,
        "tags": ["feature_engineering", "real_time", "feature_store"]
    },
    {
        "concept_id": "vrec_cold_start_new_user",
        "module": "06-video-recommendation",
        "question": "A new user signs up with no watch history. How does the recommendation system handle this cold start?",
        "choices": [
            "A. Show the globally most popular videos",
            "B. Use onboarding signals (explicitly asked preferences), device/location as weak signals, collaborative filtering on similar new-user profiles, and quickly adapt based on first few interactions",
            "C. Show random videos to collect data",
            "D. Don't recommend until the user watches 10 videos"
        ],
        "correct": "B",
        "hint": "You have zero behavioral data but you do have some information. What is it?",
        "explanation": "Cold start mitigation: (1) Onboarding explicit feedback (ask 3 interest topics), (2) Demographic signals (location → language, time → timezone, device → demographics proxy), (3) Population-level priors (new users like this demographic tend to watch X), (4) Rapid online learning: after the first 1-2 interactions, update user embedding in real-time to reflect emerging interests.",
        "difficulty": 3,
        "tags": ["cold_start", "new_user", "onboarding"]
    },
    {
        "concept_id": "vrec_diversity_novelty",
        "module": "06-video-recommendation",
        "question": "How do you prevent your recommendation feed from showing 5 very similar cooking videos in a row?",
        "choices": [
            "A. Randomly shuffle the top-K ranked results",
            "B. Post-processing diversity re-ranking (MMR — Maximal Marginal Relevance): maximize both relevance to user AND diversity across selected items in a single pass",
            "C. Show only one video per category",
            "D. Increase the number of recommendations"
        ],
        "correct": "B",
        "hint": "You want the set of recommendations to be diverse, not just individually relevant.",
        "explanation": "Maximal Marginal Relevance (MMR): iteratively select the next item that maximizes relevance × (1 - max_similarity_to_already_selected_items). This balances exploration and exploitation within a single recommendation slate. Lambda parameter controls the relevance-diversity tradeoff. Determinantal Point Processes (DPP) provide a more principled probabilistic approach to diverse subset selection.",
        "difficulty": 4,
        "tags": ["diversity", "mmr", "re_ranking"]
    },
    {
        "concept_id": "vrec_feedback_loop",
        "module": "06-video-recommendation",
        "question": "Your recommendation model is trained on watch data which was generated by your previous recommendation model. What systemic problem does this create?",
        "choices": [
            "A. No problem — this is standard practice",
            "B. Feedback loop / exposure bias: the model only ever learns about videos it already surfaces (survivorship bias), and popular-content bias compounds over time — already-popular content gets recommended more, gets more watches, gets trained on more",
            "C. The model trains too slowly",
            "D. Users see too much variety"
        ],
        "correct": "B",
        "hint": "Can your model learn that a video it has never recommended might be great?",
        "explanation": "Closed feedback loop: Model A recommends videos → generates watch data → trains Model B → Model B recommends similar videos → same feedback. Content never recommended never accumulates training signal, making it invisible forever. Solutions: forced exploration (randomly surface non-recommended content), counterfactual learning (offline policy evaluation), inverse propensity scoring to debias the training data.",
        "difficulty": 5,
        "tags": ["feedback_loop", "exposure_bias", "debiasing"]
    },
    {
        "concept_id": "vrec_long_term_value",
        "module": "06-video-recommendation",
        "question": "Optimizing for immediate next-click maximizes short-term engagement but may hurt long-term user retention. How do you account for long-term value?",
        "choices": [
            "A. Only optimize for immediate clicks",
            "B. Incorporate long-term signals: user return rate (did they come back tomorrow?), session length trends, explicit satisfaction surveys, and consider RL-based approaches that optimize multi-step reward",
            "C. Long-term value cannot be measured",
            "D. Use a longer recommendation horizon (top-100 instead of top-10)"
        ],
        "correct": "B",
        "hint": "What signals indicate a user is getting long-term value vs. just clicking compulsively?",
        "explanation": "Short-term vs. long-term optimization tension: immediate CTR can be maximized by clickbait, but this reduces trust and long-term retention. Long-term value proxies: D30 return rate, weekly active rate, satisfaction survey scores, subscription rate. RL-based recommenders (Reinforcement Learning for Recommendations) model the recommendation as a sequential decision problem with delayed rewards.",
        "difficulty": 4,
        "tags": ["long_term_value", "rl", "retention"]
    },
    {
        "concept_id": "vrec_serving_architecture",
        "module": "06-video-recommendation",
        "question": "YouTube serves 2B+ users daily. Describe the serving architecture for recommendation inference.",
        "choices": [
            "A. One large model server handles all users",
            "B. Retrieval (ANN lookup over pre-computed item embeddings) → Scoring (features fetched from feature store, model inference on candidates) → Re-ranking/filtering → Response, with each stage horizontally scaled",
            "C. Pre-compute all recommendations offline nightly",
            "D. Use a single API call to a cloud ML endpoint"
        ],
        "correct": "B",
        "hint": "At 2B users with < 200ms latency, what can't you do?",
        "explanation": "Production recommendation serving: (1) User embedding lookup (feature store, < 5ms), (2) ANN retrieval — fast lookup over pre-computed video embeddings to get 100-1000 candidates (< 20ms), (3) Feature hydration — fetch real-time and cached features for candidates from feature store (< 30ms), (4) Scoring model — forward pass for all candidates (< 100ms, GPU), (5) Post-processing diversity + business rules (< 10ms). Total: < 200ms P99.",
        "difficulty": 4,
        "tags": ["serving_architecture", "latency", "scale"]
    },
]
