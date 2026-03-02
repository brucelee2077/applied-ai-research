"""Quiz questions for Module 10: Personalized News Feed"""

QUESTIONS = [
    {
        "concept_id": "nf_ranking_objective",
        "module": "10-personalized-news-feed",
        "question": "Facebook's news feed ranking went from chronological to ML-ranked. What is the primary objective function?",
        "choices": [
            "A. Maximize total posts shown",
            "B. Predict P(meaningful interaction) — a weighted combination of long-form engagement (likes, comments, shares, hide, 'see less') that proxies user value better than raw clicks",
            "C. Maximize ad revenue directly",
            "D. Rank by friend network size"
        ],
        "correct": "B",
        "hint": "Not all interactions are equal — a 'hide this post' is a very different signal from a 'like'.",
        "explanation": "Facebook's ranking signal is 'meaningful social interaction': weighted sum of positive signals (comments > shares > reactions > clicks) and negative signals (hide post, unfollow, report). Simple click maximization incentivizes outrage/clickbait content. Comments and shares indicate deeper engagement. 'Hide' is a strong negative signal. This multi-signal formulation is more aligned with long-term user value.",
        "difficulty": 3,
        "tags": ["ranking_objective", "meaningful_interaction", "optimization_target"]
    },
    {
        "concept_id": "nf_inventory_scoring",
        "module": "10-personalized-news-feed",
        "question": "A user's feed has 1500 candidate posts but only 200 are shown. Describe the efficient ranking pipeline.",
        "choices": [
            "A. Score all 1500 posts with the full ranking model",
            "B. Multi-stage: (1) Light model scores all 1500 candidates fast, (2) Medium model re-scores top 500, (3) Heavy model re-scores top 200 for final ranking — with each stage filtering and the heavy model only running on high-quality candidates",
            "C. Randomly select 200 posts",
            "D. Show only the 200 most recent posts"
        ],
        "correct": "B",
        "hint": "Running a heavy transformer model on all 1500 candidates would take too long. How do you balance quality and speed?",
        "explanation": "Multi-stage news feed ranking: (1) Fast scorer (logistic regression on sparse features) filters 1500 → 500 in < 10ms, (2) Medium scorer (shallow DNN) reduces 500 → 200 in < 30ms, (3) Heavy scorer (deep model with all features) ranks final 200 in < 100ms. This cascade allows expensive models to run on only the most promising candidates, achieving the quality of a heavy model at a fraction of the cost.",
        "difficulty": 3,
        "tags": ["multi_stage_ranking", "cascade", "efficiency"]
    },
    {
        "concept_id": "nf_content_understanding",
        "module": "10-personalized-news-feed",
        "question": "A post contains text, images, and a link to an article. How does the feed ranking model represent this post?",
        "choices": [
            "A. Use only the text",
            "B. Multimodal fusion: text embeddings (BERT-style), image embeddings (CNN/ViT), link article embeddings (headline + body text), combined with post metadata (author, engagement history) into a unified post representation",
            "C. Use only the image",
            "D. Use only engagement count"
        ],
        "correct": "B",
        "hint": "A post is more than its text — what other signals does it contain?",
        "explanation": "Post representation for news feed: text encoder processes caption/text, vision encoder processes attached images/videos, link encoder processes the linked article (headline, body, URL domain). These modality embeddings are fused (concatenated or cross-attention) with structured features (post_age, author_follower_count, post_type) into a unified post embedding. Multimodal representation is crucial for content understanding beyond text keywords.",
        "difficulty": 3,
        "tags": ["multimodal", "content_understanding", "post_representation"]
    },
    {
        "concept_id": "nf_user_modeling",
        "module": "10-personalized-news-feed",
        "question": "How do you model a user's evolving interests for personalized news feed ranking?",
        "choices": [
            "A. Use only the user's profile demographics",
            "B. Multi-timescale user modeling: long-term interest embedding (stable interests over months), short-term interest embedding (recent behavior in last 24-48h), and real-time session context (last 5 interactions)",
            "C. Use only the user's last action",
            "D. Use a static user cluster label"
        ],
        "correct": "B",
        "hint": "Your long-term interests (sports) and what you're interested in RIGHT NOW (a breaking news story) are different.",
        "explanation": "Multi-timescale user modeling: (1) Long-term embedding (learned from months of interactions, stable, captures core interests), (2) Short-term embedding (last 1-7 days, captures mood/recent topics), (3) Session context (last 5 posts interacted with in this session, captures immediate intent). These are combined with learned weights or attention mechanisms. This avoids two failure modes: ignoring current context (showing sports when user is reading politics today) and over-reacting to session context (one politics click ≠ permanent interest shift).",
        "difficulty": 4,
        "tags": ["user_modeling", "temporal", "multi_timescale"]
    },
    {
        "concept_id": "nf_misinformation",
        "module": "10-personalized-news-feed",
        "question": "Misinformation spreads faster than corrections on social networks. What ML signals help detect and limit its spread?",
        "choices": [
            "A. Block all political content",
            "B. Engagement pattern anomalies (unusual sharing velocity, high comment-to-like ratio indicating controversy), fact-checking API integration, source credibility scoring, and cross-reference with known misinformation databases",
            "C. Only user-reported content gets reviewed",
            "D. ML cannot detect misinformation"
        ],
        "correct": "B",
        "hint": "What is unusual about how misinformation spreads compared to normal viral content?",
        "explanation": "Misinformation signals: (1) Sharing velocity anomaly (shares 10× faster than similar organic content), (2) Engagement pattern: high share-to-like ratio (sharing without reading), (3) Source credibility (publisher trust score from independent fact-checkers), (4) Semantic similarity to known false claims (nearest neighbor in misinformation database), (5) User report rate. These features feed a classifier that limits amplification (not removal) of likely-false content.",
        "difficulty": 4,
        "tags": ["misinformation", "content_moderation", "detection"]
    },
    {
        "concept_id": "nf_near_duplicate",
        "module": "10-personalized-news-feed",
        "question": "The same news story is posted by 50 different pages. How do you prevent showing the user 50 versions of the same story?",
        "choices": [
            "A. Only show the first post chronologically",
            "B. Story clustering: deduplicate near-duplicate content using embedding similarity + time proximity clustering, then select the best representative (highest quality source, most engagement) or surface diversity of perspectives",
            "C. Show all 50 posts",
            "D. Only show content from major publishers"
        ],
        "correct": "B",
        "hint": "50 copies of the same story wastes the user's feed capacity. How do you collapse them?",
        "explanation": "Near-duplicate detection pipeline: (1) Compute content embeddings for all posts, (2) Cluster posts that are semantically similar AND temporally proximate (same story within a 24-48h window), (3) Select cluster representative: highest engagement OR most authoritative source, (4) Option: show 'X others posted about this story' with link to all versions (respects diverse perspectives). This is related to story deduplication in Google News.",
        "difficulty": 3,
        "tags": ["deduplication", "clustering", "story_clustering"]
    },
    {
        "concept_id": "nf_engagement_bait",
        "module": "10-personalized-news-feed",
        "question": "A post says 'Like this if you love your mom'. It gets massive likes but provides no value. How do you prevent the ranking model from rewarding engagement bait?",
        "choices": [
            "A. Remove all posts asking for engagement",
            "B. Train a classifier to detect engagement bait patterns, then heavily downweight or nullify engagement signals (likes, comments) from posts classified as bait",
            "C. Let users report it",
            "D. Cap maximum likes per post"
        ],
        "correct": "B",
        "hint": "The engagement signal is still 'real' (people liked it), but it doesn't represent genuine interest.",
        "explanation": "Engagement bait detection: fine-tune a classifier on labeled examples of engagement bait ('like if X', 'tag a friend who Y', 'comment below'). When detected, the engagement signals from that post are discounted or zeroed out in training data and ranking. Facebook explicitly built this system in 2017 after engagement bait was exploited to game the feed algorithm. This is a form of label cleaning — the signal is technically real but doesn't reflect what the ranking should optimize for.",
        "difficulty": 3,
        "tags": ["engagement_bait", "signal_quality", "adversarial"]
    },
    {
        "concept_id": "nf_ad_feed_integration",
        "module": "10-personalized-news-feed",
        "question": "How is ad insertion integrated into the organic news feed ranking?",
        "choices": [
            "A. Ads are always shown in fixed positions (1, 5, 10)",
            "B. Ads compete with organic content via a unified auction: ads are ranked alongside organic posts, with revenue value (P(click) × bid) added to the ad's organic-equivalent relevance score to determine insertion position",
            "C. Ads are shown after all organic content",
            "D. Ads are never shown in the feed"
        ],
        "correct": "B",
        "hint": "If ads are always in fixed positions, users learn to ignore them. What is the alternative?",
        "explanation": "Unified auction for feed ranking: organic posts and ads compete in the same ranking. An ad's 'rank score' = organic relevance score + f(P(click) × bid). This ensures ads appear at positions where they would organically fit (high relevance to user) while maximizing revenue. It also means ads don't push out highly relevant organic content, improving user experience. The tradeoff is complex optimization across organic engagement and ad revenue objectives.",
        "difficulty": 4,
        "tags": ["ads", "auction", "feed_ranking"]
    },
    {
        "concept_id": "nf_regional_cultural",
        "module": "10-personalized-news-feed",
        "question": "A global news feed serves users in 100+ countries with different languages and cultural norms. What are the main ML challenges?",
        "choices": [
            "A. Translation is the only challenge",
            "B. Multi-language content understanding (multilingual models), cultural engagement pattern differences (what counts as 'meaningful interaction' differs), regional content policies (legal differences), and data imbalance (most training data is English/Western)",
            "C. Use one global model for everything",
            "D. Build separate models per country"
        ],
        "correct": "B",
        "hint": "An emoji reaction's meaning, a post format's popularity, and acceptable content all vary by country.",
        "explanation": "Global feed challenges: (1) Multilingual encoding (mBERT, XLM-R), (2) Cultural calibration: 'shares' dominate in India, 'reactions' in US — engagement signals need regional normalization, (3) Legal content policies vary by jurisdiction (Germany: no Nazi content, Singapore: blasphemy laws), (4) Training data imbalance: most interactions are US/Western, causing the model to underfit Asian/African/Latin American content patterns.",
        "difficulty": 4,
        "tags": ["internationalization", "multilingual", "cultural_bias"]
    },
    {
        "concept_id": "nf_filter_bubble_feed",
        "module": "10-personalized-news-feed",
        "question": "A user only engages with political content from one party. Should the feed algorithm amplify this?",
        "choices": [
            "A. Yes — user engagement is king",
            "B. Balance personalization with exposure diversity: optimize for short-term engagement (what the user clicks now) AND long-term health signals (do they return? Do they diversify over time?). Inject diverse viewpoints at low cost to the overall feed quality.",
            "C. Show equal content from all political parties regardless of user interest",
            "D. Remove all political content"
        ],
        "correct": "B",
        "hint": "Pure engagement optimization on politically one-sided users creates filter bubbles. How do you counteract this without overriding user intent?",
        "explanation": "Filter bubble mitigation: (1) Measure diversity of content types in feed (topic, source, viewpoint diversity), (2) Add small 'diversity bonus' to ranking for content types the user rarely sees but occasionally engages with, (3) Monitor long-term health metrics: do users' engagement patterns diversify over time or narrow? (4) This is a values-laden product decision that goes beyond pure ML — requires explicit policy choices about what a 'healthy' information diet looks like.",
        "difficulty": 5,
        "tags": ["filter_bubble", "diversity", "long_term_health"]
    },
]
