"""Quiz questions for Module 09: Similar Listing (Airbnb-style)"""

QUESTIONS = [
    {
        "concept_id": "sl_listing_embeddings",
        "module": "09-similar-listing",
        "question": "Airbnb trains listing embeddings using a Word2Vec-style approach on booking sequences. What is a 'session' in this context?",
        "choices": [
            "A. A user's web browsing session",
            "B. A user's historical booking sequence: the sequence of listings a user has booked over time, treated like a 'sentence' where each listing is a 'word'",
            "C. The sequence of photos in a listing",
            "D. A sequence of price changes for a listing"
        ],
        "correct": "B",
        "hint": "How do you apply Word2Vec to non-text data?",
        "explanation": "Airbnb's Real-time Personalization at Airbnb (KDD 2018) treats each user's booking history as a 'sentence' of listing IDs. Skip-gram Word2Vec learns listing embeddings where listings that are commonly booked by the same users (in similar temporal proximity) are embedded close together. This captures latent style preferences (beach house lovers cluster together) without manual feature engineering.",
        "difficulty": 3,
        "tags": ["embeddings", "word2vec", "listing"]
    },
    {
        "concept_id": "sl_negative_sampling_listings",
        "module": "09-similar-listing",
        "question": "When training listing embeddings with negative sampling, what is the key design choice for negative examples?",
        "choices": [
            "A. Randomly sample any listing",
            "B. Use in-market negatives: randomly sample listings from the same market (city) as the positive listing, so the model learns fine-grained within-market differentiation, not just city-level geography",
            "C. Use globally popular listings as negatives",
            "D. Use listings the user explicitly rejected"
        ],
        "correct": "B",
        "hint": "You want the model to learn to distinguish similar listings, not just 'Paris vs. Tokyo'.",
        "explanation": "Random global negatives make the task too easy — a Paris listing is obviously different from a Tokyo listing. In-market negatives (other Paris listings) force the model to learn nuanced within-market similarities: style, price tier, neighborhood characteristics. This produces embeddings where 'similar' means same city + same style + same price range, not just same city.",
        "difficulty": 4,
        "tags": ["negative_sampling", "in_market", "embeddings"]
    },
    {
        "concept_id": "sl_similar_listing_ranking",
        "module": "09-similar-listing",
        "question": "Beyond embedding similarity, what other signals matter for ranking 'similar listings'?",
        "choices": [
            "A. Embedding cosine similarity alone is sufficient",
            "B. Personalization signals (user's price preference, past locations, travel party size), availability matching (is the listing available for the user's dates?), and quality signals (review score, response rate)",
            "C. Only price similarity",
            "D. Only geographic distance"
        ],
        "correct": "B",
        "hint": "A listing that looks similar but costs 10× more than what the user can afford is a poor recommendation.",
        "explanation": "Similar listing ranking is a multi-factor problem: (1) Embedding cosine similarity (visual/style match), (2) Price proximity (within user's historical price range ±20%), (3) Availability (must overlap with user's dates — a hard filter), (4) Quality (review score, superhost status), (5) Personalization (match user's historical accommodation style). Pure embedding similarity ignores whether the user can afford or book the listing.",
        "difficulty": 3,
        "tags": ["ranking", "personalization", "multi_signal"]
    },
    {
        "concept_id": "sl_host_guest_alignment",
        "module": "09-similar-listing",
        "question": "Airbnb must satisfy both guests (find great listings) and hosts (get bookings). How does this dual-sided nature affect the ML system?",
        "choices": [
            "A. Only optimize for guest satisfaction",
            "B. Multi-objective optimization: balance guest satisfaction (click, book, review score) with host outcomes (occupancy rate, revenue). Surfacing a listing the guest loves but the host never responds to is a failure for the platform.",
            "C. Only optimize for platform revenue",
            "D. These objectives never conflict"
        ],
        "correct": "B",
        "hint": "A listing with 5-star reviews but a 20% response rate creates great guest intent but failed bookings.",
        "explanation": "Marketplace ML has two-sided objectives: (1) Guest: recommend listings they'll love (high review scores, match preferences), (2) Host: recommend listings they can actually book (available, host responsive, price appropriate). Airbnb's ranking model includes host responsiveness, booking acceptance rate, and capacity signals alongside guest preference signals. Optimizing purely for guest clicks leads to recommending unresponsive hosts.",
        "difficulty": 3,
        "tags": ["marketplace", "multi_objective", "two_sided_market"]
    },
    {
        "concept_id": "sl_query_listing_embed",
        "module": "09-similar-listing",
        "question": "A user is viewing a listing in Malibu for 4 guests with a pool, priced at $400/night. How do you represent their 'query' for similar listings?",
        "choices": [
            "A. Just use the listing ID",
            "B. Combine the listing embedding with session context (price preference, party size, location, dates) to form a query embedding that captures both the listing style AND the user's constraints",
            "C. Use only the listing description text",
            "D. Use GPS coordinates only"
        ],
        "correct": "B",
        "hint": "The listing embedding captures style. What else do you know about the user's needs?",
        "explanation": "Query representation for similar listings: the anchor listing embedding provides style/feature similarity, but the context provides constraint similarity. A query embedding = f(listing_embed, price_range, party_size, dates, location_preference). This ensures 'similar listings' means similar style AND feasible for the user's specific trip, not just visually/stylistically similar.",
        "difficulty": 3,
        "tags": ["query_embedding", "context", "representation"]
    },
    {
        "concept_id": "sl_price_elasticity",
        "module": "09-similar-listing",
        "question": "Should you recommend a listing that is $50/night more expensive but has significantly better reviews than the anchor listing?",
        "choices": [
            "A. Always recommend the cheaper option",
            "B. Model price elasticity per user: budget-sensitive users prefer cheaper; quality-sensitive users prefer better reviews even at higher price. Use user's historical price-vs-quality tradeoffs to personalize this decision.",
            "C. Always recommend the higher-quality option",
            "D. Show both and let the user choose without ML"
        ],
        "correct": "B",
        "hint": "Different users have different price sensitivity — how do you capture this?",
        "explanation": "Price elasticity personalization: from booking history, estimate each user's price-quality utility function. Power users who consistently book highly-rated listings despite high prices have low price sensitivity; users who consistently book the cheapest available option have high price sensitivity. This learned preference should weight the similar listing ranking — don't show a luxury user only budget options or vice versa.",
        "difficulty": 4,
        "tags": ["price_elasticity", "personalization", "utility"]
    },
    {
        "concept_id": "sl_visual_similarity",
        "module": "09-similar-listing",
        "question": "A user is attracted to a listing with a modern minimalist interior. How do you find visually similar listings?",
        "choices": [
            "A. Use text descriptions only ('modern', 'minimalist')",
            "B. Photo embeddings: encode each listing's photos with a CNN trained on interior design similarity, then retrieve listings with similar photo embeddings via ANN search",
            "C. Use price as a proxy for interior quality",
            "D. Use host-provided category tags"
        ],
        "correct": "B",
        "hint": "Text descriptions are inconsistent and often inaccurate. What signal is more reliable?",
        "explanation": "Photo embeddings for listing similarity: (1) Extract CNN features from all listing photos (pre-trained on interior design or fine-tuned with contrastive learning), (2) Pool across multiple photos per listing (average or attention-weighted), (3) ANN retrieval finds listings with similar visual style. Photo embeddings capture actual style better than text tags, which are often missing, inconsistent, or misleading.",
        "difficulty": 3,
        "tags": ["visual_similarity", "photo_embeddings", "cnn"]
    },
    {
        "concept_id": "sl_availability_freshness",
        "module": "09-similar-listing",
        "question": "A highly similar listing becomes unavailable for the user's dates after being retrieved. How do you handle this in the pipeline?",
        "choices": [
            "A. Show it anyway and let the user figure it out",
            "B. Apply availability as a hard post-retrieval filter, and use predicted availability (likelihood of opening up) as a soft ranking signal for nearly-available listings",
            "C. Remove all unavailable listings from the ANN index permanently",
            "D. Show a waitlist button"
        ],
        "correct": "B",
        "hint": "Showing an unavailable listing wastes the user's attention but removing it from the index is too aggressive.",
        "explanation": "Availability filtering pipeline: (1) ANN retrieval ignores availability for speed (availability changes too frequently to maintain in a real-time ANN index), (2) Post-retrieval hard filter removes unavailable listings for user's dates, (3) For 'not quite available' listings (host hasn't updated calendar), use predicted availability model based on host response patterns. Removing from ANN index is wrong — the listing will be available for other date ranges.",
        "difficulty": 3,
        "tags": ["availability", "pipeline", "freshness"]
    },
    {
        "concept_id": "sl_seasonality",
        "module": "09-similar-listing",
        "question": "A beach house in Malibu is highly demanded in July but low demand in January. How should seasonality affect similar listing recommendations?",
        "choices": [
            "A. Ignore seasonality — recommend based on all-time popularity",
            "B. Use seasonality-adjusted demand signals: during peak season, recommend alternatives to prevent user frustration with unavailability; during off-peak, surface normally hard-to-get listings",
            "C. Only recommend in-season listings",
            "D. Remove seasonal demand from the model"
        ],
        "correct": "B",
        "hint": "In July, the best Malibu beach houses are fully booked. What are you actually optimizing for?",
        "explanation": "Seasonal demand adjustment: (1) During high demand (July), top similar listings may all be unavailable — the system should proactively surface slightly less similar but available alternatives, (2) Demand forecasting (predict occupancy by week) helps pre-compute availability-adjusted rankings, (3) During low demand, opportunistically surface premium listings that are normally unavailable to price-sensitive users.",
        "difficulty": 3,
        "tags": ["seasonality", "demand_forecasting", "availability"]
    },
    {
        "concept_id": "sl_embedding_drift",
        "module": "09-similar-listing",
        "question": "A listing gets renovated and its photos/description change significantly. How do you update the listing embedding?",
        "choices": [
            "A. Keep the old embedding forever",
            "B. Trigger re-embedding when content changes significantly (photo update, major description change). Use change detection signals to prioritize re-embedding high-traffic listings first.",
            "C. Rebuild all embeddings every hour",
            "D. Only re-embed listings that haven't been booked in 90 days"
        ],
        "correct": "B",
        "hint": "An old embedding of a renovated listing will recommend similar 'old-style' listings that no longer match.",
        "explanation": "Embedding freshness: (1) Event-driven re-embedding: when a listing's photos or key features change (detected by hashing image content), trigger re-embedding, (2) Prioritize by traffic: high-traffic listings need faster updates, (3) Drift detection: monitor the cosine similarity between old and new embeddings after re-embedding — large drift signals significant listing changes worth alerting hosts about, (4) Efficient incremental update vs. full index rebuild.",
        "difficulty": 3,
        "tags": ["embedding_freshness", "drift", "update_strategy"]
    },
]
