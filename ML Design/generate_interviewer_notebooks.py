"""
Generator script for 11 Interviewer Perspective notebooks.
Creates 05_interviewer_perspective.ipynb (or 04_ for module 05) in each module dir.
Run with: python generate_interviewer_notebooks.py
"""
import json
import os

BASE = "/Users/ruifengli/Desktop/applied-ai-research/ML Design"

# ── Per-module configuration ─────────────────────────────────────────────────
MODULES = [
    {
        "num": "01",
        "dir": "01-ml-design-prep",
        "title": "ML Design Prep Framework",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "ML interview coaching and preparation frameworks",
        "problem": "Design a scalable ML interview preparation and coaching system",
        "q1_arch": "Why a structured 7-step framework over ad-hoc coaching? What breaks if candidates skip clarification?",
        "q1_alts": ["Unstructured coaching", "Rote memorization of solutions", "Single deep-dive vs breadth approach"],
        "q2_training": "How do you measure 'interview readiness'? What labels, what loss function? How do you handle the sparse feedback problem (few real interview outcomes)?",
        "q3_data": "How do you capture candidate progress? What is the label delay (weeks to interview)? How do you avoid training-serving skew when mock sessions differ from real interviews?",
        "q4_serving": "How do you serve personalized coaching at scale? Latency budget for real-time feedback? Degraded-mode if question bank is unavailable?",
        "q5_metrics": "Offline: session completion rate vs. interview pass rate — why the gap? How do you A/B test coaching curriculum changes?",
        "q6_edge": "candidate skips framework steps, rushes clarification — how does the system detect and correct this pattern?",
        "q6_signals": [
            "No Hire: Cannot explain why clarification matters or what goes wrong without it",
            "Weak Hire: Detects skip but only suggests 'go back and clarify' without root cause",
            "Hire: Diagnoses WHY the candidate rushes (anxiety, habit), proposes targeted drills",
            "Strong Hire: Builds a pattern library of skip behaviors; surfaces cross-candidate insights to improve curriculum",
        ],
        "q7_principal": "How to build internal ML interview prep tooling at org scale — standardizing rubrics, calibrating interviewers, feeding aggregate signal back into candidate prep?",
        "q7_signals": [
            "No Hire: Treats this as a tooling problem only; misses the organizational calibration angle",
            "Weak Hire: Proposes a rubric spreadsheet; no ML signal loop",
            "Hire: Designs feedback loop from interview outcomes to curriculum updates; mentions interviewer calibration",
            "Strong Hire: Proposes a platform — shared rubric store, cross-team calibration, anomaly detection on interviewer scoring patterns",
        ],
    },
    {
        "num": "02",
        "dir": "02-visual-search",
        "title": "Visual Search",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "visual similarity search and image retrieval",
        "problem": "Design a visual search system (find visually similar products from a query image)",
        "q1_arch": "Why contrastive learning / two-tower for embeddings over a classification head? What breaks if you use off-the-shelf ImageNet features without fine-tuning?",
        "q1_alts": ["Perceptual hashing", "Global color histograms", "ImageNet features without fine-tuning", "Contrastive (SimCLR/CLIP)"],
        "q2_training": "How do you construct triplets/pairs for contrastive learning? What is your mining strategy? How do you handle the extreme class imbalance (billions of negatives)?",
        "q3_data": "How do you keep embeddings fresh when catalog changes? What is the rebuild frequency? How do you avoid embedding drift between training and the live ANN index?",
        "q4_serving": "Latency budget for an ANN query over 1B product embeddings? How does HNSW vs. IVF-PQ trade off recall vs. latency? Fallback when GPU inference is degraded?",
        "q5_metrics": "Offline: Recall@k on a curated query set vs. online CTR — why do they diverge? How do you A/B test embedding model updates without re-indexing all products?",
        "q6_edge": "near-duplicates and adversarial visual perturbations — pixel noise that flips retrieval results. How does the system detect and handle these?",
        "q6_signals": [
            "No Hire: Not aware that small pixel perturbations can change embedding similarity drastically",
            "Weak Hire: Suggests deduplication post-retrieval; no adversarial robustness discussion",
            "Hire: Proposes augmentation-based training to improve robustness; discusses perceptual hash dedup pre-index",
            "Strong Hire: Designs adversarial evaluation suite; proposes ensemble of embeddings from models trained with different augmentation policies",
        ],
        "q7_principal": "Should visual search embeddings be shared with the product recommendation system? What are the alignment challenges, governance tradeoffs, and platform design?",
        "q7_signals": [
            "No Hire: Says 'yes, sharing saves compute' without understanding objective misalignment",
            "Weak Hire: Recognizes the trade-off but only proposes separate models",
            "Hire: Proposes a multi-task embedding tower with task-specific heads; discusses when to share vs. split",
            "Strong Hire: Designs an embedding platform with versioning, A/B testing of shared vs. task-specific, and consumer SLAs",
        ],
    },
    {
        "num": "03",
        "dir": "03-google-street-view",
        "title": "Google Street View Blurring",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "street-level imagery analysis and privacy blurring",
        "problem": "Design a system to automatically detect and blur faces/license plates in Street View imagery",
        "q1_arch": "Why a two-stage detect-then-blur pipeline over end-to-end segmentation? What breaks if you skip detection and blur everything above a saliency threshold?",
        "q1_alts": ["End-to-end semantic segmentation", "Saliency-based blurring", "Two-stage detect + blur", "Rule-based color/shape heuristics"],
        "q2_training": "How do you construct labels for a privacy blurring system at scale? What is your strategy for hard negatives (mannequins, posters)? How do you handle class imbalance between faces and plates?",
        "q3_data": "How do you handle GPS noise corrupting image metadata? What is your training-serving skew risk given that Street View imagery is captured with specialized hardware? How do you keep the model current as urban environments change?",
        "q4_serving": "Latency budget for processing billions of Street View frames? Batch vs. streaming architecture? Degraded-mode if the face detector is down — do you blur everything or nothing?",
        "q5_metrics": "Offline: detection recall vs. false positive rate — what is your operating point and why? Online: How do you measure privacy compliance without ground truth at scale?",
        "q6_edge": "GPS noise, occluded imagery (obstructions, bad lighting), and geographic gaps (rural areas with few training examples). How does the system gracefully handle these?",
        "q6_signals": [
            "No Hire: Treats all failure modes as 'model quality' without decomposing by data characteristic",
            "Weak Hire: Suggests re-collecting data in rural areas; no systematic approach",
            "Hire: Designs geo-stratified evaluation; proposes conservative high-recall fallback for low-confidence regions",
            "Strong Hire: Builds an active learning loop targeting geographic gaps; designs a confidence-calibrated operating threshold per region",
        ],
        "q7_principal": "Should the team build a universal street-level perception platform vs. task-specific models for blurring, signage reading, and address detection?",
        "q7_signals": [
            "No Hire: Treats each task as independent without considering shared infrastructure costs",
            "Weak Hire: Suggests a shared backbone but misses the conflicting annotation requirements",
            "Hire: Proposes multi-task learning with task-specific heads; discusses label schema unification",
            "Strong Hire: Designs a perception platform with standardized interfaces, shared annotation tooling, and cross-task transfer learning policies",
        ],
    },
    {
        "num": "04",
        "dir": "04-youtube-video-search",
        "title": "YouTube Video Search",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "video search and multimodal information retrieval",
        "problem": "Design a video search system (query → ranked list of relevant videos)",
        "q1_arch": "Why a multi-modal two-tower (text query + video transcript/title/thumbnail) over BM25 keyword matching? What breaks if you use text-only retrieval for a video platform?",
        "q1_alts": ["BM25 keyword search", "Text-only dense retrieval", "Multimodal two-tower", "Learning-to-rank on top of BM25"],
        "q2_training": "How do you construct relevance labels for a search system? Click-through as positive — what are the biases? How do you handle position bias in training data?",
        "q3_data": "How do you handle feature freshness for trending queries? What is the label delay for implicit feedback (watch time)? How do you avoid training-serving skew when query logs differ from batch training data?",
        "q4_serving": "Latency budget breakdown for a YouTube-scale search: ANN retrieval + ranking + re-ranking. What degrades gracefully if the heavy ranker is slow?",
        "q5_metrics": "Offline: nDCG@10 on editorial judgments vs. online CTR and watch time — when do they diverge and why? How do you design an A/B test that is not contaminated by position effects?",
        "q6_edge": "multilingual query-video mismatch (English query, foreign-language video with relevant content) and zero-result queries. How does the system handle both?",
        "q6_signals": [
            "No Hire: Treats multilingual as a 'translation problem' without considering embedding alignment",
            "Weak Hire: Proposes language detection + separate per-language index; no unified approach",
            "Hire: Proposes multilingual embeddings (mBERT, LASER) with cross-lingual retrieval; fallback to translated titles",
            "Strong Hire: Designs cross-lingual evaluation suite; discusses when to surface foreign-language results and UX signals for user language preference inference",
        ],
        "q7_principal": "Should the search team share an embedding layer with the recommendation team? What are the technical and organizational challenges?",
        "q7_signals": [
            "No Hire: Immediately agrees to share without discussing objective misalignment (search relevance vs. engagement)",
            "Weak Hire: Recognizes the tension but only proposes separate systems",
            "Hire: Proposes a multi-task embedding with search and recommendation heads; evaluates on both tasks",
            "Strong Hire: Designs a shared embedding service with consumer-specific fine-tuning; discusses SLA ownership and rollback policy when shared model degrades one consumer",
        ],
    },
    {
        "num": "05",
        "dir": "05-harmful-content-detection",
        "title": "Harmful Content Detection",
        "filename": "04_interviewer_perspective.ipynb",
        "domain": "content safety, harmful content detection, and trust & safety",
        "problem": "Design a harmful content detection system for a social media platform",
        "q1_arch": "Why a multi-task model (hate speech + spam + nudity jointly) over specialized per-category classifiers? What breaks if you train them independently?",
        "q1_alts": ["Per-category binary classifiers", "Single multi-label classifier", "Multi-task with shared backbone", "LLM-based zero-shot classification"],
        "q2_training": "How do you define and construct harmful labels — who decides what is harmful? How do you handle labeling disagreement? What is your strategy for rapidly evolving harm categories?",
        "q3_data": "How do you handle training-serving skew when harmful content evolves adversarially? What is your feature freshness requirement? How do you prevent the model from learning spurious correlations with platform demographics?",
        "q4_serving": "What is the latency budget for real-time content moderation at posting time? How does your serving architecture differ for async review queues vs. synchronous blocking decisions?",
        "q5_metrics": "Offline: precision-recall tradeoff at your operating threshold — how do you set it? Online: appeals rate, valid appeals rate, harmful impressions per user. How do you A/B test a new harm category?",
        "q6_edge": "adversarial probing of classifier boundaries — users crafting content that barely evades detection — and rapidly evolving meme variants. How does the system adapt?",
        "q6_signals": [
            "No Hire: Treats adversarial probing as a model accuracy problem without recognizing the cat-and-mouse dynamic",
            "Weak Hire: Suggests retraining when adversarial patterns are detected; no proactive defense",
            "Hire: Proposes adversarial training, ensemble disagreement monitoring, and rapid retraining pipelines with human-in-the-loop for emerging patterns",
            "Strong Hire: Designs a red-team feedback loop; proposes detection of distribution shift from adversarial probing as an early warning system; discusses the policy vs. ML boundary",
        ],
        "q7_principal": "Should you build vs. buy a content moderation platform? What are the implications of outsourcing ML moderation vs. building in-house? Who owns the boundary between policy and ML decisions?",
        "q7_signals": [
            "No Hire: Treats this as a pure buy-vs-build cost analysis",
            "Weak Hire: Recognizes the policy sensitivity but does not address org structure",
            "Hire: Discusses the risk of outsourcing policy-encoded ML decisions; proposes a hybrid with platform providing infrastructure and teams encoding policy in labels",
            "Strong Hire: Designs a platform where policy is declaratively specified, ML translates policy to classifiers, and policy changes can be audited and rolled back independently of model updates",
        ],
    },
    {
        "num": "06",
        "dir": "06-video-recommendation",
        "title": "Video Recommendation",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "video recommendation and multi-stage retrieval ranking",
        "problem": "Design a homepage video recommendation system for a YouTube-scale platform",
        "q1_arch": "Why a two-tower model over pure matrix factorization for candidate generation? What breaks if you use MF alone for a platform with 10B videos?",
        "q1_alts": ["Pure matrix factorization (WALS)", "Item-based collaborative filtering", "Two-tower neural network", "Content-based with pre-trained embeddings only"],
        "q2_training": "What is your label construction strategy — clicks vs. watch time? How do you handle position bias in the training data? What loss function and why?",
        "q3_data": "How do you handle training-serving skew when features computed at training time differ from serving time? What is your feature freshness requirement? How does label delay affect model quality?",
        "q4_serving": "How does your multi-stage pipeline (retrieval → ranking → re-ranking) fit within a 200ms budget? What degrades gracefully if the heavy ranker is slow? Caching strategy?",
        "q5_metrics": "Offline: mAP vs. online watch time — when do they diverge? How do you detect that your A/B test has a novelty effect contaminating results?",
        "q6_edge": "feedback loops leading to filter bubbles and cold-start at platform scale (new users, new videos, new markets). How does the system address these simultaneously?",
        "q6_signals": [
            "No Hire: Not aware that optimizing engagement can create filter bubbles",
            "Weak Hire: Mentions diversity in re-ranking but treats it as a post-hoc fix",
            "Hire: Designs diversity-aware ranking objective; uses multiple candidate generators with different exploration objectives; cold start handled via demographics + content features",
            "Strong Hire: Proposes a long-term value model (LTV) that trades off short-term engagement for user health; designs counterfactual evaluation to measure filter bubble effect",
        ],
        "q7_principal": "Should the team build a cross-surface embedding platform shared between homepage recommendations, Shorts, and search? What are the tradeoffs?",
        "q7_signals": [
            "No Hire: Immediately agrees without understanding objective divergence across surfaces",
            "Weak Hire: Proposes separate models with shared infrastructure",
            "Hire: Designs multi-task embedding with surface-specific heads; evaluates on per-surface metrics",
            "Strong Hire: Proposes an embedding platform with consumer SLAs, A/B testing infrastructure for shared model changes, and clear deprecation policy when divergence is detected",
        ],
    },
    {
        "num": "07",
        "dir": "07-event-recommendation",
        "title": "Event Recommendation",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "event recommendation with extreme temporal and geographic sparsity",
        "problem": "Design an event recommendation system for a platform like Eventbrite or Facebook Events",
        "q1_arch": "Why a hybrid content + collaborative approach over pure CF? What breaks when most events are unique one-time occurrences with no repeat interaction history?",
        "q1_alts": ["Pure collaborative filtering", "Geographic proximity ranking only", "Content-based with event category matching", "Hybrid with temporal decay"],
        "q2_training": "How do you define 'relevant' for an event — RSVP, attendance, page view? How do you handle the cold-start for new events that haven't occurred yet? What is your label construction strategy?",
        "q3_data": "How do you handle the extreme label delay — events happen once and feedback arrives weeks later? What feature freshness is required for time-sensitive recommendations (concerts this weekend)?",
        "q4_serving": "Events expire — how does your serving architecture handle TTL and cache invalidation for time-sensitive inventory? What is your fallback when local events are sparse?",
        "q5_metrics": "Offline: precision@k on RSVPs vs. online attendance rate — why is the gap large for events? How do you A/B test when the treatment group cannot attend the same event as control?",
        "q6_edge": "extreme data sparsity for events (they happen once), geographic sparsity (rural areas with few events), and the cold start for both new users and new event organizers",
        "q6_signals": [
            "No Hire: Applies standard CF without addressing the one-shot event problem",
            "Weak Hire: Suggests content-based fallback for new events; no systematic sparsity handling",
            "Hire: Designs a hybrid system that transfers learning from organizer history, event category, and geographic signals; uses implicit signals (page views, shares) to compensate for sparse RSVPs",
            "Strong Hire: Proposes a hierarchical model that learns event-class-level patterns to bootstrap individual event models; designs geographic interpolation for sparse regions",
        ],
        "q7_principal": "How should the event recommendation system inform the event creation product — surfacing demand signals to organizers and closing the supply-demand loop?",
        "q7_signals": [
            "No Hire: Treats recommendation as purely a consumer-facing feature",
            "Weak Hire: Suggests showing 'popular event categories in your area' to organizers",
            "Hire: Designs a demand signal API for organizers; discusses how recommendation feedback shapes supply; addresses gaming risk",
            "Strong Hire: Proposes a two-sided marketplace optimization where recommendation and supply creation are jointly optimized; discusses mechanism design to prevent organizer gaming",
        ],
    },
    {
        "num": "08",
        "dir": "08-ad-click-prediction",
        "title": "Ad Click Prediction",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "ad click prediction, CTR estimation, and auction systems",
        "problem": "Design an ad click prediction system for a social media platform",
        "q1_arch": "Why DeepFM over a vanilla DNN for ad CTR prediction? What breaks if you use a DNN without explicit feature interaction modeling?",
        "q1_alts": ["Logistic Regression baseline", "GBDT (XGBoost)", "Vanilla DNN", "DeepFM / DCN with explicit interaction modeling"],
        "q2_training": "How do you construct negative labels for ad impressions? What is the dwell time threshold and why? How do you handle position bias in click data? What happens to calibration after negative downsampling?",
        "q3_data": "How do you avoid training-serving skew when online features (session clicks, recent impressions) differ between training logs and serving time? How often must the model be updated for an ad system?",
        "q4_serving": "What is the latency budget for CTR prediction in an ad auction? How do you serve a model that must score millions of ad candidates per second? Fallback strategy when the ranking model is down?",
        "q5_metrics": "Offline: NCE (Normalized Cross-Entropy) vs. AUC — why is NCE preferred for ad systems? Online: revenue lift vs. CTR — when can CTR increase while revenue decreases?",
        "q6_edge": "click fraud (adversarial actors inflating CTR) and concept drift from external events (elections, viral news). How does the system detect and respond to both?",
        "q6_signals": [
            "No Hire: Not aware of click fraud as a systematic threat or how it corrupts training data",
            "Weak Hire: Suggests post-hoc filtering of fraud clicks; no model-level defense",
            "Hire: Proposes real-time fraud detection as a separate signal fed into training; discusses robust training against label noise; describes drift detection using sliding-window NCE monitoring",
            "Strong Hire: Designs an adversarial training setup that regularizes against fraudulent click patterns; proposes causal inference to separate true CTR signal from external event confounders",
        ],
        "q7_principal": "Should calibration be built as a shared platform service across ad formats (display, video, sponsored content)? What are the design and governance considerations?",
        "q7_signals": [
            "No Hire: Treats calibration as a model-level detail, not a platform concern",
            "Weak Hire: Agrees calibration should be shared but does not address format-specific distribution differences",
            "Hire: Proposes a calibration service with format-specific Platt scaling layers; discusses the interface contract between the prediction model and the calibration layer",
            "Strong Hire: Designs a calibration platform with monitoring, A/B testing, and automatic recalibration triggers; discusses auction theory implications of miscalibration and platform-level liability",
        ],
    },
    {
        "num": "09",
        "dir": "09-similar-listing",
        "title": "Similar Listing (Airbnb)",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "similar listing recommendation for short-term rental marketplaces",
        "problem": "Design a 'similar listings' recommendation system for a marketplace like Airbnb",
        "q1_arch": "Why contrastive learning with booking co-occurrence over simple attribute similarity matching? What breaks if you use only price + location + amenity features?",
        "q1_alts": ["Attribute-based similarity (price, location, amenities)", "Matrix factorization on booking co-occurrence", "Contrastive learning with hard negative mining", "Graph neural network on booking sessions"],
        "q2_training": "How do you construct positive and negative pairs for listing similarity? What makes a good hard negative (same city, different neighborhood vs. different city)? How do you handle listing inventory churn?",
        "q3_data": "How do you keep embeddings fresh when listing attributes change (price updates, new photos, seasonal availability)? What is the rebuild frequency? How do you avoid stale embeddings in the ANN index?",
        "q4_serving": "How does your ANN search scale to tens of millions of listing embeddings? What is the latency budget for a 'similar listings' carousel? Fallback when the embedding service is unavailable?",
        "q5_metrics": "Offline: Recall@k on held-out booking sessions vs. online booking conversion rate — why is the correlation weak? How do you design an A/B test for a widget that has a long conversion funnel?",
        "q6_edge": "duplicate listings (same property listed multiple times), price manipulation (sudden price spikes to game ranking), and geographic clustering bias (all results from the same block)",
        "q6_signals": [
            "No Hire: Not aware that duplicate listings are a real scale problem or how they degrade quality",
            "Weak Hire: Suggests deduplication as a post-processing step; no geographic diversity mechanism",
            "Hire: Proposes a perceptual hash + text similarity deduplication pipeline; designs a geographic diversity constraint in re-ranking; flags price-manipulated listings via anomaly detection",
            "Strong Hire: Builds a listing quality score that feeds into both retrieval and ranking; proposes a marketplace health dashboard tracking duplicate rate, price manipulation signals, and geographic diversity metrics",
        ],
        "q7_principal": "Should cross-marketplace embedding sharing (Airbnb Rooms vs. Experiences vs. long-term stays) be a platform investment? What are the alignment and governance challenges?",
        "q7_signals": [
            "No Hire: Treats each marketplace as completely independent without considering shared user behavior signals",
            "Weak Hire: Proposes sharing infrastructure but not models",
            "Hire: Identifies user interest alignment across products as a valid signal; proposes cross-product transfer learning with product-specific fine-tuning",
            "Strong Hire: Designs a user interest embedding platform that serves multiple products with different retrieval objectives; discusses cold-start for new marketplace categories leveraging existing user signals",
        ],
    },
    {
        "num": "10",
        "dir": "10-personalized-news-feed",
        "title": "Personalized News Feed",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "personalized news feed ranking and content quality scoring",
        "problem": "Design a personalized news feed ranking system for a social media platform",
        "q1_arch": "Why a multi-objective ranking model (relevance + quality + freshness) over a single engagement-optimized model? What breaks when you only optimize for clicks?",
        "q1_alts": ["Single engagement model (maximize clicks)", "Chronological feed", "Multi-objective ranking with weighted sum", "Constrained ranking with quality floor"],
        "q2_training": "How do you construct 'quality' labels for news content? What is your labeling strategy — crowd workers, expert annotators, behavioral signals? How do you handle the subjectivity of quality?",
        "q3_data": "How do you handle the breaking news recency problem — a high-quality article from 3 hours ago vs. a low-quality article from 3 minutes ago? What is the feature freshness requirement for engagement signals?",
        "q4_serving": "What is the latency budget for news feed ranking? How do you handle the spike in traffic during breaking news events? What is your caching strategy for a feed that must balance recency and personalization?",
        "q5_metrics": "Offline: nDCG on editorial quality judgments vs. online time-in-feed — when is time-in-feed a misleading metric? How do you A/B test feed changes without contamination from social sharing?",
        "q6_edge": "echo chambers (users only seeing content that confirms their views) and breaking news recency vs. quality tradeoff (viral misinformation spreading before fact-checking)",
        "q6_signals": [
            "No Hire: Treats echo chambers as a 'user preference' and not a platform design problem",
            "Weak Hire: Suggests injecting diverse content as a post-hoc fix; no measurement framework",
            "Hire: Designs a viewpoint diversity metric; proposes a constrained ranking approach that enforces minimum diversity; discusses the breaking news pipeline with a fast-path for authoritative sources",
            "Strong Hire: Proposes a long-term user health model that measures the causal effect of feed composition on user polarization; designs a counterfactual evaluation framework for echo chamber measurement",
        ],
        "q7_principal": "Should the same ranker be used across different content types (text posts, images, videos, links) on the same platform, or should content-specific rankers be maintained?",
        "q7_signals": [
            "No Hire: Argues for a single ranker without addressing cross-type feature incompatibility",
            "Weak Hire: Proposes separate rankers per content type without considering cross-type calibration",
            "Hire: Designs a unified ranker with content-type embeddings and content-specific feature towers; discusses cross-type calibration to prevent one content type from dominating the feed",
            "Strong Hire: Proposes a platform ranker with content-type-specific fine-tuning; designs a cross-type fairness constraint and an experimental framework for testing new content type introductions",
        ],
    },
    {
        "num": "11",
        "dir": "11-people-you-may-know",
        "title": "People You May Know (PYMK)",
        "filename": "05_interviewer_perspective.ipynb",
        "domain": "friend recommendation using social graph embeddings",
        "problem": "Design a People You May Know (PYMK) friend recommendation system",
        "q1_arch": "Why graph neural networks or node2vec for PYMK over simple mutual friends counting? What breaks if you only use 2nd-degree connections?",
        "q1_alts": ["Mutual friends count (heuristic)", "Graph random walk (node2vec)", "Graph Neural Network (GraphSAGE)", "Collaborative filtering on social interactions"],
        "q2_training": "How do you construct positive and negative training pairs for friend recommendation? What makes a good hard negative (connected in the graph but unlikely to friend-request)? How do you handle the class imbalance (billions of non-edges vs. thousands of new edges)?",
        "q3_data": "How do you keep graph embeddings fresh as the social graph evolves (new users, new edges daily)? What is the incremental update strategy vs. full retraining? How do you handle embedding drift?",
        "q4_serving": "How do you serve real-time PYMK for a user whose graph neighborhood changes frequently? What is the latency budget? Caching strategy for a user's candidate pool?",
        "q5_metrics": "Offline: AUC on held-out friend requests vs. online friend request acceptance rate — what drives the gap? How do you A/B test PYMK without network effects contaminating the control group?",
        "q6_edge": "privacy attacks via inference (inferring sensitive attributes from PYMK suggestions) and spam account injection into the social graph. How does the system defend against both?",
        "q6_signals": [
            "No Hire: Not aware that PYMK suggestions can reveal sensitive information (e.g., surfacing connections between users who want to keep separate social circles private)",
            "Weak Hire: Suggests blocking spam accounts post-detection; no privacy-by-design approach",
            "Hire: Proposes differential privacy in graph embedding training; designs spam detection as a quality signal that suppresses candidates from suspected spam clusters; discusses the policy for sensitive connection inference",
            "Strong Hire: Designs a privacy-preserving PYMK pipeline with formal DP guarantees; proposes adversarial evaluation of inference attacks; discusses the organizational tension between growth (more connections) and privacy (fewer inferences)",
        ],
        "q7_principal": "Should the social graph embedding service be shared across PYMK, feed ranking, and search? What are the platform design and governance challenges?",
        "q7_signals": [
            "No Hire: Treats PYMK, feed, and search as independent problems with separate graph data stores",
            "Weak Hire: Proposes a shared graph storage layer but not shared embeddings",
            "Hire: Designs a graph embedding service with PYMK-specific, feed-specific, and search-specific fine-tuning heads; discusses the challenge of different update frequencies per consumer",
            "Strong Hire: Proposes a graph embedding platform with versioning, consumer-specific SLAs, rollback policy, and a governance model for changes that could affect multiple product surfaces simultaneously",
        ],
    },
]

# ── Notebook cell builder helpers ────────────────────────────────────────────

def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }

def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

# ── Shared code blocks ────────────────────────────────────────────────────────

CELL2_CODE = '''import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Hiring Level Framework Visualization ───────────────────────────────────

levels = [
    ("No Hire",     "#d32f2f", "Mid→Staff attempt\\nMissing fundamentals;\\ncannot design working system"),
    ("Weak Hire",   "#f57c00", "Strong Senior at Staff bar\\nCorrect but naive;\\nlacks production/scale awareness"),
    ("Hire",        "#388e3c", "Staff Engineer\\nProduction-ready design;\\nproactively addresses tradeoffs"),
    ("Strong Hire", "#1b5e20", "Principal Engineer\\nExpands scope; platform thinking;\\nfinds tradeoffs not asked about"),
]

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('Hiring Level Calibration (Staff/Principal Bar)',
             fontsize=14, fontweight='bold', pad=15)

for i, (level, color, desc) in enumerate(levels):
    x = i * 3.5
    rect = mpatches.FancyBboxPatch((x + 0.2, 0.3), 3.0, 3.3,
        boxstyle='round,pad=0.1', facecolor=color, edgecolor='#222', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + 1.7, 3.2, level, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(x + 1.7, 1.8, desc, ha='center', va='center',
            fontsize=8, color='white', linespacing=1.5)

# Gradient arrow
ax.annotate('', xy=(13.8, 0.08), xytext=(0.1, 0.08),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#444'))
ax.text(7, -0.3, 'Increasing Seniority →', ha='center', fontsize=10, color='#444')

plt.tight_layout()
plt.savefig('/tmp/hiring_levels.png', dpi=100, bbox_inches='tight')
plt.show()

# ─── Helper Functions ────────────────────────────────────────────────────────

def print_rubric(rubric: dict):
    """Print a formatted signal checklist for a single question rubric."""
    level_colors = {
        "No Hire":     "\\033[91m",   # red
        "Weak Hire":   "\\033[93m",   # yellow
        "Hire":        "\\033[92m",   # green
        "Strong Hire": "\\033[32m",   # dark green
    }
    reset = "\\033[0m"
    print("\\n" + "=" * 72)
    print(f"  Q{rubric.get('q_num', '?')}: {rubric.get('title', '')}")
    print(f"  Category: {rubric.get('category', '')}  |  Tests: {rubric.get('tests', '')}")
    print("=" * 72)
    for level, signals in rubric.get('signals', {}).items():
        color = level_colors.get(level, '')
        print(f"\\n{color}  [{level}]{reset}")
        for s in signals:
            print(f"    • {s}")
    if rubric.get('disqualifiers'):
        print("\\n\\033[91m  ⚠ DISQUALIFYING SIGNALS (override adequate answers):\\033[0m")
        for d in rubric['disqualifiers']:
            print(f"    ✗ {d}")
    print()

def render_rubric_table(rubric: dict):
    """Render a 4-column matplotlib table for a question rubric."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    ax.set_title(f"Q{rubric.get('q_num', '?')}: {rubric.get('title', '')} — Rubric Table",
                 fontsize=12, fontweight='bold', pad=10)
    col_labels = ["No Hire", "Weak Hire", "Hire", "Strong Hire"]
    col_colors = ["#d32f2f", "#f57c00", "#388e3c", "#1b5e20"]
    signals = rubric.get('signals', {})
    rows = max(len(v) for v in signals.values()) if signals else 1
    cell_data = []
    for r in range(rows):
        row = []
        for level in col_labels:
            sigs = signals.get(level, [])
            row.append(sigs[r] if r < len(sigs) else "")
        cell_data.append(row)
    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j, color in enumerate(col_colors):
        tbl[0, j].set_facecolor(color)
        tbl[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(1, rows + 1):
            tbl[i, j].set_facecolor(color + '22')  # light tint
    plt.tight_layout()
    plt.show()

print("Helper functions loaded: print_rubric(), render_rubric_table()")
print("Hiring level framework displayed above.")'''


def build_question_map_code(m: dict) -> str:
    return f'''# ─── Question Category Map ──────────────────────────────────────────────────

questions = {{
    "Q1": {{"category": "ML Fundamentals – Model Choice",
           "what_it_tests": "Architecture selection and trade-off reasoning",
           "one_liner": "{m["q1_arch"][:80]}..."}},
    "Q2": {{"category": "ML Fundamentals – Training Strategy",
           "what_it_tests": "Loss function, label construction, class imbalance",
           "one_liner": "{m["q2_training"][:80]}..."}},
    "Q3": {{"category": "Systems & Scale – Data Pipeline",
           "what_it_tests": "Feature freshness, training-serving skew, label delay",
           "one_liner": "{m["q3_data"][:80]}..."}},
    "Q4": {{"category": "Systems & Scale – Serving Architecture",
           "what_it_tests": "Latency budget, caching, degraded-mode fallback",
           "one_liner": "{m["q4_serving"][:80]}..."}},
    "Q5": {{"category": "Evaluation & Metrics",
           "what_it_tests": "Offline vs online gap, A/B design, gaming risk",
           "one_liner": "{m["q5_metrics"][:80]}..."}},
    "Q6": {{"category": "Edge Cases & Failure Modes",
           "what_it_tests": "Module-specific: {m["q6_edge"][:60]}",
           "one_liner": "How does the system handle adversarial or unusual inputs?"}},
    "Q7": {{"category": "Principal-Level Thinking",
           "what_it_tests": "Build vs buy, platform/cross-surface, org implications",
           "one_liner": "{m["q7_principal"][:80]}..."}},
}}

print("=" * 72)
print(f"  QUESTION MAP — {m["title"].upper()}")
print("=" * 72)
for qid, info in questions.items():
    print(f"\\n  {{qid}} | {{info['category']}}")
    print(f"       Tests: {{info['what_it_tests']}}")
    print(f"       Focus: {{info['one_liner']}}")
'''


def q_markdown(q_num: int, title: str, category: str, tests: str,
               phrasing: str, probes: list[str]) -> str:
    probe_lines = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(probes))
    return f"""## Q{q_num}: {title}

**Category:** {category}
**What it tests:** {tests}

---

### Exact phrasing to use

> "{phrasing}"

### Three follow-up probes

{probe_lines}

---

*See the code cell below for the four-level rubric, signal checklist, and disqualifying signals.*"""


def q_code(q_num: int, title: str, category: str, tests: str,
           signals: dict, disqualifiers: list[str]) -> str:
    rubric_dict = {
        "q_num": q_num,
        "title": title,
        "category": category,
        "tests": tests,
        "signals": signals,
        "disqualifiers": disqualifiers,
    }
    return f'''rubric_q{q_num} = {json.dumps(rubric_dict, indent=4)}

print_rubric(rubric_q{q_num})
render_rubric_table(rubric_q{q_num})'''


# ── Per-module notebook builder ───────────────────────────────────────────────

def build_notebook(m: dict) -> dict:
    num = m["num"]
    title = m["title"]
    domain = m["domain"]

    # ── Cell 1: Header markdown ──────────────────────────────────────────────
    cell1 = md_cell(f"""# Interviewer Perspective: {title}

## The Big Idea (For a 12-Year-Old)

Imagine you are a judge on a talent show for ML engineers. Your job is not to perform —
it is to watch carefully and decide: does this person really understand what they are doing,
or are they just memorizing patterns?

You have a **score card** with four boxes: *No Hire*, *Weak Hire*, *Hire*, and *Strong Hire*.
You fill in the box based on whether the candidate shows they can build real systems that
work at scale, handle failures gracefully, and think about the business beyond just the model.

---

## Staff-Level Technical Summary

This notebook equips you to **interview** a candidate on {domain}.
You will find:
- 7 deep-dive questions with exact phrasing
- 3 follow-up probes per question
- 4-level rubric (No Hire → Strong Hire) calibrated to **Staff / Principal Engineer** bar
- Disqualifying signals that override otherwise-adequate answers
- Cross-cutting patterns that distinguish Staff from Senior engineers
- A sample scorecard and 45-minute interviewer agenda

**Calibration anchor:**
- *No Hire* = Mid-level to Staff-attempt: missing fundamentals, cannot design a working system
- *Weak Hire* = Strong Senior at Staff bar: correct but naive, lacks production/scale awareness
- *Hire* = Staff Engineer: production-ready design, proactively addresses tradeoffs
- *Strong Hire* = Principal Engineer: expands scope, platform thinking, finds tradeoffs not asked about

**Key rule:** Disqualifying signals override otherwise-adequate answers.""")

    # ── Cell 2: Hiring level framework + helpers ─────────────────────────────
    cell2 = code_cell(CELL2_CODE)

    # ── Cell 3: Question map ─────────────────────────────────────────────────
    cell3 = code_cell(build_question_map_code(m))

    # ── Q1: Model Choice ─────────────────────────────────────────────────────
    q1_probes = [
        f"You chose {m['q1_alts'][2] if len(m['q1_alts']) > 2 else 'that architecture'}. Walk me through why specifically over {m['q1_alts'][0]}.",
        "What is the minimum viable version of this system? Could you launch with just a baseline?",
        "If your compute budget were cut in half, what would you remove first and why?",
    ]
    cell4 = md_cell(q_markdown(
        1, "Model Choice & Architecture Trade-offs",
        "ML Fundamentals – Model Choice",
        "Whether the candidate can reason about architecture choices, not just name models",
        m["q1_arch"],
        q1_probes,
    ))
    cell5 = code_cell(q_code(
        1, "Model Choice & Architecture Trade-offs",
        "ML Fundamentals – Model Choice",
        "Architecture selection, trade-off reasoning, simplicity vs. power",
        {
            "No Hire": [
                "Names a model without justification ('I'd use a transformer')",
                "Cannot explain what breaks with a simpler baseline",
                "Unaware of the alternatives listed in the problem",
            ],
            "Weak Hire": [
                "Identifies the right architecture but cannot explain the mechanism",
                "Aware of alternatives but frames them as 'worse' without trade-off depth",
                "Does not connect architecture choice to scale constraints",
            ],
            "Hire": [
                "Compares at least 3 alternatives with specific pros/cons",
                "Connects architecture to scale and latency constraints",
                "Proactively mentions what breaks with the simpler approach",
            ],
            "Strong Hire": [
                "Proposes a staged rollout: launch with baseline, migrate to full model when justified",
                "Identifies non-obvious failure modes of the recommended architecture",
                "Connects architecture choice to downstream serving costs and org implications",
            ],
        },
        ["Cannot explain why the simplest baseline is insufficient",
         "Proposes a model that is computationally infeasible at stated scale"],
    ))

    # ── Q2: Training Strategy ─────────────────────────────────────────────────
    cell6 = md_cell(q_markdown(
        2, "Training Strategy & Label Construction",
        "ML Fundamentals – Training Strategy",
        "Whether the candidate understands label quality, loss function design, and class imbalance",
        m["q2_training"],
        [
            "How would you handle the class imbalance between positive and negative labels?",
            "What is your loss function and why — specifically, why not cross-entropy in this case?",
            "What happens if your label construction is wrong — what does the model learn instead?",
        ],
    ))
    cell7 = code_cell(q_code(
        2, "Training Strategy & Label Construction",
        "ML Fundamentals – Training Strategy",
        "Label quality, loss function design, class imbalance handling",
        {
            "No Hire": [
                "Uses raw clicks or events as labels without questioning quality",
                "Cannot explain what loss function to use or why",
                "Unaware of class imbalance as a practical problem",
            ],
            "Weak Hire": [
                "Identifies the label quality issue but proposes naive fix (oversample positives)",
                "Chooses an appropriate loss function but cannot explain alternatives",
                "Mentions class imbalance but only proposes resampling, not loss weighting or focal loss",
            ],
            "Hire": [
                "Designs a principled label construction strategy with threshold justification",
                "Proposes appropriate loss function (weighted BCE, focal loss) with reasoning",
                "Addresses class imbalance with multiple strategies and discusses trade-offs",
            ],
            "Strong Hire": [
                "Anticipates label noise propagation and proposes a noise-robust training objective",
                "Discusses the relationship between label construction choices and business metric alignment",
                "Proposes an evaluation of label quality as a prerequisite to model training",
            ],
        },
        ["Proposes a label construction that is biased in a way that contradicts the business objective",
         "Cannot identify what the model would learn if given noisy labels"],
    ))

    # ── Q3: Data Pipeline ─────────────────────────────────────────────────────
    cell8 = md_cell(q_markdown(
        3, "Data Pipeline & Training-Serving Skew",
        "Systems & Scale – Data Pipeline",
        "Whether the candidate understands feature freshness, skew, and label delay",
        m["q3_data"],
        [
            "What features in your design are most at risk of training-serving skew and why?",
            "How do you detect skew after deployment — what does it look like in metrics?",
            "If your feature store goes down, what degrades gracefully vs. catastrophically?",
        ],
    ))
    cell9 = code_cell(q_code(
        3, "Data Pipeline & Training-Serving Skew",
        "Systems & Scale – Data Pipeline",
        "Feature freshness, training-serving skew, label delay, feature store design",
        {
            "No Hire": [
                "Unaware that training and serving feature computation can diverge",
                "No concept of feature freshness requirements or staleness impact",
                "Cannot describe what a feature store does",
            ],
            "Weak Hire": [
                "Aware of training-serving skew conceptually but cannot identify specific skew risks",
                "Mentions feature store but treats it as a simple database",
                "Addresses label delay only by 'waiting longer' before training",
            ],
            "Hire": [
                "Identifies the specific features at risk of skew in this system",
                "Proposes shared feature computation between training and serving to prevent skew",
                "Designs a freshness SLA per feature type (batch vs. real-time vs. near-real-time)",
            ],
            "Strong Hire": [
                "Designs a skew detection system (distribution comparison between training logs and live traffic)",
                "Proposes a graceful degradation strategy when real-time features are unavailable",
                "Discusses label delay compensation strategies (delayed feedback models, importance weighting)",
            ],
        },
        ["Proposes a serving architecture that is architecturally impossible to keep fresh",
         "Unaware that offline model performance can appear good while live performance degrades due to skew"],
    ))

    # ── Q4: Serving Architecture ──────────────────────────────────────────────
    cell10 = md_cell(q_markdown(
        4, "Serving Architecture & Latency Design",
        "Systems & Scale – Serving Architecture",
        "Whether the candidate can design a production-ready serving system under latency constraints",
        m["q4_serving"],
        [
            "Walk me through your latency budget breakdown — how many milliseconds does each stage get?",
            "What happens to the user experience if your heavy ranking model is slow by 2x?",
            "How do you cache in this system — what is the cache key, TTL, and invalidation strategy?",
        ],
    ))
    cell11 = code_cell(q_code(
        4, "Serving Architecture & Latency Design",
        "Systems & Scale – Serving Architecture",
        "Latency budget allocation, caching, graceful degradation, multi-stage pipeline",
        {
            "No Hire": [
                "Proposes single-stage scoring of all items (computationally infeasible at scale)",
                "No concept of latency budget allocation across pipeline stages",
                "No caching or fallback strategy",
            ],
            "Weak Hire": [
                "Describes a multi-stage pipeline but cannot allocate latency budgets to stages",
                "Mentions caching but only for the final result (no intermediate caching)",
                "Fallback is 'use default recommendations' without discussing staleness impact",
            ],
            "Hire": [
                "Explicitly allocates latency budget across retrieval, ranking, re-ranking stages",
                "Designs intermediate caching (user embedding cache, candidate pre-fetch)",
                "Proposes a graceful degradation mode that degrades quality not availability",
            ],
            "Strong Hire": [
                "Designs an adaptive pipeline that dynamically adjusts stage depth based on remaining latency budget",
                "Discusses the cache warming strategy for new models and cold starts",
                "Connects serving architecture choices to infrastructure cost and organizational ownership",
            ],
        },
        ["Proposes a serving latency that is technically impossible given stated scale",
         "No awareness of the multi-stage funnel requirement for billion-scale candidate spaces"],
    ))

    # ── Q5: Evaluation & Metrics ─────────────────────────────────────────────
    cell12 = md_cell(q_markdown(
        5, "Evaluation Design & Offline-Online Gap",
        "Evaluation & Metrics",
        "Whether the candidate can design rigorous evaluation and understands metric gaming",
        m["q5_metrics"],
        [
            "What metric can your system game without actually improving user value?",
            "How long do you run your A/B test, and how do you account for day-of-week effects?",
            "Your offline metric went up 3% but online metric went down — what are the three most likely causes?",
        ],
    ))
    cell13 = code_cell(q_code(
        5, "Evaluation Design & Offline-Online Gap",
        "Evaluation & Metrics",
        "Offline metric selection, A/B test design, gaming risk, offline-online gap",
        {
            "No Hire": [
                "Proposes only one metric (e.g., accuracy or CTR) without considering gaming or gaps",
                "No awareness that offline improvements do not always translate online",
                "Cannot design an A/B test or explain what 'statistical significance' means",
            ],
            "Weak Hire": [
                "Mentions both offline and online metrics but cannot explain when they diverge",
                "Designs a simple A/B test but misses confounders (novelty effects, day-of-week)",
                "Identifies one gameable metric but cannot propose a non-gameable alternative",
            ],
            "Hire": [
                "Pairs a hard-to-game offline metric with the right online business metric",
                "Designs an A/B test with appropriate duration, sample size, and holdout strategy",
                "Explains at least two root causes of offline-online divergence (distribution shift, feedback loops)",
            ],
            "Strong Hire": [
                "Designs a counter-metric (guardrail) to prevent gaming while optimizing the primary metric",
                "Proposes a holdback experiment design to measure long-term vs. short-term effects",
                "Discusses the cost of running A/B tests (opportunity cost, traffic splitting) and when to use interleaving or bandits instead",
            ],
        },
        ["Proposes an evaluation design that has a causal identification problem (e.g., no holdout for long-term effects)",
         "Unaware that their primary metric can be gamed by the model in a way that hurts users"],
    ))

    # ── Q6: Edge Cases (module-specific) ────────────────────────────────────
    cell14 = md_cell(q_markdown(
        6, f"Edge Cases & Failure Modes: {m['q6_edge'][:50]}...",
        "Edge Cases & Failure Modes",
        "Module-specific edge cases, adversarial inputs, and systemic failure patterns",
        f"Walk me through how your system handles: {m['q6_edge']}",
        [
            "How does this failure mode manifest in your metrics — what is the first signal you would see?",
            "Is this failure mode detectable before it reaches users, or only after?",
            "What is the worst-case scenario if this failure mode goes undetected for 24 hours?",
        ],
    ))
    cell15 = code_cell(q_code(
        6, f"Edge Cases: {m['q6_edge'][:50]}...",
        "Edge Cases & Failure Modes",
        f"Handling: {m['q6_edge'][:80]}",
        {level: [signal] for level, signal in zip(
            ["No Hire", "Weak Hire", "Hire", "Strong Hire"],
            m["q6_signals"]
        )},
        [
            "Cannot identify the mechanism by which this failure mode propagates",
            "Proposes a mitigation that introduces a new, worse failure mode",
        ],
    ))

    # ── Q7: Principal-Level (module-specific) ────────────────────────────────
    cell16 = md_cell(q_markdown(
        7, "Principal-Level: Platform & Org Thinking",
        "Principal-Level Thinking",
        "Build vs. buy, platform/cross-surface architecture, organizational implications",
        m["q7_principal"],
        [
            "Who owns this platform if it is built — which team, and what is their mandate?",
            "What is the SLA you would commit to for downstream consumers of this platform?",
            "How do you handle a breaking change to the shared platform that degrades one consumer?",
        ],
    ))
    cell17 = code_cell(q_code(
        7, "Principal-Level: Platform & Org Thinking",
        "Principal-Level Thinking",
        f"Build vs. buy, platform thinking: {m['q7_principal'][:80]}",
        {level: [signal] for level, signal in zip(
            ["No Hire", "Weak Hire", "Hire", "Strong Hire"],
            m["q7_signals"]
        )},
        [
            "Treats this as purely a technical decision without org/ownership discussion",
            "Proposes a platform that creates a single point of failure without mitigation",
        ],
    ))

    # ── Cell 18: Cross-cutting patterns ─────────────────────────────────────
    cell18 = md_cell(f"""## Cross-Cutting Patterns: Staff vs. Senior

The questions above probe individual dimensions.
Here is what distinguishes a **Staff Engineer** from a **Strong Senior** across all 7 questions.

| Dimension | Strong Senior (Weak Hire at Staff bar) | Staff Engineer (Hire) |
|-----------|----------------------------------------|-----------------------|
| Problem framing | Picks one approach and defends it | Compares approaches, chooses with explicit trade-off reasoning |
| Scale awareness | Knows scale is a constraint | Quantifies the constraint and derives architecture from it |
| Data thinking | Assumes clean labels exist | Designs label construction and proactively flags data quality risks |
| Metrics | Names the right metric | Explains when the metric fails and proposes a complementary counter-metric |
| Production | Describes the model | Describes the full system: pipeline, serving, monitoring, fallback |
| Edge cases | Addresses edge cases when asked | Proactively surfaces edge cases before being asked |
| Org/platform | Treats this as a one-off system | Asks "should this be a shared platform?" and reasons about the answer |

### The Principal Multiplier

A **Principal Engineer (Strong Hire)** does everything the Staff engineer does, plus:
1. **Expands scope without being asked** — identifies that solving this problem creates a platform opportunity
2. **Quantifies second-order effects** — feedback loops, org incentives, tech debt from the current design
3. **Drives to a decision under uncertainty** — does not need complete information to recommend a path
4. **Identifies the problem not asked** — the most important constraint or risk that the interviewer did not raise

### Red Flags That Override Adequate Answers

These signals disqualify a candidate even if their technical answer is otherwise correct:

1. **No clarification** — jumping to a solution without asking about scale, latency, or data
2. **One-option thinking** — proposing a solution without comparing alternatives
3. **Offline-only evaluation** — no mention of A/B testing or online metrics
4. **No production awareness** — designing a model with no serving, monitoring, or fallback plan
5. **Confident incorrectness** — stating something technically wrong with high confidence and not updating when probed""")

    # ── Cell 19: Sample candidate scorecard ──────────────────────────────────
    cell19 = code_cell(f'''# ─── Sample Candidate Scorecard Visualization ──────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Example scores for a "Weak Hire → Hire" candidate
# 0 = No Hire, 1 = Weak Hire, 2 = Hire, 3 = Strong Hire
SAMPLE_SCORES = {{
    "Q1 Model Choice":       2,   # Hire
    "Q2 Training Strategy":  1,   # Weak Hire
    "Q3 Data Pipeline":      2,   # Hire
    "Q4 Serving Arch":       2,   # Hire
    "Q5 Evaluation":         1,   # Weak Hire
    "Q6 Edge Cases":         1,   # Weak Hire  (module-specific)
    "Q7 Principal Level":    1,   # Weak Hire
}}

level_names = ["No Hire", "Weak Hire", "Hire", "Strong Hire"]
level_colors = ["#d32f2f", "#f57c00", "#388e3c", "#1b5e20"]

fig, (ax_bar, ax_text) = plt.subplots(1, 2, figsize=(16, 6),
    gridspec_kw={{"width_ratios": [3, 1]}})

# Bar chart of scores
questions = list(SAMPLE_SCORES.keys())
scores = list(SAMPLE_SCORES.values())
bar_colors = [level_colors[s] for s in scores]

bars = ax_bar.barh(questions, scores, color=bar_colors, edgecolor="#333", linewidth=1.5)
ax_bar.set_xlim(-0.2, 3.5)
ax_bar.set_xticks([0, 1, 2, 3])
ax_bar.set_xticklabels(level_names, fontsize=9)
ax_bar.set_title(f"Sample Candidate Scorecard — {title}", fontsize=13, fontweight="bold")
ax_bar.axvline(1.5, color="#999", linestyle="--", linewidth=1, alpha=0.7, label="Hire threshold")
ax_bar.legend(loc="lower right", fontsize=9)
for bar, score in zip(bars, scores):
    ax_bar.text(score + 0.05, bar.get_y() + bar.get_height()/2,
                level_names[score], va="center", fontsize=9, fontweight="bold")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# Decision summary panel
avg_score = np.mean(scores)
hire_count = sum(1 for s in scores if s >= 2)
decision_color = "#388e3c" if avg_score >= 1.5 else "#d32f2f"
decision_text = "HIRE" if avg_score >= 2.0 else ("WEAK HIRE" if avg_score >= 1.5 else "NO HIRE")

ax_text.axis("off")
ax_text.add_patch(mpatches.FancyBboxPatch((0.05, 0.3), 0.9, 0.6,
    boxstyle="round,pad=0.05", facecolor=decision_color, edgecolor="#222", linewidth=2, alpha=0.85))
ax_text.text(0.5, 0.72, "FINAL DECISION", ha="center", va="center",
             fontsize=11, fontweight="bold", color="white")
ax_text.text(0.5, 0.55, decision_text, ha="center", va="center",
             fontsize=22, fontweight="bold", color="white")
ax_text.text(0.5, 0.40, f"Avg score: {{avg_score:.1f}} / 3.0\\nHire-level Qs: {{hire_count}} / 7",
             ha="center", va="center", fontsize=10, color="white")
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)
ax_text.text(0.5, 0.20,
    "Decision rule:\\n≥2.0 avg = Hire\\n1.5–2.0 avg = Weak Hire\\n<1.5 avg = No Hire",
    ha="center", va="center", fontsize=8, color="#333",
    bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="#999"))

plt.tight_layout()
plt.show()

print(f"\\nScorecard Summary for {title}:")
for q, s in SAMPLE_SCORES.items():
    marker = "✓" if s >= 2 else ("~" if s == 1 else "✗")
    print(f"  {{marker}} {{q}}: {{level_names[s]}}")
print(f"\\n  Average: {{avg_score:.2f}} → {{decision_text}}")
print("\\nNote: Disqualifying signals override the average score.")''')

    # ── Cell 20: 45-minute interviewer agenda ────────────────────────────────
    cell20 = code_cell(f'''# ─── Interviewer 45-Minute Agenda ─────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

agenda = [
    (0,  5,  "Opening & Problem Setup\\n(frame the problem, confirm scope)", "#fff9c4", "#f57f17"),
    (5,  8,  "Q1: Model Choice\\n(architecture trade-offs)", "#e3f2fd", "#1565c0"),
    (13, 6,  "Q2: Training Strategy\\n(labels, loss, imbalance)", "#e3f2fd", "#1565c0"),
    (19, 6,  "Q3: Data Pipeline\\n(freshness, skew, label delay)", "#e8f5e9", "#2e7d32"),
    (25, 5,  "Q4: Serving Architecture\\n(latency, caching, fallback)", "#e8f5e9", "#2e7d32"),
    (30, 5,  "Q5: Evaluation & Metrics\\n(offline/online gap, A/B design)", "#f3e5f5", "#6a1b9a"),
    (35, 5,  "Q6: Edge Cases (module-specific)\\n(adversarial, failure modes)", "#fff3e0", "#e65100"),
    (40, 5,  "Q7: Principal-Level\\n(platform, build vs. buy, org)", "#e0f7fa", "#00695c"),
]

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 50)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title(f"Interviewer 45-Minute Agenda — {title}",
             fontsize=14, fontweight="bold", pad=15)

for (start, dur, label, fc, ec) in agenda:
    rect = mpatches.FancyBboxPatch((start + 0.15, 1.5), dur - 0.3, 3.0,
        boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(start + dur/2, 3.2, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=ec)
    ax.text(start + dur/2, 1.9, f"{{dur}}m", ha="center", fontsize=8, style="italic")

# Timeline
for t in range(0, 46, 5):
    ax.text(t, 1.0, str(t), ha="center", fontsize=9)
    ax.plot([t, t], [1.1, 1.3], "k-", linewidth=1)
ax.plot([0, 45], [1.2, 1.2], "k-", linewidth=2)
ax.text(22.5, 0.3, "Minutes", ha="center", fontsize=10, color="#444")

plt.tight_layout()
plt.show()

print("""
Interviewer Notes:
  - Spend the first 5 min framing the problem clearly; do NOT skip this.
  - Q1-Q2 are warm-up: calibrate the candidate's level before going deep.
  - Q6 and Q7 are differentiators: most candidates struggle here.
  - If the candidate is clearly a Strong Hire after Q4, skip Q6 and go deep on Q7.
  - If the candidate is clearly a No Hire after Q2, you can wrap early with Q5 for data.
  - Always leave 2 min at the end for candidate questions — it is part of the evaluation.
""")''')

    # ── Cell 21: Key takeaways ────────────────────────────────────────────────
    cell21 = md_cell(f"""## Key Takeaways for the Interviewer

### What You Are Evaluating

1. **Structured thinking** — Does the candidate decompose the problem systematically or jump around?
2. **Depth of trade-off reasoning** — Can they compare approaches at a technical level, not just name them?
3. **Production awareness** — Do they design systems or models? Systems include serving, monitoring, and fallback.
4. **Proactive edge case coverage** — Do they surface problems before being asked, or only when prompted?
5. **Principal-level scope expansion** — Do they identify platform opportunities or org implications unprompted?

### Calibration Reminders

- A **Weak Hire** is not a bad engineer — they are a great senior who needs more time at this scope.
- A **Strong Hire** should make you think "I want them on my team today and would work for them in 3 years."
- **Disqualifying signals override averages** — a candidate who scores Hire on 6 questions but
  confidently states something technically wrong on Q2 and does not update when probed is a No Hire.

### After the Interview

1. Write your feedback immediately — memory degrades within 30 minutes.
2. Anchor on concrete signals ("candidate said X when asked about Y") not impressions ("felt confident").
3. Calibrate with the hiring committee using this rubric — ensure interviewers share the same bar.
4. If unsure between Weak Hire and Hire: ask yourself "would I trust this person to own a critical production system solo?" Staff = yes, Senior = not yet.

---

*This notebook is part of the ML Design interview preparation series.
For the candidate-facing preparation notebook, see `04_interview_walkthrough.ipynb` in this module.*""")

    # ── Assemble all cells ────────────────────────────────────────────────────
    cells = [
        cell1, cell2, cell3,
        cell4, cell5,
        cell6, cell7,
        cell8, cell9,
        cell10, cell11,
        cell12, cell13,
        cell14, cell15,
        cell16, cell17,
        cell18,
        cell19,
        cell20,
        cell21,
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    created = []
    for m in MODULES:
        module_dir = os.path.join(BASE, m["dir"])
        out_path = os.path.join(module_dir, m["filename"])
        nb = build_notebook(m)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        created.append(out_path)
        print(f"  ✓  {m['num']} — {m['title']}: {out_path}")

    print(f"\nCreated {len(created)} notebooks.")
