"""Quiz questions for Module 08: Ad Click Prediction"""

QUESTIONS = [
    {
        "concept_id": "ads_ctr_formulation",
        "module": "08-ad-click-prediction",
        "question": "Ad click prediction is formulated as which type of ML problem?",
        "choices": [
            "A. Regression predicting exact click count",
            "B. Binary classification predicting P(click | user, ad, context) — calibrated probability is critical because it feeds into auction pricing (expected revenue = P(click) × bid)",
            "C. Multi-class classification predicting which ad category gets clicked",
            "D. Clustering users by click behavior"
        ],
        "correct": "B",
        "hint": "The output must be a probability, not just a ranking score. Why?",
        "explanation": "P(click) is directly used in Vickrey-style ad auctions: expected revenue = P(click) × advertiser bid. A miscalibrated model that predicts 0.5 for a 0.01% CTR ad causes the auction to massively overprice that ad slot. Calibration (Platt scaling, isotonic regression) is a required post-processing step. Simple AUC optimization doesn't guarantee calibration.",
        "difficulty": 3,
        "tags": ["formulation", "calibration", "auction"]
    },
    {
        "concept_id": "ads_sparse_features",
        "module": "08-ad-click-prediction",
        "question": "Ad click models have billions of sparse categorical features (user IDs, ad IDs, keyword IDs). How do you handle this?",
        "choices": [
            "A. One-hot encode everything",
            "B. Learned embedding tables: each categorical ID maps to a dense embedding vector (16-256 dim), trained end-to-end. IDs with few occurrences are hashed to shared embeddings.",
            "C. Use only numerical features",
            "D. Drop rare IDs from training"
        ],
        "correct": "B",
        "hint": "One-hot encoding 1B user IDs creates a 1B-dim vector. What's the alternative?",
        "explanation": "Embedding tables map each categorical ID to a dense low-dimensional vector. The table for user IDs has one row per user (up to 1B rows × 32 dims = 32 GB). For rare IDs (appearing < 5 times), use feature hashing to map to a shared embedding bucket. This dramatically reduces dimensionality while capturing latent patterns. Embedding tables are the core of models like DLRM (Facebook) and Wide & Deep (Google).",
        "difficulty": 3,
        "tags": ["sparse_features", "embeddings", "feature_engineering"]
    },
    {
        "concept_id": "ads_wide_deep",
        "module": "08-ad-click-prediction",
        "question": "Google's Wide & Deep model for ad click prediction combines a 'wide' and 'deep' component. What does each component capture?",
        "choices": [
            "A. Wide: recent data; Deep: older data",
            "B. Wide: memorization of specific feature co-occurrences (hand-crafted cross features like user_city × app_category); Deep: generalization via embedding-based representations that handle unseen feature combinations",
            "C. Wide: text features; Deep: image features",
            "D. Wide: user features; Deep: ad features"
        ],
        "correct": "B",
        "hint": "Memorization vs. generalization — what does each component specialize in?",
        "explanation": "Wide component: linear model on raw and crossed features — memorizes patterns like 'users in SF who searched for coffee click coffee shop ads'. Deep component: feed-forward network on dense embeddings — generalizes to new user/ad combinations by learning latent representations. Wide without deep: can't generalize to unseen combos. Deep without wide: misses specific high-frequency patterns that are hard to generalize.",
        "difficulty": 4,
        "tags": ["wide_deep", "architecture", "memorization_generalization"]
    },
    {
        "concept_id": "ads_feature_cross",
        "module": "08-ad-click-prediction",
        "question": "Why are feature interactions ('crosses') so important in CTR prediction?",
        "choices": [
            "A. They reduce model size",
            "B. Individual features have limited predictive power; the interaction (e.g., 'gender=male AND category=sports') is much more predictive than either feature alone",
            "C. Feature crosses improve training speed",
            "D. They eliminate the need for deep networks"
        ],
        "correct": "B",
        "hint": "Knowing someone is male is mildly predictive of sports clicks. Knowing they searched for 'NFL' is predictive. Knowing both is much more predictive than either alone.",
        "explanation": "CTR signals are highly non-linear and context-dependent. A user who is female AND searching for beauty products AND the ad is for a beauty brand has very high CTR probability — no single feature alone captures this. Hand-crafted crosses (Wide component) and learned crosses (DCN, DeepFM, xDeepFM) both try to capture these multiplicative feature interactions.",
        "difficulty": 3,
        "tags": ["feature_interaction", "feature_cross", "ctr"]
    },
    {
        "concept_id": "ads_delayed_feedback",
        "module": "08-ad-click-prediction",
        "question": "Ad conversions (purchases) can happen hours or days after a click. How does this affect model training?",
        "choices": [
            "A. No impact — clicks are sufficient labels",
            "B. Delayed label problem: training on recent data requires waiting for conversion labels to arrive. Solutions: use click as immediate proxy label, use conversion with a delay window, or use a joint model that handles delayed feedback explicitly.",
            "C. Only use data older than 30 days",
            "D. Delete unconverted clicks from training"
        ],
        "correct": "B",
        "hint": "A click from 1 hour ago has a 'conversion = 0' label, but the user might convert in 3 hours.",
        "explanation": "Delayed feedback in conversion prediction: attributing a 0 label to recent clicks that haven't converted yet creates noisy labels. Solutions: (1) Elapsed time features (how long since click), (2) Training with a delay window (only use clicks > 24h old as training examples), (3) Fake negative correction: model that estimates P(conversion | click, elapsed_time) and reweights accordingly.",
        "difficulty": 4,
        "tags": ["delayed_feedback", "conversion", "label_noise"]
    },
    {
        "concept_id": "ads_auction_mechanics",
        "module": "08-ad-click-prediction",
        "question": "In a second-price auction for ad slots, what is the winning advertiser's payment?",
        "choices": [
            "A. Their bid × P(click)",
            "B. The second-highest bid (or the minimum needed to beat the second-place bidder)",
            "C. The first-place bid",
            "D. A fixed rate per impression"
        ],
        "correct": "B",
        "hint": "Second-price auction: you win by bidding highest, but pay the second-highest price.",
        "explanation": "Second-price (Vickrey) auctions: the winner pays the second-highest bid, not their own bid. This makes truthful bidding the dominant strategy — there's no advantage to strategic underbidding. In practice, Google/Facebook use modified VCG auctions that also incorporate ad quality (CTR × bid score), not just the raw bid, to prevent low-quality ads from dominating by overbidding.",
        "difficulty": 3,
        "tags": ["auction_mechanics", "pricing", "mechanism_design"]
    },
    {
        "concept_id": "ads_position_bias_ads",
        "module": "08-ad-click-prediction",
        "question": "An ad shown in position 1 gets 5× more clicks than the same ad in position 5. How do you prevent position bias from corrupting your CTR model?",
        "choices": [
            "A. Only train on position-1 impressions",
            "B. Include position as a feature at training time but remove it at inference time (so the model scores the ad's intrinsic quality, not its position). Or use inverse propensity scoring to reweight training examples.",
            "C. Multiply all CTR predictions by 5 for position-1",
            "D. Ignore position — it averages out"
        ],
        "correct": "B",
            "hint": "You want the model to learn ad quality, not ad position. How do you prevent them from conflating?",
        "explanation": "Position bias: an ad in position 1 gets more clicks purely due to visibility, not quality. If you train naively, the model learns 'position 1 ads are high CTR' rather than 'high-quality ads get shown in position 1'. Solution: add position as a training feature (model can correct for it) but zero it out at inference time (model evaluates ad quality independent of where it will be shown). Inverse propensity scoring: reweight training examples by 1/P(shown at that position).",
        "difficulty": 4,
        "tags": ["position_bias", "debiasing", "feature_engineering"]
    },
    {
        "concept_id": "ads_online_learning",
        "module": "08-ad-click-prediction",
        "question": "Why is online learning (continuous model updates from streaming data) especially important for ad click prediction?",
        "choices": [
            "A. It reduces model size",
            "B. User interests and ad relevance shift rapidly (trends, news, promotions). A model trained yesterday misses today's trending topics. Online learning keeps the model fresh.",
            "C. Online learning is always better than batch training",
            "D. It eliminates the need for feature engineering"
        ],
        "correct": "B",
        "hint": "Think about a major news event happening today. Would a model trained last week know about it?",
        "explanation": "Ad CTR has strong temporal patterns: a political event causes interest in related news ads to spike immediately. Seasonal trends (holidays, sports events) shift user intent daily. Online learning systems update model parameters continuously from streaming click data (using SGD variants that support incremental updates). This keeps the model aligned with current user intent without waiting for weekly batch retraining.",
        "difficulty": 3,
        "tags": ["online_learning", "streaming", "concept_drift"]
    },
    {
        "concept_id": "ads_negative_sampling",
        "module": "08-ad-click-prediction",
        "question": "In ad click prediction, impressions (ads shown but not clicked) vastly outnumber clicks. How do you handle this extreme class imbalance?",
        "choices": [
            "A. Upsample clicks to match impressions",
            "B. Downsample negative impressions (non-clicks) to a manageable ratio (e.g., 1:10 positive:negative), then correct the model's predicted probabilities using calibration",
            "C. Use class weights to upweight clicks",
            "D. Ignore non-clicks in training"
        ],
        "correct": "B",
        "hint": "Training on all impressions when CTR is 0.1% would make the dataset 999x larger than necessary.",
        "explanation": "Random downsampling of negatives: if CTR is 0.1%, train on all clicks + 10× random sample of non-clicks (10:1 ratio instead of 1000:1). After training, the model's raw score is systematically too high — correct with: p_corrected = p_raw / (p_raw + (1-p_raw)/sampling_rate). This recalibration is critical for maintaining accurate probability estimates for the auction.",
        "difficulty": 4,
        "tags": ["class_imbalance", "negative_sampling", "calibration"]
    },
    {
        "concept_id": "ads_multi_task_learning",
        "module": "08-ad-click-prediction",
        "question": "Ad systems predict both P(click) and P(conversion | click). Why train these jointly rather than sequentially?",
        "choices": [
            "A. Sequential training is always inferior",
            "B. Joint training allows shared representations to benefit from both signals — conversion labels are sparser but richer, improving the click model's embeddings. Also eliminates training-serving skew between two separate models.",
            "C. Joint training is faster",
            "D. There is no benefit — train them separately"
        ],
        "correct": "B",
        "hint": "Conversion data is sparse. Can it still help the click prediction model?",
        "explanation": "Multi-task ad models (like Facebook's MMOE): shared bottom layers learn user/ad representations that are useful for both click and conversion prediction. Conversion labels (much sparser, ~1% of clicks convert) provide a richer quality signal that regularizes the click model. Joint optimization also ensures the two outputs are consistent, simplifying downstream auction integration.",
        "difficulty": 4,
        "tags": ["multi_task", "conversion", "joint_training"]
    },
]
