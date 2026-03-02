"""Quiz questions for Module 01: ML Design Prep Framework"""

QUESTIONS = [
    {
        "concept_id": "prep_clarification_first",
        "module": "01-ml-design-prep",
        "question": "In an ML system design interview, why should you spend the first 3-5 minutes on clarification rather than jumping into architecture?",
        "choices": [
            "A. It buys time to think of the architecture",
            "B. The problem scope (scale, latency, precision/recall tradeoff) fundamentally changes the architecture — getting this wrong invalidates the whole design",
            "C. Interviewers expect you to ask questions as a formality",
            "D. Clarification only matters for coding interviews, not ML design"
        ],
        "correct": "B",
        "hint": "Think about what changes between a prototype for 1K users vs. a production system for 1B users.",
        "explanation": "Scale (1M vs 1B users) changes whether you need distributed training, approximate nearest neighbors, or sharded serving. Latency SLA (real-time vs. batch) changes the entire inference stack. Precision vs recall priority changes the loss function. Getting these wrong wastes 40 minutes designing the wrong system.",
        "difficulty": 2,
        "tags": ["interview_strategy", "framework"]
    },
    {
        "concept_id": "prep_7step_order",
        "module": "01-ml-design-prep",
        "question": "What is the correct order of the 7-step ML design framework?",
        "choices": [
            "A. Data → Model → Metrics → Problem → Training → Serving → Monitoring",
            "B. Problem Definition → Metrics → Data → Features → Model → Training → Serving",
            "C. Model → Data → Metrics → Problem → Features → Serving → Training",
            "D. Metrics → Problem → Data → Model → Features → Training → Serving"
        ],
        "correct": "B",
        "hint": "You must know what you're solving and measuring before deciding how to model it.",
        "explanation": "Problem Definition first anchors everything. Metrics second because they drive model choice. Data and features before model because you need to know what's available. Model after data. Training after model. Serving last because it depends on the trained model architecture.",
        "difficulty": 2,
        "tags": ["framework", "interview_strategy"]
    },
    {
        "concept_id": "prep_proxy_metric",
        "module": "01-ml-design-prep",
        "question": "An interviewer asks: 'How do you measure success for a content recommendation system?' What is the strongest answer?",
        "choices": [
            "A. Accuracy on a held-out test set",
            "B. Click-through rate (CTR)",
            "C. A hierarchy: business metric (revenue/DAU) → proxy metric (CTR, watch time) → offline metric (AUC, NDCG), with a plan to validate proxy tracks business",
            "D. User satisfaction surveys"
        ],
        "correct": "C",
        "hint": "Think about what you can optimize directly (offline) vs. what actually matters (business).",
        "explanation": "Business metrics (revenue, DAU) are the true goal but can't be directly optimized. Offline metrics (AUC, NDCG) are optimizable but may not correlate with business outcomes. A strong answer defines all three layers and explains how to validate the proxy-business correlation via A/B tests.",
        "difficulty": 3,
        "tags": ["metrics", "interview_strategy"]
    },
    {
        "concept_id": "prep_train_serve_skew",
        "module": "01-ml-design-prep",
        "question": "What is training-serving skew and why is it critical to prevent?",
        "choices": [
            "A. When the training dataset is larger than the serving dataset",
            "B. When features computed at training time differ from features computed at inference time, causing silent accuracy degradation in production",
            "C. When the model is trained on GPU but served on CPU",
            "D. When batch size at training differs from batch size at serving"
        ],
        "correct": "B",
        "hint": "Think about how a feature like 'user's last 7 days of clicks' could be computed differently during training vs. live serving.",
        "explanation": "Training-serving skew happens when: (1) feature pipelines use different code paths at train vs. serve time, (2) historical data is preprocessed differently than live data, (3) label leakage causes a feature to encode future information at training but not serving. It's insidious because the model looks fine offline but degrades silently in production.",
        "difficulty": 3,
        "tags": ["training", "serving", "production"]
    },
    {
        "concept_id": "prep_online_offline_gap",
        "module": "01-ml-design-prep",
        "question": "Your model achieves 0.92 AUC offline but only 0.5% CTR improvement in A/B test. What are the most likely root causes?",
        "choices": [
            "A. The model overfits to the training set",
            "B. AUC measures ranking quality on a static dataset; CTR is affected by selection bias (users only click shown items), position bias, and novelty effects not captured offline",
            "C. The A/B test sample size was too small",
            "D. The model needs more features"
        ],
        "correct": "B",
        "hint": "What does your offline test set actually represent vs. what the model sees in production?",
        "explanation": "Offline evaluation uses logged data with selection bias (only items previously shown). High AUC means the model ranks previously-shown items well but doesn't predict how users react to newly-surfaced items. Additionally, novelty, diversity, and position effects dominate live CTR but aren't in the offline dataset.",
        "difficulty": 4,
        "tags": ["metrics", "offline_evaluation", "bias"]
    },
    {
        "concept_id": "prep_label_definition",
        "module": "01-ml-design-prep",
        "question": "For a 'time-to-hire' ML model predicting interview success, what is the most important data challenge?",
        "choices": [
            "A. Class imbalance (most candidates don't get hired)",
            "B. Label delay: you don't know the outcome for weeks/months, making it hard to train on recent data",
            "C. Feature dimensionality is too high",
            "D. The model needs a GPU to train"
        ],
        "correct": "B",
        "hint": "How long after a candidate applies do you know whether they were hired?",
        "explanation": "Label delay means your training data always lags reality. If the hiring process takes 6 weeks, you can't use data from the last 6 weeks for training (labels are unknown). This creates a bias toward older hiring patterns and makes the model stale for recent candidates.",
        "difficulty": 3,
        "tags": ["data", "labels", "training"]
    },
    {
        "concept_id": "prep_capacity_estimation",
        "module": "01-ml-design-prep",
        "question": "A system serves 1M users/day with a 100ms P99 latency SLA and each model inference takes 50ms. What is the minimum number of servers needed (assume 1 request/user/day, CPU-only, single-threaded)?",
        "choices": [
            "A. 1 server",
            "B. About 12 servers",
            "C. 100 servers",
            "D. 1000 servers"
        ],
        "correct": "B",
        "hint": "Calculate QPS first, then how many requests a single thread can handle per second.",
        "explanation": "1M requests/day = ~11.6 QPS. One thread handling 50ms inference can do 20 req/sec. So 11.6/20 ≈ 0.58 threads needed. But with P99 latency headroom, redundancy, and non-uniform traffic (peak ~3x average), you need ~3x → 2 servers minimum, but best practice is ~12 for reliability and traffic spikes.",
        "difficulty": 5,
        "tags": ["capacity_planning", "serving", "math"]
    },
    {
        "concept_id": "prep_precision_recall_tradeoff",
        "module": "01-ml-design-prep",
        "question": "For a medical diagnosis model, you should optimize for:",
        "choices": [
            "A. High precision (minimize false positives)",
            "B. High recall (minimize false negatives)",
            "C. Equal precision and recall (F1 score)",
            "D. High accuracy"
        ],
        "correct": "B",
        "hint": "Think about the cost asymmetry: missing a sick patient vs. over-diagnosing a healthy one.",
        "explanation": "Missing a true positive (diseased patient classified as healthy) can be fatal. A false positive (healthy patient classified as diseased) leads to further tests, not harm. High recall minimizes false negatives. For fraud detection or content moderation, the tradeoff is different — excessive false positives frustrate users.",
        "difficulty": 2,
        "tags": ["metrics", "loss_function"]
    },
    {
        "concept_id": "prep_ab_test_design",
        "module": "01-ml-design-prep",
        "question": "You want to A/B test a new recommendation model. What is the minimum sample size consideration you must discuss?",
        "choices": [
            "A. At least 100 users in each bucket",
            "B. Statistical power (typically 80%), significance level (typically 5%), and minimum detectable effect — together these determine the required sample size",
            "C. Exactly 50% of traffic in each bucket",
            "D. At least 1 week of data"
        ],
        "correct": "B",
        "hint": "Think about what 'statistically significant' actually means quantitatively.",
        "explanation": "Power = P(detect true effect). At 80% power and 5% significance, a 0.1% CTR lift on a baseline of 2% requires ~200K users per bucket. Smaller effects need exponentially more users. Getting this wrong means either underpowered tests (miss real improvements) or overpowered tests (waste months waiting).",
        "difficulty": 4,
        "tags": ["ab_testing", "statistics"]
    },
    {
        "concept_id": "prep_model_selection",
        "module": "01-ml-design-prep",
        "question": "An interviewer says 'design a CTR prediction model'. What should you ask before choosing between logistic regression vs. a deep neural network?",
        "choices": [
            "A. Whether the team prefers PyTorch or TensorFlow",
            "B. Scale (# features, # training examples), latency SLA, interpretability requirements, and team ML maturity",
            "C. Whether the interviewer prefers simpler models",
            "D. How many GPUs are available"
        ],
        "correct": "B",
        "hint": "The 'best' model depends entirely on constraints you don't know yet.",
        "explanation": "Logistic regression wins when: low latency needed (< 10ms), interpretability required (legal/compliance), sparse features (millions of categorical IDs), small team. Neural networks win when: dense feature interactions exist, data is abundant (billions of examples), latency budget is generous, team has ML infra. Jumping to 'use a transformer' without asking is a red flag.",
        "difficulty": 3,
        "tags": ["model_selection", "interview_strategy"]
    },
    {
        "concept_id": "prep_data_splits",
        "module": "01-ml-design-prep",
        "question": "For a time-series prediction model (e.g. next-day stock movement), why is random train/test split wrong?",
        "choices": [
            "A. Random splits use too much memory",
            "B. Random splits cause data leakage: the model sees future data during training, inflating offline metrics and creating training-serving skew",
            "C. Random splits are less computationally efficient",
            "D. Random splits only work for classification, not regression"
        ],
        "correct": "B",
        "hint": "If your test set contains data from Jan 2023 and training data contains data from Dec 2023, what happened?",
        "explanation": "With time-series data, you must split chronologically: train on past, validate/test on future. Random split lets the model 'see the future' — e.g., training on Dec 2023 data while being tested on Jan 2023. This causes catastrophically optimistic offline metrics that fail to generalize.",
        "difficulty": 3,
        "tags": ["data_splits", "time_series", "leakage"]
    },
]
