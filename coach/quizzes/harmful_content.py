"""Quiz questions for Module 05: Harmful Content Detection"""

QUESTIONS = [
    {
        "concept_id": "hcd_precision_recall",
        "module": "05-harmful-content-detection",
        "question": "For a hate speech detection model, which is worse: a false positive (flagging benign content) or a false negative (missing hate speech)?",
        "choices": [
            "A. False positive — penalizes innocent users",
            "B. False negative — allows harmful content to reach millions",
            "C. They are equally bad",
            "D. Depends only on the platform's revenue model"
        ],
        "correct": "B",
        "hint": "Scale matters — a single missed piece of hate speech can reach millions of users.",
        "explanation": "This is platform- and context-dependent, but for high-virality platforms, false negatives are typically more damaging. A false positive (removing a benign post) affects one user. A false negative (missing hate speech that goes viral) can incite real-world harm at scale. However, over-correction on false positives creates censorship concerns. The correct answer is: it depends on context, but at scale the asymmetry usually favors higher recall.",
        "difficulty": 3,
        "tags": ["precision_recall", "content_moderation", "tradeoffs"]
    },
    {
        "concept_id": "hcd_multimodal_hate",
        "module": "05-harmful-content-detection",
        "question": "A meme contains a benign image and benign text, but combined they create hate speech (e.g., a dog whistle). How do you detect this?",
        "choices": [
            "A. Only analyze the text",
            "B. Multimodal fusion: encode image and text separately then jointly analyze the combination, since hate speech often emerges from the image-text interaction, not either modality alone",
            "C. Only analyze the image",
            "D. Use keyword blocking"
        ],
        "correct": "B",
        "hint": "Hateful memes deliberately use benign components that are only harmful in combination.",
        "explanation": "Facebook's Hateful Memes Challenge showed that unimodal models (text-only or image-only) achieve only ~65% accuracy on hateful memes because each modality alone is benign. Multimodal models that learn cross-modal interactions achieve 80%+. This requires joint fusion (late fusion, cross-attention between image and text embeddings, or multimodal transformers like ViLBERT).",
        "difficulty": 4,
        "tags": ["multimodal", "memes", "fusion"]
    },
    {
        "concept_id": "hcd_class_imbalance",
        "module": "05-harmful-content-detection",
        "question": "Harmful content is 0.1% of your training data (1M harmful out of 1B posts). What is the risk of naively training a classifier?",
        "choices": [
            "A. The model will be too slow",
            "B. The model achieves 99.9% accuracy by predicting 'safe' for everything — class imbalance causes trivial solutions that appear good on accuracy but fail at the actual task",
            "C. The model will overfit",
            "D. The training data is too large"
        ],
        "correct": "B",
        "hint": "What accuracy does a model achieve that predicts 'safe' for every single input?",
        "explanation": "With 0.1% harmful content, a trivial 'always safe' classifier gets 99.9% accuracy while catching zero harmful content. Solutions: (1) Class-weighted loss (upweight harmful class 1000x), (2) oversampling harmful examples (SMOTE or simple oversampling), (3) undersampling safe examples, (4) use PR-AUC / F1 on harmful class as the metric, not overall accuracy.",
        "difficulty": 2,
        "tags": ["class_imbalance", "metrics", "training"]
    },
    {
        "concept_id": "hcd_adversarial_users",
        "module": "05-harmful-content-detection",
        "question": "Bad actors learn your harmful content detector's rules and start bypassing it (e.g., substituting 'l3t5 k1ll th3m'). How do you design for adversarial robustness?",
        "choices": [
            "A. Update the keyword blocklist weekly",
            "B. Semantic-level detection (intent classification, not keyword matching), continuous adversarial training (add bypass attempts to training data), and human review escalation for near-threshold cases",
            "C. Block all messages with numbers",
            "D. Rate-limit users who post frequently"
        ],
        "correct": "B",
        "hint": "Keyword blocking is a game of whack-a-mole. What is the adversarially robust alternative?",
        "explanation": "Keyword-based systems are brittle against obfuscation. Semantic models that understand intent (not surface text) are harder to bypass. Adversarial training: add known bypass patterns to training data. Human-in-the-loop for borderline cases. Red team: hire people to find bypasses and add those patterns to training data before bad actors find them.",
        "difficulty": 4,
        "tags": ["adversarial_robustness", "content_moderation", "red_teaming"]
    },
    {
        "concept_id": "hcd_policy_ml_separation",
        "module": "05-harmful-content-detection",
        "question": "Should your ML model encode content policy rules (e.g., 'graphic violence is allowed in news contexts but not entertainment') directly in the model weights?",
        "choices": [
            "A. Yes — bake all rules into training labels",
            "B. No — policies change frequently; the ML model should output a score/classification, and a separate policy layer applies rules (context, user age, region) on top. This allows policy updates without retraining.",
            "C. Yes — one model per policy rule",
            "D. Policies don't matter for ML"
        ],
        "correct": "B",
        "hint": "How often does content policy change vs. how often can you retrain a large model?",
        "explanation": "Baking policies into model weights means every policy change requires retraining (weeks/months). The correct architecture separates concerns: (1) ML model produces a raw score (how harmful is this content?), (2) a rule engine applies context-dependent policies (news exception, regional laws, user age verification). Policy changes are then instant deployments of rule updates.",
        "difficulty": 3,
        "tags": ["system_design", "policy", "modularity"]
    },
    {
        "concept_id": "hcd_human_review",
        "module": "05-harmful-content-detection",
        "question": "How should you design the interface between ML classifiers and human reviewers?",
        "choices": [
            "A. Send all content to human reviewers",
            "B. ML classifier triages: high-confidence harmful → auto-remove, high-confidence safe → no review, uncertain (near decision boundary) → human review queue with ML-generated context",
            "C. Human reviewers only review appeals after users complain",
            "D. Use ML only and never involve humans"
        ],
        "correct": "B",
        "hint": "Human reviewers are expensive. How do you use their capacity most efficiently?",
        "explanation": "Three-tier triage: (1) High-confidence violations → auto-action (remove/restrict) with appeal mechanism, (2) Low-confidence/unclear → human review queue prioritized by virality/severity, (3) Clearly safe → no action. Human reviewers see ML confidence scores and reasoning to accelerate decision-making. This reduces human review volume by 90%+ while maintaining accuracy on borderline cases.",
        "difficulty": 3,
        "tags": ["human_review", "triage", "hybrid_system"]
    },
    {
        "concept_id": "hcd_context_dependence",
        "module": "05-harmful-content-detection",
        "question": "The word 'shoot' appears in a post. How should your content moderation system handle it?",
        "choices": [
            "A. Always flag it as potentially violent",
            "B. Use full contextual understanding: 'shoot a basketball' is safe; 'shoot that person' requires action. Context-aware models (BERT-style) understand surrounding text, not just keywords.",
            "C. Block all uses of the word",
            "D. Only act if a user reports it"
        ],
        "correct": "B",
        "hint": "English is ambiguous. The same word has drastically different meanings in context.",
        "explanation": "Keyword-level moderation fails on lexical ambiguity — 'shoot' appears in photography, basketball, filmmaking, and violent speech. Context-aware models (BERT, RoBERTa fine-tuned on moderation data) encode full sentence context. The challenge is also code-switching, sarcasm, and cultural context — these require increasingly sophisticated models and sometimes domain-specific fine-tuning.",
        "difficulty": 2,
        "tags": ["context", "nlp", "keyword_vs_semantic"]
    },
    {
        "concept_id": "hcd_speed_vs_accuracy",
        "module": "05-harmful-content-detection",
        "question": "Content must be moderated before it goes viral. How do you balance the speed-accuracy tradeoff?",
        "choices": [
            "A. Always use the most accurate model regardless of latency",
            "B. Tiered approach: fast lightweight model (< 10ms) for all content at upload time, escalate to heavy model for uncertain/high-virality content, heavy model runs async for non-urgent cases",
            "C. Only moderate content after it gets 100 views",
            "D. Use batch processing once per hour"
        ],
        "correct": "B",
        "hint": "Viral content can reach millions within minutes. When must moderation happen?",
        "explanation": "Tiered moderation: (1) Real-time screening (< 10ms, lightweight distilbert/fasttext) at upload — catches obvious violations immediately, (2) Near-real-time (< 1 min) medium model for uncertain cases, (3) Async deep analysis (minutes) for high-priority content. This architecture catches viral harmful content before it spreads while not blocking the upload pipeline.",
        "difficulty": 4,
        "tags": ["latency", "tiered_system", "real_time"]
    },
    {
        "concept_id": "hcd_annotator_disagreement",
        "module": "05-harmful-content-detection",
        "question": "Human annotators disagree 30% of the time on whether content is 'borderline' harmful. How do you handle this?",
        "choices": [
            "A. Take the majority vote and ignore disagreement",
            "B. Model annotator disagreement explicitly: use multiple annotations per item, treat it as a soft label (0.7 harmful), measure annotator agreement (Fleiss kappa), and focus model training on high-agreement examples",
            "C. Only use unambiguous examples in training",
            "D. Replace human annotators with GPT-4"
        ],
        "correct": "B",
        "hint": "Disagreement contains information — borderline content is genuinely ambiguous.",
        "explanation": "High annotator disagreement signals inherently ambiguous content. Best practices: (1) multiple annotations per item (3-5 annotators), (2) model soft labels (probability of harmful) rather than binary, (3) track inter-annotator agreement (Cohen's/Fleiss kappa) and audit low-kappa items for unclear policy, (4) train models to express uncertainty on high-disagreement examples rather than forcing a binary decision.",
        "difficulty": 4,
        "tags": ["annotation", "soft_labels", "uncertainty"]
    },
    {
        "concept_id": "hcd_explainability",
        "module": "05-harmful-content-detection",
        "question": "A user appeals a content removal decision. What does your ML system need to provide?",
        "choices": [
            "A. Just a confidence score",
            "B. Explainability: which parts of the content triggered the decision (attention visualization, LIME/SHAP token importance), which policy was violated, and how the user can modify content to comply",
            "C. The model architecture details",
            "D. A random sample of similar removed content"
        ],
        "correct": "B",
        "hint": "Users need to understand why they were moderated to (a) appeal effectively and (b) learn what is acceptable.",
        "explanation": "Explainable AI for content moderation: SHAP/LIME attributions show which tokens/regions contributed most to the decision. This serves three purposes: (1) user-facing: 'these specific words triggered the policy', (2) human reviewer: context for appeal decision, (3) model debugging: discover systemic biases or error patterns. Legal requirements in some jurisdictions (EU DSA) mandate explainability for automated moderation.",
        "difficulty": 3,
        "tags": ["explainability", "appeals", "regulatory_compliance"]
    },
]
