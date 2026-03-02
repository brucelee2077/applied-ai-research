"""Quiz questions for Module 03: Google Street View"""

QUESTIONS = [
    {
        "concept_id": "gsv_localization_task",
        "module": "03-google-street-view",
        "question": "Street View address localization is best framed as which ML task?",
        "choices": [
            "A. Pure image classification (which city?)",
            "B. Object detection + OCR + geo-spatial inference — detect address numbers, read them, map to GPS coordinates",
            "C. Image segmentation",
            "D. Generative image synthesis"
        ],
        "correct": "B",
        "hint": "Think about the pipeline: you have a raw street-level photo and need a precise GPS coordinate.",
        "explanation": "Street View localization is a multi-stage pipeline: (1) detect address number regions with an object detector (YOLO/Faster-RCNN), (2) OCR to read the number, (3) geo-spatial database lookup to map number + street context to GPS coordinates. Pure classification can't handle unseen addresses.",
        "difficulty": 2,
        "tags": ["pipeline", "object_detection", "ocr"]
    },
    {
        "concept_id": "gsv_data_augmentation",
        "module": "03-google-street-view",
        "question": "Street View imagery varies massively by weather, time of day, and region. What is the primary augmentation strategy to handle this?",
        "choices": [
            "A. Only train on clear daytime images",
            "B. Photometric augmentations (brightness, contrast, color jitter, rain simulation) + geometric augmentations (rotation, perspective transform) to make the model invariant to conditions",
            "C. Train separate models per region",
            "D. Use dropout to handle variability"
        ],
        "correct": "B",
        "hint": "How do you make a model that works in Tokyo rain and Phoenix noon sun?",
        "explanation": "Photometric augmentations simulate lighting and weather conditions. Geometric augmentations handle camera angle and mounting variations. Together they train models robust to the extreme distribution shift between sunny California and rainy London. Region-specific models are operationally infeasible at Google scale.",
        "difficulty": 3,
        "tags": ["data_augmentation", "robustness"]
    },
    {
        "concept_id": "gsv_map_accuracy",
        "module": "03-google-street-view",
        "question": "What is the appropriate offline metric to evaluate address number detection accuracy?",
        "choices": [
            "A. Accuracy (% of correctly classified images)",
            "B. mAP (mean Average Precision) at IoU 0.5 for detection, plus character-level accuracy for OCR",
            "C. F1 score",
            "D. AUC-ROC"
        ],
        "correct": "B",
        "hint": "You have both a detection component and an OCR component — each needs its own metric.",
        "explanation": "mAP@0.5 measures whether the bounding box around the address number is correctly located (IoU ≥ 0.5). Character-level OCR accuracy measures whether the digits are correctly read. A detected box with wrong digits is a failed localization even with 100% detection mAP.",
        "difficulty": 3,
        "tags": ["metrics", "map", "ocr"]
    },
    {
        "concept_id": "gsv_scale_challenge",
        "module": "03-google-street-view",
        "question": "Google has 220 billion Street View images across 100+ countries. What is the primary infrastructure challenge for training?",
        "choices": [
            "A. Finding enough annotators",
            "B. Distributed training across thousands of GPUs with data parallelism, plus efficient data pipeline to avoid I/O bottlenecking GPU compute",
            "C. Choosing the right model architecture",
            "D. Licensing the images"
        ],
        "correct": "B",
        "hint": "With 220B images, a single epoch would take years on one GPU. What does that imply?",
        "explanation": "At 220B images, even at 10,000 images/second/GPU, a single pass takes 220M seconds ≈ 7 years on one GPU. Solution: 1000s of GPUs in a data-parallel setup, with a highly optimized pipeline that prefetches and preprocesses data to keep GPU utilization > 90%. Data parallelism bottlenecks at gradient synchronization (solved with Ring-AllReduce or parameter servers).",
        "difficulty": 4,
        "tags": ["distributed_training", "scale", "infrastructure"]
    },
    {
        "concept_id": "gsv_transfer_learning",
        "module": "03-google-street-view",
        "question": "You're building an address number detector for a new country with very little labeled data. What is the correct approach?",
        "choices": [
            "A. Train a model from scratch on synthetic data",
            "B. Fine-tune a pre-trained backbone (e.g., ResNet or EfficientDet trained on COCO) on your small country-specific dataset",
            "C. Use a rule-based OCR system",
            "D. Wait until you have more labeled data"
        ],
        "correct": "B",
        "hint": "Low-data regime + similar visual features to existing datasets → what technique do you use?",
        "explanation": "Transfer learning from a model pre-trained on a large dataset (COCO, ImageNet) provides rich visual features (edges, textures, shapes) that generalize. Fine-tuning adapts the final layers to the new domain. For very limited data, only fine-tune the head; for more data, fine-tune deeper layers. This dramatically reduces the amount of labeled data needed.",
        "difficulty": 2,
        "tags": ["transfer_learning", "low_data"]
    },
    {
        "concept_id": "gsv_active_learning",
        "module": "03-google-street-view",
        "question": "With 220B images to label, you can only annotate 1M per year. How do you choose which images to label next?",
        "choices": [
            "A. Randomly sample images",
            "B. Active learning: have the model identify images where it is most uncertain (high entropy predictions), and prioritize labeling those",
            "C. Label the most recent images",
            "D. Label images from the most populous countries"
        ],
        "correct": "B",
        "hint": "What images will improve the model most per labeling dollar?",
        "explanation": "Active learning maximizes label efficiency by querying for labels on the examples the model is most uncertain about. An image the model classifies as 95% class A provides little new signal. An image with entropy close to maximum (equal probability across classes) likely falls in a region of input space the model hasn't learned well — labeling it improves generalization most efficiently.",
        "difficulty": 3,
        "tags": ["active_learning", "label_efficiency"]
    },
    {
        "concept_id": "gsv_privacy_blurring",
        "module": "03-google-street-view",
        "question": "Street View must blur faces and license plates before publishing. What ML approach detects these at 220B image scale?",
        "choices": [
            "A. Manual review by humans",
            "B. Real-time face/plate detection models (YOLO or similar) running at ingestion time, with a high-recall threshold to minimize missed detections at the cost of some false positives",
            "C. Blur the entire image",
            "D. Only blur upon user request"
        ],
        "correct": "B",
        "hint": "Privacy violations are asymmetric: a missed face is worse than an over-blurred non-face.",
        "explanation": "Privacy protection requires high recall (minimize missed faces/plates) even at the cost of some false positives (blurring a non-face causes minor image quality degradation, not a privacy violation). High-recall detection at ingestion time with human review of edge cases. Models are optimized for recall, not F1.",
        "difficulty": 3,
        "tags": ["privacy", "detection", "threshold_tuning"]
    },
    {
        "concept_id": "gsv_geo_temporal",
        "module": "03-google-street-view",
        "question": "A building's address number in a 2015 Street View image has changed. How does the ML system handle temporal consistency?",
        "choices": [
            "A. Keep only the most recent image",
            "B. Use temporal metadata to weight more recent captures, and trigger re-labeling when change detection models flag significant appearance changes",
            "C. Average predictions across time",
            "D. Show both images to the user"
        ],
        "correct": "B",
        "hint": "How do you know when the ground truth has changed?",
        "explanation": "Temporal consistency requires: (1) change detection models that identify when a location's appearance has significantly changed (new construction, address update), (2) recapture triggers, (3) recency-weighted confidence scoring so newer images override older ones. Map data staleness is a real problem — Google recaptures high-change areas more frequently.",
        "difficulty": 4,
        "tags": ["temporal_data", "data_freshness", "change_detection"]
    },
    {
        "concept_id": "gsv_multi_task",
        "module": "03-google-street-view",
        "question": "Instead of separate models for (1) address detection, (2) storefront classification, (3) road sign reading — when should you use a multi-task model?",
        "choices": [
            "A. Always use multi-task — it's always better",
            "B. When tasks share a visual backbone and training data, multi-task learning provides regularization and amortizes compute. But if tasks conflict (different optimal feature scales), use separate models.",
            "C. Never use multi-task — it's too complex",
            "D. Only when you have unlimited GPU memory"
        ],
        "correct": "B",
        "hint": "Think about whether detecting a storefront and reading an address number benefit from the same low-level visual features.",
        "explanation": "Multi-task learning benefits: shared backbone trains on more data, lower serving cost (one inference for multiple outputs), implicit regularization from auxiliary tasks. Risks: tasks with conflicting gradient directions hurt each other (negative transfer), tasks with different optimal resolution scales need different heads. Evaluate per task metric in multi-task vs. single-task before committing.",
        "difficulty": 4,
        "tags": ["multi_task_learning", "architecture"]
    },
    {
        "concept_id": "gsv_model_compression",
        "module": "03-google-street-view",
        "question": "You need to run address detection on a mobile device with 256MB RAM and 50ms latency budget. What techniques do you apply?",
        "choices": [
            "A. Use a full ResNet-152",
            "B. Knowledge distillation (train a small student model to mimic a large teacher), quantization (FP32 → INT8), and pruning — combined they can achieve 10-100x size reduction with < 5% accuracy loss",
            "C. Reduce the training dataset",
            "D. Use the server-side model via API"
        ],
        "correct": "B",
        "hint": "Three standard techniques for model compression — do you know all three?",
        "explanation": "Quantization (FP32 → INT8): 4× size reduction, 2-4× speedup, < 1% accuracy loss with calibration. Pruning: remove near-zero weights (20-80% of parameters), often combined with fine-tuning. Knowledge distillation: student model (MobileNet) learns to match teacher model (ResNet) soft label outputs, retaining much of the teacher's knowledge in a much smaller model.",
        "difficulty": 3,
        "tags": ["model_compression", "quantization", "knowledge_distillation", "mobile"]
    },
]
