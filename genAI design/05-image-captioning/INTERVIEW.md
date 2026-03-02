# Staff / Principal Engineer Interview Guide — Image Captioning Systems

---

## How to Use This Guide

This guide is designed for interviewers assessing Staff and Principal Engineer candidates on their ability to design, build, evaluate, and operate production-grade image captioning systems. Image captioning is a representative multimodal ML problem that touches every major discipline a senior engineer must command: problem framing, data engineering, model architecture, evaluation science, infrastructure, and operational monitoring.

**Who this guide is for.** Use it for Staff Engineer, Principal Engineer, and Senior ML Research Engineer loops. The questions probe both breadth (system design) and depth (mathematical and architectural understanding). A Staff candidate is expected to own one complete vertical; a Principal candidate is expected to synthesize trade-offs across all verticals and influence the broader organization's technical direction.

**How to run the interview.** Each section maps to a step in the 7-step ML system design framework. Walk through the framework sequentially for a 60-minute system design loop. For a 45-minute deep-dive loop, select 2-3 Deep Technical Probe questions and use the tiered answer rubrics to calibrate the candidate's signal precisely.

**How to score.** Each question includes four labeled answer tiers: **No Hire**, **Weak Hire**, **Hire**, and **Strong Hire**. Match the candidate's actual answer against these tiers holistically. A single answer that hits Strong Hire criteria does not override consistent Weak Hire answers elsewhere. Aggregate across sections before making a final recommendation. The Hiring Decision Summary table at the end provides a calibration checklist.

**Interviewer etiquette.** Do not lead the candidate. If a candidate stalls, offer a single clarifying prompt (e.g., "How would your answer change for a medical imaging context?"). Do not reveal scoring criteria mid-interview. Take contemporaneous notes keyed to each question number.

**Ground rules.** Candidates may ask clarifying questions before answering. This is desirable — clarifying requirements is itself a scored dimension. Silence is acceptable during thinking. Expect Staff candidates to sketch architectures; Principal candidates should anticipate second- and third-order implications without prompting.

---

## The Problem Context

An image captioning system takes an image as input and produces a natural language sentence or paragraph describing its content. Real-world applications span a wide spectrum:

- **Accessibility**: Facebook's automatic alt-text for screen reader users; generating descriptions for visually impaired users
- **Search and discovery**: Google Lens product search; e-commerce product description generation
- **Medical imaging**: Radiology report generation from chest X-rays or CT slices
- **Content moderation**: Describing images to downstream text-based filters
- **Document understanding**: Captioning charts, diagrams, and figures in scientific papers

Each application has radically different quality criteria, latency requirements, and tolerance for hallucination (generating plausible-sounding but factually incorrect content). A system designed for Facebook alt-text cannot be directly repurposed for radiology reporting without fundamental changes to evaluation criteria, confidence handling, and human-in-the-loop architecture.

---

## Step 1: Requirements Clarification

### Question 1.1

**What clarifying questions would you ask before designing an image captioning system, and why does each answer change your design?**

**No Hire:** The candidate lists generic questions like "what's the scale?" or "what language?" without connecting each answer to a concrete design implication. The candidate treats requirements as a checkbox exercise rather than a design-shaping dialogue. Answers are shallow (one sentence per question) and do not reveal awareness that, for example, latency requirements affect whether beam search is even feasible. The candidate moves to architecture too quickly without recognizing that "image captioning" is wildly underspecified. There is no mention of hallucination tolerance, domain specificity, or quality criteria differences across use cases.

**Weak Hire:** The candidate identifies several important dimensions — latency, language, scale — but does not connect them rigorously to design choices. For example, they ask about latency but do not explain that real-time accessibility captioning at p99 < 200ms may preclude large ViT encoders or multi-beam decoding. They ask about medical vs. consumer use cases but do not surface the hallucination tolerance implication (near-zero tolerance in radiology vs. acceptable tolerance in e-commerce). The candidate covers breadth at the cost of depth. They would produce a workable system but might miss critical constraints.

**Hire:** The candidate systematically probes five or more dimensions and connects each to a concrete downstream design decision. They ask about the use case domain (accessibility vs. search vs. medical) and immediately note that medical requires a human-review fallback and confidence thresholding, while accessibility prioritizes fluency and coverage. They ask about latency budget and note that real-time requires greedy or small-beam decoding with a compact encoder, while batch processing for e-commerce allows large ViT + beam search + reranking. They ask about language support, noting that a multilingual decoder requires a multilingual pretrained language model (e.g., mT5 or multilingual LLM). They ask about input image quality and note that low-resolution or noisy inputs require different preprocessing and potentially a super-resolution preprocessing step. They ask about hallucination tolerance explicitly, stating that low tolerance domains require CHAIR metric monitoring and explicit confidence calibration.

**Strong Hire:** The candidate treats requirements clarification as a risk identification exercise, not a form-filling exercise. They proactively surface non-obvious tensions: for example, that accessibility users need high recall (describe everything present) while search users need high precision (only describe salient objects that match query intent). They note that hallucination has asymmetric costs — a wrong caption for accessibility is a nuisance, but a wrong finding in a radiology report is a patient safety event. They ask about ground truth availability for ongoing evaluation, noting that without periodic human annotation, model drift goes undetected. They ask about downstream consumers of the caption (human readers vs. NLP pipeline) because NLP pipelines are sensitive to stylistic inconsistency. They flag that "natural language caption" is itself underspecified — is a single sentence sufficient, or is a structured report with anatomical terminology required? They connect each answer to a specific technology choice, evaluation metric, or operational monitoring decision.

---

### Question 1.2

**How do the quality criteria differ between an accessibility captioning system and a medical imaging captioning system? How would you operationalize each quality criterion as a measurable metric?**

**No Hire:** The candidate gives a vague answer about "accuracy being important in both" without distinguishing the two domains. They may suggest using BLEU or ROUGE for both without recognizing that BLEU rewards n-gram overlap and is poorly correlated with clinical utility. They do not mention hallucination as a distinct quality dimension from accuracy. There is no awareness that accessibility captions are evaluated by human users while medical captions may be evaluated against structured gold-standard reports written by radiologists. The candidate cannot name a hallucination metric.

**Weak Hire:** The candidate recognizes that medical imaging requires higher accuracy and mentions that hallucination is more problematic there, but cannot quantify the difference or propose measurement methodologies. They mention BLEU for accessibility and "maybe some clinical score" for medical without being able to specify what that clinical score is. They recognize that human evaluation is the gold standard but cannot describe how to structure it (adequacy vs. fluency vs. specificity rubrics). They do not mention CHAIR. They understand the directional difference between use cases but lack the metric vocabulary to operationalize it.

**Hire:** The candidate clearly distinguishes accessibility quality criteria (fluency, coverage of salient objects, grammatical correctness) from medical quality criteria (factual accuracy, zero hallucination, clinical terminology alignment, structured finding completeness). For accessibility, they propose CIDEr as the primary automatic metric, supplemented by human evaluation on adequacy and fluency, and CHAIR to measure hallucination rate on a held-out set. For medical, they propose expert radiologist evaluation as the primary signal, CHAIR adapted with medical ontology (UMLS/RadLex entity lists), and a separate factual accuracy rubric that scores mention of anatomical locations, severity, and laterality. They explain that BLEU alone is inappropriate because it rewards surface form overlap rather than clinical content accuracy.

**Strong Hire:** The candidate builds a measurement framework from first principles, distinguishing automatic offline metrics from human evaluation from production monitoring. For accessibility, they propose a multi-axis human evaluation rubric: adequacy (is every important object described?), fluency (is it grammatically natural?), and specificity (does it go beyond generic descriptions?). They note that CHAIR can be estimated offline but that production monitoring requires a sampling strategy to maintain statistical power while controlling annotation cost. For medical imaging, they propose a clinical information extraction pipeline — parse generated reports using a medical NER model to extract findings, compare against structured gold-standard annotations, and compute finding-level precision and recall separately for high-stakes findings (e.g., pneumothorax, mass) vs. incidental findings. They flag that standard NLP metrics fail catastrophically on medical text because synonymy is high (e.g., "opacity", "consolidation", "infiltrate" can describe the same finding) and METEOR's stemming does not capture clinical synonym relationships without a medical ontology.

---

## Step 2: ML Problem Framing

### Question 2.1

**Frame image captioning as a machine learning problem. Define the input space, output space, loss function, and key architectural choices you need to make.**

**No Hire:** The candidate describes image captioning as a classification problem or confuses it with image tagging. They cannot articulate that caption generation is an autoregressive sequence prediction problem. They may mention "CNN + LSTM" as the architecture without being able to explain why these components are combined or what role each plays. They cannot state the training objective (cross-entropy loss over token predictions). They do not distinguish between the encoder (visual understanding) and decoder (language generation) roles.

**Weak Hire:** The candidate correctly identifies the encoder-decoder structure and knows that the loss is cross-entropy at each decoder step. They mention teacher forcing but cannot explain why it is necessary (exposure bias during training vs. inference) or what its failure mode is (the model never learns to recover from its own mistakes during inference). They know CNN and ViT are options for the encoder but cannot articulate when each is preferred. They know that cross-attention connects the encoder to the decoder but cannot explain the mechanics: query comes from the decoder (text), key and value come from the encoder (image). They have a correct but shallow framing.

**Hire:** The candidate gives a complete and precise framing. Input: image pixel tensor of shape (H, W, C), typically 224x224x3 after preprocessing. Output: token sequence of variable length. The model is trained to maximize the log-likelihood of the reference caption given the image: sum over timesteps of log P(token_t | image, token_1..t-1). Training uses teacher forcing — the ground truth token at each step is fed as input to the next step, not the model's own prediction. Loss: cross-entropy averaged over all token positions in the batch. Key architectural choices: (1) encoder type (CNN vs. ViT), (2) how visual features are projected into the decoder's attention space (cross-attention or prefix tokens), (3) decoder type (RNN vs. Transformer), (4) whether to pretrain the encoder and decoder separately before joint fine-tuning.

**Strong Hire:** The candidate frames the problem rigorously and then immediately identifies the second-order implications of each choice. They note that teacher forcing creates a train-inference mismatch: at inference time, the model conditions on its own (potentially wrong) previous predictions, but it was never trained to do this. This is the exposure bias problem, and it explains why SCST (reinforcement learning to optimize evaluation metrics directly) can improve over purely supervised training. They note that cross-entropy loss treats all tokens equally, but tokens like "the" and "a" are easy to predict and dominate the loss, while rare informative tokens like "frisbee" contribute little signal — motivating weighted loss or CIDEr optimization via SCST. They flag that the output space is combinatorially large (vocabulary size ^ sequence length) and that the model must learn both what to say (content selection from the image) and how to say it (language fluency), which are distinct capacities that may require different training stages.

---

## Step 3: Data Strategy

### Question 3.1

**Describe the key training datasets for image captioning. What are the quality trade-offs between them, and how would you handle noisy web-scraped data?**

**No Hire:** The candidate cannot name any standard image captioning dataset. They may vaguely mention "ImageNet" (which is a classification dataset, not a captioning dataset). They have no strategy for handling noisy data beyond "clean it manually" or "just use it as-is." They do not recognize that data quality affects downstream hallucination rates. They do not mention the fact that multiple captions per image are needed for reliable evaluation with metrics like CIDEr.

**Weak Hire:** The candidate knows MS COCO and possibly Conceptual Captions but cannot describe their characteristics precisely. They know COCO has human-written captions and that web-scraped data is noisy but cannot propose a specific filtering strategy. They suggest removing pairs where the alt-text contains no nouns, or filtering on caption length, but cannot go deeper. They do not mention CLIP-based alignment filtering or why CLIP is especially well-suited to this task (joint vision-language embedding space allows direct similarity measurement).

**Hire:** The candidate describes MS COCO (330K images, 5 human-written captions each — gold standard for evaluation, high quality but small scale), Conceptual Captions (3M web-scraped image-alt-text pairs — large but noisy, alt-texts are often not literal descriptions), and SBU Captions (1M Flickr images with user-provided captions — intermediate quality). They explain that COCO's multiple captions per image are essential for CIDEr, which rewards n-grams that are informative relative to the set of reference captions. They propose CLIP-based filtering for Conceptual Captions: compute CLIP cosine similarity between image and alt-text embeddings, retain only pairs above a threshold (e.g., 0.28), which removes pairs where the alt-text describes something not visible in the image. They note that this filtering trades recall for precision and that the threshold is a hyperparameter calibrated against a human-annotated validation set.

**Strong Hire:** The candidate builds a complete data quality pipeline and connects each step to a downstream metric impact. They note that COCO has an annotation bias: captions describe objects that annotators found salient, which may not match the salience hierarchy of a visually impaired user (who may want spatial layout described, not just object identity). They propose a data mixing strategy: pretrain on Conceptual Captions after CLIP filtering to learn broad visual vocabulary, fine-tune on COCO for quality. They note that data augmentation must be semantically consistent: horizontal flips are safe (a dog on the left becomes a dog on the right, and captions rarely specify left/right), but vertical flips would produce unnatural images and should not be used. Color jitter is safe for most captioning tasks but is contraindicated for medical imaging where color encodes clinical information (e.g., fluorescence microscopy, histopathology stains). They propose caption quality scoring using a separate model (e.g., CLIP text-image similarity + grammar model) to create a quality curriculum, training on easier (high-quality) examples first.

---

## Step 4: Model Development

### Question 4.1

**Compare a CNN encoder (e.g., ResNet) to a Vision Transformer (ViT) encoder for image captioning. When would you choose each?**

**No Hire:** The candidate knows CNN and ViT exist but cannot articulate a principled trade-off. They may say "ViT is newer so it's better" without being able to explain the inductive biases that make each appropriate for different settings. They cannot explain what "translation equivariance" means or why it matters. They do not know that ViT patchifies the image and processes patches as tokens. They cannot connect encoder architecture choice to downstream captioning behavior.

**Weak Hire:** The candidate knows that ViT requires more data than CNN because it lacks the translation equivariance inductive bias and that ViT scales better with data and compute. They know that CNN is hierarchical and that ViT processes the image as a flat sequence of patches. However, they cannot quantify the patchification process (16x16 patches on 224x224 → 196 tokens) or explain why patch size matters for caption quality. They prefer ViT in high-data regimes and CNN in low-data regimes but cannot give a concrete recommendation for specific applications.

**Hire:** The candidate gives a thorough comparison. CNN (ResNet): hierarchical feature extraction through convolutional layers, translation equivariance means the model detects the same feature regardless of its position, works well with limited data due to strong inductive biases, spatial feature maps preserve local structure well. ViT: patchify the image into 16x16 non-overlapping patches, producing 196 tokens for a 224x224 input, linear projection + 1D positional encoding, bidirectional self-attention across all patches allows global context from the first layer. ViT requires more data to learn spatial relationships that CNN gets for free via convolution but achieves better performance at scale. For captioning: ViT is preferred when a large pretrained checkpoint is available (e.g., CLIP ViT-L/14) and the dataset is large; CNN is preferred in medical imaging or specialized domains with limited data where transfer from general-purpose pretrained CNNs is effective.

**Strong Hire:** The candidate goes beyond the standard comparison to discuss the implications for the decoder. ViT produces a flat sequence of patch tokens — all 196 tokens are given equal weight in cross-attention, which is computationally expensive for the decoder. CNN produces a spatial feature map (e.g., 7x7x2048 for ResNet-101 after removing the classification head), which can be flattened to 49 spatial tokens — cheaper for cross-attention. The candidate notes that 1D positional encoding in ViT loses 2D spatial structure (row vs. column information), which matters for captions that reference spatial relationships ("the ball on the left"). 2D positional encoding or relative positional bias (as in Swin Transformer) partially addresses this. For very fine-grained captioning tasks — medical imaging, OCR, spatial relationship descriptions — the candidate recommends Swin Transformer (hierarchical ViT with local window attention) because it preserves multi-scale spatial information better than flat ViT while achieving ViT-level performance at scale.

---

### Question 4.2

**Explain the cross-attention mechanism that connects the visual encoder to the text decoder. How does it work mechanically, and what would break if you removed it?**

**No Hire:** The candidate knows the term "cross-attention" but cannot distinguish it from self-attention. They may say the encoder and decoder are "connected by attention" without being able to specify what the query, key, and value are in cross-attention. They cannot explain the information flow: the decoder asks questions (query) about the image (key/value). They do not know that removing cross-attention would make the decoder a language model that cannot condition on the image at all.

**Weak Hire:** The candidate knows that queries come from the decoder and keys/values come from the encoder but cannot explain the computation precisely. They know the output is a weighted sum of values, where weights are computed from query-key dot products, but they may not know that the softmax normalization and the 1/sqrt(d_k) scaling factor are part of the standard mechanism. They know removing cross-attention breaks conditioning on the image but cannot articulate what failure mode would result in generated captions (the decoder would produce a generic language-model output, averaging over all possible scenes it has seen during training).

**Hire:** The candidate gives a complete mechanical description. At each decoder layer and each decoder position t, cross-attention computes: Q = decoder_hidden_state * W_Q, K = encoder_output * W_K, V = encoder_output * W_V. Attention weights A = softmax(QK^T / sqrt(d_k)). Output = AV. The decoder can attend to any subset of image patches at each generation step. For example, when generating the word "cat," the decoder should attend heavily to the patch containing the cat. Self-attention in the decoder is causal (masked): position t can only attend to positions 1..t-1, enforcing autoregressive generation. Cross-attention is not masked: the decoder can see all encoder positions. Removing cross-attention: the decoder becomes a standard language model conditioned only on previously generated tokens, producing grammatically fluent but visually ungrounded captions.

**Strong Hire:** The candidate connects cross-attention mechanics to hallucination analysis and training efficiency. They note that cross-attention weight maps can be used diagnostically: if the model hallucinates "cat," inspecting the cross-attention weights when the token "cat" was generated should show diffuse attention (spread across many patches) rather than concentrated attention on a cat-containing region. If the attention is diffuse, the token was generated from the language prior rather than from visual evidence — this is the mechanistic signature of hallucination. The candidate also notes that modern architectures like BLIP-2's Q-Former do not use standard cross-attention in the same way: the Q-Former uses learned query tokens that attend to the frozen encoder via cross-attention, compressing 196 image tokens into 32 query token representations. This reduces the computational cost of cross-attention in the downstream LLM by 6x. In contrast, LLaVA simply projects all 256 ViT tokens into the LLM's embedding space using a linear layer (or MLP), preserving all spatial detail at the cost of longer LLM context.

---

## Step 5: Evaluation

### Question 5.1

**Explain how CIDEr works. What are its strengths and weaknesses relative to BLEU? What does it mean to "game" CIDEr, and how would you detect it?**

**No Hire:** The candidate cannot explain CIDEr beyond "it measures caption quality." They may confuse it with BLEU or say they are the same thing. They cannot explain what TF-IDF weighting accomplishes or why it is well-suited to captioning. They do not know what "gaming a metric" means in the context of SCST training. They cannot name the CHAIR metric or any other complementary evaluation approach.

**Weak Hire:** The candidate knows CIDEr uses TF-IDF weighting and computes cosine similarity between candidate and reference n-gram vectors. They know BLEU measures n-gram precision and CIDEr rewards informative words over common words. They can name SCST as a method that optimizes CIDEr directly via reinforcement learning. However, they cannot explain TF-IDF precisely (term frequency times log of inverse document frequency), cannot give a concrete example of a high-IDF vs. low-IDF word in captioning, and cannot propose a diagnostic for detecting CIDEr gaming in production.

**Hire:** The candidate explains CIDEr completely. TF-IDF: for each n-gram in the candidate caption, compute TF (how many times it appears in the caption) times IDF (log of total number of reference captions divided by number of captions containing the n-gram). This weights n-grams by how informative they are — "frisbee" is rare across the caption corpus so it has high IDF and contributes more to the score; "a" and "the" are near-universal so their IDF is near zero. CIDEr averages TF-IDF weighted cosine similarity across n-gram orders 1-4 and averages over all reference captions. BLEU: n-gram precision only, no IDF weighting, penalizes short captions via a brevity penalty but not common words. CIDEr better correlates with human judgment on captioning benchmarks. CIDEr gaming via SCST: the model learns to produce high-IDF training-set n-grams even when they are not appropriate for a given image, because SCST rewards any caption that scores well against the training references — including captions with domain-specific jargon that sounds specific but is not grounded in the image.

**Strong Hire:** The candidate proposes a complete diagnostic and mitigation framework. To detect gaming: (1) monitor vocabulary novelty — track the fraction of generated n-grams that do not appear in the training reference captions; a gaming model will produce fewer novel n-grams over time as it converges to memorized high-IDF phrases; (2) track diversity within a batch — generate 5 captions for 1000 images and compute pairwise CIDEr variance; a gaming model will converge to a small set of high-scoring templates; (3) run CHAIR alongside CIDEr — a model that games CIDEr by inventing specific objects that are not in the image will show increasing CHAIR hallucination rate. To mitigate: mix the SCST reward — optimize 0.5 * CIDEr + 0.5 * human_preference_reward, or add a diversity regularization term. The candidate also notes a deeper problem: CIDEr is computed against a fixed reference set. When the reference set is from COCO (a specific annotation style), CIDEr rewards matching that style, not necessarily producing captions useful for the target application. Domain shift between COCO annotation style and the target use case (e.g., medical) means COCO-trained CIDEr scores are not predictive of production quality.

---

### Question 5.2

**What is the CHAIR metric, and how would you use it as part of an evaluation and monitoring pipeline for a production image captioning system?**

**No Hire:** The candidate has not heard of CHAIR or cannot explain what hallucination means in the context of image captioning. They may conflate hallucination with low accuracy or grammatical errors. They do not recognize that a caption can be perfectly fluent and grammatically correct while being factually wrong (e.g., describing a skateboard as a surfboard). They have no production monitoring framework.

**Weak Hire:** The candidate knows CHAIR measures hallucination — the fraction of objects mentioned in the caption that are not actually in the image — but cannot describe the computation precisely. They know it requires ground-truth object annotations. They suggest using it as an offline evaluation metric but have no plan for production monitoring where ground truth is not available in real time. They do not know how to adapt CHAIR to domains beyond COCO (where standard object categories are well-defined).

**Hire:** The candidate explains CHAIR precisely. CHAIR_i (instance-level): for each generated caption, count the fraction of object words that do not appear in the image's ground-truth object list. CHAIR_s (sentence-level): fraction of captions that contain at least one hallucinated object. To compute CHAIR, you need: (1) generated captions, (2) a ground-truth object list per image (from segmentation annotations or detection model), (3) a noun extraction step on the captions (NER or constituency parse to extract object mentions). For production monitoring: since ground truth annotations are not available for every new image, use a proxy — run a strong object detection model (e.g., DINO or YOLO-v8) on the input image to generate a pseudo-ground-truth object list, then compute CHAIR against that list. Track CHAIR_i over time; alert when it increases more than 2 standard deviations above the baseline measured at deployment.

**Strong Hire:** The candidate builds a complete monitoring architecture and flags its limitations. They note that proxy-CHAIR (using a detection model for pseudo-ground-truth) has its own false positive and false negative rate — if the detection model misses an object, a correct caption mentioning it will be penalized as a hallucination. To handle this: calibrate the proxy-CHAIR against human-annotated CHAIR on a held-out set to estimate the bias and variance of the proxy. Use proxy-CHAIR for high-frequency monitoring and human-annotated CHAIR for monthly deep-dive audits. Additionally, CHAIR only covers objects and is blind to attribute hallucination (describing a blue car as red) and relational hallucination (describing a dog chasing a cat when the cat is chasing the dog). For comprehensive hallucination monitoring, supplement CHAIR with a VQA-based consistency check: generate a set of yes/no questions about the caption's claims (using an LLM) and answer them using a VQA model applied to the original image; flag captions where VQA answers contradict the caption's claims.

---

## Step 6: System Design

### Question 6.1

**Design the end-to-end inference pipeline for a real-time image captioning system serving accessibility captions at p99 < 300ms. What are the bottlenecks, and how do you address each one?**

**No Hire:** The candidate sketches a simple "call the model" pipeline without identifying bottlenecks. They do not know what beam search is or how it affects latency. They do not mention KV-cache, batching strategies, or model quantization. They treat the problem as a software engineering problem (fast server, load balancer) rather than an ML systems problem. They cannot estimate the computational cost of ViT encoding or autoregressive decoding.

**Weak Hire:** The candidate identifies model inference as the bottleneck and suggests using a GPU, smaller model, or quantization. They mention beam search but do not know that greedy decoding (beam width = 1) is significantly faster. They mention caching but confuse it with standard HTTP caching rather than KV-cache for the transformer decoder. They have the right instincts but lack the technical depth to propose a precise optimization strategy.

**Hire:** The candidate designs a complete pipeline: Image upload → Preprocessing service (resize to 224x224, normalize, GPU-resident) → ViT encoder (batched, GPU) → Cross-attention bridge → Autoregressive decoder (KV-cache) → Safety filter → Response. Bottleneck analysis: (1) ViT encoding is fast (~5ms for a single image with ViT-B/16 on A100) but batching amortizes GPU launch overhead. (2) Autoregressive decoder is the primary latency bottleneck — each token requires a forward pass through the decoder, and captions are typically 10-20 tokens, so decoding takes ~50-100ms for a large decoder. (3) Beam search (width=5) multiplies this by 5x — for real-time, use greedy decoding or width=2. Optimizations: KV-cache stores past key-value pairs in the decoder so each step only computes the new token's attention, reducing per-step cost from O(L^2) to O(L). FP16 or INT8 quantization of encoder. Separate encoder and decoder serving so they can be independently scaled.

**Strong Hire:** The candidate designs for tail latency, not just median. They note that the p99 constraint is more demanding than p50 because GPU queuing, garbage collection pauses, and batch assembly delays can create long-tail latency spikes. To address p99: (1) continuous batching — do not wait for a full batch; process requests as they arrive with a maximum wait time of 5ms to assemble micro-batches; (2) pre-warm the encoder for common image resolutions to avoid JIT compilation delays; (3) use speculative decoding — a small draft model generates candidate tokens that the full model verifies in parallel, reducing the number of full-model forward passes; (4) monitor per-step decoding latency, not just end-to-end latency, to isolate whether tail latency comes from encoding or decoding. For the ViT encoder specifically, they note that encoder inference is parallelizable (no autoregressive dependency) and can be batched aggressively, while decoder inference is sequential and dominates latency at small batch sizes. They recommend profiling the system under realistic traffic load (including image size distribution and caption length distribution) rather than benchmarking on a single canonical input.

---

## Step 7: Deployment and Monitoring

### Question 7.1

**How would you monitor a deployed image captioning system over time? What signals would you track, and how would you detect model degradation?**

**No Hire:** The candidate suggests monitoring CPU/GPU utilization and error rates (5xx) without any ML-specific monitoring. They do not know that model quality can degrade silently — the system remains up and responsive but produces worse captions. They have no plan for detecting data drift or distribution shift. They cannot distinguish between infrastructure monitoring and model quality monitoring.

**Weak Hire:** The candidate knows to monitor model quality metrics in production but is vague about how. They suggest "logging captions and reviewing them periodically" without a structured sampling strategy or rubric. They mention A/B testing as a way to compare models but do not propose a control group or statistical test. They know distribution shift is a concern but cannot propose a specific signal (e.g., CLIP score distribution shift) that would detect it without ground truth labels.

**Hire:** The candidate proposes a multi-layer monitoring system. Infrastructure layer: latency (p50, p95, p99), throughput (requests/sec), GPU memory, error rate. Model quality layer: (1) CLIP alignment score — compute CLIP cosine similarity between generated caption and input image; track distribution over time; a drop signals quality degradation or input distribution shift; no ground truth needed; (2) proxy-CHAIR hallucination rate on a sampled subset; (3) caption length distribution — abnormal shortening or lengthening may indicate model drift; (4) vocabulary entropy — a drop in vocabulary entropy signals the model is collapsing to generic templates. Human evaluation layer: weekly stratified sample of 200 captions reviewed by annotators using adequacy/fluency/specificity rubric; track mean scores and alert on statistically significant drops. Alert conditions: CLIP score drops by >0.03 from 30-day rolling average, CHAIR rate increases by >2%, human evaluation score drops by >0.1 on a 5-point scale.

**Strong Hire:** The candidate builds a complete observability stack with causal diagnosis capabilities. They distinguish between three failure modes with different root causes and different detection signals: (1) distribution shift — the input image distribution changes (e.g., a new camera sensor generates images with different color profiles), detected by monitoring CLIP embedding distribution drift (e.g., Maximum Mean Discrepancy between current week and baseline week); (2) model degradation — the model weights are unchanged but performance drops because the world changed (e.g., new products appear in e-commerce that were not in training data), detected by declining CLIP alignment on new product categories; (3) catastrophic failure — a code change or infrastructure issue causes systematic errors, detected by sudden step changes in latency or error rate. They propose a shadow deployment strategy for new model versions: run the candidate model in shadow mode alongside the production model for 48 hours, compare CLIP scores and CHAIR rates, promote only if the candidate is statistically significantly better (paired t-test, p < 0.01). For medical imaging specifically, they propose a zero-tolerance hallucination protocol: any caption mentioning a finding not confirmed by a detection model triggers human review before the caption is delivered, with an SLA of 4 hours for radiologist review.

---

## Deep Technical Probe Sections

### Probe 1: ViT Patch Size Sensitivity

**If you are building a captioning system for chest X-rays, and the clinically significant finding is a 1-2mm pulmonary nodule, what problems does a standard ViT-B/16 encoder have, and how would you address them?**

**No Hire:** The candidate does not know what a patch is or cannot compute how many patches a 224x224 image produces with 16x16 patches (answer: 196). They cannot reason about spatial resolution loss. They suggest "using a bigger model" without identifying the fundamental patch-size problem. They do not know what a pulmonary nodule is in imaging terms (a small round opacity in the lung) and cannot connect physical lesion size to pixel representation.

**Weak Hire:** The candidate understands that 16x16 patches can miss small structures but cannot quantify the problem. They suggest using a higher-resolution input but do not know that this quadruples the number of tokens (and quadruples compute). They do not know about hierarchical ViT (Swin) as an alternative. They cannot explain 1D vs. 2D positional encoding or why it matters for spatial captions.

**Hire:** The candidate quantifies the problem precisely. A 224x224 image with 16x16 patches gives 196 tokens. A 1-2mm nodule on a 224px image occupies roughly 1-2 pixels — less than one patch. The nodule is completely lost before the encoder even processes it. Solutions: (1) increase input resolution to 448x448 — gives 784 tokens with 16x16 patches, quadrupling compute, but the nodule now occupies 4-8 pixels and spans at least one patch; (2) reduce patch size to 8x8 — also gives 784 tokens; (3) use a hierarchical ViT like Swin Transformer, which processes patches at multiple scales (4x4, 8x8, 16x16 equivalents) hierarchically, capturing fine-grained detail while maintaining tractable compute. They also note the 1D positional encoding problem: standard ViT uses 1D positional encoding (patch index 0-195), which loses the 2D spatial structure (row, column). A caption that says "nodule in the right upper lobe" requires the model to know spatial position, not just patch order. Fix: use 2D sinusoidal positional encoding or relative position bias (as in Swin).

**Strong Hire:** The candidate designs a complete medical imaging captioning architecture that addresses resolution, spatial encoding, and hallucination concerns together. For resolution: use 448x448 with ViT-L/14 (the CLIP encoder used in LLaVA) — 1024 tokens at 32x32 patches, giving 2x2 pixel resolution per patch, sufficient for most nodule detection. For positional encoding: use 2D factored positional encoding (separate encodings for row and column, combined additively) to preserve 2D spatial structure. For hallucination: in medical captioning, the asymmetry of errors demands that the model never confidently claim a finding that is not present, even if it misses some findings. Implement a confidence-gated generation protocol: after generating a candidate report, run a medical NLP model to extract clinical findings, verify each finding against an independent detection model, and replace unverified findings with a hedged alternative ("consider clinical correlation for possible nodule") rather than asserting them as fact. They note that this is a systems-level change, not just a model-level change, and requires collaboration with radiologists to define the acceptable false-negative vs. false-positive trade-off for each finding type.

---

### Probe 2: Hallucination Mechanisms and Diagnosis

**A model trained on COCO generates "a dog sitting on a mat" for an image that contains only a mat and no dog. Walk me through the mechanistic cause and a diagnostic procedure to confirm it.**

**No Hire:** The candidate explains hallucination as "the model making stuff up" without a mechanistic account. They cannot distinguish language-prior hallucination from attention failure hallucination. They have no diagnostic procedure. They may suggest retraining the model or getting more data as the fix without understanding the root cause.

**Weak Hire:** The candidate knows that the model has learned a statistical association between "mat" and "dog" from training data (dogs on mats are common in COCO) and that this prior dominates over the visual evidence. They may mention attention maps as a diagnostic but cannot describe attention rollout or explain why raw attention maps from intermediate layers are misleading.

**Hire:** The candidate gives a complete mechanistic account. COCO contains many images of dogs sitting on mats. The language model component of the decoder learns a strong prior P("dog" | "sitting on a mat") from these co-occurrences during training. At inference time, if the visual features are ambiguous (the mat texture triggers a soft match to the mat-with-dog distribution), the language prior can override weak visual evidence to produce "dog." Diagnostic procedure: (1) extract cross-attention maps for the decoder token "dog" — these show which image patches the decoder attended to when generating "dog"; (2) a grounded, non-hallucinating model would show high attention mass concentrated on the spatial region where a dog exists; a hallucinating model shows diffuse, spatially random attention or attention concentrated on an irrelevant region; (3) use CHAIR to confirm: compare "dog" against the ground-truth object list for the image; (4) counterfactual test — blank out the mat region in the image; if "dog" disappears from the caption, the mat patch triggered the dog generation via learned co-occurrence.

**Strong Hire:** The candidate proposes a full diagnosis-to-mitigation pipeline. For attention diagnosis, they note that raw attention maps from a single layer are not faithful — you must use attention rollout (Abnar & Zuidema 2020), which propagates attention weights through all layers via matrix multiplication to produce a single map showing the true influence of each patch on each token. Rollout reveals long-range dependencies that single-layer attention maps obscure. For mitigation: (1) contrastive decoding — at each step, subtract the log-probabilities from a language-only decoder (no image conditioning) from the full model's log-probabilities; this penalizes tokens that the language model would generate anyway, forcing the model to rely more heavily on visual evidence; (2) hard negative training — add training examples where common co-occurrences are deliberately absent (mat without dog) and supervise the model to not generate the absent object; (3) POPE benchmark evaluation — Polling-based Object Probing Evaluation directly measures object hallucination rate by asking yes/no questions about randomly, frequently, and adversarially sampled objects; track POPE alongside CHAIR. The candidate also notes that hallucination rate is strongly correlated with the model's visual grounding: models with stronger visual encoders (larger ViT, higher CLIP score) hallucinate less because the visual signal dominates the language prior.

---

### Probe 3: BLIP-2 Q-Former vs. LLaVA MLP Projection

**BLIP-2 uses a Q-Former with 32 learned queries to compress image features. LLaVA uses a simple MLP projection. When would you use each, and what does the Q-Former actually do mechanically?**

**No Hire:** The candidate has not heard of BLIP-2 or LLaVA. They cannot describe any modern multimodal architecture beyond a generic encoder-decoder. They do not know what a learned query is or how it differs from the input-derived key-value pairs in standard attention. They cannot discuss the trade-off between compression and detail preservation.

**Weak Hire:** The candidate knows BLIP-2 compresses the image into fewer tokens and LLaVA passes all tokens through, and they know compression saves compute. They cannot explain the Q-Former mechanism mechanically (32 learned queries attending to encoder output via cross-attention). They have a directional intuition about when each is better but cannot give a principled argument.

**Hire:** The candidate explains the Q-Former mechanically. Q-Former maintains 32 learned query vectors (parameters, not input-dependent). These 32 queries attend to the full ViT encoder output (196 tokens for ViT-B/16) via cross-attention: Q = learned_queries * W_Q, K = V = vit_output * W_K, W_V. The output is 32 contextually compressed image representations. The compression bottleneck forces the Q-Former to distill the most semantically important information from the 196 patches into 32 summary tokens. These 32 tokens are passed to the LLM. In contrast, LLaVA applies a linear projection (or two-layer MLP in LLaVA-1.5) to each of the 256 ViT-L/14 tokens individually, mapping each to the LLM embedding space without any compression. Use Q-Former when: LLM context length is a constraint (32 tokens vs. 256 tokens is significant for long conversations), semantic-level understanding is more important than fine-grained spatial detail. Use LLaVA MLP when: fine-grained texture, OCR, or spatial relationship understanding matters because all spatial information is preserved.

**Strong Hire:** The candidate adds InstructBLIP to the comparison and discusses task-conditioning. InstructBLIP extends Q-Former by conditioning the 32 query tokens on the instruction text — the instruction is processed by a text encoder and cross-attends into the Q-Former queries. This means the 32 compressed image tokens are task-specific: for "describe the colors in the image," the Q-Former selects color-salient information; for "describe the spatial layout," it selects positional information. This is a significant improvement for multi-task captioning. The candidate can also discuss the information-theoretic framing: Q-Former is an explicit information bottleneck (Tishby et al.) that compresses I(image; tokens) subject to a task relevance constraint. LLaVA MLP is an unconstrained projection that preserves full information at higher LLM context cost. For a production system with diverse user queries (accessibility queries that ask about color, location, and objects), InstructBLIP's task-conditioned compression is strictly superior to both fixed Q-Former and non-compressive MLP projection, at the cost of requiring the instruction to be known before encoding the image — which creates a challenge for streaming architectures where the query arrives after the image is already encoded.

---

### Probe 4: Multi-Modal Compositionality Failure

**CLIP and captioning models frequently fail at compositional understanding — for example, "dog chasing cat" vs. "cat chasing dog." Why does this happen architecturally, and how would you fix it?**

**No Hire:** The candidate does not understand what compositionality means in this context. They may say the model just needs more training data. They do not know the Winoground benchmark. They cannot identify the architectural root cause (ViT features are bag-of-patches, not relational).

**Weak Hire:** The candidate understands the problem intuitively — the model can identify "dog" and "cat" but cannot determine which is the agent and which is the patient of the chasing action. They may know the Winoground benchmark. They propose "more data with diverse relationships" as the fix but cannot describe a targeted architectural intervention.

**Hire:** The candidate identifies the architectural root cause. ViT's self-attention computes global patch-level features but does not explicitly model relationships between patches. After the CLS token aggregation or the average pooling of patch tokens, the representation is essentially a bag-of-patches — it encodes what objects are present and their individual properties but not the relational structure (who is doing what to whom, and in which direction). Winoground benchmark measures this directly: given two captions ("dog chasing cat" and "cat chasing dog") and two images, the model must correctly match each image to its caption. CLIP-style contrastive training exacerbates this because the training objective matches the image to the correct caption among negatives — but random negatives rarely include compositionally similar sentences, so the model is never forced to learn fine-grained relational distinctions. Fix: (1) NegCLIP — add hard negatives that are compositionally swapped versions of the original caption ("cat chasing dog" as a negative when the image shows "dog chasing cat"); (2) scene graph supervision — add a scene graph prediction auxiliary loss that explicitly trains the model to predict (subject, relation, object) triples from the image.

**Strong Hire:** The candidate proposes a complete research and engineering fix and discusses the fundamental limitations. The root cause is deeper than training data: standard ViT spatial features are weak at encoding directed relationships because attention is symmetric (patch A attending to patch B produces the same contribution as B attending to A). To encode directionality, you need either explicit positional relationship encoding (relative position between subject and object patches) or structured supervision that forces the model to distinguish agent and patient roles. Proposed fixes in order of complexity: (1) NegCLIP (easiest — data augmentation, no architectural change); (2) structured text decoder that generates scene graph triples first, then converts to natural language (architecturally moderate); (3) relation-aware cross-attention — augment decoder cross-attention with pairwise patch features that encode the spatial relationship between patches (computationally expensive but principled); (4) neuro-symbolic hybrid — generate symbolic scene graphs using a separate model and ground the decoder in the scene graph rather than raw patch features. The candidate notes the performance-generalization trade-off: structured approaches improve compositionality on benchmarks but may produce less fluent natural language captions because the generation is constrained by the scene graph vocabulary. A production system must balance compositionality against fluency for the target use case.

---

### Probe 5: SCST and Reward Engineering

**Self-Critical Sequence Training uses reinforcement learning to optimize captioning metrics directly. Explain the mechanism, the reward variance problem, and how you would design the reward function for a medical captioning system.**

**No Hire:** The candidate does not know SCST. They may know that RL can be used to optimize non-differentiable metrics but cannot connect this to captioning. They do not know what a self-critical baseline is or why a baseline is needed in REINFORCE. They cannot explain why reward variance is a problem.

**Weak Hire:** The candidate knows SCST optimizes CIDEr directly and uses the model's own greedy decoding as a baseline reward, taking the difference between sampled reward and greedy reward as the REINFORCE signal. They know reward variance is high but cannot explain why or propose a variance reduction strategy. They know SCST is used after supervised pretraining but do not know why the order matters (the model must already produce reasonable captions before RL is stable).

**Hire:** The candidate explains SCST completely. Standard supervised training maximizes log P(caption | image). Non-differentiable metrics (CIDEr, BLEU) cannot be optimized with standard backpropagation. SCST uses REINFORCE: sample a caption sequence from the model, compute its CIDEr score (reward r_s), also compute the CIDEr score of the greedy decoding (baseline reward r_b), and update the model to increase the probability of the sample if r_s > r_b and decrease it if r_s < r_b. The gradient is: (r_s - r_b) * sum_t log P(token_t | ...). The self-critical baseline (greedy decoding) reduces reward variance compared to a fixed baseline because it tracks the current model's performance. Reward variance is still high because the reward is sparse (computed only at the end of the full sequence, not at each step) and CIDEr scores for similar captions can vary significantly due to IDF weighting. Mitigation: use a mixed loss — start with supervised cross-entropy and gradually increase the RL objective weight (curriculum).

**Strong Hire:** The candidate designs a medical captioning reward function from first principles. Standard CIDEr is inappropriate for medical captioning because it rewards surface n-gram similarity, not clinical accuracy. Proposed reward: R = alpha * clinical_recall + beta * clinical_precision - gamma * hallucination_penalty - delta * severity_mismatch_penalty. Clinical recall: fraction of ground-truth findings extracted from the reference report that appear in the generated report (using a medical NER model). Clinical precision: fraction of generated findings that are verified by the reference. Hallucination penalty: binary penalty for each generated finding that has no support in the image (detected using a separate detection model). Severity mismatch penalty: extra penalty when the severity of a finding is wrong (e.g., "mild" when reference says "severe") because severity errors have higher clinical consequence than omission. The candidate notes that this reward function requires a medical NER model as a reward component, which is itself imperfect and may introduce reward hacking (the captioning model learns to exploit NER model errors). To address: periodically retrain the reward NER model and add a human evaluation regularization step where a radiologist reviews a sample of generated reports and provides binary adequacy scores as an additional reward signal.

---

## Red Flags

| Red Flag | What It Signals | Severity |
|---|---|---|
| Cannot explain cross-attention query/key/value roles | Does not understand the core mechanism connecting vision and language | Disqualifying for Staff+ |
| Uses BLEU as the sole evaluation metric without qualification | Unfamiliar with captioning-specific evaluation; will miss hallucination and informativeness problems | High |
| Cannot name a single hallucination metric | Has not worked on production captioning systems; blind to a critical failure mode | High |
| Recommends vertical flip as a data augmentation | Fundamental misunderstanding of image augmentation semantics | Moderate |
| Says ViT is always better than CNN with no caveat | Does not understand inductive bias trade-offs; will make poor architecture choices in data-limited domains | Moderate |
| Cannot explain teacher forcing or exposure bias | Does not understand autoregressive training; will be unable to debug training instability | High |
| Treats medical imaging captioning identically to consumer captioning | No understanding of domain-specific quality criteria and risk asymmetry | High for medical domains |
| Cannot quantify ViT patch count (196 for 224x224 with 16x16 patches) | Does not have working-level knowledge of ViT; cannot reason about resolution/detail trade-offs | Moderate |
| Has never heard of COCO or Conceptual Captions | Has not engaged with the standard benchmarks in this field | High |
| Proposes beam search for real-time system without noting latency cost | Has not thought through the latency-quality trade-off; will violate SLAs | Moderate |
| Cannot distinguish BLIP-2 Q-Former from LLaVA MLP projection | Unfamiliar with the current generation of multimodal architectures | Moderate for Principal+ |
| Says hallucination is "just a data quality problem" | Does not understand that hallucination arises from language prior interaction with vision — a model architecture and training problem | High |
| Cannot define CIDEr TF-IDF weighting | Surface-level metric knowledge; cannot diagnose metric gaming | Moderate |
| No monitoring strategy beyond infrastructure metrics | Has not deployed ML systems in production; will be surprised by silent quality degradation | High for Staff+ |
| Cannot explain why SCST gradient requires a baseline | Does not understand REINFORCE variance reduction; will produce unstable RL training | Moderate |
| Proposes a single model for all captioning use cases | Does not understand that quality criteria, latency, and hallucination tolerance vary radically by domain | High |

---

## Hiring Decision Summary

| Section | Competency Assessed | Min Expectation (Staff) | Min Expectation (Principal) |
|---|---|---|---|
| Step 1: Requirements | Problem framing, domain awareness | Identifies 5+ dimensions with design implications | Surfaces non-obvious tensions and risk asymmetries proactively |
| Step 1: Quality Criteria | Metric selection for different domains | CIDEr for consumer, acknowledges medical needs expert eval | Full multi-axis evaluation framework with per-finding medical metrics |
| Step 2: ML Framing | Loss function, training objective | Cross-entropy with teacher forcing, encoder-decoder framing | Exposure bias, weighted loss, CIDEr optimization motivation |
| Step 3: Data Strategy | Dataset knowledge, noise handling | Knows COCO/Conceptual Captions, CLIP filtering | Data mixing curriculum, augmentation semantics, caption quality scoring |
| Step 4: Architecture — Encoder | CNN vs. ViT trade-offs | Correct preference with data-size rationale | Inductive bias, patch count arithmetic, Swin for hierarchical tasks |
| Step 4: Architecture — Cross-Attention | Mechanism and role | Q/K/V roles, causal vs. non-causal masking | Diagnostic use of attention maps, Q-Former vs. MLP projection trade-off |
| Step 5: CIDEr | Metric mechanics and gaming | TF-IDF explanation, SCST connection | Gaming detection pipeline, vocabulary novelty monitoring |
| Step 5: CHAIR | Hallucination measurement | CHAIR definition, proxy-ground-truth approach | Proxy calibration, VQA consistency check supplement, production sampling strategy |
| Step 6: System Design | End-to-end inference pipeline | Identifies decoder as bottleneck, KV-cache, greedy decoding | Speculative decoding, p99 tail latency analysis, continuous batching |
| Step 7: Deployment | Production monitoring | CLIP alignment score, CHAIR proxy, human sampling | Causal failure mode diagnosis, shadow deployment, medical zero-tolerance protocol |
| Probe 1: Patch Size | Resolution and spatial encoding | Patch arithmetic, resolution increase solution | Swin architecture, 2D positional encoding, full medical architecture |
| Probe 2: Hallucination | Mechanistic diagnosis | Language prior explanation, attention map diagnosis | Attention rollout, contrastive decoding, POPE benchmark |
| Probe 3: Q-Former vs. MLP | Modern multimodal architecture | Q-Former compression vs. MLP preservation trade-off | InstructBLIP task-conditioning, information bottleneck framing |
| Probe 4: Compositionality | Architectural limitation analysis | Winoground, NegCLIP hard negatives | Relation-aware attention, scene graph supervision, performance-generalization trade-off |
| Probe 5: SCST | RL for metric optimization | REINFORCE gradient, self-critical baseline, reward variance | Custom medical reward function, reward hacking prevention, curriculum |

### Overall Hiring Thresholds

| Recommendation | Criteria |
|---|---|
| **Strong Hire** | Strong Hire or Hire on at least 10 of 15 competencies, with Strong Hire on at least 4 Deep Technical Probes; no Disqualifying red flags |
| **Hire** | Hire or above on at least 10 of 15 competencies; no Disqualifying red flags; one or two Moderate red flags acceptable |
| **Weak Hire** | Hire or above on 7-9 competencies; demonstrates genuine depth in at least one domain (e.g., evaluation or architecture) even if breadth is limited; no Disqualifying red flags |
| **No Hire** | Hire or above on fewer than 7 competencies; any Disqualifying red flag; cannot explain cross-attention mechanics or fails to identify hallucination as a distinct failure mode |

### Calibration Notes for Interviewers

A Staff Engineer candidate should demonstrate the ability to make correct trade-off decisions independently, own a complete system from data to deployment, and mentor others on the team. They are not expected to have deep familiarity with every paper referenced (e.g., InstructBLIP architecture details) but must be able to reason from first principles to arrive at similar conclusions when prompted.

A Principal Engineer candidate must demonstrate the ability to set technical direction across multiple teams, identify non-obvious systemic risks before they become problems, and frame engineering decisions in terms of organizational impact. In the context of this interview, this means proactively raising concerns like the mismatch between COCO annotation style and medical use case, the patient safety implications of hallucination in radiology, or the organizational cost of retraining the reward model in SCST. These are insights that a Staff Engineer might produce when prompted but that a Principal Engineer should raise without prompting.

If a candidate demonstrates exceptional depth in one domain (e.g., they have clearly shipped a production medical imaging system and know every detail of CHAIR, clinical NER, and radiologist workflow integration) but is shallow on another (e.g., they are not familiar with Q-Former), weight their practical depth heavily — the gaps can be closed on the job, but genuine production experience cannot be simulated in an interview.

---

*Guide version: 2026-02. Calibrated against COCO 2017 benchmark, BLIP-2 (ICML 2023), LLaVA-1.5 (NeurIPS 2023), and InstructBLIP (NeurIPS 2023) state of the art.*
