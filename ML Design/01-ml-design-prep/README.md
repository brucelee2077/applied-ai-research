# ML System Design Interview Prep -- The Complete Guide

> Based on *Machine Learning System Design Interview* (Chapter 1: Introduction and Overview)
>
> Written as if explaining to a 12-year-old, but with staff-level ML engineering depth.

---

## Table of Contents

1. [What Is an ML System Design Interview?](#what-is-an-ml-system-design-interview)
2. [The 7-Step Framework](#the-7-step-framework)
3. [Step 1: Clarifying Requirements](#step-1-clarifying-requirements)
4. [Step 2: Frame the Problem as an ML Task](#step-2-frame-the-problem-as-an-ml-task)
5. [Step 3: Data Preparation](#step-3-data-preparation)
6. [Step 4: Model Development](#step-4-model-development)
7. [Step 5: Evaluation](#step-5-evaluation)
8. [Step 6: Deployment and Serving](#step-6-deployment-and-serving)
9. [Step 7: Monitoring and Infrastructure](#step-7-monitoring-and-infrastructure)
10. [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)
11. [Interview Tips and Strategies](#interview-tips-and-strategies)
12. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## What Is an ML System Design Interview?

**Simple analogy:** Imagine your teacher asks you to "design a library that recommends books to every student in school." You wouldn't just say "use a computer to pick books." You'd need to think about: How do we know what books exist? How do we know what each student likes? How do we make it fast enough so students don't wait forever? What if a student's tastes change? That is what an ML system design interview is about -- designing the *whole system*, not just the algorithm.

**Technical reality:** Most engineers think of ML algorithms (logistic regression, neural networks) as the entirety of an ML system. In reality, a production ML system is far more complex. It includes:

- **Data stacks** to manage data
- **Serving infrastructure** to make the system available to millions of users
- **Evaluation pipelines** to measure performance
- **Monitoring** to ensure the model doesn't degrade over time

At an ML system design interview, you are given **open-ended questions** like "design a movie recommendation system" or "design a video search engine." There is no single correct answer. The interviewer wants to evaluate:

1. Your **thought process** -- how you break down a big fuzzy problem
2. Your **in-depth understanding** of various ML topics
3. Your ability to design an **end-to-end system**
4. Your **design choices** based on trade-offs

---

## The 7-Step Framework

Think of this framework like a recipe. When you bake a cake, you follow steps: gather ingredients, mix, bake, decorate. Similarly, when designing an ML system, you follow these steps:

```
Step 1: Clarifying Requirements
Step 2: Frame the Problem as an ML Task
Step 3: Data Preparation
Step 4: Model Development
Step 5: Evaluation
Step 6: Deployment and Serving
Step 7: Monitoring and Infrastructure
```

**Important caveat:** Every interview is different because the problem is open-ended. There is no one-size-fits-all approach. The framework helps you *structure your thoughts*, but you should be flexible. If an interviewer is mainly interested in model development, follow their lead.

---

## Step 1: Clarifying Requirements

**Simple analogy:** Imagine someone tells you "build me a cool robot." Before you start, you'd ask: "What should the robot do? How big should it be? Does it need to fly? How much money can we spend?" That is what clarifying requirements means -- asking questions before you start building.

ML system design questions are **intentionally vague**. For example: "Design an event recommendation system." Your first job is to ask clarifying questions.

### Categories of Clarifying Questions

| Category | What to Ask | Example |
|----------|------------|---------|
| **Business Objective** | What is the ultimate goal? | "Are we trying to increase bookings or increase revenue?" |
| **Features the System Needs** | What user interactions exist? | "Can users 'like' or 'dislike' recommended videos?" |
| **Data** | What data sources exist? How much? Labeled? | "Is the data user-generated or system-generated?" |
| **Constraints** | Computing power? Cloud or on-device? | "Should the model improve automatically over time?" |
| **Scale** | How many users? How many items? Growth rate? | "How many videos are we dealing with?" |
| **Performance** | Latency requirements? Accuracy vs. speed? | "Is a real-time solution expected?" |

### Other Important Topics

- **Privacy and ethics** -- always consider these
- **Bias** -- is there potential for unfair outcomes?

### What to Do After Asking

Write down the list of requirements and constraints you gather. This ensures everyone is on the same page. Think of it as your "contract" with the interviewer.

---

## Step 2: Frame the Problem as an ML Task

**Simple analogy:** Suppose your friend says "I want more people to come to my lemonade stand." That is a *business problem*, not something you can directly tell a computer to solve. You need to translate it: "Let's predict which neighbors are most likely to want lemonade today, and put signs on their street." Now it is an ML task -- a prediction problem.

### Three Sub-Steps

#### 2a. Define the ML Objective

A business objective like "increase sales by 20%" is not something you can train a model on directly. You must translate it:

| Application | Business Objective | ML Objective |
|------------|-------------------|--------------|
| Event ticket selling app | Increase ticket sales | Recommend events the user is likely to purchase |
| Video streaming app | Increase user engagement | Recommend videos the user is likely to watch |
| Ad click prediction | Increase user clicks | Predict the probability a user clicks on an ad |
| Harmful content detection | Improve platform safety | Predict whether a post is harmful or not |
| Friend recommendation | Increase network growth rate | Recommend users who are likely to become friends |

#### 2b. Specify the System's Input and Output

Once you have the ML objective, define what goes in and what comes out.

**Example:** For harmful content detection:
- **Input:** A social media post (text, image, video)
- **Output:** Whether the post is harmful or not (binary classification)

**Key insight:** Sometimes the system has *multiple* ML models. For example, harmful content detection might use:
- Model 1: Predicts violence
- Model 2: Predicts nudity
- The system combines both to decide if a post is harmful

**Another key insight:** There are often *multiple ways* to specify a model's input/output. Discuss these options with the interviewer and explain your reasoning.

#### 2c. Choose the Right ML Category

Most problems fit into one of these categories:

```
Machine Learning
├── Supervised Learning
│   ├── Classification
│   │   ├── Binary Classification (is this spam? yes/no)
│   │   └── Multiclass Classification (is this a dog, cat, or rabbit?)
│   ├── Regression (predict a continuous number, like house price)
│   └── Ranking (order items by relevance)
├── Unsupervised Learning (find patterns without labels)
└── Reinforcement Learning (learn by trial-and-error, like AlphaGo)
```

**In practice:** The vast majority of real-world ML systems use **supervised learning** because models learn better when labeled training data is available.

### Talking Points for This Step

- What is a good ML objective? Pros and cons of different objectives?
- What are the inputs and outputs?
- If multiple models are involved, what are the inputs/outputs of each?
- Supervised or unsupervised?
- Regression or classification? Binary or multiclass?

---

## Step 3: Data Preparation

**Simple analogy:** Before you cook dinner, you need to buy ingredients, wash them, chop them up, and measure them out. Data preparation is the same idea for ML -- getting your data clean, organized, and in the right format.

Data preparation has two main parts: **Data Engineering** and **Feature Engineering**.

### 3a. Data Engineering

Data engineering is the practice of designing and building pipelines for collecting, storing, retrieving, and processing data.

#### Data Sources

An ML system can work with data from many different sources. Knowing the sources helps answer context questions:
- Who collected it?
- How clean is the data?
- Can the data source be trusted?
- Is it user-generated or system-generated?

#### Data Storage

Different databases serve different use cases:
- **Relational databases** (SQL) -- structured data with rows and columns
- **NoSQL databases** -- flexible schema for unstructured data
- **Data warehouses** -- optimized for analytical queries
- **Data lakes** -- raw storage for any type of data

#### ETL (Extract, Transform, Load)

This is a three-phase process:
1. **Extract:** Pull data from different data sources
2. **Transform:** Clean, map, and transform data into a specific format
3. **Load:** Put the transformed data into the target destination

#### Data Types

| Type | Description | Examples | Best Models |
|------|------------|----------|-------------|
| **Structured** | Follows a predefined schema | Dates, phone numbers, credit card numbers, addresses | Traditional ML (trees, linear models) |
| **Unstructured** | No underlying schema | Images, audio, video, text | Deep learning (CNNs, Transformers) |

Within structured data:
- **Numerical (Continuous):** House prices (any value in a range)
- **Numerical (Discrete):** Number of houses sold (distinct integers)
- **Categorical (Nominal):** Gender (no inherent order)
- **Categorical (Ordinal):** Rating from "bad" to "excellent" (has order)

### 3b. Feature Engineering

**Simple analogy:** Imagine you're describing your friend to someone who has never met them. You wouldn't just hand over their birth certificate. You'd say "they're tall, have brown hair, love soccer, and always wear red shoes." You're *choosing the most useful features* to help someone recognize your friend. Feature engineering is doing the same for ML models.

Feature engineering has two processes:
1. Using **domain knowledge** to select and extract predictive features from raw data
2. **Transforming** those features into a format usable by the model

#### Common Feature Engineering Operations

**Handling Missing Values:**

| Method | How It Works | Drawback |
|--------|-------------|----------|
| Row Deletion | Remove rows with missing values | Reduces training data |
| Column Deletion | Remove features with too many missing values | Loses potentially useful features |
| Imputation (defaults) | Fill missing values with default values | May introduce noise |
| Imputation (mean/median/mode) | Fill with statistical measures | May introduce noise |

**Feature Scaling:**

Why? Many ML models struggle when features have different ranges (e.g., age 0-100 vs. income 0-1,000,000).

- **Normalization (min-max scaling):** Scales all values to [0, 1]
  - Formula: `x_normalized = (x - x_min) / (x_max - x_min)`
- **Standardization (z-score):** Transforms to mean=0, std=1
  - Formula: `x_standardized = (x - mean) / std`
- **Log transformation:** Reduces skewness in distributions

**Discretization (Bucketing):**

Converting continuous values into buckets. Example:

| Bucket | Age Range |
|--------|-----------|
| 1 | 0-9 |
| 2 | 10-19 |
| 3 | 20-29 |
| 4 | 30-39 |
| 5 | 40+ |

**Encoding Categorical Features:**

Since ML models need numbers, not words:

1. **Integer Encoding:** Assign integers (Excellent=1, Good=2, Bad=3). Only works when there is a natural ordinal relationship.

2. **One-Hot Encoding:** Create a new binary column for each unique value.
   - Red -> [1, 0, 0]
   - Green -> [0, 1, 0]
   - Blue -> [0, 0, 1]

3. **Embedding Learning:** Map each category to an N-dimensional vector. Useful when the number of unique values is very large (one-hot would create huge sparse vectors).

### Talking Points for Data Preparation

- Data availability and collection: sources, size, frequency
- Data storage: cloud vs. device, format, multimodal data handling
- Feature engineering: processing raw data, handling missing data, normalization
- Privacy: sensitivity of data, anonymization, on-device storage
- Biases: types of biases present, correction methods

---

## Step 4: Model Development

**Simple analogy:** Now you've got your ingredients ready and measured (data prepared). Time to choose your recipe (model) and start cooking (training). Should you make a simple sandwich, or a complex multi-layer cake? It depends on what you need.

### Model Selection

A typical process:

1. **Establish a simple baseline** -- e.g., recommend the most popular videos to everyone
2. **Experiment with simple models** -- logistic regression, decision trees (quick to train)
3. **Switch to complex models** -- deep neural networks (if simple models aren't good enough)
4. **Use an ensemble** -- combine multiple models for better accuracy (bagging, boosting, stacking)

#### Common Model Options

| Model | When to Consider |
|-------|-----------------|
| Logistic Regression | Linear tasks, fast training, interpretable |
| Linear Regression | Continuous output prediction |
| Decision Trees | Interpretable, handles non-linear relationships |
| Gradient Boosted Trees / Random Forests | Strong performance on structured/tabular data |
| Support Vector Machines | Good for high-dimensional data |
| Naive Bayes | Text classification, fast baseline |
| Factorization Machines (FM) | Recommendation systems with sparse features |
| Neural Networks | Complex patterns, unstructured data |

#### Key Considerations When Choosing a Model

- **Amount of training data needed**
- **Training speed**
- **Hyperparameters to tune**
- **Possibility of continual learning**
- **Compute requirements** (GPU vs. CPU)
- **Interpretability** -- more complex models may be less interpretable
- **Inference latency** -- can the model serve predictions fast enough?
- **Can it be deployed on-device?**

### Model Training

#### Constructing the Dataset

Five steps:
1. **Collect raw data** (covered in data preparation)
2. **Identify features and labels** (task-specific, the most important part)
3. **Select a sampling strategy** (convenience, snowball, stratified, reservoir, importance sampling)
4. **Split the data** (training / validation / test)
5. **Address class imbalances**

#### Getting Labels

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Hand Labeling** | Human annotators label data | Accurate labels | Expensive, slow, introduces bias, needs domain knowledge, privacy concerns |
| **Natural Labeling** | Labels inferred automatically from user behavior | Free, scalable, no annotation needed | May be noisy, limited to certain tasks |

**Natural labeling example:** In a news feed system, you train a model to predict whether a user will "like" a post. The label is 1 if the user liked it, 0 if they didn't. The labels come naturally from user interactions -- no human annotator needed.

#### Handling Class Imbalance

When one class vastly outnumbers another (e.g., 99% non-spam, 1% spam):

**Approach 1: Resampling**
- **Oversampling** the minority class (duplicate minority examples)
- **Undersampling** the majority class (remove majority examples)

**Approach 2: Altering the Loss Function**
- Give more weight to minority class data points
- **Class-balanced loss:** Weights inversely proportional to class frequency
- **Focal loss:** Down-weights easy examples, focuses on hard ones

#### Choosing the Loss Function

The loss function measures how wrong the model's predictions are. Common choices:

| Task | Common Loss Functions |
|------|---------------------|
| Classification | Cross-entropy |
| Regression | MSE (Mean Squared Error), MAE (Mean Absolute Error), Huber loss |
| Ranking | Pairwise loss, Listwise loss |
| Imbalanced classes | Focal loss, Class-balanced loss |

#### Regularization

Prevents overfitting:
- **L1 regularization** (Lasso) -- encourages sparsity
- **L2 regularization** (Ridge) -- penalizes large weights
- **Dropout** -- randomly deactivates neurons during training
- **K-fold cross-validation** -- better use of limited data

#### Training from Scratch vs. Fine-Tuning

- **Training from scratch:** Start with random weights, train on your data
- **Fine-tuning:** Take a pre-trained model, continue training on your specific data with small parameter updates

#### Distributed Training

When models or datasets are too large for one machine:
- **Data parallelism:** Split data across multiple worker nodes, each trains on a subset
- **Model parallelism:** Split the model itself across multiple machines

### Talking Points for Model Development

- Model selection: which models are suitable, pros/cons
- Training time, data requirements, compute resources, latency
- Interpretability trade-offs
- Dataset labels: how to obtain them, annotation quality, natural labels
- Loss function choice (Cross-entropy, MSE, MAE, Huber, etc.)
- Regularization (L1, L2, dropout, K-fold CV)
- Backpropagation and optimization methods (SGD, AdaGrad, Momentum, RMSProp)
- Activation functions (ReLU, Sigmoid, Tanh, ELU) and why
- Handling imbalanced datasets
- Bias-variance trade-off
- Overfitting and underfitting: causes and solutions
- Continual learning: how often to retrain (daily, weekly, monthly, yearly)
- Neural network architecture choices (ResNet, Transformer)
- Hyperparameter tuning

---

## Step 5: Evaluation

**Simple analogy:** After baking your cake, you don't just serve it immediately. You taste-test it first (offline evaluation). Then you give a slice to a few friends to see if they like it (online evaluation). Only if everyone's happy do you serve it at the party.

### Offline Evaluation

Offline evaluation measures model performance during development, before it sees real users.

| Task | Common Offline Metrics |
|------|----------------------|
| **Classification** | Precision, Recall, F1 Score, Accuracy, ROC-AUC, PR-AUC, Confusion Matrix |
| **Regression** | MSE, MAE, RMSE |
| **Ranking** | Precision@k, Recall@k, MRR (Mean Reciprocal Rank), mAP (mean Average Precision), nDCG (normalized Discounted Cumulative Gain) |
| **Image Generation** | FID (Frechet Inception Distance), Inception Score |
| **NLP** | BLEU, METEOR, ROUGE, CIDEr, SPICE |

### Online Evaluation

Online evaluation measures how the model performs in production with real users. These metrics are tied to business objectives.

| Problem | Online Metrics |
|---------|---------------|
| **Ad click prediction** | Click-through rate, revenue lift |
| **Harmful content detection** | Prevalence, valid appeals |
| **Video recommendation** | Click-through rate, total watch time, number of completed videos |
| **Friend recommendation** | Number of requests sent per day, number of requests accepted per day |

### Talking Points for Evaluation

- Which online metrics measure the ML system's effectiveness?
- How do these metrics relate to the business objective?
- Which offline metrics best evaluate the model during development?
- **Fairness and bias:** Does the model have potential for bias across attributes like age, gender, or race?
- What if someone with malicious intent gets access to your system?

---

## Step 6: Deployment and Serving

**Simple analogy:** Your cake is tested and delicious. Now you need to figure out how to get it to the party. Do you deliver it yourself (cloud deployment), or give each friend the recipe so they can bake it at home (on-device deployment)?

### Cloud vs. On-Device Deployment

| Dimension | Cloud | On-Device |
|-----------|-------|-----------|
| **Simplicity** | Simple to deploy and manage | More complex to deploy |
| **Cost** | Cloud costs can be high | No cloud cost |
| **Network Latency** | Present (data travels to server and back) | None |
| **Inference Latency** | Usually faster (more powerful machines) | Slower (limited hardware) |
| **Hardware Constraints** | Fewer constraints | Limited memory, battery, etc. |
| **Privacy** | Less private (user data goes to cloud) | More private (data stays on device) |
| **Internet Dependency** | Needs internet connection | No internet needed |

### Model Compression

Making models smaller for faster inference and smaller deployments:

1. **Knowledge Distillation:** Train a small "student" model to mimic a larger "teacher" model
2. **Pruning:** Find the least useful parameters and set them to zero, creating sparser models
3. **Quantization:** Use fewer bits to represent parameters (e.g., 32-bit -> 8-bit), reducing model size

### Testing in Production

The only way to ensure a model works well in production is to test it with real traffic:

#### Shadow Deployment
Deploy the new model alongside the existing model. Both receive every request, but only the old model's predictions are served to users. This lets you compare without risk.
- **Pro:** Zero risk to users
- **Con:** Doubles compute cost

#### A/B Testing
Deploy the new model in parallel. Route a portion of traffic to the new model and the rest to the old model.
- Traffic routing must be **random**
- Must run on a **sufficient number of data points** for statistical significance

#### Other Methods
- **Canary release:** Gradually roll out to a small percentage of users
- **Interleaving experiments:** Mix results from both models in the same response
- **Bandits:** Dynamically allocate traffic based on observed performance

### Prediction Pipeline

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Batch Prediction** | Predictions computed periodically ahead of time | No latency worries once pre-computed | Less responsive to changing preferences; only works if you know what to compute in advance |
| **Online Prediction** | Predictions generated in real-time when requests arrive | Responsive to user needs; works for any input | Model must be fast enough for real-time serving |

**Rule of thumb:**
- Use **online prediction** when you don't know what to compute in advance (e.g., language translation)
- Use **batch prediction** when processing high volumes and real-time results aren't needed

### Talking Points for Deployment

- Is model compression needed? Which techniques?
- Online prediction or batch prediction? Trade-offs?
- Is real-time access to features possible? Challenges?
- How should we test the deployed model?
- What components does the system have, and what are each component's responsibilities?
- What technologies ensure fast and scalable serving?

---

## Step 7: Monitoring and Infrastructure

**Simple analogy:** Even after the party starts and everyone is eating your cake, you keep an eye on things. Is anyone getting sick? Are you running out of slices? Is the frosting melting in the sun? That is monitoring -- keeping watch over your system after it's live.

### Why Systems Fail in Production

The most common reason: **Data Distribution Shift**

This means the data the model sees in production differs from what it saw during training. Example: You trained a cup-recognition model on front-view images, but production images include cups at weird angles.

**How to handle it:**
1. **Train on large datasets** so the model learns a comprehensive distribution
2. **Regularly retrain** the model using labeled data from the new distribution

### What to Monitor

| Category | Metrics |
|----------|---------|
| **Operation-Related** | Average serving time, throughput, number of prediction requests, CPU/GPU utilization |
| **ML-Specific: Inputs/Outputs** | Monitor the model's input data quality and output distribution |
| **ML-Specific: Drifts** | Detect changes in underlying distribution of inputs or outputs |
| **ML-Specific: Model Accuracy** | Expect accuracy to be within a specific range |
| **ML-Specific: Model Versions** | Track which version is currently deployed |

### Infrastructure

Infrastructure is the foundation for training, deploying, and maintaining ML systems. Not all interviews require deep infrastructure knowledge, but roles like DevOps and MLOps may need it.

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Jumping Straight to Model Selection
**What happens:** You hear "design a recommendation system" and immediately say "I'd use a Transformer."
**Why it's bad:** You skipped understanding the problem, the data, and the constraints.
**Fix:** Always start with clarifying requirements.

### Pitfall 2: Giving Unstructured Answers
**What happens:** You ramble about different topics without a clear flow.
**Why it's bad:** The interviewer can't follow your thought process.
**Fix:** Follow the 7-step framework. Tell the interviewer your plan upfront.

### Pitfall 3: Ignoring the Business Objective
**What happens:** You optimize for the wrong thing. You design a perfectly accurate model that doesn't actually help the business.
**Why it's bad:** ML models exist to serve business goals, not the other way around.
**Fix:** Always connect your ML objective to the business objective.

### Pitfall 4: Not Discussing Trade-Offs
**What happens:** You pick one approach without explaining why or what alternatives exist.
**Why it's bad:** The interviewer wants to see that you understand the landscape.
**Fix:** For every major decision, mention at least one alternative and explain the trade-offs.

### Pitfall 5: Skipping Data Discussion
**What happens:** You assume perfect data exists.
**Why it's bad:** In reality, data is messy, biased, incomplete, and expensive to label.
**Fix:** Discuss data sources, quality, labeling strategy, and potential biases.

### Pitfall 6: Forgetting About Production
**What happens:** You design a beautiful model that can't be served at scale.
**Why it's bad:** A model that can't run in production is useless.
**Fix:** Always discuss deployment, serving latency, and monitoring.

### Pitfall 7: Being Inflexible
**What happens:** You rigidly follow your framework even when the interviewer signals they want to go deeper on a specific topic.
**Why it's bad:** The interview is a conversation, not a presentation.
**Fix:** Be ready to go with the interviewer's flow while maintaining overall structure.

---

## Interview Tips and Strategies

### Before the Interview

1. **Practice the framework** on 10-15 common ML design problems
2. **Know common ML categories** (classification, regression, ranking, etc.) and when to use each
3. **Understand metrics** -- be able to explain precision, recall, F1, AUC, nDCG, etc.
4. **Study real-world systems** -- how do YouTube, Netflix, Instagram actually work?
5. **Prepare talking points** for generic topics (deployment, monitoring, infrastructure)

### During the Interview

1. **Drive the conversation** -- don't wait for the interviewer to guide you step by step
2. **Communicate your thought process** -- the interviewer can't read your mind
3. **Write down requirements** -- shows thoroughness and organization
4. **Be honest about what you don't know** -- it's better than making things up
5. **Draw diagrams** -- visual representations of your system architecture are powerful
6. **Discuss trade-offs for every decision** -- this is the #1 signal of a strong candidate
7. **Be flexible** -- if the interviewer raises a question, go with their flow
8. **Time management** -- don't spend 30 minutes on requirements and run out of time for model design

### The Role Matters

Different roles emphasize different parts of the framework:

| Role | Focus Areas |
|------|------------|
| **Data Science** | Data engineering, feature engineering, model development |
| **Applied ML Engineer** | Model development, deployment, production |
| **ML Infrastructure / MLOps** | Deployment, serving, monitoring, infrastructure |
| **Research Scientist** | Model development, novel architectures, loss functions |

### The "No Best Algorithm" Principle

There is no single best algorithm that solves all problems. The interviewer wants to see if you:
- Understand different ML algorithms and their pros/cons
- Can choose a model based on requirements and constraints
- Can explain *why* you made a particular choice

---

## Quick Reference Cheat Sheet

### The 7-Step Framework at a Glance

```
1. CLARIFY REQUIREMENTS
   Ask about: business objective, features, data, constraints, scale, performance

2. FRAME AS ML TASK
   Define ML objective | Specify inputs/outputs | Choose ML category

3. DATA PREPARATION
   Data Engineering: sources, storage, ETL, data types
   Feature Engineering: missing values, scaling, encoding

4. MODEL DEVELOPMENT
   Model Selection: baseline -> simple -> complex -> ensemble
   Model Training: dataset construction, loss function, training strategy

5. EVALUATION
   Offline: precision, recall, F1, AUC, nDCG, MSE, etc.
   Online: CTR, revenue lift, watch time, etc.

6. DEPLOYMENT & SERVING
   Cloud vs. on-device | Compression | A/B testing | Batch vs. online prediction

7. MONITORING & INFRASTRUCTURE
   Data distribution shift | Operational metrics | ML-specific metrics
```

### Common Interview Questions (from the book)

The book covers these specific ML system design problems:

1. **Visual Search System**
2. **Google Street View Blurring System**
3. **YouTube Video Search**
4. **Harmful Content Detection**
5. **Video Recommendation System**
6. **Event Recommendation System**
7. **Ad Click Prediction on Social Platforms**
8. **Similar Listings on Vacation Rental Platforms**
9. **Personalized News Feed**
10. **People You May Know**

### Key Formulas to Remember

| Formula | Use Case |
|---------|----------|
| `Precision = TP / (TP + FP)` | Of all positive predictions, how many were correct? |
| `Recall = TP / (TP + FN)` | Of all actual positives, how many did we find? |
| `F1 = 2 * (Precision * Recall) / (Precision + Recall)` | Harmonic mean of precision and recall |
| `x_norm = (x - x_min) / (x_max - x_min)` | Min-max normalization |
| `x_std = (x - mean) / std` | Standardization (z-score) |

---

## Final Words

Remember: an ML system design interview is not about finding the "right answer." It is about demonstrating that you can think through a complex problem systematically, understand the trade-offs at every decision point, and design a system that works end-to-end in the real world.

No engineer can be an expert in every aspect of the ML lifecycle. Some specialize in deployment, others in model development. What matters is that you can **drive the conversation**, show **breadth across the full pipeline**, and go **deep where it matters** for the specific role you're interviewing for.

Now let's get hands-on with the notebooks.

---

## Notebooks in This Module

| Notebook | Topic |
|----------|-------|
| `01_ml_design_framework.ipynb` | Complete 7-step framework: requirements, framing, data prep, model dev, evaluation, deployment, monitoring -- all with runnable code |
| `02_interview_strategy.ipynb` | Interview strategy, communication templates, 3 mini-walkthroughs, 50 key vocabulary terms, self-assessment checklist |
