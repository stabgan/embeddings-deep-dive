# Chapter 8: Fine-Tuning Embeddings — A Deep Dive

## Table of Contents

1. [Why Fine-Tune?](#1-why-fine-tune)
2. [The Fine-Tuning Pipeline](#2-the-fine-tuning-pipeline)
3. [Multiple Negatives Ranking Loss (MNRL)](#3-multiple-negatives-ranking-loss-mnrl)
4. [Hard Negative Mining](#4-hard-negative-mining)
5. [Knowledge Distillation for Embeddings](#5-knowledge-distillation-for-embeddings)
6. [Data Augmentation for Embeddings](#6-data-augmentation-for-embeddings)
7. [SimCSE (2021)](#7-simcse-2021)
8. [Practical Fine-Tuning Recipe](#8-practical-fine-tuning-recipe)
9. [What Happens During Fine-Tuning — Layer by Layer](#9-what-happens-during-fine-tuning--layer-by-layer)

---

## 1. Why Fine-Tune?

### 1.1 The Domain Shift Problem

Pre-trained embedding models like BERT, RoBERTa, and their sentence-transformer variants
are trained on broad web corpora — Wikipedia, BookCorpus, Common Crawl. These models learn
a general-purpose representation of language. But language in specialized domains diverges
sharply from general text.

Consider the word "transformer" in three domains:

| Domain | Meaning of "transformer" | Nearest neighbors (general model) |
|--------|--------------------------|-----------------------------------|
| NLP | Attention-based neural architecture | model, network, architecture |
| Electrical Engineering | Voltage conversion device | inductor, capacitor, coil |
| Entertainment | Robots in disguise (franchise) | movie, cartoon, toy |

A general-purpose embedding model conflates these meanings into a single region of the
vector space. When you deploy this model for electrical engineering document retrieval,
queries about "transformer efficiency" return results about neural network training speed
instead of power conversion losses.

**This is domain shift**: the distribution of text your model was trained on differs from
the distribution it encounters at deployment.

### 1.2 Quantifying Domain Shift

We can measure domain shift by comparing vocabulary overlap and embedding space geometry:

**Vocabulary divergence**: For a target domain corpus D_target and the pre-training corpus
D_pretrain, compute the fraction of domain-specific tokens absent or rare in pre-training:

```
Divergence = |{w ∈ vocab(D_target) : freq(w, D_pretrain) < threshold}| / |vocab(D_target)|
```

Typical values:

| Domain | Vocabulary Divergence (threshold=100) |
|--------|---------------------------------------|
| Biomedical (PubMed) | 38% |
| Legal (case law) | 22% |
| Financial (SEC filings) | 19% |
| General web text | 3% |

### 1.3 Task-Specific Optimization

Beyond domain shift, general embeddings are not optimized for any specific task. A model
trained with masked language modeling (MLM) learns token-level representations, but these
don't directly optimize for:

- **Semantic similarity**: Are two sentences paraphrases?
- **Information retrieval**: Does this document answer this query?
- **Clustering**: Do these documents belong to the same topic?
- **Classification**: What category does this text belong to?

Each task imposes different geometric requirements on the embedding space:

| Task | Geometric Requirement | What Fine-Tuning Optimizes |
|------|----------------------|---------------------------|
| Similarity | Paraphrases close, non-paraphrases far | Cosine distance between semantic equivalents |
| Retrieval | Relevant docs close to query | Asymmetric query-document similarity |
| Clustering | Intra-cluster tight, inter-cluster separated | Cluster compactness and separation |
| Classification | Linear separability by class | Decision boundary geometry |

### 1.4 Before and After: A Concrete Example

Let's measure the impact of fine-tuning on a biomedical retrieval task. We use the
NFCorpus dataset (nutrition and fitness) and measure NDCG@10:

```
Base model: all-MiniLM-L6-v2 (general purpose)
  NDCG@10 on NFCorpus: 0.312

Fine-tuned on 10K biomedical query-passage pairs:
  NDCG@10 on NFCorpus: 0.387  (+24.0% relative improvement)

Fine-tuned on 10K pairs + hard negatives:
  NDCG@10 on NFCorpus: 0.421  (+34.9% relative improvement)
```

The embedding space geometry changes dramatically. Before fine-tuning, biomedical concepts
are scattered across the space. After fine-tuning, related concepts cluster tightly:

```
Before fine-tuning — cosine similarities:
  sim("insulin resistance", "type 2 diabetes")     = 0.41
  sim("insulin resistance", "glucose metabolism")   = 0.38
  sim("insulin resistance", "car insurance")        = 0.12

After fine-tuning:
  sim("insulin resistance", "type 2 diabetes")     = 0.82
  sim("insulin resistance", "glucose metabolism")   = 0.79
  sim("insulin resistance", "car insurance")        = 0.03
```

Fine-tuning amplifies the signal (related pairs become more similar) and suppresses the
noise (unrelated pairs become more distant).

---

## 2. The Fine-Tuning Pipeline

### 2.1 Overview

The embedding fine-tuning pipeline consists of five steps:

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Choose Base Model                                      │
│    ↓                                                            │
│  Step 2: Prepare Training Data (pairs, triplets, or labeled)    │
│    ↓                                                            │
│  Step 3: Select Loss Function                                   │
│    ↓                                                            │
│  Step 4: Train with Appropriate Hyperparameters                 │
│    ↓                                                            │
│  Step 5: Evaluate on Held-Out Data                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Step 1: Choose a Base Model

The base model determines your starting point in embedding space. Key considerations:

| Model | Parameters | Max Seq Len | Speed | Quality | Best For |
|-------|-----------|-------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 22M | 256 | ★★★★★ | ★★★ | Fast prototyping, edge |
| all-mpnet-base-v2 | 109M | 384 | ★★★ | ★★★★ | Balanced performance |
| bge-base-en-v1.5 | 109M | 512 | ★★★ | ★★★★★ | Retrieval tasks |
| e5-large-v2 | 335M | 512 | ★★ | ★★★★★ | Maximum quality |
| BERT-base-uncased | 110M | 512 | ★★★ | ★★★ | Custom training from scratch |
| RoBERTa-base | 125M | 512 | ★★★ | ★★★★ | Robust pre-training |

**Decision framework**:

1. **If you have < 10K training pairs**: Start with a strong pre-trained sentence
   transformer (bge-base, e5-base). The pre-training does most of the work.
2. **If you have 10K–100K pairs**: Any base model works. Larger models benefit more.
3. **If you have > 100K pairs**: Consider starting from a raw BERT/RoBERTa checkpoint
   and training from scratch with your loss function.

### 2.3 Step 2: Prepare Training Data

Training data format depends on your loss function. The three main formats:

**Format A: Pairs (query, positive)**
```python
# For Multiple Negatives Ranking Loss
training_pairs = [
    ("What causes diabetes?", "Diabetes is caused by insulin resistance..."),
    ("Python list comprehension", "List comprehensions provide a concise way..."),
    ("How to fix a flat tire", "To repair a punctured tire, first remove..."),
]
```

**Format B: Triplets (anchor, positive, negative)**
```python
# For Triplet Loss or Triplet-based training
training_triplets = [
    ("What causes diabetes?",
     "Diabetes is caused by insulin resistance...",      # positive
     "Diabetes was first described in ancient Egypt..."), # hard negative
]
```

**Format C: Pairs with scores (sentence_a, sentence_b, score)**
```python
# For CosineSimilarityLoss or regression-style training
training_scored = [
    ("A man is playing guitar", "A musician performs on stage", 0.85),
    ("A cat sits on a mat", "The stock market crashed today", 0.02),
]
```

**Data quality matters more than quantity.** A common finding in embedding research:

```
5K high-quality pairs  >  50K noisy pairs  (in downstream performance)
```

### 2.4 Step 3: Select Loss Function

The loss function is the most critical decision. Here's a decision tree:

```
Do you have labeled similarity scores?
├── Yes → CosineSimilarityLoss or CoSENT Loss
└── No
    ├── Do you have (query, positive) pairs?
    │   ├── Yes → Multiple Negatives Ranking Loss (MNRL)  ← Most common
    │   └── No
    │       ├── Do you have (anchor, positive, negative) triplets?
    │       │   ├── Yes → TripletLoss
    │       │   └── No
    │       │       └── Do you have only unlabeled sentences?
    │       │           └── Yes → SimCSE (unsupervised)
    └── Do you have a teacher cross-encoder?
        └── Yes → MarginMSE (knowledge distillation)
```

### 2.5 Step 4: Train with Appropriate Hyperparameters

Key hyperparameters for embedding fine-tuning:

| Hyperparameter | Typical Range | Notes |
|---------------|---------------|-------|
| Learning rate | 2e-5 to 5e-5 | Lower than classification fine-tuning |
| Batch size | 32 to 1024 | Larger is better for contrastive losses |
| Warmup ratio | 0.1 (10% of steps) | Prevents early divergence |
| Epochs | 1–3 | Embeddings overfit quickly |
| Weight decay | 0.01 | Standard AdamW regularization |
| Temperature τ | 0.05 to 0.1 | For contrastive losses |
| Max sequence length | 128–512 | Task-dependent |

### 2.6 Step 5: Evaluate on Held-Out Data

Evaluation metrics depend on the downstream task:

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Retrieval | NDCG@10 | MRR@10, Recall@100 |
| Similarity | Spearman correlation | Pearson correlation |
| Clustering | V-measure | Adjusted Rand Index |
| Classification | Accuracy | F1-score |
| Pair classification | AP (Average Precision) | F1 at optimal threshold |

**Always evaluate on data from your target domain**, not on general benchmarks. A model
that scores 0.85 on STS-Benchmark may score 0.45 on your domain-specific similarity task.

---

## 3. Multiple Negatives Ranking Loss (MNRL)

### 3.1 Why MNRL Is the Most Important Loss

Multiple Negatives Ranking Loss (also called InfoNCE loss or in-batch negatives loss) has
become the dominant loss function for training modern embedding models. It was popularized
by Henderson et al. (2017) for response selection and formalized in the contrastive learning
literature (Oord et al., 2018; Chen et al., 2020).

**Key insight**: You don't need explicit negative examples. Given a batch of (query,
positive) pairs, every other positive in the batch serves as a negative for each query.
This gives you B-1 negatives for free, where B is the batch size.

### 3.2 Setup and Notation

Given a batch of B training pairs:

```
{(q₁, p₁), (q₂, p₂), ..., (q_B, p_B)}
```

where:
- q_i is the i-th query (or anchor)
- p_i is the positive (relevant) passage for q_i
- B is the batch size

We compute embeddings for all queries and positives:

```
h_i^q = Encoder(q_i)    ∈ ℝ^d     (query embedding)
h_j^p = Encoder(p_j)    ∈ ℝ^d     (passage embedding)
```

The similarity between query i and passage j is:

```
s_{ij} = sim(h_i^q, h_j^p) / τ
```

where sim(·,·) is typically cosine similarity and τ is a temperature parameter.

### 3.3 The Loss Function — Step-by-Step Derivation

**Step 1: Construct the similarity matrix.**

For a batch of size B, compute the B × B similarity matrix S:

```
        p₁      p₂      p₃      ...   p_B
q₁  [ s₁₁    s₁₂    s₁₃    ...   s₁B  ]
q₂  [ s₂₁    s₂₂    s₂₃    ...   s₂B  ]
q₃  [ s₃₁    s₃₂    s₃₃    ...   s₃B  ]
...  [ ...    ...    ...    ...   ...   ]
q_B [ s_B1   s_B2   s_B3   ...   s_BB  ]
```

The diagonal entries s_{ii} are the positive (correct) similarities.
All off-diagonal entries s_{ij} (i ≠ j) are negative (incorrect) similarities.

**Step 2: Apply softmax row-wise.**

For each query q_i, we want the probability of selecting the correct positive p_i to be
high relative to all other passages in the batch:

```
P(p_i | q_i) = exp(s_{ii}) / Σ_{j=1}^{B} exp(s_{ij})
```

This is a softmax over the i-th row of the similarity matrix, where the "correct class"
is the diagonal entry.

**Step 3: Take the negative log-likelihood.**

The loss for query q_i is:

```
L_i = -log P(p_i | q_i)
     = -log [ exp(s_{ii}) / Σ_{j=1}^{B} exp(s_{ij}) ]
     = -s_{ii} + log Σ_{j=1}^{B} exp(s_{ij})
```

**Step 4: Average over the batch.**

The total MNRL loss is:

```
L_MNRL = (1/B) Σ_{i=1}^{B} L_i
       = (1/B) Σ_{i=1}^{B} [ -s_{ii} + log Σ_{j=1}^{B} exp(s_{ij}) ]
```

Expanding with the temperature and cosine similarity:

```
L_MNRL = (1/B) Σ_{i=1}^{B} [ -cos(h_i^q, h_i^p)/τ + log Σ_{j=1}^{B} exp(cos(h_i^q, h_j^p)/τ) ]
```

**Step 5: Understand the gradient signal.**

The gradient of L_i with respect to the similarity s_{ij} is:

```
∂L_i/∂s_{ij} = -𝟙[i=j] + exp(s_{ij}) / Σ_{k=1}^{B} exp(s_{ik})
              = -𝟙[i=j] + P(p_j | q_i)
```

This means:
- For the positive pair (j = i): gradient = P(p_i | q_i) - 1 (always negative → pushes
  similarity up)
- For negative pairs (j ≠ i): gradient = P(p_j | q_i) (always positive → pushes
  similarity down)
- The magnitude of the push is proportional to the current probability — hard negatives
  (high P(p_j | q_i)) get pushed harder.

### 3.4 Why Batch Size Matters Enormously

The number of negatives per query equals B - 1. This directly affects the quality of the
gradient signal:

```
Batch size B = 4:    3 negatives per query
Batch size B = 32:   31 negatives per query
Batch size B = 256:  255 negatives per query
Batch size B = 1024: 1023 negatives per query
```

**Theoretical justification**: MNRL is a lower bound on mutual information (Oord et al.,
2018). The tightness of this bound improves with more negatives:

```
I(q; p) ≥ log(B) - L_MNRL
```

With B = 4, the bound is at most log(4) ≈ 1.39 nats.
With B = 1024, the bound is at most log(1024) ≈ 6.93 nats.

More negatives → tighter bound → better gradient signal → better embeddings.

**Empirical evidence**: Performance scales log-linearly with batch size:

```
Batch Size | NDCG@10 (MS MARCO) | Negatives per query
-----------|---------------------|--------------------
32         | 0.321               | 31
64         | 0.335               | 63
128        | 0.348               | 127
256        | 0.359               | 255
512        | 0.367               | 511
1024       | 0.372               | 1023
```

Returns diminish beyond ~512, but the improvement from 32 → 256 is substantial.

### 3.5 Numerical Example with Batch Size 4

Let's walk through MNRL with a concrete batch of 4 pairs and temperature τ = 0.05.

**The batch:**

```
(q₁, p₁) = ("What is photosynthesis?", "Photosynthesis converts light energy...")
(q₂, p₂) = ("Python for loops", "A for loop in Python iterates over...")
(q₃, p₃) = ("Best pizza recipe", "For authentic Neapolitan pizza, use...")
(q₄, p₄) = ("How do vaccines work?", "Vaccines stimulate the immune system...")
```

**Step 1: Compute embeddings** (simplified to 4D for illustration):

```
h₁^q = [0.8, 0.1, 0.0, 0.2]    h₁^p = [0.7, 0.2, 0.1, 0.1]
h₂^q = [0.1, 0.9, 0.1, 0.0]    h₂^p = [0.0, 0.8, 0.2, 0.1]
h₃^q = [0.0, 0.1, 0.8, 0.3]    h₃^p = [0.1, 0.0, 0.9, 0.2]
h₄^q = [0.3, 0.0, 0.2, 0.8]    h₄^p = [0.2, 0.1, 0.1, 0.9]
```

**Step 2: Compute cosine similarities** (raw, before temperature scaling):

```
cos(h₁^q, h₁^p) = (0.56+0.02+0.00+0.02) / (0.837×0.748) = 0.60/0.626 ≈ 0.958
cos(h₁^q, h₂^p) = (0.00+0.08+0.00+0.02) / (0.837×0.837) = 0.10/0.700 ≈ 0.143
cos(h₁^q, h₃^p) = (0.08+0.00+0.00+0.04) / (0.837×0.922) = 0.12/0.772 ≈ 0.155
cos(h₁^q, h₄^p) = (0.16+0.01+0.00+0.18) / (0.837×0.949) = 0.35/0.794 ≈ 0.441
```

**Step 3: Apply temperature scaling** (τ = 0.05):

```
s₁₁ = 0.958 / 0.05 = 19.16
s₁₂ = 0.143 / 0.05 =  2.86
s₁₃ = 0.155 / 0.05 =  3.10
s₁₄ = 0.441 / 0.05 =  8.82
```

**Step 4: Compute softmax for q₁:**

```
exp(19.16) ≈ 2.11 × 10⁸
exp(2.86)  ≈ 17.46
exp(3.10)  ≈ 22.20
exp(8.82)  ≈ 6,765

Σ = 2.11 × 10⁸ + 17.46 + 22.20 + 6,765 ≈ 2.11 × 10⁸

P(p₁|q₁) = 2.11 × 10⁸ / 2.11 × 10⁸ ≈ 0.99997
```

**Step 5: Compute loss for q₁:**

```
L₁ = -log(0.99997) ≈ 0.00003
```

This is a very low loss — the model already strongly associates q₁ with p₁. The
temperature τ = 0.05 creates sharp distributions. In practice, early in training,
similarities are lower and the loss is much higher.

**Step 6: Consider a harder case.** Suppose q₄ and p₁ have cosine similarity 0.35
(vaccines and photosynthesis share some biology vocabulary):

```
s₄₁ = 0.35 / 0.05 = 7.0    exp(7.0) ≈ 1,097
s₄₂ = 0.05 / 0.05 = 1.0    exp(1.0) ≈ 2.72
s₄₃ = 0.08 / 0.05 = 1.6    exp(1.6) ≈ 4.95
s₄₄ = 0.92 / 0.05 = 18.4   exp(18.4) ≈ 9.78 × 10⁷

P(p₄|q₄) = 9.78×10⁷ / (1,097 + 2.72 + 4.95 + 9.78×10⁷) ≈ 0.99999
L₄ ≈ 0.00001
```

Even with a confusing negative (biology overlap), the temperature scaling makes the
correct positive dominate. But with a higher temperature (τ = 0.5):

```
s₄₁ = 0.35/0.5 = 0.70    exp(0.70) ≈ 2.01
s₄₄ = 0.92/0.5 = 1.84    exp(1.84) ≈ 6.30

P(p₄|q₄) = 6.30 / (2.01 + ... + 6.30) ≈ 0.65
L₄ = -log(0.65) ≈ 0.43
```

Higher temperature → softer distribution → more gradient signal from negatives.

### 3.6 Gradient Flow Analysis

Let's trace the gradient from the loss back to the model parameters.

**Layer 1: Loss → Similarity scores**

```
∂L_i/∂s_{ij} = -𝟙[i=j] + softmax(s_{i,:})_j
```

**Layer 2: Similarity scores → Embeddings**

For cosine similarity with temperature:

```
s_{ij} = (h_i^q · h_j^p) / (‖h_i^q‖ · ‖h_j^p‖ · τ)
```

The gradient with respect to the query embedding h_i^q:

```
∂s_{ij}/∂h_i^q = (1/τ) · [ h_j^p/‖h_j^p‖ · 1/‖h_i^q‖ - (h_i^q · h_j^p)/(‖h_i^q‖³ · ‖h_j^p‖) · h_i^q ]
```

Simplifying with normalized embeddings (‖h‖ = 1):

```
∂s_{ij}/∂h_i^q = (1/τ) · [ h_j^p - (h_i^q · h_j^p) · h_i^q ]
```

This is the component of h_j^p orthogonal to h_i^q, scaled by 1/τ.

**Layer 3: Embeddings → Model parameters**

The embedding h = Encoder(x) is produced by the transformer. The gradient flows through:

```
∂L/∂θ = Σ_{i,j} (∂L_i/∂s_{ij}) · (∂s_{ij}/∂h) · (∂h/∂θ)
```

where θ represents all transformer parameters (attention weights, FFN weights, layer
norms, etc.).

**Key insight**: The gradient signal for each query is a weighted combination of:
- A pull toward the positive passage embedding (weighted by 1 - P(p_i|q_i))
- Pushes away from each negative passage embedding (weighted by P(p_j|q_i))

Hard negatives (high P(p_j|q_i)) contribute more gradient, which is why they accelerate
learning.

---

## 4. Hard Negative Mining

### 4.1 Why Random Negatives Are Too Easy

In MNRL with in-batch negatives, most negatives are "easy" — they come from completely
unrelated topics. Consider a batch containing:

```
q₁: "How to train a neural network"     p₁: "Backpropagation computes gradients..."
q₂: "Best Italian restaurants in NYC"   p₂: "Carbone on Thompson Street..."
q₃: "Symptoms of the common cold"       p₃: "Runny nose, sore throat..."
q₄: "History of the Roman Empire"       p₄: "Founded in 27 BC by Augustus..."
```

For q₁, the negatives p₂, p₃, p₄ are trivially distinguishable. The cosine similarity
between "neural network training" and "Italian restaurants" is near zero. The softmax
probability for the correct positive is ~1.0, and the loss is ~0.0.

**The model learns nothing from easy negatives.** The gradient signal is vanishingly small.

### 4.2 What Makes a Negative "Hard"?

A hard negative for query q is a passage that is:
1. **Not relevant** to q (it's genuinely a negative)
2. **Superficially similar** to the positive passage (high lexical or semantic overlap)

Examples of hard negatives for "What causes type 2 diabetes?":

```
Positive:  "Type 2 diabetes is primarily caused by insulin resistance,
            where cells fail to respond to insulin properly."

Easy negative:    "The Eiffel Tower was built in 1889."
                  (cosine sim ≈ 0.02, gradient ≈ 0)

Medium negative:  "Type 1 diabetes is an autoimmune condition where the
                   immune system attacks insulin-producing cells."
                  (cosine sim ≈ 0.55, some gradient)

Hard negative:    "Insulin resistance can be measured using the HOMA-IR
                   index, calculated from fasting glucose and insulin levels."
                  (cosine sim ≈ 0.72, strong gradient — related topic but
                   doesn't answer the question)
```

### 4.3 BM25 Hard Negatives

BM25 (Best Matching 25) is a classical lexical retrieval function. It finds passages that
share many words with the query but may not be semantically relevant.

**Mining procedure:**

```
For each query q_i in training data:
    1. Index all passages using BM25 (e.g., with Elasticsearch or Pyserini)
    2. Retrieve top-K passages by BM25 score: R = BM25(q_i, top_k=100)
    3. Remove the known positive: R = R \ {p_i}
    4. Select hard negatives: n_i = R[0:num_negatives]
```

**Why BM25 negatives work**: BM25 finds passages with high lexical overlap but potentially
different semantics. These are exactly the cases where a semantic model needs to learn
the difference between "shares words" and "answers the question."

```python
from pyserini.search.lucene import LuceneSearcher

# Index passages
searcher = LuceneSearcher('path/to/index')

# Mine hard negatives
def mine_bm25_negatives(query, positive_id, num_negatives=10):
    hits = searcher.search(query, k=100)
    negatives = []
    for hit in hits:
        if hit.docid != positive_id:
            negatives.append(hit.docid)
        if len(negatives) >= num_negatives:
            break
    return negatives
```

### 4.4 Cross-Encoder Hard Negatives

A cross-encoder processes query and passage jointly through a transformer, producing a
relevance score. Cross-encoders are more accurate than bi-encoders but too slow for
retrieval. We can use them to mine high-quality hard negatives.

**Mining procedure:**

```
For each query q_i:
    1. Retrieve candidate negatives using BM25 or a bi-encoder: C = retrieve(q_i, top_k=200)
    2. Remove known positives: C = C \ {p_i}
    3. Score all candidates with cross-encoder: scores = CE(q_i, c) for c in C
    4. Select passages with HIGH cross-encoder scores as hard negatives
       (the cross-encoder thinks they're relevant, but they're not)
    5. n_i = top_k(C, by=scores, k=num_negatives)
```

**Why this works**: Cross-encoder hard negatives are passages that fool even a strong
model. Training the bi-encoder to distinguish these cases forces it to learn subtle
semantic differences.

### 4.5 Mining Strategies: Static vs. Dynamic

**Static Mining (Pre-compute)**

Mine all hard negatives before training begins:

```
Procedure: Static Hard Negative Mining
─────────────────────────────────────────
Input: Training set T = {(q_i, p_i)}_{i=1}^{N}, retrieval model M
Output: Augmented training set T' = {(q_i, p_i, n_i^1, ..., n_i^K)}

1. Build index I from all passages {p_1, ..., p_N} using model M
2. For each query q_i in T:
   a. Retrieve top-R passages: candidates = M.search(q_i, I, top_k=R)
   b. Filter out positive: candidates = candidates \ {p_i}
   c. Select top-K as hard negatives: n_i^1, ..., n_i^K = candidates[:K]
3. Return augmented dataset T'
```

Advantages:
- Simple to implement
- No overhead during training
- Reproducible

Disadvantages:
- Negatives become stale as the model improves during training
- Early-training hard negatives may be easy by epoch 2

**Dynamic Mining (Mine during training)**

Re-mine hard negatives periodically during training:

```
Procedure: Dynamic Hard Negative Mining
─────────────────────────────────────────
Input: Training set T, initial model M₀, re-mining interval R_interval

1. Initialize model M = M₀
2. Mine initial negatives using M: T' = static_mine(T, M)
3. For each training step t:
   a. Train on batch from T'
   b. If t % R_interval == 0:
      i.   Update index I using current model M
      ii.  Re-mine negatives: T' = static_mine(T, M)
      iii. Log: "Re-mined negatives at step {t}"
4. Return final model M
```

Advantages:
- Negatives stay challenging as the model improves
- Curriculum-like effect: negatives get harder over time

Disadvantages:
- Expensive: requires re-encoding and re-indexing
- Non-deterministic training

**Practical recommendation**: Start with static BM25 negatives. If you have compute
budget, add one round of re-mining after the first epoch using the partially trained model.

### 4.6 Step-by-Step Example: Mining Hard Negatives

Let's walk through mining hard negatives for a single query.

**Query**: "What is the capital of France?"
**Positive**: "Paris is the capital and most populous city of France."

**Step 1: BM25 retrieval (top 10)**

```
Rank | BM25 Score | Passage                                              | Hard?
-----|------------|------------------------------------------------------|------
1    | 18.7       | "Paris is the capital and most populous city of..."  | (positive, skip)
2    | 15.2       | "France is a country in Western Europe with..."      | Medium
3    | 14.8       | "The capital of Germany is Berlin, which..."          | Hard ✓
4    | 13.1       | "Paris, Texas is a city in Lamar County..."           | Hard ✓
5    | 12.4       | "French is the official language of France..."        | Medium
6    | 11.9       | "Lyon is the third-largest city in France..."         | Hard ✓
7    | 10.3       | "The Eiffel Tower is located in Paris..."             | Medium
8    | 9.8        | "Capital punishment was abolished in France in..."   | Hard ✓
9    | 8.5        | "The French Revolution began in 1789..."              | Easy
10   | 7.2        | "Marseille is a port city in southern France..."      | Medium
```

**Step 2: Score with cross-encoder**

```
Rank | CE Score | Passage                                              | Selection
-----|----------|------------------------------------------------------|----------
3    | 0.82     | "The capital of Germany is Berlin, which..."          | ✓ Top hard neg
6    | 0.45     | "Lyon is the third-largest city in France..."         | ✓ Hard neg
4    | 0.38     | "Paris, Texas is a city in Lamar County..."           | ✓ Hard neg
8    | 0.12     | "Capital punishment was abolished in France in..."   | ✗ Too easy for CE
```

**Step 3: Final training example**

```python
{
    "query": "What is the capital of France?",
    "positive": "Paris is the capital and most populous city of France.",
    "hard_negatives": [
        "The capital of Germany is Berlin, which...",   # Confuses "capital of [country]"
        "Lyon is the third-largest city in France...",  # Right country, wrong city
        "Paris, Texas is a city in Lamar County...",    # Right name, wrong Paris
    ]
}
```

---

## 5. Knowledge Distillation for Embeddings

### 5.1 The Teacher-Student Framework

Knowledge distillation transfers knowledge from a large, accurate "teacher" model to a
smaller, faster "student" model. For embeddings, the typical setup is:

```
Teacher: Cross-encoder (e.g., cross-encoder/ms-marco-MiniLM-L-12-v2)
  - Processes query and passage jointly
  - Very accurate but O(N) inference for N passages (too slow for retrieval)

Student: Bi-encoder (e.g., sentence-transformers/all-MiniLM-L6-v2)
  - Encodes query and passage independently
  - Less accurate but O(1) retrieval with pre-computed passage embeddings
```

The goal: make the bi-encoder's similarity scores match the cross-encoder's relevance
scores as closely as possible.

### 5.2 Margin-MSE Loss — Derivation

The Margin-MSE loss (Hofstätter et al., 2020) is the standard distillation loss for
embedding models. Instead of matching absolute scores, it matches the *margin* between
positive and negative passages.

**Setup:**

For a query q, positive passage d⁺, and negative passage d⁻:

```
Teacher scores:
  t⁺ = CE_teacher(q, d⁺)     (cross-encoder score for positive)
  t⁻ = CE_teacher(q, d⁻)     (cross-encoder score for negative)

Student scores:
  s⁺ = sim(Enc(q), Enc(d⁺))  (bi-encoder similarity for positive)
  s⁻ = sim(Enc(q), Enc(d⁻))  (bi-encoder similarity for negative)
```

**Teacher margin:**

```
m_teacher = t⁺ - t⁻
```

This represents how much more relevant the teacher thinks d⁺ is compared to d⁻.

**Student margin:**

```
m_student = s⁺ - s⁻
```

**Margin-MSE loss:**

```
L_MarginMSE = MSE(m_student, m_teacher)
            = (m_student - m_teacher)²
            = ((s⁺ - s⁻) - (t⁺ - t⁻))²
```

For a batch of N triplets:

```
L = (1/N) Σ_{i=1}^{N} ((s_i⁺ - s_i⁻) - (t_i⁺ - t_i⁻))²
```

**Why margins instead of absolute scores?**

Cross-encoder scores and bi-encoder similarities are on different scales:
- Cross-encoder: logits in [-10, 10] or probabilities in [0, 1]
- Bi-encoder: cosine similarity in [-1, 1]

Matching absolute scores requires careful calibration. Margins are scale-invariant:
if the teacher says d⁺ is 3.5 points better than d⁻, the student should also produce
a proportional gap, regardless of the absolute scale.

### 5.3 Step-by-Step Numerical Example

**Query**: "How does photosynthesis work?"

**Passages:**
```
d⁺: "Photosynthesis converts CO₂ and water into glucose using sunlight energy,
     occurring in chloroplasts through light-dependent and light-independent reactions."

d⁻₁: "Cellular respiration breaks down glucose to produce ATP, the energy currency
      of cells, through glycolysis, the Krebs cycle, and oxidative phosphorylation."

d⁻₂: "The water cycle describes how water evaporates from surfaces, condenses in
      clouds, and falls as precipitation."
```

**Step 1: Get teacher scores**

```
t(q, d⁺)  = 9.2   (highly relevant)
t(q, d⁻₁) = 3.8   (related biology topic, but wrong process)
t(q, d⁻₂) = 0.5   (unrelated)
```

**Step 2: Compute teacher margins**

```
m_teacher(d⁺, d⁻₁) = 9.2 - 3.8 = 5.4
m_teacher(d⁺, d⁻₂) = 9.2 - 0.5 = 8.7
```

**Step 3: Get student scores (current bi-encoder)**

```
s(q, d⁺)  = 0.72
s(q, d⁻₁) = 0.58   (student struggles with this hard negative)
s(q, d⁻₂) = 0.11
```

**Step 4: Compute student margins**

```
m_student(d⁺, d⁻₁) = 0.72 - 0.58 = 0.14
m_student(d⁺, d⁻₂) = 0.72 - 0.11 = 0.61
```

**Step 5: Compute loss**

We need to normalize margins to the same scale. Using min-max normalization on teacher
margins to [0, 1]:

```
m_teacher_norm(d⁺, d⁻₁) = 5.4 / 8.7 = 0.621
m_teacher_norm(d⁺, d⁻₂) = 8.7 / 8.7 = 1.000
```

```
L₁ = (0.14 - 0.621)² = (-0.481)² = 0.231
L₂ = (0.61 - 1.000)² = (-0.390)² = 0.152

L_total = (0.231 + 0.152) / 2 = 0.192
```

**Step 6: Interpret the gradient**

The loss is high for the (d⁺, d⁻₁) pair because the student's margin (0.14) is much
smaller than the teacher's normalized margin (0.621). The gradient will:
- Push s(q, d⁺) higher (increase positive similarity)
- Push s(q, d⁻₁) lower (decrease hard negative similarity)
- The push is stronger for the hard negative pair (larger loss)

### 5.4 Why Distillation from Cross-Encoders Works So Well

Cross-encoders provide richer training signal than binary labels:

```
Binary labels:     relevant=1, not_relevant=0  (2 bits of information)
Cross-encoder:     score=7.3 vs score=3.1      (continuous, nuanced signal)
```

The cross-encoder captures:
1. **Degree of relevance**: "somewhat related" vs "exactly answers the question"
2. **Fine-grained ordering**: Among 10 negatives, which are hardest?
3. **Soft labels**: Passages that are partially relevant get intermediate scores

This is analogous to label smoothing in classification — soft targets provide more
gradient information per training step than hard 0/1 labels.

**Empirical results** (Hofstätter et al., 2020):

```
Training signal          | MRR@10 (MS MARCO Dev)
-------------------------|----------------------
Binary labels only       | 0.330
BM25 hard negatives      | 0.349
Cross-encoder distill.   | 0.371
CE distill. + hard neg.  | 0.382
```

---

## 6. Data Augmentation for Embeddings

### 6.1 Why Augmentation Matters

Labeled training data for embeddings is expensive. Each training pair requires a human to
judge whether two texts are semantically related. Data augmentation creates additional
training signal from existing data.

### 6.2 Back-Translation

Back-translation generates paraphrases by translating text to another language and back:

```
Original (English):  "The cat sat on the mat"
    → Translate to French: "Le chat était assis sur le tapis"
    → Translate back to English: "The cat was sitting on the rug"
```

The back-translated version is a natural paraphrase — same meaning, different words.

**Using back-translation for embedding training:**

```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src_lang="en", pivot_lang="de"):
    # English → German
    fwd_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
    fwd_tokenizer = MarianTokenizer.from_pretrained(fwd_model_name)
    fwd_model = MarianMTModel.from_pretrained(fwd_model_name)

    # German → English
    bwd_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"
    bwd_tokenizer = MarianTokenizer.from_pretrained(bwd_model_name)
    bwd_model = MarianMTModel.from_pretrained(bwd_model_name)

    # Forward translation
    inputs = fwd_tokenizer(text, return_tensors="pt", padding=True)
    translated = fwd_model.generate(**inputs)
    pivot_text = fwd_tokenizer.decode(translated[0], skip_special_tokens=True)

    # Backward translation
    inputs = bwd_tokenizer(pivot_text, return_tensors="pt", padding=True)
    back_translated = bwd_model.generate(**inputs)
    result = bwd_tokenizer.decode(back_translated[0], skip_special_tokens=True)

    return result

# Create augmented training pair
original = "Neural networks learn hierarchical representations of data."
paraphrase = back_translate(original)
# → "Neural networks learn hierarchical representations of data."
# (or a paraphrase like "Neural networks acquire layered data representations.")

# New training pair: (original, paraphrase) — positive pair for contrastive learning
```

**Multiple pivot languages** increase diversity:

```
Original:     "The experiment yielded surprising results"
Via German:   "The experiment produced surprising results"
Via French:   "The experiment gave surprising results"
Via Chinese:  "The experimental results were surprising"
```

### 6.3 Dropout as Augmentation (SimCSE Insight)

The key insight from SimCSE (Gao et al., 2021): standard dropout in a transformer creates
implicit data augmentation. Passing the same sentence through the encoder twice with
different dropout masks produces two slightly different embeddings.

```
Input: "The cat sat on the mat"

Pass 1 (dropout mask A):  h  = [0.32, 0.71, -0.15, 0.48, ...]
Pass 2 (dropout mask B):  h' = [0.29, 0.68, -0.18, 0.51, ...]

cosine_sim(h, h') ≈ 0.95  (high but not identical)
```

This works because dropout randomly zeros out ~10% of hidden units. The two passes
"see" slightly different subnetworks, producing embeddings that agree on the core
semantics but differ in noise.

**Why this is brilliant**: No external data, no translation models, no human labels.
Just the model's own stochasticity provides positive pairs.

### 6.4 Cropping and Deletion

**Random cropping**: Take a random contiguous subsequence of the input:

```
Original:  "The quick brown fox jumps over the lazy dog near the river"
Crop 1:    "brown fox jumps over the lazy"
Crop 2:    "the lazy dog near the river"
```

These crops share semantic content and serve as positive pairs.

**Random token deletion**: Randomly remove tokens:

```
Original:  "The quick brown fox jumps over the lazy dog"
Delete:    "The brown fox jumps the lazy dog"        (removed "quick", "over")
```

**Random token replacement**: Replace tokens with [MASK] or random words:

```
Original:  "The quick brown fox jumps over the lazy dog"
Replace:   "The quick brown cat jumps over the lazy dog"  (fox → cat)
```

**Effectiveness comparison** (from SimCSE paper, STS-B Spearman correlation):

```
Augmentation Method     | STS-B (Spearman ρ)
------------------------|--------------------
No augmentation         | 74.5
Random deletion         | 75.2
Random cropping         | 75.8
Back-translation        | 76.1
Dropout (SimCSE)        | 76.8
```

Dropout augmentation outperforms all explicit augmentation strategies while being
simpler and faster.

---

## 7. SimCSE (2021)

### 7.1 Overview

SimCSE (Simple Contrastive Learning of Sentence Embeddings) by Gao et al. (2021) is a
landmark paper that showed how to train high-quality sentence embeddings with minimal
supervision. It introduced two variants:

1. **Unsupervised SimCSE**: Uses dropout as the only augmentation
2. **Supervised SimCSE**: Uses Natural Language Inference (NLI) pairs

### 7.2 Unsupervised SimCSE — The Dropout Trick

**Training procedure:**

Given a collection of sentences {x₁, x₂, ..., x_N}:

```
For each mini-batch of B sentences:
    1. Pass each sentence through the encoder TWICE with DIFFERENT dropout masks:
       h_i  = Encoder(x_i, dropout_mask=z)     # first pass
       h_i' = Encoder(x_i, dropout_mask=z')    # second pass (different mask)

    2. The positive pair for x_i is (h_i, h_i')
       All other sentences in the batch are negatives

    3. Apply contrastive loss (same as MNRL):
       L_i = -log [ exp(sim(h_i, h_i')/τ) / Σ_{j=1}^{B} exp(sim(h_i, h_j')/τ) ]
```

**The loss function in full:**

```
L_unsup = -(1/B) Σ_{i=1}^{B} log [ exp(sim(h_i, h_i')/τ) / Σ_{j=1}^{B} exp(sim(h_i, h_j')/τ) ]
```

where:
- h_i = f_θ(x_i, z_i) is the embedding with dropout mask z_i
- h_i' = f_θ(x_i, z_i') is the embedding with a different dropout mask z_i'
- τ = 0.05 is the temperature
- B is the batch size

**Note**: The denominator includes j = i, so the positive pair competes against B total
candidates (itself + B-1 negatives).

### 7.3 Why Dropout Creates Good Positive Pairs

Standard dropout in BERT/RoBERTa operates at multiple levels:

```
Layer                    | Dropout locations
-------------------------|------------------------------------------
Embedding layer          | After embedding lookup (p=0.1)
Attention                | On attention weights (p=0.1)
Feed-forward             | After first linear layer (p=0.1)
Residual connections     | Before adding residual (p=0.1)
```

With 12 layers and ~4 dropout locations per layer, there are ~48 independent dropout
masks. Each pass through the encoder samples a different combination, creating a
different "view" of the same input.

**Mathematical perspective**: Let f_θ(x, z) be the encoder output where z is the
concatenation of all dropout masks. The two passes sample:

```
z  ~ Bernoulli(1-p)^D    (D = total number of dropout positions)
z' ~ Bernoulli(1-p)^D    (independent sample)
```

The expected embedding is the same: E_z[f_θ(x, z)] = E_z'[f_θ(x, z')], but the
variance creates a small perturbation that acts as a positive augmentation.

**Key property**: Dropout perturbations are:
1. **Semantics-preserving**: The meaning doesn't change (same input)
2. **Diverse**: Different dropout masks create different perturbations
3. **Calibrated**: The perturbation magnitude is controlled by dropout rate p
4. **Free**: No additional computation, data, or models needed

### 7.4 Supervised SimCSE — NLI Pairs

For supervised SimCSE, the authors use Natural Language Inference (NLI) datasets, which
contain sentence pairs labeled as entailment, neutral, or contradiction.

**Training data construction:**

```
From NLI dataset:
  Premise:       "Two dogs are running in a park"
  Entailment:    "Animals are playing outdoors"        → positive pair
  Contradiction: "The dogs are sleeping inside"        → hard negative

Training triplet: (premise, entailment, contradiction)
```

**The supervised loss:**

```
L_sup = -(1/B) Σ_{i=1}^{B} log [ exp(sim(h_i, h_i⁺)/τ) /
        (Σ_{j=1}^{B} exp(sim(h_i, h_j⁺)/τ) + Σ_{j=1}^{B} exp(sim(h_i, h_j⁻)/τ)) ]
```

where:
- h_i is the premise embedding
- h_i⁺ is the entailment (positive) embedding
- h_j⁻ is the contradiction (hard negative) embedding

The denominator includes both in-batch positives (as negatives for other queries) AND
explicit contradiction negatives, giving 2B - 1 negatives per query.

### 7.5 Uniformity and Alignment Analysis

Gao et al. introduced two metrics to analyze embedding quality:

**Alignment**: Measures how close positive pairs are in the embedding space:

```
ℓ_align = E_{(x,x⁺)~p_pos} [ ‖f(x) - f(x⁺)‖² ]
```

Lower alignment loss → positive pairs are closer together.

**Uniformity**: Measures how uniformly distributed embeddings are on the unit hypersphere:

```
ℓ_uniform = log E_{(x,y)~p_data} [ exp(-2‖f(x) - f(y)‖²) ]
```

Lower uniformity loss → embeddings are more uniformly spread out.

**The ideal embedding space** has both low alignment (positives are close) and low
uniformity (all embeddings are spread out). These two objectives are in tension:

```
                    Alignment
                    (positives close)
                         ↑
                         |
        Collapsed ●      |      ● Ideal
        (all same)       |      (aligned + uniform)
                         |
    ─────────────────────┼──────────────→ Uniformity
                         |               (spread out)
                         |
        Random ●         |      ● Uniform but
        (neither)        |        unaligned
                         |
```

**SimCSE results on alignment and uniformity:**

```
Model                    | ℓ_align | ℓ_uniform | STS-B (Spearman)
-------------------------|---------|-----------|------------------
BERT-base (CLS, no FT)  | 0.335   | -1.41     | 20.3
BERT-base (mean pool)    | 0.278   | -2.72     | 47.3
Unsupervised SimCSE      | 0.241   | -2.86     | 76.8
Supervised SimCSE        | 0.218   | -3.01     | 81.6
```

SimCSE improves both alignment (positive pairs become closer) and uniformity (the
embedding space becomes more isotropic), explaining its strong performance.

### 7.6 The Anisotropy Problem

Pre-trained BERT embeddings suffer from **anisotropy**: embeddings occupy a narrow cone
in the high-dimensional space rather than being uniformly distributed.

```
Isotropic (uniform):          Anisotropic (BERT):
    ·  ·  ·  ·                      · · ·
  ·  ·  ·  ·  ·                   · · · ·
 ·  ·  ·  ·  ·  ·                · · · · ·
  ·  ·  ·  ·  ·                    · · ·
    ·  ·  ·  ·                       ·
(spread across sphere)        (clustered in a cone)
```

**Measuring anisotropy**: Compute the average cosine similarity between random sentence
pairs. For a perfectly isotropic space, this should be ~0. For BERT:

```
Model                    | Avg cosine (random pairs) | Interpretation
-------------------------|---------------------------|---------------
Ideal (isotropic)        | ~0.00                     | Uniform
BERT-base (CLS)          | 0.65                      | Highly anisotropic
BERT-base (mean pool)    | 0.52                      | Anisotropic
SimCSE (unsupervised)    | 0.18                      | Nearly isotropic
SimCSE (supervised)      | 0.12                      | Nearly isotropic
```

SimCSE's contrastive objective directly combats anisotropy by pushing random pairs apart
(the uniformity term), spreading embeddings across the hypersphere.

---

## 8. Practical Fine-Tuning Recipe

### 8.1 The Complete Recipe

Here is a battle-tested recipe for fine-tuning embedding models:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING FINE-TUNING RECIPE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base model:      sentence-transformers/all-MiniLM-L6-v2        │
│  Loss:            MultipleNegativesRankingLoss                  │
│  Learning rate:   2e-5                                          │
│  Batch size:      256 (or largest that fits in GPU memory)      │
│  Warmup:          10% of total steps                            │
│  Epochs:          1–3 (evaluate after each)                     │
│  Weight decay:    0.01                                          │
│  Optimizer:       AdamW                                         │
│  Scheduler:       Linear warmup + linear decay                  │
│  Temperature:     0.05 (default for MNRL)                       │
│  Evaluation:      Every 1000 steps on validation set            │
│  Early stopping:  Patience 3 evaluations                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Hyperparameter Deep Dive

**Learning Rate: 2e-5 to 5e-5**

Embedding fine-tuning uses lower learning rates than classification fine-tuning because:
1. We're adjusting the entire representation space, not just a classification head
2. Large updates can catastrophically distort pre-trained knowledge
3. The contrastive loss landscape has many local minima

```
Learning Rate | Typical Outcome
--------------|--------------------------------------------------
1e-6          | Too slow, barely moves from pre-trained model
2e-5          | Conservative, safe for small datasets (<10K pairs)
3e-5          | Good default for most scenarios
5e-5          | Aggressive, good for large datasets (>100K pairs)
1e-4          | Usually too high, causes instability
```

**Batch Size: As Large As Possible**

For MNRL, batch size directly determines the number of negatives:

```
GPU Memory | Max Batch Size (MiniLM-L6) | Max Batch Size (BERT-base)
-----------|----------------------------|---------------------------
8 GB       | 128                        | 32
16 GB      | 256                        | 64
24 GB      | 512                        | 128
40 GB      | 1024                       | 256
80 GB      | 2048                       | 512
```

If your GPU can't fit a large batch, use **gradient accumulation**:

```python
# Effective batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus
# Example: 32 × 8 × 1 = 256 effective batch size
training_args = SentenceTransformerTrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
)
```

**Important caveat**: Gradient accumulation does NOT give the same in-batch negatives as
a true large batch. With accumulation, each micro-batch of 32 only sees 31 negatives.
The gradients are accumulated, but the contrastive signal is weaker. For MNRL, true large
batches are strongly preferred.

**Warmup: 10% of Total Steps**

Warmup prevents the model from making large, destructive updates in the first few steps
when the loss landscape is unfamiliar:

```
Step:     0    100   200   300   ...   1000  ...   10000
LR:      0    2e-6  4e-6  6e-6  ...   2e-5  ...   2e-5 → 0
         |←── warmup (10%) ──→|←──── linear decay ────→|
```

**Epochs: 1–3 (Embeddings Overfit Quickly)**

Unlike classification tasks where 5-10 epochs are common, embedding models overfit
rapidly because:
1. Contrastive losses are very efficient (each example provides B-1 negative signals)
2. The model memorizes specific pairs rather than learning general semantics
3. The embedding space can collapse to a degenerate solution

```
Epoch | Train Loss | Val NDCG@10 | Status
------|------------|-------------|--------
0     | 2.45       | 0.312       | Pre-trained baseline
1     | 0.82       | 0.387       | Good improvement
2     | 0.31       | 0.395       | Marginal improvement
3     | 0.12       | 0.391       | Slight degradation (overfitting)
4     | 0.04       | 0.378       | Clear overfitting
```

### 8.3 Complete Code Example

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset

# ─────────────────────────────────────────────────────────────
# Step 1: Load base model
# ─────────────────────────────────────────────────────────────
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────
# Step 2: Prepare training data
# ─────────────────────────────────────────────────────────────
# Format: {"anchor": query, "positive": relevant_passage}
# For MNRL, we only need positive pairs — negatives come from the batch
train_examples = [
    {"anchor": "What causes diabetes?",
     "positive": "Diabetes is caused by insulin resistance..."},
    {"anchor": "How do neural networks learn?",
     "positive": "Neural networks learn through backpropagation..."},
    {"anchor": "What is photosynthesis?",
     "positive": "Photosynthesis converts light energy into chemical energy..."},
    # ... thousands more pairs
]

train_dataset = Dataset.from_list(train_examples)

# ─────────────────────────────────────────────────────────────
# Step 3: Define loss function
# ─────────────────────────────────────────────────────────────
loss = losses.MultipleNegativesRankingLoss(model)

# ─────────────────────────────────────────────────────────────
# Step 4: Configure training arguments
# ─────────────────────────────────────────────────────────────
args = SentenceTransformerTrainingArguments(
    output_dir="./output/finetuned-model",
    num_train_epochs=2,
    per_device_train_batch_size=256,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,                          # Mixed precision for speed
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_ndcg@10",
    logging_steps=100,
)

# ─────────────────────────────────────────────────────────────
# Step 5: Set up evaluation
# ─────────────────────────────────────────────────────────────
# Prepare IR evaluation data
queries = {"q1": "What causes diabetes?", "q2": "How do neural networks learn?"}
corpus = {
    "d1": "Diabetes is caused by insulin resistance...",
    "d2": "Neural networks learn through backpropagation...",
    "d3": "The Eiffel Tower is in Paris...",
}
relevant_docs = {"q1": {"d1"}, "q2": {"d2"}}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="domain-eval",
)

# ─────────────────────────────────────────────────────────────
# Step 6: Train
# ─────────────────────────────────────────────────────────────
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)

trainer.train()

# ─────────────────────────────────────────────────────────────
# Step 7: Save and use
# ─────────────────────────────────────────────────────────────
model.save_pretrained("./output/final-model")

# Use the fine-tuned model
model = SentenceTransformer("./output/final-model")
embeddings = model.encode(["New query to embed"])
```

### 8.4 Fine-Tuning with Hard Negatives

To include hard negatives, modify the data format and loss:

```python
from sentence_transformers import losses

# Training data with hard negatives
train_examples_with_negatives = [
    {
        "anchor": "What causes diabetes?",
        "positive": "Diabetes is caused by insulin resistance...",
        "negative": "Diabetes was first described in ancient Egypt...",  # hard neg
    },
    # ... more examples
]

train_dataset = Dataset.from_list(train_examples_with_negatives)

# MNRL automatically uses the "negative" column as hard negatives
# These are added to the in-batch negatives
loss = losses.MultipleNegativesRankingLoss(model)
```

### 8.5 Fine-Tuning with Knowledge Distillation

```python
from sentence_transformers import losses

# Training data with teacher scores
train_examples_distill = [
    {
        "sentence1": "What causes diabetes?",
        "sentence2": "Diabetes is caused by insulin resistance...",
        "label": 0.95,  # Teacher cross-encoder score (normalized)
    },
    {
        "sentence1": "What causes diabetes?",
        "sentence2": "Diabetes was first described in ancient Egypt...",
        "label": 0.25,  # Teacher score for hard negative
    },
    # ... more examples
]

train_dataset = Dataset.from_list(train_examples_distill)

# Use CosineSimilarityLoss to match teacher scores
loss = losses.CosineSimilarityLoss(model)
```

---

## 9. What Happens During Fine-Tuning — Layer by Layer

### 9.1 Transformer Layer Anatomy (Recap)

A BERT-base model has 12 transformer layers. Each layer contains:

```
Layer l:
  ├── Multi-Head Self-Attention (MHSA)
  │     ├── Q = W_Q^l · h^{l-1}     (query projection)
  │     ├── K = W_K^l · h^{l-1}     (key projection)
  │     ├── V = W_V^l · h^{l-1}     (value projection)
  │     └── Attention(Q,K,V) = softmax(QK^T/√d_k) · V
  ├── Layer Norm + Residual
  ├── Feed-Forward Network (FFN)
  │     ├── FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
  │     └── W_1: d→4d, W_2: 4d→d
  └── Layer Norm + Residual
```

### 9.2 How Different Layers Change During Fine-Tuning

Research (Merchant et al., 2020; Mosbach et al., 2020) shows that fine-tuning affects
transformer layers differently:

**Lower layers (1-4): Minimal change**

These layers encode basic linguistic features — syntax, morphology, part-of-speech. These
features are useful across all tasks and domains, so fine-tuning preserves them.

```
Layer | Avg weight change (L2 norm) | What it encodes
------|----------------------------|------------------
1     | 0.012                      | Token identity, basic syntax
2     | 0.018                      | Phrase structure
3     | 0.025                      | Syntactic dependencies
4     | 0.031                      | Basic semantic roles
```

**Middle layers (5-8): Moderate change**

These layers encode semantic relationships and begin to specialize for the fine-tuning
task. The attention patterns shift to focus on task-relevant tokens.

```
Layer | Avg weight change (L2 norm) | What changes
------|----------------------------|------------------
5     | 0.048                      | Semantic grouping begins to shift
6     | 0.067                      | Topic-level representations adapt
7     | 0.089                      | Task-specific features emerge
8     | 0.102                      | Similarity computation adapts
```

**Upper layers (9-12): Maximum change**

These layers undergo the most dramatic transformation. They shift from encoding general
language understanding to computing task-specific similarity.

```
Layer | Avg weight change (L2 norm) | What changes
------|----------------------------|------------------
9     | 0.134                      | Representation geometry shifts
10    | 0.178                      | Embedding space restructures
11    | 0.215                      | Final similarity features
12    | 0.267                      | Output representation (most change)
```

### 9.3 Attention Pattern Shifts

Before fine-tuning, BERT's attention patterns are general-purpose. After fine-tuning for
retrieval, attention shifts to focus on semantically important tokens.

**Example: Query "What causes type 2 diabetes?"**

```
Before fine-tuning (BERT-base, Layer 11, Head 3):
  What    causes    type    2    diabetes    ?
  0.08    0.12      0.15   0.05   0.45     0.15
  ↑ Attention focuses on content words, especially "diabetes"

After fine-tuning for medical retrieval:
  What    causes    type    2    diabetes    ?
  0.05    0.28      0.18   0.12   0.32     0.05
  ↑ Attention redistributes: "causes" gets much more weight
    because the model learns that the QUESTION TYPE matters
    for retrieval (cause vs. symptom vs. treatment)
```

**Attention head specialization**: After fine-tuning, specific attention heads develop
specialized roles:

```
Head | Before fine-tuning          | After fine-tuning
-----|----------------------------|----------------------------------
1    | General syntactic patterns  | Entity recognition
3    | Positional attention        | Question-type detection
7    | Coreference-like patterns   | Semantic matching
11   | Broad context gathering     | Key term identification
```

### 9.4 Embedding Space Geometry Changes

Fine-tuning fundamentally restructures the geometry of the embedding space.

**Before fine-tuning** (pre-trained BERT):

```
Properties:
  - Anisotropic: embeddings cluster in a narrow cone
  - High average cosine similarity between random pairs (~0.6)
  - Semantic structure exists but is entangled with syntax
  - Frequency bias: common words dominate the representation

Visualization (2D projection):
    ·  ·  ·
   · · · · ·
  · · · · · · ·     ← All embeddings in a narrow region
   · · · · ·
    ·  ·  ·
```

**After fine-tuning** (with contrastive loss):

```
Properties:
  - More isotropic: embeddings spread across the hypersphere
  - Low average cosine similarity between random pairs (~0.1)
  - Clear semantic clusters
  - Task-relevant features amplified

Visualization (2D projection):
        · ·                    · ·
       · · ·                  · · ·
                                        ← Clusters separated
  · ·              · ·
 · · ·            · · ·
```

**Quantitative changes:**

```
Metric                          | Before FT | After FT | Change
--------------------------------|-----------|----------|--------
Avg cosine (random pairs)       | 0.58      | 0.12     | -79%
Avg cosine (positive pairs)     | 0.62      | 0.85     | +37%
Avg cosine (negative pairs)     | 0.55      | 0.08     | -85%
Effective dimensionality        | 47        | 189      | +302%
Isotropy score                  | 0.23      | 0.71     | +209%
Cluster separation (silhouette) | 0.12      | 0.67     | +458%
```

### 9.5 Isotropy Improvement

**Isotropy** measures how uniformly the embedding space is utilized. A perfectly isotropic
space uses all dimensions equally; an anisotropic space wastes dimensions.

**Measuring isotropy**: Compute the singular values of the embedding matrix. For N
embeddings of dimension d, form the matrix E ∈ ℝ^{N×d} and compute SVD:

```
E = UΣV^T

Isotropy = min(σ_i) / max(σ_i)
```

where σ_i are the singular values. Isotropy = 1 means perfect uniformity; isotropy → 0
means the space is degenerate.

**Singular value spectrum before and after fine-tuning:**

```
Singular value index:  1     2     3     4     5    ...   50   ...  384
Before fine-tuning:   45.2  38.1  29.7  18.3  12.1 ...  0.8  ...  0.01
After fine-tuning:    12.8  11.9  11.2  10.5   9.8 ...  5.2  ...  2.1
```

Before fine-tuning, the first few singular values dominate — most of the variance is
captured by a handful of directions. After fine-tuning, the singular values are more
uniform, meaning the model uses more dimensions to encode information.

**Why isotropy matters for retrieval:**

```
Anisotropic space (before FT):
  - Most embeddings point in similar directions
  - Cosine similarity has low dynamic range (0.4 to 0.8)
  - Hard to distinguish relevant from irrelevant documents

Isotropic space (after FT):
  - Embeddings spread across all directions
  - Cosine similarity has high dynamic range (-0.2 to 0.95)
  - Clear separation between relevant and irrelevant documents
```

### 9.6 The Representation Bottleneck

During fine-tuning, information flows through a bottleneck at the pooling layer:

```
Token embeddings (seq_len × d):
  [CLS]  [tok1]  [tok2]  ...  [tokN]
    ↓       ↓       ↓           ↓
  h_CLS   h_1     h_2    ...  h_N

         ↓ Pooling (mean or CLS) ↓

Sentence embedding (1 × d):
  h_sentence = mean(h_CLS, h_1, ..., h_N)  or  h_CLS
```

Fine-tuning optimizes what information passes through this bottleneck:

- **Before**: The pooled representation contains general linguistic information
- **After**: The pooled representation is optimized to preserve task-relevant semantics
  and discard irrelevant details (syntax, style, formatting)

This is why fine-tuned embeddings are so much better for specific tasks — the model
learns to compress exactly the right information into the fixed-size vector.

### 9.7 Practical Implications

Understanding layer-by-layer changes has practical implications:

**1. Layer freezing**: Freeze lower layers (1-4) to speed up training and reduce
overfitting. Only fine-tune layers 5-12:

```python
# Freeze first 4 layers
for name, param in model.named_parameters():
    if any(f"layer.{i}." in name for i in range(4)):
        param.requires_grad = False
```

**2. Discriminative learning rates**: Use lower learning rates for lower layers:

```python
# Layer-wise learning rate decay
optimizer_grouped_parameters = []
lr = 2e-5
lr_decay = 0.95

for layer_idx in range(11, -1, -1):
    layer_params = [p for n, p in model.named_parameters()
                    if f"layer.{layer_idx}." in n]
    optimizer_grouped_parameters.append({
        "params": layer_params,
        "lr": lr * (lr_decay ** (11 - layer_idx))
    })
# Layer 11: lr = 2e-5
# Layer 10: lr = 2e-5 × 0.95 = 1.9e-5
# Layer 0:  lr = 2e-5 × 0.95^11 = 1.14e-5
```

**3. Early stopping by layer**: Monitor the change in upper layer weights. If they
stabilize, training is likely complete even if the loss is still decreasing (the model
may be memorizing).

---

## Summary

Fine-tuning transforms general-purpose embeddings into task-specific representations
through careful optimization. The key takeaways:

| Concept | Key Insight |
|---------|-------------|
| MNRL | In-batch negatives give you B-1 negatives for free; batch size is critical |
| Hard negatives | Easy negatives provide no gradient; mine hard negatives with BM25 or cross-encoders |
| Knowledge distillation | Cross-encoder soft labels provide richer signal than binary labels |
| SimCSE | Dropout alone creates effective positive pairs; no labels needed |
| Data augmentation | Dropout > back-translation > cropping for embedding augmentation |
| Layer changes | Upper layers change most; lower layers preserve linguistic knowledge |
| Isotropy | Fine-tuning spreads embeddings across the hypersphere, improving discrimination |
| Practical recipe | lr=2e-5, batch=256+, warmup=10%, epochs=1-3, evaluate often |

The next chapter explores modern loss functions — CoSENT, AnglE, and circle loss — that
build on these foundations with more sophisticated optimization objectives.


---

*Next chapter: [Chapter 9 — Modern Loss Functions: CoSENT, AnglE, and Circle Loss](09-modern-loss-functions.md)*
