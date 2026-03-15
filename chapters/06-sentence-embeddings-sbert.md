# Chapter 6: Sentence Embeddings and Sentence-BERT

## Table of Contents

1. [The Need for Sentence Embeddings](#1-the-need-for-sentence-embeddings)
2. [Pre-SBERT Approaches](#2-pre-sbert-approaches)
3. [Sentence-BERT (2019)](#3-sentence-bert-2019)
4. [Training Objectives for SBERT](#4-training-objectives-for-sbert)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation](#6-evaluation)

---

## 1. The Need for Sentence Embeddings

### 1.1 From Words to Sentences

In previous chapters, we built embeddings for individual words — first with static methods
(Word2Vec, GloVe, FastText), then with contextual models (ELMo, BERT). But many real-world
NLP tasks operate at the **sentence level**:

- **Semantic textual similarity (STS)**: Given two sentences, how similar are they in meaning?
- **Information retrieval / semantic search**: Given a query, find the most relevant documents.
- **Clustering**: Group sentences or paragraphs by topic.
- **Paraphrase detection**: Are two sentences saying the same thing?
- **Duplicate question detection**: Are two questions semantically equivalent?

For all of these, we need a single fixed-size vector that captures the **meaning of an entire
sentence** — not just individual tokens.

### 1.2 Semantic Textual Similarity (STS)

The STS task is the canonical benchmark for sentence embeddings. Given a pair of sentences,
predict a similarity score on a continuous scale (typically 0 to 5):

| Score | Meaning                        | Example                                                    |
|-------|--------------------------------|------------------------------------------------------------|
| 5.0   | Completely equivalent          | "A man is playing guitar" / "Someone plays a guitar"       |
| 3.0   | Roughly equivalent             | "A man is playing a flute" / "A man is playing music"      |
| 0.0   | Completely dissimilar          | "A cat sits on a mat" / "New stock exchange rules"         |

A good sentence embedding maps sentences to vectors such that cosine similarity between
vectors correlates strongly with human-judged similarity scores.

### 1.3 Information Retrieval and Semantic Search

Traditional keyword search (BM25, TF-IDF) matches on surface-level word overlap. Semantic
search retrieves documents based on **meaning**:

```
Query:    "How do I fix a flat tire?"
Match:    "Steps to repair a punctured wheel"    ← No word overlap, semantically similar
No match: "The apartment has flat pricing tiers"  ← Word overlap ("flat"), different meaning
```

For semantic search at scale, we need to encode every document into a vector (offline),
encode the query (online), and find nearest neighbors using ANN search.

This requires sentence embeddings that are both **high-quality** and **fast to compute**.

### 1.4 Why Cross-Encoders Are Too Slow

BERT (and similar transformers) can measure sentence similarity directly using a
**cross-encoder** architecture: concatenate two sentences with a [SEP] token and let
the model attend across both simultaneously:

```
Input:  [CLS] sentence_A [SEP] sentence_B [SEP]
Output: similarity score (from [CLS] token → regression head)
```

Cross-encoders are **accurate** because full cross-attention captures fine-grained
interactions. But they are **catastrophically slow** for retrieval.

**The combinatorial problem** — finding the most similar pair among n sentences:

    Number of pairs = n(n-1)/2 = O(n²)

Concrete example with n = 10,000 sentences:

    Pairs = 10,000 × 9,999 / 2 = 49,995,000 ≈ 50 million

If each BERT inference takes ~5ms on a GPU:

    Total time = 50,000,000 × 0.005s = 250,000s ≈ 2.9 days

For n = 100,000 sentences:

    Pairs = 100,000 × 99,999 / 2 ≈ 5 × 10⁹
    Total time ≈ 2.5 × 10⁷ seconds ≈ 289 days

This is clearly impractical for real-time search over large corpora.

### 1.5 Bi-Encoders: Encode Once, Compare Many Times

The solution is a **bi-encoder** (also called a dual-encoder or two-tower) architecture:

```
┌─────────────┐     ┌─────────────┐
│ Sentence A   │     │ Sentence B   │
└──────┬──────┘     └──────┬──────┘
       │                    │
       ▼                    ▼
┌─────────────┐     ┌─────────────┐
│  Encoder     │     │  Encoder     │
│  (BERT)      │     │  (BERT)      │
└──────┬──────┘     └──────┬──────┘
       │                    │
       ▼                    ▼
┌─────────────┐     ┌─────────────┐
│  Pooling     │     │  Pooling     │
└──────┬──────┘     └──────┬──────┘
       │                    │
       ▼                    ▼
    u ∈ ℝᵈ              v ∈ ℝᵈ
       │                    │
       └────────┬───────────┘
                │
                ▼
        similarity(u, v)
```

**Key insight**: Each sentence is encoded **independently**. The encoders share weights
(Siamese architecture), but there is no cross-attention between sentences.

**Complexity comparison:**

| Operation                  | Cross-Encoder        | Bi-Encoder           |
|----------------------------|----------------------|----------------------|
| Encode n sentences         | N/A (pairs only)     | O(n) BERT calls      |
| Compare all pairs          | O(n²) BERT calls     | O(n²) dot products   |
| Cost per comparison        | ~5ms (full BERT)     | ~0.001ms (dot prod)  |
| Total for n=10,000         | ~2.9 days            | ~50s encode + <1s    |
| Pre-compute embeddings?    | No                   | Yes                  |
| Use ANN index (FAISS)?     | No                   | Yes                  |

The bi-encoder trades a small amount of accuracy for **orders of magnitude** speedup.
With FAISS or ScaNN, finding nearest neighbors among millions of vectors takes milliseconds.

---

## 2. Pre-SBERT Approaches

Before Sentence-BERT, several methods attempted to produce general-purpose sentence
embeddings.

### 2.1 InferSent (Conneau et al., 2017)

**Paper**: *"Supervised Learning of Universal Sentence Representations from Natural Language
Inference Data"* (Conneau et al., 2017)

InferSent was one of the first models to show that **supervised training on NLI data**
produces sentence embeddings that transfer well to other tasks.

**Architecture:**

```
┌──────────────┐     ┌──────────────┐
│  Premise (u)  │     │ Hypothesis(v)│
└──────┬───────┘     └──────┬───────┘
       ▼                     ▼
┌──────────────┐     ┌──────────────┐
│ BiLSTM (shared)│     │ BiLSTM (shared)│
│ + Max Pool   │     │ + Max Pool   │
└──────┬───────┘     └──────┬───────┘
       ▼                     ▼
    u ∈ ℝ⁴⁰⁹⁶            v ∈ ℝ⁴⁰⁹⁶
       └────────┬────────────┘
                ▼
     [u; v; |u-v|; u*v]  (16384-dim)
                ▼
         3-class softmax
```

**Key details:**
- Encoder: BiLSTM with max pooling, embedding dim 4096 (2048 forward + 2048 backward)
- Training data: Stanford NLI (SNLI) — 570K sentence pairs
- The concatenation [u; v; |u-v|; u*v] captures both similarity and difference

**Limitations:** BiLSTM encoder is slower than transformers, uses fixed GloVe embeddings
(no subword information), and is limited to the vocabulary seen during training.

### 2.2 Universal Sentence Encoder (Cer et al., 2018)

**Paper**: *"Universal Sentence Encoder"* (Cer et al., 2018, Google)

Google released two variants: a 6-layer Transformer (higher quality, slower) and a Deep
Averaging Network (DAN, faster, lower quality).

**DAN variant** — remarkably simple:

    Step 1: Look up word/bigram embeddings
    Step 2: Average them:  h = (1/n) Σᵢ eᵢ
    Step 3: Pass through feedforward layers:  s = FFN(h)

**Training**: Multi-task learning on skip-thought-like prediction, NLI classification,
and conversational response prediction. Produces 512-dimensional embeddings.

**Limitations:** Transformer variant uses cross-attention (slow for retrieval), DAN loses
word order, not based on large-scale pre-trained LMs like BERT.

### 2.3 Skip-Thought Vectors (Kiros et al., 2015)

**Paper**: *"Skip-Thought Vectors"* (Kiros et al., 2015)

Inspired by Skip-gram (but at the sentence level), Skip-Thought trains an encoder-decoder
to predict surrounding sentences given the current one:

```
    Sentence (i-1)          Sentence (i)          Sentence (i+1)
         ▲                      │                      ▲
         │                      ▼                      │
    ┌─────────┐          ┌─────────────┐          ┌─────────┐
    │ Decoder  │◄─────── │   Encoder    │────────►│ Decoder  │
    │ (prev)   │         │   (GRU)      │         │ (next)   │
    └─────────┘          └─────────────┘          └─────────┘
                               │
                               ▼
                         s ∈ ℝ²⁴⁰⁰
```

**Limitations:** Requires ordered text (books, articles), GRU encoder is slow compared
to transformers, and embedding quality lags behind supervised approaches.

### 2.4 The Gap Before SBERT

By 2018, BERT achieved state-of-the-art results on virtually every NLP benchmark, but
naive BERT sentence embeddings are surprisingly bad:

| Method                          | STS Benchmark (Spearman ρ) |
|---------------------------------|----------------------------|
| GloVe average                   | 58.02                      |
| InferSent (BiLSTM)              | 68.03                      |
| USE (Transformer)               | 74.92                      |
| BERT [CLS] token                | 20.29                      |
| BERT mean of all tokens         | 46.35                      |
| SBERT (mean pooling)            | **84.67**                  |

The BERT [CLS] token produces **terrible** sentence embeddings — worse than GloVe
averaging! This happens because BERT's [CLS] was pre-trained for next-sentence prediction
(binary), token representations are optimized for MLM, and the representation space is
**anisotropic** (embeddings cluster in a narrow cone).

This gap — powerful BERT representations that fail as sentence embeddings — is exactly
what Sentence-BERT was designed to close.

---

## 3. Sentence-BERT (2019)

### 3.1 The Paper

**"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"**
Nils Reimers and Iryna Gurevych (2019), UKP Lab, TU Darmstadt

The core idea: **fine-tune BERT in a Siamese/triplet network architecture** to produce
sentence embeddings that are semantically meaningful and can be compared using cosine
similarity.

Key contributions:
- Fine-tuning BERT with a Siamese objective improves sentence embeddings from ~46 to ~85
  Spearman ρ on STS
- Reduced time to find the most similar pair among 10,000 sentences from 65 hours to ~5s
- Released the `sentence-transformers` Python library

### 3.2 Siamese Network Architecture

A Siamese network consists of two (or more) identical sub-networks that share the same
weights. In SBERT, both sub-networks are copies of the same BERT model:

```
          Sentence A                              Sentence B
    "A man plays guitar"                    "Someone plays music"
              │                                       │
              ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │       BERT        │ ◄── shared ──►    │       BERT        │
    │   (12 layers)     │     weights       │   (12 layers)     │
    └─────────┬─────────┘                   └─────────┬─────────┘
              │                                       │
              ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │  Token embeddings  │                   │  Token embeddings  │
    │  h₁, h₂, ..., hₘ  │                   │  h₁, h₂, ..., hₖ  │
    └─────────┬─────────┘                   └─────────┬─────────┘
              │                                       │
              ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │  Pooling Strategy  │                   │  Pooling Strategy  │
    └─────────┬─────────┘                   └─────────┬─────────┘
              │                                       │
              ▼                                       ▼
         u ∈ ℝ⁷⁶⁸                               v ∈ ℝ⁷⁶⁸
              │                                       │
              └──────────────┬────────────────────────┘
                             │
                             ▼
                    Training Objective
```

**"Shared weights"** means both BERT copies are the same model — gradients from both
branches are accumulated and applied to a single set of parameters.

### 3.3 Pooling Strategies

BERT outputs a sequence of token embeddings h₁, h₂, ..., hₙ (each ∈ ℝ⁷⁶⁸ for BERT-base).
We need to collapse these into a single sentence vector e ∈ ℝ⁷⁶⁸. SBERT investigates
three pooling strategies:

#### Strategy 1: [CLS] Token Pooling

Use the output embedding of the special [CLS] token as the sentence representation:

    e = h_[CLS] = h₁

This is the simplest approach. However, as we saw in Section 2.4, raw BERT [CLS]
embeddings perform poorly for similarity tasks. After SBERT fine-tuning, [CLS] pooling
improves but still underperforms mean pooling.

#### Strategy 2: Mean Pooling (Best)

Average all token embeddings (excluding padding tokens):

                    1   n
    e = mean(H) = ─── · Σ  hᵢ
                    n  i=1

where n is the number of non-padding tokens and hᵢ is the BERT output for token i.

**Step-by-step numerical example:**

Consider a short sentence with 4 tokens (using 4-dimensional embeddings for clarity):

    Token 1 ("A"):       h₁ = [0.2,  0.8, -0.1,  0.5]
    Token 2 ("man"):     h₂ = [0.6,  0.3,  0.7,  0.2]
    Token 3 ("plays"):   h₃ = [0.1,  0.5,  0.4,  0.9]
    Token 4 ("guitar"):  h₄ = [0.8,  0.1,  0.3,  0.6]

    Mean pooling:
    e₁ = (0.2 + 0.6 + 0.1 + 0.8) / 4 = 1.7 / 4 = 0.425
    e₂ = (0.8 + 0.3 + 0.5 + 0.1) / 4 = 1.7 / 4 = 0.425
    e₃ = (-0.1 + 0.7 + 0.4 + 0.3) / 4 = 1.3 / 4 = 0.325
    e₄ = (0.5 + 0.2 + 0.9 + 0.6) / 4 = 2.2 / 4 = 0.550

    e = [0.425, 0.425, 0.325, 0.550]

**With attention mask** (to handle padding in batched inputs):

                  Σᵢ (maskᵢ · hᵢ)
    e = ─────────────────────────
                   Σᵢ maskᵢ

where maskᵢ = 1 for real tokens and maskᵢ = 0 for padding tokens.

#### Strategy 3: Max Pooling

Take the element-wise maximum across all token embeddings:

    eⱼ = max(h₁ⱼ, h₂ⱼ, ..., hₙⱼ)    for each dimension j

Or equivalently:

    eⱼ = maxᵢ(hᵢⱼ)    for j = 1, 2, ..., d

**Step-by-step numerical example** (same tokens as above):

    Token 1 ("A"):       h₁ = [0.2,  0.8, -0.1,  0.5]
    Token 2 ("man"):     h₂ = [0.6,  0.3,  0.7,  0.2]
    Token 3 ("plays"):   h₃ = [0.1,  0.5,  0.4,  0.9]
    Token 4 ("guitar"):  h₄ = [0.8,  0.1,  0.3,  0.6]

    Max pooling:
    e₁ = max(0.2, 0.6, 0.1, 0.8) = 0.8
    e₂ = max(0.8, 0.3, 0.5, 0.1) = 0.8
    e₃ = max(-0.1, 0.7, 0.4, 0.3) = 0.7
    e₄ = max(0.5, 0.2, 0.9, 0.6) = 0.9

    e = [0.8, 0.8, 0.7, 0.9]

Max pooling captures the **most activated feature** for each dimension across all tokens.

### 3.4 Why Mean Pooling Works Best

Reimers & Gurevych found that mean pooling consistently outperforms both [CLS] and max
pooling across tasks:

| Pooling Strategy | STS-B (Spearman ρ) | NLI Transfer Accuracy |
|------------------|--------------------|-----------------------|
| [CLS] token      | 80.78              | 79.19                 |
| Max pooling       | 82.35              | 80.44                 |
| Mean pooling      | **84.67**          | **80.78**             |

**Why does mean pooling win?**

1. **Information preservation**: Averaging retains contributions from all tokens, while
   [CLS] compresses everything into one token and max pooling only keeps extremes.

2. **Smooth gradients**: The gradient distributes evenly to all tokens (∂e/∂hᵢ = 1/n · I),
   so every token receives gradient signal during training.

3. **Alignment with pre-training**: BERT's MLM objective trains every token position to
   carry contextual information. Mean pooling leverages all positions.

---

## 4. Training Objectives for SBERT

SBERT uses different training objectives depending on the available labeled data.

### 4.1 Classification Objective (NLI Data)

**When to use**: Sentence pairs with categorical labels (e.g., NLI: entailment, neutral,
contradiction).

**Architecture for classification:**

```
     u ∈ ℝ⁷⁶⁸          v ∈ ℝ⁷⁶⁸
          │                   │
          │      |u - v|      │
          │      ∈ ℝ⁷⁶⁸      │
          ▼        ▼          ▼
     ┌──────────────────────────────┐
     │  Concatenation: [u; v; |u-v|] │
     │       ∈ ℝ²³⁰⁴ (3 × 768)     │
     └──────────────┬───────────────┘
                    ▼
     ┌──────────────────────────────┐
     │  Linear: W ∈ ℝ³ˣ²³⁰⁴ + b    │
     └──────────────┬───────────────┘
                    ▼
              Softmax → P(y|u,v) ∈ ℝ³
```

**Step 1: Compute the concatenated feature vector**

Given sentence embeddings u and v:

    |u - v| = element-wise absolute difference
    feature = [u; v; |u - v|] ∈ ℝ³ᵈ    (3 × 768 = 2304 dimensions)

**Why include |u - v|?** The absolute difference encodes how the two sentences differ.
Combined with u and v, the classifier sees what each sentence says and how they differ.

**Step 2: Linear transformation**

    z = W · feature + b

where W ∈ ℝᵏˣ³ᵈ and b ∈ ℝᵏ, with k = 3 (number of NLI classes).

**Step 3: Softmax**

    P(y = c | u, v) = softmax(z)_c = exp(z_c) / Σⱼ exp(z_j)

for c ∈ {entailment, neutral, contradiction}.

**Step 4: Cross-entropy loss**

For a single training example with true label c*:

    L = -log P(y = c* | u, v)
      = -log [ exp(z_c*) / Σⱼ exp(z_j) ]
      = -z_c* + log Σⱼ exp(z_j)

**Numerical example:**

Suppose we have 4-dimensional sentence embeddings (for clarity):

    u = [0.5, 0.3, -0.2, 0.8]     (sentence A embedding)
    v = [0.1, 0.7,  0.4, 0.6]     (sentence B embedding)

Step 1: Compute features
    |u - v| = [|0.5-0.1|, |0.3-0.7|, |-0.2-0.4|, |0.8-0.6|]
            = [0.4, 0.4, 0.6, 0.2]

    feature = [u; v; |u-v|]
            = [0.5, 0.3, -0.2, 0.8, 0.1, 0.7, 0.4, 0.6, 0.4, 0.4, 0.6, 0.2]
              (12-dimensional vector: 3 × 4 = 12)

Step 2: Linear transformation (using small example weights)
    Suppose W (3×12) and b (3×1) give us:
    z = [2.1, 0.5, -0.8]

Step 3: Softmax
    exp(z) = [8.166, 1.649, 0.449],  sum = 10.264

    P(entailment) = 0.796,  P(neutral) = 0.161,  P(contradiction) = 0.044

Step 4: Cross-entropy loss (true label = entailment)
    L = -log(0.796) = 0.229

If the true label were contradiction:
    L = -log(0.044) = 3.130    (much higher — model is wrong)

### 4.2 Regression Objective (STS Data)

**When to use**: Sentence pairs with continuous similarity scores (e.g., STS Benchmark,
scores 0–5).

**Architecture for regression:**

```
     u ∈ ℝ⁷⁶⁸              v ∈ ℝ⁷⁶⁸
          └───────┬───────────┘
                  ▼
          cosine_similarity(u, v) → ŷ ∈ [-1, 1]
                  ▼
          MSE Loss: L = (ŷ - y)²
```

**Step 1: Cosine similarity**

                        u · v           Σⱼ uⱼvⱼ
    cos(u, v) = ─────────────── = ───────────────────────
                 ‖u‖ · ‖v‖      √(Σⱼ uⱼ²) · √(Σⱼ vⱼ²)

**Step 2: Mean Squared Error loss**

    L = (cos(u, v) - y)²

where y is the normalized ground-truth similarity score (scaled to [-1, 1] or [0, 1]).

**Numerical example:**

    u = [0.5, 0.3, -0.2, 0.8]
    v = [0.1, 0.7,  0.4, 0.6]

Step 1: u · v = (0.5)(0.1) + (0.3)(0.7) + (-0.2)(0.4) + (0.8)(0.6) = 0.66

Step 2: ‖u‖ = √1.02 = 1.010,  ‖v‖ = √1.02 = 1.010

Step 3: cos(u, v) = 0.66 / (1.010 × 1.010) = 0.66 / 1.020 = 0.647

Step 4: MSE loss (ground truth y = 0.8)
    L = (0.647 - 0.8)² = (-0.153)² = 0.0234

The loss is small but nonzero — the model predicts 0.647 similarity but ground truth
is 0.8. The gradient will push the embeddings to be more aligned.

**Gradient of the regression loss with respect to u:**

    ∂L/∂u = 2(cos(u,v) - y) · ∂cos/∂u

    ∂cos     1        v      cos(u,v)
    ───── = ────── · (─── - ────────── · u)
     ∂u    ‖u‖      ‖v‖      ‖u‖²

This pushes u toward v when cos < y and away when cos > y.

### 4.3 Triplet Objective

**When to use**: Training data consists of (anchor, positive, negative) triplets.

**Architecture for triplet loss:**

```
    Anchor (a)          Positive (p)         Negative (n)
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐          ┌─────────┐
    │  BERT    │         │  BERT    │          │  BERT    │
    │ (shared) │         │ (shared) │          │ (shared) │
    └────┬────┘         └────┬────┘          └────┬────┘
         ▼                    ▼                    ▼
      a ∈ ℝᵈ              p ∈ ℝᵈ              n ∈ ℝᵈ
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
               L = max(‖a-p‖ - ‖a-n‖ + ε, 0)
```

**Triplet loss formula:**

    L = max(‖a - p‖ - ‖a - n‖ + ε, 0)

where a = anchor, p = positive, n = negative, ε = margin (typically 1.0), and ‖·‖ is
Euclidean distance.

**Intuition**: The loss is zero when the negative is sufficiently farther than the
positive (by at least margin ε). Otherwise, the loss penalizes proportionally.

**Numerical example:**

    a = [0.5, 0.3, -0.2, 0.8]    (anchor: "A dog runs in the park")
    p = [0.4, 0.4, -0.1, 0.7]    (positive: "A puppy is running outside")
    n = [0.9, -0.5, 0.6, 0.1]    (negative: "The stock market crashed")
    ε = 1.0                        (margin)

Step 1: Compute ‖a - p‖ (anchor-positive distance)
    a - p = [0.1, -0.1, -0.1, 0.1]
    ‖a - p‖ = √(0.01 + 0.01 + 0.01 + 0.01) = √0.04 = 0.2

Step 2: Compute ‖a - n‖ (anchor-negative distance)
    a - n = [-0.4, 0.8, -0.8, 0.7]
    ‖a - n‖ = √(0.16 + 0.64 + 0.64 + 0.49) = √1.93 = 1.3892

Step 3: Compute triplet loss
    L = max(‖a-p‖ - ‖a-n‖ + ε, 0)
      = max(0.2 - 1.3892 + 1.0, 0)
      = max(-0.1892, 0)
      = 0

The loss is zero! The negative is far enough from the anchor (1.389 vs. 0.2 for the
positive), with a gap of 1.189 > ε = 1.0.

**Example where loss is nonzero:**

    a = [0.5, 0.3, -0.2, 0.8]
    p = [0.1, 0.7,  0.4, 0.6]    (positive, but somewhat distant)
    n = [0.6, 0.2, -0.1, 0.7]    (negative, but close to anchor!)
    ε = 1.0

Step 1: ‖a - p‖
    a - p = [0.4, -0.4, -0.6, 0.2]
    ‖a - p‖ = √(0.16 + 0.16 + 0.36 + 0.04) = √0.72 = 0.8485

Step 2: ‖a - n‖
    a - n = [-0.1, 0.1, -0.1, 0.1]
    ‖a - n‖ = √(0.01 + 0.01 + 0.01 + 0.01) = √0.04 = 0.2

Step 3: Triplet loss
    L = max(0.8485 - 0.2 + 1.0, 0)
      = max(1.6485, 0)
      = 1.6485

The loss is large! The negative (0.2 away) is closer to the anchor than the positive
(0.849 away). The model must push the negative farther and pull the positive closer.

**Gradient derivation for the triplet loss:**

When L > 0 (the constraint is violated), the gradients are:

    ∂L       ∂‖a - p‖     (a - p)
    ── = + ─────────── = ─────────
    ∂a         ∂a         ‖a - p‖

              ∂‖a - n‖     (a - n)
           - ─────────── = ─────────
                ∂a         ‖a - n‖

So:
              (a - p)     (a - n)
    ∂L/∂a = ───────── - ─────────
             ‖a - p‖    ‖a - n‖

This gradient has two components:
- **(a - p) / ‖a - p‖**: Pushes a toward p (reducing anchor-positive distance)
- **-(a - n) / ‖a - n‖**: Pushes a away from n (increasing anchor-negative distance)

For the positive and negative:

    ∂L/∂p = (p - a) / ‖a - p‖     (pushes p toward a)
    ∂L/∂n = (a - n) / ‖a - n‖     (pushes n away from a)

When L = 0, all gradients are zero.

**Continuing the numerical example (L = 1.6485):**

    ∂L/∂a = (a-p)/‖a-p‖ - (a-n)/‖a-n‖

    (a-p)/‖a-p‖ = [0.4, -0.4, -0.6, 0.2] / 0.8485 = [0.471, -0.471, -0.707, 0.236]
    (a-n)/‖a-n‖ = [-0.1, 0.1, -0.1, 0.1] / 0.2     = [-0.5, 0.5, -0.5, 0.5]

    ∂L/∂a = [0.971, -0.971, -0.207, -0.264]

With learning rate η = 0.01:
    a_new = a - η · ∂L/∂a
          = [0.5, 0.3, -0.2, 0.8] - 0.01 × [0.971, -0.971, -0.207, -0.264]
          = [0.490, 0.310, -0.198, 0.803]

The anchor has moved slightly toward the positive and away from the negative.

### 4.4 Comparison of Training Objectives

| Objective       | Data Format              | Loss Function                    | Best For                    |
|-----------------|--------------------------|----------------------------------|-----------------------------|
| Classification  | (s₁, s₂, label)         | Cross-entropy on softmax         | NLI data (3 classes)        |
| Regression      | (s₁, s₂, score)         | MSE on cosine similarity         | STS data (continuous)       |
| Triplet         | (anchor, pos, neg)       | max(‖a-p‖ - ‖a-n‖ + ε, 0)      | Retrieval, ranking          |

In practice, the classification objective on NLI data provides the best general-purpose
sentence embeddings: NLI data is abundant (~1M pairs), the three-way classification
forces fine-grained semantic distinctions, and the |u - v| term explicitly teaches
about semantic differences.

---

## 5. Training Pipeline

### 5.1 Two-Stage Training: NLI Pre-training → STS Fine-tuning

SBERT's best results come from a two-stage training process:

```
Stage 1: NLI Pre-training                    Stage 2: STS Fine-tuning
┌─────────────────────────────┐              ┌─────────────────────────────┐
│  Data: SNLI + MultiNLI      │              │  Data: STS Benchmark        │
│  (~1M pairs, 3 classes)     │    ────►     │  (~8.6K pairs, continuous)  │
│  Objective: Classification   │              │  Objective: Regression      │
│  Epochs: 1, Batch: 16      │              │  Epochs: 4, Batch: 16      │
│  LR: 2e-5, Warmup: 10%     │              │  LR: 2e-5, Warmup: 10%     │
└─────────────────────────────┘              └─────────────────────────────┘
```

**Why two stages?**

Stage 1 (NLI) teaches broad semantic understanding from a large, diverse dataset. The
classification objective with [u; v; |u-v|] forces the model to learn what makes sentences
similar vs. different. Stage 2 (STS) fine-tunes specifically for similarity scoring,
directly optimizing cosine similarity to match human judgments.

**Training details from the paper:**

| Hyperparameter     | Value                                    |
|--------------------|------------------------------------------|
| Base model         | BERT-base-uncased (110M params)          |
| Pooling            | Mean pooling                             |
| Optimizer          | Adam with weight decay                   |
| Learning rate      | 2 × 10⁻⁵                                |
| Max sequence length| 128 tokens                               |

### 5.2 Data Requirements and Augmentation

**NLI datasets:**

| Dataset   | Size      | Labels                          |
|-----------|-----------|----------------------------------|
| SNLI      | 570K      | Entailment, Neutral, Contradict |
| MultiNLI  | 433K      | Entailment, Neutral, Contradict |

**STS Benchmark:** 5,749 training / 1,500 dev / 1,379 test pairs, scored 0.0–5.0.

**Data augmentation strategies** (used in later work, not the original SBERT paper)
include back-translation, synonym replacement, and dropout-as-augmentation (SimCSE).

### 5.3 Batch Size Effects

Batch size has a moderate impact on SBERT's classification and regression objectives:

| Batch Size | STS-B (Spearman ρ) | Notes                        |
|------------|--------------------|-----------------------------|
| 8          | 83.91              | Slightly underfitting        |
| 16         | 84.67              | Default (best)               |
| 32         | 84.42              | Slightly worse               |
| 64         | 84.15              | Diminishing returns          |

Batch size 16 works well because each example is an independent (premise, hypothesis,
label) triple — there is no in-batch negative mining. For the triplet objective, larger
batches can help by providing more hard negatives.

---

## 6. Evaluation

### 6.1 STS Benchmark

The **Semantic Textual Similarity Benchmark (STS-B)** is the primary evaluation metric
for sentence embeddings. It consists of sentence pairs drawn from three domains:

- **News headlines** (from MSR Paraphrase Corpus)
- **Image captions** (from Flickr and PASCAL datasets)
- **User forum posts** (from SMT and OntoNotes)

Each pair is annotated with a similarity score from 0.0 to 5.0, averaged over multiple
human annotators.

**Evaluation protocol:**
1. Encode all sentences (no fine-tuning on test data).
2. Compute cosine similarity between each pair's embeddings.
3. Compute Spearman rank correlation between predicted and gold scores.

### 6.2 Spearman Rank Correlation

Spearman's ρ measures the monotonic relationship between two ranked variables. Unlike
Pearson's r, it does not assume linearity.

**Computation:**

Step 1: Rank both the predicted similarities and gold scores.
Step 2: Compute:

                    6 Σᵢ dᵢ²
    ρ = 1 - ─────────────────
              n(n² - 1)

where dᵢ = rank(ŷᵢ) - rank(yᵢ) is the difference in ranks for pair i, and n is the
number of pairs.

**Numerical example** (5 sentence pairs):

| Pair | Gold Score (y) | Predicted cos(u,v) | Gold Rank | Pred Rank | d    | d²   |
|------|----------------|---------------------|-----------|-----------|------|------|
| 1    | 4.2            | 0.85                | 5         | 5         | 0    | 0    |
| 2    | 1.0            | 0.22                | 2         | 1         | 1    | 1    |
| 3    | 3.5            | 0.71                | 4         | 4         | 0    | 0    |
| 4    | 0.5            | 0.35                | 1         | 2         | -1   | 1    |
| 5    | 2.8            | 0.58                | 3         | 3         | 0    | 0    |

    Σ dᵢ² = 0 + 1 + 0 + 1 + 0 = 2

              6 × 2         12
    ρ = 1 - ─────── = 1 - ──── = 1 - 0.1 = 0.90
             5(25-1)       120

A Spearman ρ of 0.90 indicates strong monotonic agreement. Perfect agreement gives ρ = 1.0.

### 6.3 SBERT Results: How SBERT Improved Over BERT

The following table shows SBERT's improvement over naive BERT and prior methods on the
STS Benchmark (test set, Spearman ρ × 100):

| Model                                  | STS-B (ρ × 100) | Notes                              |
|----------------------------------------|------------------|------------------------------------|
| Avg. GloVe embeddings                  | 58.02            | Simple baseline                    |
| Avg. BERT embeddings (no fine-tuning)  | 46.35            | Worse than GloVe!                  |
| BERT [CLS] (no fine-tuning)            | 20.29            | Terrible                           |
| InferSent (Conneau et al., 2017)       | 68.03            | BiLSTM + NLI                       |
| USE (Cer et al., 2018)                 | 74.92            | Transformer + multi-task           |
| SBERT-NLI (base)                       | 77.03            | NLI only                           |
| SBERT-NLI (large)                      | 79.23            | NLI only, BERT-large               |
| **SBERT-NLI-STS (base)**              | **84.67**        | **NLI + STS fine-tuning**          |
| **SBERT-NLI-STS (large)**             | **85.29**        | **NLI + STS, BERT-large**          |
| BERT cross-encoder (upper bound)       | 87.13            | Full cross-attention (slow)        |

**Key observations:**

1. **SBERT improved over avg. BERT by ~38 points** (46.35 → 84.67) — fine-tuning with a
   Siamese objective transforms BERT from a poor sentence encoder into an excellent one.

2. **SBERT improved over the best prior method (USE) by ~10 points** (74.92 → 84.67).

3. **SBERT approaches cross-encoder quality** (84.67 vs. 87.13) while being orders of
   magnitude faster.

4. **Two-stage training matters**: NLI-only scores 77.03; adding STS pushes to 84.67.

### 6.4 Speed Comparison

The practical impact is best illustrated by speed benchmarks:

| n (sentences) | BERT Cross-Encoder | SBERT              | Speedup     |
|---------------|--------------------|--------------------|-------------|
| 1,000         | ~42 minutes        | ~2.5 seconds       | ~1,000×     |
| 10,000        | ~65 hours          | ~25 seconds        | ~9,400×     |
| 1,000,000     | ~75 years          | ~40 minutes        | ~1,000,000× |

With approximate nearest neighbor indexing (FAISS), even the 1M case drops to seconds.

### 6.5 The Legacy of SBERT

SBERT established the paradigm that dominates modern sentence embeddings: start with a
pre-trained transformer, fine-tune with a Siamese architecture on paired data, use mean
pooling, and evaluate with cosine similarity. This recipe has been extended by SimCSE
(2021, contrastive learning with dropout), E5 (2022, instruction-tuned embeddings),
GTE (2023, multi-stage training), and many others — all building on SBERT's foundation.

---

## Summary

This chapter traced the journey from the need for sentence-level representations to
Sentence-BERT.

**Key takeaways:**

| Concept                    | Key Insight                                              |
|----------------------------|----------------------------------------------------------|
| Cross-encoders             | Accurate but O(n²) — impractical for retrieval           |
| Bi-encoders                | Encode once, compare with dot products — O(n) encoding   |
| Naive BERT embeddings      | Surprisingly poor (worse than GloVe averaging)           |
| SBERT architecture         | Siamese BERT + mean pooling                              |
| Best training objective    | Classification on NLI data with [u; v; \|u-v\|]         |
| Two-stage training         | NLI pre-training → STS fine-tuning                       |
| Result                     | ~38-point improvement over naive BERT on STS-B           |

In the next chapter, we will explore how contrastive learning methods further improved
sentence embeddings and how modern embedding models scale to billions of training pairs.

---

## References

1. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks*. EMNLP 2019.

2. Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). *Supervised
   Learning of Universal Sentence Representations from Natural Language Inference Data*.
   EMNLP 2017.

3. Cer, D., Yang, Y., Kong, S., Hua, N., Limtiaco, N., St. John, R., ... & Kurzweil, R.
   (2018). *Universal Sentence Encoder*. arXiv:1803.11175.

4. Kiros, R., Zhu, Y., Salakhutdinov, R., Zemel, R., Urtasun, R., Torralba, A., & Fidler,
   S. (2015). *Skip-Thought Vectors*. NeurIPS 2015.

5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep
   Bidirectional Transformers for Language Understanding*. NAACL 2019.

6. Agirre, E., Cer, D., Diab, M., Gonzalez-Agirre, A., & Guo, W. (2012–2017). *SemEval
   Semantic Textual Similarity Shared Tasks*. SemEval 2012–2017.

7. Gao, T., Yao, X., & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence
   Embeddings*. EMNLP 2021.


---

*Next chapter: [Chapter 7 — Training Objectives and Loss Functions](07-training-objectives-and-loss-functions.md)*
