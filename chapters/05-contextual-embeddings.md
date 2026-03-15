# Chapter 5: Contextual Embeddings — When Words Learn Their Surroundings

## Table of Contents

1. [The Problem with Static Embeddings](#1-the-problem-with-static-embeddings)
2. [ELMo (2018)](#2-elmo-2018)
3. [BERT Embeddings (2018)](#3-bert-embeddings-2018)
4. [GPT-style Embeddings](#4-gpt-style-embeddings)
5. [The Anisotropy Problem](#5-the-anisotropy-problem)
6. [From Token Embeddings to Sentence Embeddings](#6-from-token-embeddings-to-sentence-embeddings)

---

## 1. The Problem with Static Embeddings

In Chapters 2–4 we built powerful embedding methods — Word2Vec, GloVe, FastText — that
assign a single, fixed vector to every word in the vocabulary. These *static* embeddings
revolutionized NLP, but they carry a fundamental flaw: **they cannot distinguish between
different meanings of the same word**.

### 1.1 Polysemy: One Word, Many Meanings

Consider the word **"bank"**:

```
Sentence A:  "I deposited money at the bank."
                                        ^^^^
                                   financial institution

Sentence B:  "We sat on the bank of the river."
                            ^^^^
                        sloping land beside water
```

In Word2Vec or GloVe, both sentences map "bank" to the **exact same vector**:

```
  Static Embedding Space
  ──────────────────────────────────────────────
  
       "bank" ──► [0.23, -0.41, 0.87, ...]   ← ONE vector for ALL meanings
  
       "money"  ──► [0.19, -0.38, 0.91, ...]   (close to "bank")
       "river"  ──► [-0.52, 0.73, 0.11, ...]   (also somewhat close)
  
  The single "bank" vector is a COMPROMISE — an average
  of all its senses, optimal for none.
  ──────────────────────────────────────────────
```

This is not a rare edge case. Polysemy is pervasive in natural language:

| Word     | Meaning 1          | Meaning 2            | Meaning 3         |
|----------|--------------------|-----------------------|--------------------|
| bank     | financial inst.    | river edge            | to bank (tilt)     |
| bat      | flying mammal      | sports equipment      | —                  |
| cell     | biological unit    | prison room           | phone (cell phone) |
| spring   | season             | water source          | coiled metal       |
| crane    | bird               | construction machine  | to crane (stretch) |

### 1.2 One Vector Per Word Is Fundamentally Limiting

The problem goes deeper than polysemy. Even words with a single dictionary meaning
shift their emphasis depending on context:

```
  "The movie was not good."        ← "good" is negated
  "The movie was really good."     ← "good" is amplified
  "Good morning!"                  ← "good" is a greeting token
```

Static embeddings encode **type-level** information (what a word generally means),
but language operates at the **token-level** (what a word means *right here, right now*).

```
  ┌─────────────────────────────────────────────────────────┐
  │  STATIC EMBEDDINGS          CONTEXTUAL EMBEDDINGS       │
  │                                                         │
  │  word type → vector         word token → vector         │
  │                              (depends on context)       │
  │                                                         │
  │  "bank" → v₁                "bank" in sent A → v_A     │
  │                             "bank" in sent B → v_B     │
  │                                                         │
  │  v₁ = v₁  (always)         v_A ≠ v_B  (different!)    │
  └─────────────────────────────────────────────────────────┘
```

The insight that launched the contextual embedding revolution:

> **Instead of learning a fixed vector for each word, learn a function that
> produces a vector for each word *given its surrounding context*.**

---

## 2. ELMo (2018)

**Paper:** Peters et al., "Deep contextualized word representations" (NAACL 2018)

ELMo (Embeddings from Language Models) was the first major model to produce
context-dependent word representations by leveraging deep bidirectional language models.

### 2.1 Core Idea

ELMo runs a multi-layer bidirectional LSTM over the input sentence. Each layer of the
LSTM produces a different representation of each token. The final ELMo embedding is a
**learned weighted combination** of all layer representations.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    ELMo Architecture                         │
  │                                                              │
  │  Layer 2 (biLSTM):  h₁²  ──  h₂²  ──  h₃²  ──  h₄²       │
  │                      ↑↓       ↑↓       ↑↓       ↑↓         │
  │  Layer 1 (biLSTM):  h₁¹  ──  h₂¹  ──  h₃¹  ──  h₄¹       │
  │                      ↑↓       ↑↓       ↑↓       ↑↓         │
  │  Layer 0 (char CNN): h₁⁰  ──  h₂⁰  ──  h₃⁰  ──  h₄⁰     │
  │                      ↑        ↑        ↑        ↑          │
  │  Input tokens:      "The"   "river"  "bank"   "flooded"    │
  │                                                              │
  │  ELMo_k = γ (s₀·h_k⁰ + s₁·h_k¹ + s₂·h_k²)               │
  └──────────────────────────────────────────────────────────────┘
```

### 2.2 The Bidirectional Language Model

ELMo is trained on a **bidirectional language model** objective. Given a sequence of
N tokens (t₁, t₂, ..., t_N), we train two separate LSTMs:

**Forward LM** — predicts each token from left-to-right context:

```
  P(t_k | t₁, t₂, ..., t_{k-1})
  
  "The"  →  predict "river"
  "The river"  →  predict "bank"
  "The river bank"  →  predict "flooded"
```

**Backward LM** — predicts each token from right-to-left context:

```
  P(t_k | t_{k+1}, t_{k+2}, ..., t_N)
  
  "flooded"  →  predict "bank"
  "bank flooded"  →  predict "river"
  "river bank flooded"  →  predict "The"
```

### 2.3 Combined Training Objective

The biLM maximizes the joint log-likelihood of both directions:

```
  L = Σ_{k=1}^{N} [ log P_fwd(t_k | t₁, ..., t_{k-1}; Θ_x, Θ_fwd, Θ_s)
                   + log P_bwd(t_k | t_{k+1}, ..., t_N; Θ_x, Θ_bwd, Θ_s) ]
```

Where:
- **Θ_x** — parameters of the token embedding layer (shared between directions)
- **Θ_fwd** — parameters of the forward LSTM
- **Θ_bwd** — parameters of the backward LSTM
- **Θ_s** — parameters of the softmax output layer (shared)

Key design choice: the forward and backward LSTMs have **separate parameters**
(Θ_fwd ≠ Θ_bwd), but they **share** the token embedding layer and the softmax layer.

### 2.4 Layer Representations

For each token position k, the biLM computes a set of 2L + 1 representations
(where L = number of LSTM layers, typically L = 2):

```
  R_k = { h_{k,j}^{LM} | j = 0, 1, ..., L }

  where:
    h_{k,0}^{LM} = x_k                          (character-level CNN embedding)
    h_{k,j}^{LM} = [h_{k,j}^{fwd} ; h_{k,j}^{bwd}]   for j = 1, ..., L
```

Each layer captures different linguistic properties:

```
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 0 (char CNN):   Morphological / surface features │
  │                        "un-" prefix, "-ing" suffix      │
  │                                                         │
  │  Layer 1 (biLSTM 1):  Syntactic features                │
  │                        POS tags, dependency structure    │
  │                                                         │
  │  Layer 2 (biLSTM 2):  Semantic features                 │
  │                        Word sense, semantic role         │
  └─────────────────────────────────────────────────────────┘
```

This was a key finding of the ELMo paper: **different layers encode different
types of linguistic information**, and downstream tasks benefit from different
layer combinations.

### 2.5 The ELMo Representation Formula

The final ELMo vector for token k in a downstream task is:

```
  ELMo_k^{task} = γ^{task} · Σ_{j=0}^{L} s_j^{task} · h_{k,j}^{LM}
```

Where:
- **s_j^{task}** — softmax-normalized weights (task-specific):

```
  s_j^{task} = exp(a_j) / Σ_{j'=0}^{L} exp(a_{j'})
  
  where a_j are learnable scalar parameters
  and Σ_j s_j = 1  (they form a proper distribution)
```

- **γ^{task}** — a scalar scaling factor (task-specific)

The weights s_j and scalar γ are the **only parameters learned during fine-tuning**.
The biLM itself is frozen.

### 2.6 Step-by-Step Example: Computing ELMo Vectors

Let's trace through a concrete example with the sentence:

```
  "The bank issued a statement"
```

**Step 1: Character-level CNN (Layer 0)**

Each word is processed through a character-level CNN to produce h⁰:

```
  "The"       → char CNN → h₁⁰ = [0.12, -0.34, 0.56, 0.78]
  "bank"      → char CNN → h₂⁰ = [0.45, 0.23, -0.67, 0.11]
  "issued"    → char CNN → h₃⁰ = [-0.22, 0.89, 0.33, -0.45]
  "a"         → char CNN → h₄⁰ = [0.01, -0.05, 0.02, 0.03]
  "statement" → char CNN → h₅⁰ = [0.67, -0.12, 0.44, 0.55]
```

**Step 2: Forward LSTM Layer 1**

Process left-to-right:

```
  h₁¹_fwd = LSTM_fwd¹("The",       init)    = [0.21, 0.43, -0.11, 0.65]
  h₂¹_fwd = LSTM_fwd¹("bank",      h₁¹_fwd) = [0.55, -0.22, 0.78, 0.33]
  h₃¹_fwd = LSTM_fwd¹("issued",    h₂¹_fwd) = [0.34, 0.67, -0.45, 0.12]
  h₄¹_fwd = LSTM_fwd¹("a",         h₃¹_fwd) = [0.11, 0.05, 0.22, -0.08]
  h₅¹_fwd = LSTM_fwd¹("statement", h₄¹_fwd) = [0.78, -0.33, 0.56, 0.44]
```

**Step 3: Backward LSTM Layer 1**

Process right-to-left:

```
  h₅¹_bwd = LSTM_bwd¹("statement", init)    = [0.33, 0.11, -0.22, 0.67]
  h₄¹_bwd = LSTM_bwd¹("a",         h₅¹_bwd) = [-0.05, 0.44, 0.33, 0.21]
  h₃¹_bwd = LSTM_bwd¹("issued",    h₄¹_bwd) = [0.56, -0.11, 0.67, 0.45]
  h₂¹_bwd = LSTM_bwd¹("bank",      h₃¹_bwd) = [0.22, 0.78, -0.33, 0.56]
  h₁¹_bwd = LSTM_bwd¹("The",       h₂¹_bwd) = [-0.44, 0.55, 0.12, 0.33]
```

**Step 4: Concatenate for Layer 1**

```
  h₂¹ = [h₂¹_fwd ; h₂¹_bwd]
       = [0.55, -0.22, 0.78, 0.33, 0.22, 0.78, -0.33, 0.56]
         \_______fwd_________/  \________bwd_________/
```

(Similarly for Layer 2, producing h₂².)

**Step 5: Apply task-specific weights**

Suppose for a sentiment analysis task, the learned weights are:

```
  a₀ = -0.5,  a₁ = 1.2,  a₂ = 0.8

  Softmax normalization:
    exp(-0.5) = 0.607,  exp(1.2) = 3.320,  exp(0.8) = 2.226
    sum = 6.153

    s₀ = 0.607 / 6.153 = 0.099
    s₁ = 3.320 / 6.153 = 0.540
    s₂ = 2.226 / 6.153 = 0.362
```

**Step 6: Compute final ELMo vector**

```
  ELMo₂ = γ · (s₀ · h₂⁰ + s₁ · h₂¹ + s₂ · h₂²)
         = γ · (0.099 · h₂⁰ + 0.540 · h₂¹ + 0.362 · h₂²)
```

With γ = 1.3 (learned), the final vector is a scaled, weighted combination
of all three layer representations for the word "bank" — **in this specific
context** of financial language.

### 2.7 Why Different Layers Capture Different Information

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Input: "The bank issued a statement"                           │
  │                                                                  │
  │  Layer 0 (char CNN):                                            │
  │    "bank" ≈ "banks" ≈ "banker" ≈ "banking"                     │
  │    → Captures morphological similarity                          │
  │    → Doesn't know if it's a river bank or financial bank        │
  │                                                                  │
  │  Layer 1 (biLSTM 1):                                            │
  │    "bank" is a NOUN, subject of "issued"                        │
  │    → Captures syntactic role                                    │
  │    → Knows it's a noun that can issue things                    │
  │                                                                  │
  │  Layer 2 (biLSTM 2):                                            │
  │    "bank" = financial institution (because of "issued"          │
  │             and "statement" in context)                          │
  │    → Captures semantic meaning / word sense                     │
  │    → Disambiguates polysemy                                     │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

Peters et al. demonstrated this empirically:
- **POS tagging** (syntactic task): Layer 1 gets highest weight s₁
- **Word sense disambiguation** (semantic task): Layer 2 gets highest weight s₂
- **Named entity recognition** (mixed): Balanced weights across layers

### 2.8 ELMo's Impact and Limitations

**Impact:** ELMo improved state-of-the-art on 6 NLP benchmarks by simply
concatenating ELMo vectors with existing model inputs.

**Limitations:**
- Sequential processing (LSTMs cannot parallelize across time steps)
- Fixed context window in practice
- Bidirectionality is "shallow" — forward and backward are independent
- The two directions never directly attend to each other

```
  ELMo's "bidirectionality":
  
  Forward:   The → river → bank → flooded
  Backward:  The ← river ← bank ← flooded
  
  These are concatenated but never truly interact.
  "bank" sees left context OR right context, never both simultaneously.
```

This limitation motivated the development of truly bidirectional models...


---

## 3. BERT Embeddings (2018)

**Paper:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding" (NAACL 2019, arXiv 2018)

BERT replaced LSTMs with the **Transformer** architecture, enabling truly
bidirectional context modeling through **self-attention**.

### 3.1 From LSTMs to Self-Attention: The Key Insight

The fundamental problem with LSTMs (and ELMo) is sequential processing:

```
  LSTM: Each token must wait for the previous token to be processed.
  
  t₁ → t₂ → t₃ → t₄ → t₅
  
  To connect t₁ and t₅, information must flow through t₂, t₃, t₄.
  Path length: O(n)
```

Self-attention connects every token to every other token **directly**:

```
  Self-Attention: Every token attends to every other token simultaneously.
  
  t₁ ←→ t₂
  t₁ ←→ t₃     Every pair has a DIRECT connection.
  t₁ ←→ t₄     Path length: O(1)
  t₁ ←→ t₅
  t₂ ←→ t₃
  ...           (all pairs)
```

### 3.2 Self-Attention Mechanism: Step by Step

Given an input sequence X ∈ ℝ^{n×d} (n tokens, each of dimension d), self-attention
computes three matrices:

```
  Q = X · W_Q     (Queries)    W_Q ∈ ℝ^{d×d_k}
  K = X · W_K     (Keys)       W_K ∈ ℝ^{d×d_k}
  V = X · W_V     (Values)     W_V ∈ ℝ^{d×d_v}
```

The attention output is:

```
  Attention(Q, K, V) = softmax( Q · K^T / √d_k ) · V
```

Let's break this down piece by piece:

**Step 1: Compute Q, K, V**

Each token's embedding is projected into three different spaces:

```
  For token i:
    q_i = x_i · W_Q    ← "What am I looking for?"
    k_i = x_i · W_K    ← "What do I contain?"
    v_i = x_i · W_V    ← "What do I provide if attended to?"
```

**Step 2: Compute attention scores**

```
  score(i, j) = q_i · k_j^T
  
  This is a dot product measuring: "How relevant is token j to token i?"
```

**Step 3: Scale by √d_k**

```
  scaled_score(i, j) = (q_i · k_j^T) / √d_k
```

Why scale? Without scaling, when d_k is large, the dot products grow large in
magnitude, pushing the softmax into regions with extremely small gradients:

```
  d_k = 64:   √d_k = 8
  
  Without scaling:  score = 48.0  →  softmax → [0.999..., 0.000..., ...]
                                      (nearly one-hot, gradient ≈ 0)
  
  With scaling:     score = 48/8 = 6.0  →  softmax → [0.73, 0.18, 0.09]
                                            (smooth distribution, healthy gradient)
```

**Step 4: Apply softmax to get attention weights**

```
  α_{i,j} = exp(scaled_score(i,j)) / Σ_m exp(scaled_score(i,m))
  
  Σ_j α_{i,j} = 1   (weights sum to 1 for each query token i)
```

**Step 5: Compute weighted sum of values**

```
  output_i = Σ_j α_{i,j} · v_j
```

### 3.3 Numerical Example: Self-Attention in Action

Let's compute self-attention for a 3-token sentence with d = 4, d_k = d_v = 3.

**Input embeddings** (after positional encoding):

```
  X = ┌                          ┐
      │  1.0   0.5  -0.3   0.8  │   ← "The"
      │  0.2  -0.7   1.1   0.4  │   ← "bank"
      │ -0.5   0.9   0.6  -0.2  │   ← "flooded"
      └                          ┘
```

**Weight matrices** (simplified for illustration):

```
  W_Q = ┌                  ┐     W_K = ┌                  ┐     W_V = ┌                  ┐
        │  0.1  0.3  0.5   │           │  0.2  0.4  0.1   │           │  0.3  0.1  0.6   │
        │  0.2  0.1  0.4   │           │  0.1  0.3  0.5   │           │  0.5  0.2  0.3   │
        │  0.4  0.2  0.1   │           │  0.5  0.1  0.2   │           │  0.1  0.4  0.2   │
        │  0.3  0.5  0.2   │           │  0.3  0.2  0.4   │           │  0.2  0.3  0.5   │
        └                  ┘           └                  ┘           └                  ┘
```

**Step 1: Compute Q = X · W_Q**

For "The" (row 1 of X):
```
  q₁ = [1.0, 0.5, -0.3, 0.8] · W_Q
     = [1.0×0.1 + 0.5×0.2 + (-0.3)×0.4 + 0.8×0.3,
        1.0×0.3 + 0.5×0.1 + (-0.3)×0.2 + 0.8×0.5,
        1.0×0.5 + 0.5×0.4 + (-0.3)×0.1 + 0.8×0.2]
     = [0.10 + 0.10 - 0.12 + 0.24,
        0.30 + 0.05 - 0.06 + 0.40,
        0.50 + 0.20 - 0.03 + 0.16]
     = [0.32, 0.69, 0.83]
```

For "bank" (row 2):
```
  q₂ = [0.2, -0.7, 1.1, 0.4] · W_Q
     = [0.02 - 0.14 + 0.44 + 0.12,
        0.06 - 0.07 + 0.22 + 0.20,
        0.10 - 0.28 + 0.11 + 0.08]
     = [0.44, 0.41, 0.01]
```

For "flooded" (row 3):
```
  q₃ = [-0.5, 0.9, 0.6, -0.2] · W_Q
     = [-0.05 + 0.18 + 0.24 - 0.06,
        -0.15 + 0.09 + 0.12 - 0.10,
        -0.25 + 0.36 + 0.06 - 0.04]
     = [0.31, -0.04, 0.13]
```

```
  Q = ┌                     ┐
      │  0.32  0.69  0.83   │   ← q₁ ("The")
      │  0.44  0.41  0.01   │   ← q₂ ("bank")
      │  0.31 -0.04  0.13   │   ← q₃ ("flooded")
      └                     ┘
```

(K and V are computed similarly — we'll use representative values.)

```
  K = ┌                     ┐        V = ┌                     ┐
      │  0.51  0.58  0.47   │            │  0.62  0.38  0.89   │
      │  0.43  0.02  0.35   │            │  0.15  0.51  0.43   │
      │  0.29  0.41  0.28   │            │  0.44  0.33  0.52   │
      └                     ┘            └                     ┘
```

**Step 2: Compute Q · K^T**

```
  Q · K^T = ┌                                    ┐
            │ q₁·k₁  q₁·k₂  q₁·k₃              │
            │ q₂·k₁  q₂·k₂  q₂·k₃              │
            │ q₃·k₁  q₃·k₂  q₃·k₃              │
            └                                    ┘
```

Computing q₁ · k₁:
```
  q₁ · k₁ = 0.32×0.51 + 0.69×0.58 + 0.83×0.47
           = 0.163 + 0.400 + 0.390
           = 0.953
```

Full matrix (computed similarly):
```
  Q · K^T = ┌                     ┐
            │  0.953  0.443  0.608 │
            │  0.436  0.202  0.295 │
            │  0.185  0.186  0.109 │
            └                     ┘
```

**Step 3: Scale by √d_k = √3 ≈ 1.732**

```
  Scaled = ┌                     ┐
           │  0.550  0.256  0.351 │
           │  0.252  0.117  0.170 │
           │  0.107  0.107  0.063 │
           └                     ┘
```

**Step 4: Apply softmax (row-wise)**

For row 1 (attention from "The"):
```
  exp([0.550, 0.256, 0.351]) = [1.733, 1.292, 1.421]
  sum = 4.446
  α₁ = [0.390, 0.291, 0.320]
```

For row 2 (attention from "bank"):
```
  exp([0.252, 0.117, 0.170]) = [1.287, 1.124, 1.185]
  sum = 3.596
  α₂ = [0.358, 0.313, 0.330]
```

For row 3 (attention from "flooded"):
```
  exp([0.107, 0.107, 0.063]) = [1.113, 1.113, 1.065]
  sum = 3.291
  α₃ = [0.338, 0.338, 0.324]
```

```
  Attention Weights:
  
            "The"   "bank"  "flooded"
  "The"    [ 0.390   0.291   0.320  ]   ← "The" attends most to itself
  "bank"   [ 0.358   0.313   0.330  ]   ← "bank" attends to all tokens
  "flooded"[ 0.338   0.338   0.324  ]   ← "flooded" attends to "The" and "bank"
```

**Step 5: Compute output = Attention_weights · V**

For "bank" (row 2):
```
  output₂ = 0.358 × v₁ + 0.313 × v₂ + 0.330 × v₃
           = 0.358 × [0.62, 0.38, 0.89]
           + 0.313 × [0.15, 0.51, 0.43]
           + 0.330 × [0.44, 0.33, 0.52]
           = [0.222, 0.136, 0.319]
           + [0.047, 0.160, 0.135]
           + [0.145, 0.109, 0.172]
           = [0.414, 0.405, 0.626]
```

This output vector for "bank" now encodes information from **all tokens
in the sentence simultaneously** — true bidirectional context.


### 3.4 Multi-Head Attention

Instead of performing a single attention function, BERT uses **multi-head attention**,
which runs h parallel attention operations with different learned projections:

```
  MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h) · W_O
  
  where head_i = Attention(X · W_Q^i, X · W_K^i, X · W_V^i)
```

Each head uses smaller dimensions: d_k = d_v = d_model / h

```
  BERT-base: d_model = 768, h = 12 heads, d_k = 768/12 = 64
  BERT-large: d_model = 1024, h = 16 heads, d_k = 1024/16 = 64
```

**Why multiple heads?** Each head can learn to attend to different types of
relationships:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Sentence: "The animal didn't cross the street because      │
  │             it was too tired"                                │
  │                                                              │
  │  Head 1 (syntactic):    "it" ──attends to──► "animal"       │
  │                         (coreference resolution)             │
  │                                                              │
  │  Head 2 (positional):   "it" ──attends to──► "tired"        │
  │                         (local context)                      │
  │                                                              │
  │  Head 3 (structural):   "it" ──attends to──► "cross"        │
  │                         (verb-subject relationship)          │
  │                                                              │
  │  Combined: "it" gets a rich representation informed by       │
  │            syntactic, positional, and structural cues.       │
  └──────────────────────────────────────────────────────────────┘
```

**Derivation of the output dimension:**

```
  Each head_i ∈ ℝ^{n × d_v}           (n tokens, d_v = 64)
  Concat(head₁,...,head_h) ∈ ℝ^{n × (h·d_v)}  = ℝ^{n × 768}
  W_O ∈ ℝ^{(h·d_v) × d_model}        = ℝ^{768 × 768}
  
  Output ∈ ℝ^{n × d_model}            = ℝ^{n × 768}
  
  → Output dimension matches input dimension (required for residual connections)
```

### 3.5 The Full Transformer Block

Each BERT layer consists of:

```
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  Input: X                                           │
  │    │                                                │
  │    ├──► Multi-Head Attention(X, X, X)               │
  │    │         │                                      │
  │    └──► Add (residual connection)                   │
  │              │                                      │
  │         Layer Norm                                  │
  │              │                                      │
  │    ┌─────────┤                                      │
  │    │         │                                      │
  │    │    Feed-Forward Network                        │
  │    │    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂         │
  │    │         │                                      │
  │    └──► Add (residual connection)                   │
  │              │                                      │
  │         Layer Norm                                  │
  │              │                                      │
  │  Output: X'  (same shape as input)                  │
  │                                                     │
  └─────────────────────────────────────────────────────┘
  
  BERT-base stacks 12 of these blocks.
  BERT-large stacks 24 of these blocks.
```

### 3.6 BERT's Pre-training Objectives

BERT is pre-trained with two objectives:

#### 3.6.1 Masked Language Model (MLM)

Randomly mask 15% of input tokens and predict them:

```
  Input:    "The  [MASK]  issued  a  [MASK]"
  Target:   "The  bank    issued  a  statement"
  
  Of the 15% selected tokens:
    80% → replaced with [MASK]
    10% → replaced with a random word
    10% → kept unchanged
```

The MLM loss for a masked token at position k:

```
  L_MLM = -log P(t_k = "bank" | context)
        = -log softmax(h_k · E^T + b)_{bank}
  
  where:
    h_k   = BERT's output vector at position k
    E     = token embedding matrix
    b     = bias vector
```

Why is MLM important for embeddings? Unlike ELMo's separate forward/backward LMs,
MLM forces each token's representation to encode **both left and right context
simultaneously**:

```
  ELMo:  "bank" sees [The] from left, [issued a statement] from right
         → Two separate, concatenated views
  
  BERT:  "bank" sees [The ___ issued a statement] all at once
         → Single, unified bidirectional view
```

#### 3.6.2 Next Sentence Prediction (NSP)

Given two sentences A and B, predict whether B follows A in the original text:

```
  Input:   [CLS] The bank issued a statement [SEP] It concerned interest rates [SEP]
  Label:   IsNext (positive pair)
  
  Input:   [CLS] The bank issued a statement [SEP] Penguins live in Antarctica [SEP]
  Label:   NotNext (negative pair — randomly sampled)
```

The NSP loss:

```
  L_NSP = -log P(IsNext | h_{[CLS]})
  
  where h_{[CLS]} is BERT's output at the [CLS] position
```

**Total pre-training loss:**

```
  L = L_MLM + L_NSP
```

Note: Later work (RoBERTa, Liu et al. 2019) showed that NSP actually *hurts*
performance and can be removed. The MLM objective alone is sufficient.

### 3.7 Extracting Embeddings from BERT

Once BERT is pre-trained, we can extract embeddings in several ways:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Input: [CLS] The bank flooded [SEP]                            │
  │                                                                  │
  │  BERT Layer 12:  h_[CLS]¹²  h_The¹²  h_bank¹²  h_flooded¹²   │
  │  BERT Layer 11:  h_[CLS]¹¹  h_The¹¹  h_bank¹¹  h_flooded¹¹   │
  │       ...                                                       │
  │  BERT Layer 1:   h_[CLS]¹   h_The¹   h_bank¹   h_flooded¹    │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

#### Method 1: [CLS] Token Embedding

Use the output vector at the [CLS] position from the final layer:

```
  sentence_embedding = h_{[CLS]}^{12}    ∈ ℝ^{768}
```

The [CLS] token is designed to aggregate sentence-level information (trained
via NSP). However, without fine-tuning, [CLS] is often a poor sentence
representation.

#### Method 2: Mean Pooling

Average all token embeddings from the final layer:

```
  sentence_embedding = (1/n) Σ_{i=1}^{n} h_i^{12}    ∈ ℝ^{768}
```

This is simple and often works better than [CLS] for out-of-the-box use.

#### Method 3: Layer Selection / Concatenation

Different layers capture different information (similar to ELMo):

```
  Layer 12:  Most task-specific (fine-tuned representations)
  Layer 11:  High-level semantics
  Layers 7-9: Syntactic information peaks
  Layers 1-4: Low-level, more general features
  
  Common strategy: Concatenate last 4 layers
    embedding = [h_k^9 ; h_k^10 ; h_k^11 ; h_k^12]    ∈ ℝ^{3072}
```

```
  ┌──────────────────────────────────────────────────────────┐
  │  Embedding Extraction Comparison                         │
  │                                                          │
  │  Method          Dim     Quality (no fine-tune)          │
  │  ─────────────   ─────   ──────────────────────          │
  │  [CLS] token     768     Mediocre                        │
  │  Mean pooling    768     Good                            │
  │  Last 4 concat   3072    Good (but high-dimensional)     │
  │  Second-to-last  768     Often best single layer         │
  └──────────────────────────────────────────────────────────┘
```


### 3.8 BERT vs ELMo: A Structural Comparison

```
  ┌────────────────────┬──────────────────────┬──────────────────────┐
  │  Feature           │  ELMo                │  BERT                │
  ├────────────────────┼──────────────────────┼──────────────────────┤
  │  Architecture      │  biLSTM (2 layers)   │  Transformer (12/24) │
  │  Bidirectionality  │  Shallow (concat)    │  Deep (self-attn)    │
  │  Context window    │  ~50 tokens (LSTM)   │  512 tokens (fixed)  │
  │  Parallelization   │  Sequential          │  Fully parallel      │
  │  Pre-train obj.    │  Bidirectional LM    │  MLM + NSP           │
  │  Fine-tuning       │  Feature extraction  │  Full fine-tuning    │
  │  Parameters        │  94M                 │  110M / 340M         │
  │  Token embeddings  │  Char CNN            │  WordPiece           │
  └────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 4. GPT-style Embeddings

While BERT uses bidirectional attention, the GPT family (Radford et al., 2018)
uses **causal (autoregressive) attention**, which produces a fundamentally
different type of embedding.

### 4.1 Causal Attention: Looking Only Left

In GPT, each token can only attend to tokens that came **before** it (and itself).
This is enforced by masking future positions:

```
  Causal Attention Mask:
  
              t₁    t₂    t₃    t₄    t₅
  t₁  →   [  1     0     0     0     0  ]    t₁ sees only t₁
  t₂  →   [  1     1     0     0     0  ]    t₂ sees t₁, t₂
  t₃  →   [  1     1     1     0     0  ]    t₃ sees t₁, t₂, t₃
  t₄  →   [  1     1     1     1     0  ]    t₄ sees t₁ through t₄
  t₅  →   [  1     1     1     1     1  ]    t₅ sees everything
  
  (1 = can attend, 0 = masked / cannot attend)
```

The masked positions are set to -∞ before softmax, ensuring zero attention weight:

```
  Causal_Attention(Q, K, V) = softmax( (Q·K^T / √d_k) + M ) · V
  
  where M_{i,j} = 0     if j ≤ i   (allowed)
                = -∞    if j > i   (masked)
```

### 4.2 How GPT Embeddings Differ from BERT

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Sentence: "The river bank flooded the town"                │
  │                                                              │
  │  BERT (bidirectional):                                      │
  │    "bank" sees: [The, river, ___, flooded, the, town]       │
  │    → Knows about flooding → river bank                      │
  │                                                              │
  │  GPT (causal):                                              │
  │    "bank" sees: [The, river, ___]                           │
  │    → Only knows "The river" precedes it                     │
  │    → Cannot use "flooded" to disambiguate                   │
  │                                                              │
  │  BUT: The LAST token in GPT sees EVERYTHING:                │
  │    "town" sees: [The, river, bank, flooded, the, ___]      │
  │    → Has full left context of the entire sentence           │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

### 4.3 Last Token Pooling

Because only the last token has seen the entire input, GPT-style models
typically use **last token pooling** for sentence embeddings:

```
  Input:  "The river bank flooded the town"
  
  Token positions:  t₁    t₂     t₃    t₄      t₅   t₆
                   "The" "river" "bank" "flooded" "the" "town"
  
  Sentence embedding = h_{t₆}^{final_layer}
                     = output vector of "town" at the last layer
```

This contrasts with BERT's approaches:

```
  ┌──────────────────────────────────────────────────────────┐
  │  Pooling Strategy Comparison                             │
  │                                                          │
  │  BERT:                                                   │
  │    [CLS] pooling  → h_{[CLS]}  (special token)         │
  │    Mean pooling   → avg(h₁, h₂, ..., h_n)              │
  │                                                          │
  │  GPT:                                                    │
  │    Last token     → h_n  (last token sees all context)  │
  │                                                          │
  │  Why not mean pooling for GPT?                          │
  │    Early tokens have very limited context.               │
  │    h₁ only sees itself — averaging it in adds noise.    │
  └──────────────────────────────────────────────────────────┘
```

### 4.4 GPT's Training Objective

GPT is trained with a standard autoregressive language model objective:

```
  L = -Σ_{k=1}^{N} log P(t_k | t₁, t₂, ..., t_{k-1})
```

This is simpler than BERT's MLM — no masking strategy needed, just predict
the next token. The trade-off:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  BERT (MLM):                                               │
  │    ✓ Bidirectional context for every token                 │
  │    ✗ Only 15% of tokens provide training signal            │
  │    ✗ [MASK] token mismatch between pre-training & usage    │
  │                                                             │
  │  GPT (Autoregressive):                                     │
  │    ✓ Every token provides training signal (100%)           │
  │    ✓ No train/test mismatch                                │
  │    ✗ Unidirectional — each token misses right context      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

Despite the unidirectional limitation, GPT-style models have scaled remarkably
well (GPT-2, GPT-3, GPT-4), and their embeddings — especially from the last
token — have proven highly effective for many tasks.


---

## 5. The Anisotropy Problem

Despite their power, BERT (and other Transformer) embeddings suffer from a
subtle but significant geometric problem: **anisotropy**.

### 5.1 What Is Anisotropy?

In an **isotropic** embedding space, vectors are uniformly distributed across
all directions. In an **anisotropic** space, vectors cluster in a narrow cone,
occupying only a small region of the available space.

```
  ISOTROPIC (ideal)                ANISOTROPIC (BERT)
  
        *    *                           * * *
      *   *    *                        * * * *
    *  *    *   *                      * * * * *
   * *   *   *  *                     * * * * * *
    *  *   *  * *                      * * * * *
      *  *   *                          * * * *
        *   *                            * * *
                                          
  Vectors spread uniformly          Vectors clustered in a
  across the sphere                 narrow cone
  
  cos(v_i, v_j) varies widely      cos(v_i, v_j) ≈ 0.6-0.99
  from -1 to +1                    for almost ALL pairs
```

### 5.2 Ethayarajh (2019): Measuring Anisotropy in Context

Kawin Ethayarajh's paper "How Contextual are Contextualized Word Representations?"
(EMNLP 2019) provided the first systematic analysis of this phenomenon.

**Key findings:**

**Finding 1: Embeddings become more anisotropic in higher layers.**

```
  Average cosine similarity between random word pairs:
  
  Layer    BERT-base    GPT-2     ELMo
  ─────    ─────────    ─────     ────
  0        0.60         0.58      0.54
  3        0.63         0.65      0.56
  6        0.68         0.72      —
  9        0.73         0.80      —
  12       0.74         0.85      —
  
  In a perfectly isotropic space, random pairs would have
  average cosine similarity ≈ 0.0
```

**Finding 2: Context-specificity increases with depth.**

Ethayarajh measured how much a word's representation changes across different
contexts using **self-similarity** — the average cosine similarity of a word
with itself across different sentences:

```
  Self-similarity of "bank" across 1000 sentences:
  
  Layer 0:   self-sim ≈ 0.99   (nearly identical — barely contextual)
  Layer 6:   self-sim ≈ 0.73   (moderately contextual)
  Layer 12:  self-sim ≈ 0.54   (highly contextual — different in each context)
  
  Lower self-similarity = more context-dependent = more "contextual"
```

**Finding 3: The anisotropy confounds cosine similarity.**

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Without anisotropy correction:                             │
  │                                                              │
  │    cos("bank_financial", "bank_river") = 0.82               │
  │    cos("bank_financial", "money")      = 0.85               │
  │    cos("bank_financial", "penguin")    = 0.78               │
  │                                                              │
  │  Everything looks similar! The high baseline cosine          │
  │  similarity drowns out meaningful differences.               │
  │                                                              │
  │  With anisotropy correction (subtract mean, normalize):     │
  │                                                              │
  │    cos("bank_financial", "bank_river") = 0.31               │
  │    cos("bank_financial", "money")      = 0.72               │
  │    cos("bank_financial", "penguin")    = 0.05               │
  │                                                              │
  │  Now the similarities are meaningful!                        │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

### 5.3 Why Does Anisotropy Occur?

The root cause is tied to the language model training objective and the
softmax output layer:

```
  The output layer computes:  P(word_i) = softmax(h · E^T)
  
  where E is the embedding matrix (shared with input embeddings in BERT).
  
  The softmax pushes the hidden state h to be close to the embedding
  of the correct word and far from all others. But with a vocabulary
  of 30,000+ words, the model finds it efficient to:
  
  1. Push ALL hidden states into a small region of the space
  2. Use small angular differences within that region to distinguish words
  
  This creates the narrow cone structure.
```

Mathematically, if we decompose the hidden states:

```
  h_i = μ + δ_i
  
  where μ is the mean direction (the "cone axis")
  and δ_i is the deviation from the mean
  
  ||μ|| >> ||δ_i|| for most tokens
  
  Therefore: cos(h_i, h_j) ≈ cos(μ + δ_i, μ + δ_j) ≈ 1 - small_correction
```

### 5.4 Implications for Embedding Quality

The anisotropy problem means that **raw BERT embeddings are poor for
similarity-based tasks** like semantic search or clustering:

```
  Task: Find the most similar sentence to "How do I open a bank account?"
  
  Using raw BERT [CLS] embeddings:
    "What are the bank's opening hours?"     cos = 0.94  ← WRONG (highest)
    "How to create a new savings account?"   cos = 0.92
    "The river bank was muddy"               cos = 0.89
    "I like pizza"                           cos = 0.85
    
  The scores are all compressed into [0.85, 0.94] — barely distinguishable.
```

This problem motivates two lines of work:
1. **Post-hoc corrections** (whitening, centering) to fix the geometry
2. **Training better sentence embeddings** (SBERT, covered in Chapter 6)


---

## 6. From Token Embeddings to Sentence Embeddings

We've seen how ELMo, BERT, and GPT produce rich **token-level** embeddings.
But many real-world tasks need **sentence-level** (or document-level) embeddings:

```
  Token-level tasks:           Sentence-level tasks:
  ─────────────────            ──────────────────────
  Named Entity Recognition     Semantic search
  POS tagging                  Sentence similarity
  Token classification         Document clustering
                               Retrieval / RAG
                               Paraphrase detection
```

### 6.1 The Gap Between Token and Sentence Representations

BERT produces a 768-dimensional vector for **each token**. But how do we get
a single vector for an entire sentence?

```
  Input: "The quick brown fox jumps over the lazy dog"
  
  BERT output: 9 vectors, each ∈ ℝ^768
  
    h_The    = [0.12, -0.34, 0.56, ..., 0.23]     ┐
    h_quick  = [0.45, 0.23, -0.67, ..., 0.11]     │
    h_brown  = [-0.22, 0.89, 0.33, ..., -0.45]    │
    h_fox    = [0.67, -0.12, 0.44, ..., 0.55]     │  9 token vectors
    h_jumps  = [0.33, 0.56, -0.11, ..., 0.78]     │  How to combine
    h_over   = [-0.15, 0.42, 0.28, ..., 0.19]     │  into ONE vector?
    h_the    = [0.08, -0.23, 0.61, ..., 0.34]     │
    h_lazy   = [0.51, 0.17, -0.39, ..., 0.62]     │
    h_dog    = [0.29, -0.55, 0.73, ..., 0.41]     ┘
    
    sentence_embedding = ???    ∈ ℝ^768
```

### 6.2 Why Naive Averaging of BERT Tokens Is Suboptimal

The simplest approach — mean pooling — has several problems:

```
  sentence_embedding = (1/n) Σ_{i=1}^{n} h_i
```

**Problem 1: Function words dominate**

```
  "The quick brown fox jumps over the lazy dog"
   ^^^                          ^^^
   Function words ("the", "a", "over", "is") appear frequently
   and contribute equally to the average, diluting content words.
```

**Problem 2: Anisotropy corrupts the average**

As we saw in Section 5, BERT embeddings cluster in a narrow cone. Averaging
vectors that are already highly similar produces a result that's close to
the mean of the entire embedding space — not a meaningful sentence representation.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Averaging anisotropic vectors:                             │
  │                                                              │
  │       * * *                                                  │
  │      * * * *          avg(*)  ≈  center of cone             │
  │     * * * * *           ↓                                    │
  │      * * * *          This point is nearly the same          │
  │       * * *           for ANY set of tokens!                 │
  │                                                              │
  │  Result: All sentence embeddings look similar.              │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

**Problem 3: No training signal for sentence-level meaning**

BERT was trained with MLM (predict masked tokens) and NSP (predict sentence
order). Neither objective explicitly teaches the model to produce good
**sentence-level** representations:

```
  MLM:  Optimizes token-level predictions
        → Token embeddings are good, sentence embedding is a side effect
  
  NSP:  Uses [CLS] for binary classification
        → [CLS] learns "same topic?" not "semantic similarity"
```

### 6.3 Empirical Evidence: BERT Sentence Embeddings Underperform

Reimers and Gurevych (2019) showed that BERT sentence embeddings (via mean
pooling or [CLS]) actually perform **worse** than GloVe averaging on several
semantic textual similarity (STS) benchmarks:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  STS Benchmark Performance (Spearman correlation)           │
  │                                                              │
  │  Method                              STS-B    SICK-R        │
  │  ──────────────────────────────────  ─────    ──────        │
  │  GloVe average                       58.02    53.76         │
  │  BERT [CLS] (no fine-tuning)         20.29    42.42         │
  │  BERT mean pooling (no fine-tuning)  46.35    58.40         │
  │  SBERT (fine-tuned)                  84.67    80.47         │
  │                                                              │
  │  Raw BERT [CLS] is WORSE than simple GloVe averaging!      │
  └──────────────────────────────────────────────────────────────┘
```

This is a striking result: a 110M-parameter model produces worse sentence
embeddings than averaging 300-dimensional static vectors.

### 6.4 The Cross-Encoder vs Bi-Encoder Problem

BERT *can* compute excellent sentence similarity — but only as a **cross-encoder**,
where both sentences are fed together:

```
  Cross-Encoder (accurate but slow):
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  Input: [CLS] sentence_A [SEP] sentence_B [SEP]   │
  │                    ↓                                │
  │               BERT (12 layers)                      │
  │                    ↓                                │
  │              h_{[CLS]}                              │
  │                    ↓                                │
  │           similarity score                          │
  │                                                     │
  │  ✓ Full cross-attention between A and B            │
  │  ✗ Must process EVERY pair — O(n²) for n sentences │
  │  ✗ Cannot pre-compute embeddings                   │
  │                                                     │
  │  10,000 sentences → 50,000,000 BERT inferences     │
  │  At 50ms each → ~29 days                           │
  └─────────────────────────────────────────────────────┘
  
  Bi-Encoder (fast but needs good embeddings):
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  sentence_A → BERT → pool → emb_A ─┐              │
  │                                      ├→ cos(A,B)   │
  │  sentence_B → BERT → pool → emb_B ─┘              │
  │                                                     │
  │  ✓ Pre-compute all embeddings once — O(n)          │
  │  ✓ Compare with cosine similarity — O(1) per pair  │
  │  ✗ No cross-attention — needs good pooling         │
  │                                                     │
  │  10,000 sentences → 10,000 BERT inferences         │
  │  At 50ms each → ~8 minutes                         │
  └─────────────────────────────────────────────────────┘
```

The bi-encoder approach is 5000× faster but requires sentence embeddings
that capture semantic meaning — which raw BERT doesn't provide.

### 6.5 Setting the Stage for SBERT

The gap is clear:

1. We need **sentence-level** embeddings for practical applications
2. Raw BERT embeddings are **poor** at the sentence level
3. Cross-encoders are **too slow** for real-world retrieval
4. We need a way to **fine-tune** BERT to produce good sentence embeddings

This is exactly what Sentence-BERT (SBERT) addresses, which we'll cover
in the next chapter. SBERT uses a **siamese network** architecture to
fine-tune BERT with a contrastive objective, producing embeddings where
cosine similarity directly reflects semantic similarity.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  The Contextual Embeddings Journey:                         │
  │                                                              │
  │  Static          Contextual         Sentence                │
  │  Embeddings  →   Token Emb.    →    Embeddings              │
  │  (Ch. 2-4)      (Ch. 5)            (Ch. 6 — next)          │
  │                                                              │
  │  Word2Vec        ELMo               SBERT                   │
  │  GloVe           BERT               SimCSE                  │
  │  FastText        GPT                E5, BGE, ...            │
  │                                                              │
  │  One vector      Context-dependent  Whole-sentence           │
  │  per word        per token          vectors optimized        │
  │                                     for similarity           │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

---

## Summary

| Model | Year | Architecture | Context | Key Innovation |
|-------|------|-------------|---------|----------------|
| ELMo | 2018 | biLSTM (2L) | Shallow bidir. | Layer-weighted representations |
| BERT | 2018 | Transformer (12/24L) | Deep bidir. | Masked LM + self-attention |
| GPT | 2018 | Transformer (12L) | Causal (L→R) | Autoregressive, scales well |

**Key takeaways:**

1. Static embeddings assign one vector per word — contextual embeddings assign
   one vector per word **per context**.

2. ELMo pioneered contextual embeddings using bidirectional LSTMs with
   task-specific layer weighting: ELMo_k = γ Σ_j s_j h_{k,j}.

3. BERT introduced deep bidirectional context through self-attention:
   Attention(Q,K,V) = softmax(QK^T/√d_k)V, trained with masked language modeling.

4. GPT uses causal attention (left-to-right only), with last-token pooling
   for sentence representations.

5. Raw Transformer embeddings suffer from **anisotropy** — vectors cluster in
   a narrow cone, making cosine similarity unreliable without correction.

6. Token-level contextual embeddings do not automatically yield good
   sentence-level embeddings — dedicated training (SBERT) is needed.

---

*Next chapter: [Chapter 6 — Sentence Embeddings and SBERT](06-sentence-embeddings-sbert.md)*
