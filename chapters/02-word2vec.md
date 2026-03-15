# Chapter 2: Word2Vec — Learning Distributed Word Representations

## Table of Contents

1. [Introduction and Historical Context](#1-introduction-and-historical-context)
2. [Continuous Bag of Words (CBOW)](#2-continuous-bag-of-words-cbow)
3. [Skip-gram](#3-skip-gram)
4. [The Softmax Bottleneck](#4-the-softmax-bottleneck)
5. [Hierarchical Softmax](#5-hierarchical-softmax)
6. [Negative Sampling](#6-negative-sampling)
7. [Subsampling of Frequent Words](#7-subsampling-of-frequent-words)
8. [Properties of Word2Vec Embeddings](#8-properties-of-word2vec-embeddings)

---

## 1. Introduction and Historical Context

### 1.1 The Paper That Changed NLP

In 2013, Tomáš Mikolov and colleagues at Google published *"Efficient Estimation of Word
Representations in Vector Space"* (Mikolov et al., 2013), introducing two model architectures
— Continuous Bag of Words (CBOW) and Skip-gram — that could learn high-quality distributed
word representations from massive unlabeled corpora in a fraction of the time required by
prior neural language models.

A follow-up paper, *"Distributed Representations of Words and Phrases and their
Compositionality"* (Mikolov et al., 2013b), introduced negative sampling, hierarchical
softmax, and subsampling of frequent words. Together, these two papers form **Word2Vec**.

### 1.2 Why Word2Vec Was Revolutionary

Before Word2Vec, learning word embeddings meant training full neural language models
(Bengio et al., 2003) with hidden layers, non-linear activations, and softmax output layers
over the entire vocabulary. These models were slow (weeks to train on large corpora),
complex (multiple non-linear hidden layers), and limited to small vocabularies.

Word2Vec's key insight was radical simplicity: **remove the hidden non-linearity**. By using
a shallow, log-linear model, Mikolov et al. achieved orders-of-magnitude speedups, enabling
training on billion-word corpora with million-word vocabularies — and discovered that the
resulting vector space exhibited striking linear algebraic structure (the famous
`king - man + woman ≈ queen` relationship).

### 1.3 Notation

Throughout this chapter, we use the following notation:

| Symbol       | Meaning                                                        |
|--------------|----------------------------------------------------------------|
| V            | Vocabulary size (number of unique words)                       |
| N            | Embedding dimension (hidden layer size)                        |
| W            | Input (center/context) embedding matrix, shape V × N          |
| W'           | Output embedding matrix, shape N × V                          |
| x            | One-hot input vector, shape V × 1                             |
| h            | Hidden layer vector, shape N × 1                              |
| u            | Output score vector (pre-softmax), shape V × 1                |
| y            | Output probability vector (post-softmax), shape V × 1         |
| v_w          | Input embedding vector for word w (row of W)                   |
| v'_w         | Output embedding vector for word w (column of W')              |
| C            | Context window size (number of context words)                  |
| w_t          | Center (target) word                                           |
| w_c          | Context word                                                   |
| w_O          | Output (predicted) word                                        |
| w_I          | Input word                                                     |

---

## 2. Continuous Bag of Words (CBOW)

### 2.1 Architecture Overview

CBOW predicts a **center word** from its surrounding **context words**. The context words
are averaged in the projection layer (no non-linearity), and the result is fed through a
softmax to produce a probability distribution over the vocabulary.

```
    Context Words (one-hot)          Projection (shared W)         Output
    ┌─────────┐
    │ x_{c-2} │──┐
    └─────────┘  │
    ┌─────────┐  │    ┌───────────────────┐      ┌──────────┐      ┌──────────┐
    │ x_{c-1} │──┼───▶│  h = (1/C) Σ Wx_c │─────▶│ u = W'ᵀh │─────▶│ softmax  │──▶ ŷ
    └─────────┘  │    └───────────────────┘      └──────────┘      └──────────┘
    ┌─────────┐  │         (N × 1)                  (V × 1)          P(w_O|ctx)
    │ x_{c+1} │──┤
    └─────────┘  │
    ┌─────────┐  │
    │ x_{c+2} │──┘
    └─────────┘
     (V × 1 each)

    Legend:
    ─────  W ∈ ℝ^{V×N}  (input embeddings, shared across all context positions)
    ─────  W' ∈ ℝ^{N×V} (output embeddings)
    C = number of context words (4 in this diagram, window = 2)
```

### 2.2 Forward Pass — Step by Step

Suppose we have a vocabulary of V words and want N-dimensional embeddings. Given C context
words {w_{c₁}, w_{c₂}, ..., w_{c_C}}, we want to predict the center word w_O.

**Step 1: One-hot encode each context word.**

Each context word w_{c_i} is represented as a one-hot vector x_{c_i} ∈ ℝ^V, where:

```
x_{c_i}[j] = 1  if j = index(w_{c_i})
x_{c_i}[j] = 0  otherwise
```

**Step 2: Look up embeddings via the input matrix W.**

Multiplying W^T by the one-hot vector simply selects the corresponding row of W:

```
v_{c_i} = W^T x_{c_i}    ∈ ℝ^N
```

This is equivalent to a table lookup: `v_{c_i} = W[index(w_{c_i}), :]`.

**Step 3: Average the context embeddings.**

```
h = (1/C) Σ_{i=1}^{C} v_{c_i} = (1/C) Σ_{i=1}^{C} W^T x_{c_i}    ∈ ℝ^N
```

**Step 4: Compute the output score vector.**

```
u = W'^T h    ∈ ℝ^V
```

where u_j = v'_j^T · h is the score for the j-th word in the vocabulary.

**Step 5: Apply softmax.**

```
P(w_O | w_{c₁}, ..., w_{c_C}) = y_j = exp(u_j) / Σ_{k=1}^{V} exp(u_k)
```

where j = index(w_O).

### 2.3 Loss Function

We minimize the negative log-likelihood of the correct center word:

```
J = -log P(w_O | context)
  = -log [ exp(u_{j*}) / Σ_{k=1}^{V} exp(u_k) ]
  = -u_{j*} + log Σ_{k=1}^{V} exp(u_k)
```

where j* = index(w_O) is the index of the true center word.

### 2.4 Gradient Derivation — Step by Step

We need gradients with respect to both W' (output embeddings) and W (input embeddings).

#### 2.4.1 Gradient with respect to u

Define the prediction error e_j = y_j - t_j, where t is the one-hot target (t_{j*} = 1).

**Derivation:**

```
∂J/∂u_j = ∂/∂u_j [ -u_{j*} + log Σ_k exp(u_k) ]
```

For j ≠ j*:

```
∂J/∂u_j = 0 + exp(u_j) / Σ_k exp(u_k) = y_j = y_j - 0 = y_j - t_j
```

For j = j*:

```
∂J/∂u_{j*} = -1 + exp(u_{j*}) / Σ_k exp(u_k) = -1 + y_{j*} = y_{j*} - 1 = y_{j*} - t_{j*}
```

So in both cases: **∂J/∂u_j = y_j - t_j = e_j**. In vector form:

```
∂J/∂u = y - t = e    ∈ ℝ^V
```

#### 2.4.2 Gradient with respect to W'

Since u = W'^T h, for each column v'_j of W':

```
∂J/∂v'_j = ∂J/∂u_j · ∂u_j/∂v'_j = e_j · h
```

In matrix form:

```
∂J/∂W' = h · e^T    ∈ ℝ^{N×V}
```

**Update rule for W':**

```
W'^{(new)} = W'^{(old)} - η · h · e^T
```

where η is the learning rate.

#### 2.4.3 Gradient with respect to h

**Step 3.** Since u = W'^T h:

```
∂J/∂h = W' · e    ∈ ℝ^N
```

This is the error signal backpropagated to the hidden layer.

#### 2.4.4 Gradient with respect to W

**Step 4.** Recall h = (1/C) Σ v_{c_i}. For each context word c_i:

```
∂J/∂v_{c_i} = (1/C) · ∂J/∂h = (1/C) · W' · e
```

Since v_{c_i} is the row of W for word c_i:

```
v_{c_i}^{(new)} = v_{c_i}^{(old)} - η · (1/C) · W' · e
```

Only the rows of W for the context words are updated per training step.

### 2.5 Numerical Example — CBOW

**Setup:** V = 5, N = 3, context window = 1 (one word on each side).

Vocabulary: {the, cat, sat, on, mat} indexed 0–4.

Training sentence: "the cat sat on mat"

Training pair: context = {cat, on} → center = sat (index 2).

**Input matrix W (V×N = 5×3):**

```
W = [ 0.1  0.2  0.3 ]   ← the (0)
    [ 0.4  0.5  0.6 ]   ← cat (1)
    [ 0.7  0.8  0.9 ]   ← sat (2)
    [ 0.2  0.1  0.4 ]   ← on  (3)
    [ 0.5  0.3  0.2 ]   ← mat (4)
```

**Output matrix W' (N×V = 3×5):**

```
W' = [ 0.2  0.3  0.1  0.4  0.5 ]
     [ 0.1  0.4  0.2  0.3  0.1 ]
     [ 0.3  0.2  0.5  0.1  0.4 ]
```

**Step 1: One-hot encode context words.**

```
x_cat = [0, 1, 0, 0, 0]^T
x_on  = [0, 0, 0, 1, 0]^T
```

**Step 2: Look up embeddings.**

```
v_cat = W^T x_cat = W[1,:] = [0.4, 0.5, 0.6]
v_on  = W^T x_on  = W[3,:] = [0.2, 0.1, 0.4]
```

**Step 3: Average to get hidden layer.**

```
h = (1/2)(v_cat + v_on) = (1/2)([0.4, 0.5, 0.6] + [0.2, 0.1, 0.4])
  = (1/2)[0.6, 0.6, 1.0]
  = [0.3, 0.3, 0.5]
```

**Step 4: Compute output scores.**

```
u = W'^T h

u_0 = 0.2(0.3) + 0.1(0.3) + 0.3(0.5) = 0.06 + 0.03 + 0.15 = 0.24
u_1 = 0.3(0.3) + 0.4(0.3) + 0.2(0.5) = 0.09 + 0.12 + 0.10 = 0.31
u_2 = 0.1(0.3) + 0.2(0.3) + 0.5(0.5) = 0.03 + 0.06 + 0.25 = 0.34
u_3 = 0.4(0.3) + 0.3(0.3) + 0.1(0.5) = 0.12 + 0.09 + 0.05 = 0.26
u_4 = 0.5(0.3) + 0.1(0.3) + 0.4(0.5) = 0.15 + 0.03 + 0.20 = 0.38

u = [0.24, 0.31, 0.34, 0.26, 0.38]
```

**Step 5: Apply softmax.**

```
exp(u) = [1.2712, 1.3634, 1.4049, 1.2969, 1.4623]

Σ exp(u_k) = 1.2712 + 1.3634 + 1.4049 + 1.2969 + 1.4623 = 6.7987

y_0 = 1.2712 / 6.7987 = 0.1870
y_1 = 1.3634 / 6.7987 = 0.2005
y_2 = 1.4049 / 6.7987 = 0.2066
y_3 = 1.2969 / 6.7987 = 0.1907
y_4 = 1.4623 / 6.7987 = 0.2151

y = [0.1870, 0.2005, 0.2066, 0.1907, 0.2151]
```

**Loss:**

```
J = -log(y_2) = -log(0.2066) = 1.5762
```

**Error vector (target is index 2):**

```
e = y - t = [0.1870, 0.2005, 0.2066 - 1, 0.1907, 0.2151]
          = [0.1870, 0.2005, -0.7934, 0.1907, 0.2151]
```

The negative value at index 2 drives the model to increase the score for "sat" and decrease
all others.

---

## 3. Skip-gram

### 3.1 Architecture Overview

Skip-gram inverts the CBOW task: given a **center word**, predict the surrounding **context
words**. This reversal means each (center, context) pair is a separate training example,
giving rare words more gradient updates relative to their frequency.

```
    Input (one-hot)     Projection      Output         Predictions
    ┌─────────┐       ┌──────────┐    ┌────────┐    ┌──────────────┐
    │  x_{w_t}│──────▶│ h = Wx_t │───▶│ W'ᵀh   │───▶│ P(w_c | w_t) │ × C
    └─────────┘       └──────────┘    └────────┘    └──────────────┘
     (V × 1)            (N × 1)        (V × 1)      (one per context pos)

    The SAME output score vector u = W'ᵀh is used for all context positions.
    Each position shares the same softmax but has a different target.
```

### 3.2 Forward Pass — Step by Step

Given center word w_t, predict each context word w_{c_i} for i = 1, ..., C.

**Step 1: One-hot encode the center word.**

```
x_{w_t} ∈ ℝ^V,  where x_{w_t}[j] = 1 if j = index(w_t), else 0
```

**Step 2: Project to hidden layer.**

```
h = W^T x_{w_t} = v_{w_t}    ∈ ℝ^N
```

**Step 3: Compute output scores.**

```
u = W'^T h    ∈ ℝ^V
```

**Step 4: Apply softmax for each context position.**

```
P(w_{c_i} | w_t) = exp(u_{c_i}) / Σ_{k=1}^{V} exp(u_k)
```

### 3.3 Loss Function

The Skip-gram objective maximizes the average log-probability of all context words given the
center word:

```
J = -(1/C) Σ_{i=1}^{C} log P(w_{c_i} | w_t)
```

Expanding and noting the log-sum term is constant across i:

```
J = -(1/C) Σ_{i=1}^{C} u_{c_i} + log Σ_{k=1}^{V} exp(u_k)
```

### 3.4 Full Gradient Derivation

#### 3.4.1 Gradient with respect to u

For a single context word w_{c_i} at position i, the loss contribution is:

```
J_i = -u_{c_i} + log Σ_{k=1}^{V} exp(u_k)
```

**Step 1.** Differentiate with respect to u_j:

```
∂J_i/∂u_j = -𝟙(j = c_i) + exp(u_j) / Σ_k exp(u_k)
           = -𝟙(j = c_i) + y_j
           = y_j - t_j^{(i)}
```

where t^{(i)} is the one-hot target for the i-th context word.

Summing over all C context positions:

```
∂J/∂u_j = (1/C) Σ_{i=1}^{C} (y_j - t_j^{(i)})
```

Define the aggregated error:

```
e_j = y_j - (1/C) Σ_{i=1}^{C} t_j^{(i)}
```

#### 3.4.2 Gradient with respect to W'

**Step 2.** Since u_j = v'_j^T h:

```
∂J/∂v'_j = e_j · h    ∈ ℝ^N
```

In matrix form:

```
∂J/∂W' = h · e^T    ∈ ℝ^{N×V}
```

**Update rule:**

```
v'_j^{(new)} = v'_j^{(old)} - η · e_j · h
```

#### 3.4.3 Gradient with respect to h

**Step 3.** Backpropagate through u = W'^T h:

```
∂J/∂h = W' · e = Σ_{j=1}^{V} e_j · v'_j    ∈ ℝ^N
```

#### 3.4.4 Gradient with respect to W

**Step 4.** Since h = v_{w_t}:

```
∂J/∂v_{w_t} = ∂J/∂h = W' · e    ∈ ℝ^N
```

Only the row of W corresponding to the center word is updated.

### 3.5 Numerical Example — Skip-gram

**Setup:** Same vocabulary and matrices as the CBOW example.

Vocabulary: {the, cat, sat, on, mat} indexed 0–4. V = 5, N = 3.

Training pair: center = sat (index 2) → context = {cat (1), on (3)}, C = 2.

**W and W' are the same as before.**

**Step 1: One-hot encode center word.**

```
x_sat = [0, 0, 1, 0, 0]^T
```

**Step 2: Hidden layer (just the embedding of "sat").**

```
h = W^T x_sat = W[2,:] = [0.7, 0.8, 0.9]
```

**Step 3: Output scores.**

```
u = W'^T h

u_0 = 0.2(0.7) + 0.1(0.8) + 0.3(0.9) = 0.14 + 0.08 + 0.27 = 0.49
u_1 = 0.3(0.7) + 0.4(0.8) + 0.2(0.9) = 0.21 + 0.32 + 0.18 = 0.71
u_2 = 0.1(0.7) + 0.2(0.8) + 0.5(0.9) = 0.07 + 0.16 + 0.45 = 0.68
u_3 = 0.4(0.7) + 0.3(0.8) + 0.1(0.9) = 0.28 + 0.24 + 0.09 = 0.61
u_4 = 0.5(0.7) + 0.1(0.8) + 0.4(0.9) = 0.35 + 0.08 + 0.36 = 0.79

u = [0.49, 0.71, 0.68, 0.61, 0.79]
```

**Step 4: Softmax.**

```
exp(u) = [1.6323, 2.0340, 1.9739, 1.8405, 2.2034]

Σ exp(u_k) = 1.6323 + 2.0340 + 1.9739 + 1.8405 + 2.2034 = 9.6841

y_0 = 1.6323 / 9.6841 = 0.1686
y_1 = 2.0340 / 9.6841 = 0.2100
y_2 = 1.9739 / 9.6841 = 0.2038
y_3 = 1.8405 / 9.6841 = 0.1900
y_4 = 2.2034 / 9.6841 = 0.2275

y = [0.1686, 0.2100, 0.2038, 0.1900, 0.2275]
```

**Loss (averaged over both context words):**

```
J = -(1/2)[log(y_1) + log(y_3)]
  = -(1/2)[log(0.2100) + log(0.1900)]
  = -(1/2)[-1.5606 + (-1.6607)]
  = -(1/2)(-3.2213)
  = 1.6107
```

**Error vector:**

Target for context word "cat" (index 1): t^{(1)} = [0,1,0,0,0]
Target for context word "on" (index 3):  t^{(2)} = [0,0,0,1,0]

Averaged target: t_avg = (1/2)([0,1,0,0,0] + [0,0,0,1,0]) = [0, 0.5, 0, 0.5, 0]

```
e = y - t_avg = [0.1686, 0.2100 - 0.5, 0.2038, 0.1900 - 0.5, 0.2275]
              = [0.1686, -0.2900, 0.2038, -0.3100, 0.2275]
```

The negative values at indices 1 and 3 push the model to increase scores for "cat" and "on".

## 4. The Softmax Bottleneck

### 4.1 The Computational Problem

The softmax denominator requires computing v'_j^T h for **every word j** in the vocabulary:

```
P(w_O | w_I) = exp(v'_{w_O}^T h) / Σ_{j=1}^{V} exp(v'_j^T h)
```

Computing this requires O(V · N) for the forward pass and O(V · N) for the gradient —
total **O(V · N) per training example**.

### 4.2 Why This Is Impractical

For realistic settings (V = 1,000,000, N = 300):

```
Operations per training example: ~600 million floating-point ops
Training corpus: ~10 billion training examples
Total: ~6 × 10^18 operations → completely infeasible
```

### 4.3 Two Solutions

Mikolov et al. proposed two alternatives:

1. **Hierarchical Softmax**: Replace the flat softmax with a binary tree, reducing cost from
   O(V) to O(log V).

2. **Negative Sampling**: Replace the multi-class problem with binary classification,
   reducing cost from O(V) to O(K) where K ≪ V is the number of negative samples.

Both approaches modify only the **output layer** — the input embeddings and projection
remain identical. We cover each in detail below.

## 5. Hierarchical Softmax

### 5.1 Core Idea

Hierarchical softmax organizes the vocabulary as leaves of a **binary tree**. Each word is
reached by a unique path from the root, and the probability of a word is the product of
binary decisions along its path.

### 5.2 Huffman Tree Construction

Word2Vec uses a **Huffman tree**, which assigns shorter paths to more frequent words.
The construction algorithm repeatedly merges the two lowest-frequency nodes until a single
root remains.

**Example with our vocabulary:**

```
Word frequencies: the=50, cat=10, sat=5, on=40, mat=3

Merge steps:
  mat(3) + sat(5) → n1(8)
  n1(8) + cat(10) → n2(18)
  n2(18) + on(40) → n3(58)
  n3(58) + the(50) → root(108)

Resulting Huffman tree:

                    root(108)
                   /          \
               n3(58)        the(50)
              /      \          code: 1
          n2(18)    on(40)
          /    \      code: 01
      n1(8)  cat(10)
      /   \    code: 001
   mat(3) sat(5)
   code:  code:
   0000   0001

Huffman codes:
  the → 1        (1 bit  — most frequent)
  on  → 01       (2 bits)
  cat → 001      (3 bits)
  sat → 0001     (4 bits)
  mat → 0000     (4 bits — least frequent)
```

### 5.3 Path Probability

Each internal node n has a parameter vector v'_n ∈ ℝ^N. At each node, we make a binary
decision: go left (coded as 1, "positive") or right (coded as 0, "negative").

The probability of taking the left branch at node n:

```
P(left | n, h) = σ(v'_n^T h)
```

The probability of taking the right branch:

```
P(right | n, h) = 1 - σ(v'_n^T h) = σ(-v'_n^T h)
```

where σ(x) = 1/(1 + exp(-x)) is the sigmoid function.

The probability of word w is the product along its path from root to leaf:

```
P(w | w_I) = Π_{l=1}^{L(w)} σ(d_l · v'_{n_l}^T h)
```

where:
- L(w) = length of the path to word w
- n_l = the l-th internal node on the path
- d_l = +1 if the path goes left at node n_l, -1 if right
- h = hidden layer vector (same as before)

### 5.4 Step-by-Step Example: Traversing the Tree

Let's compute P(sat | h) using our Huffman tree, with h = [0.3, 0.3, 0.5] (from the CBOW
example).

The path from root to "sat" passes through: root → n3 → n2 → n1 → sat.

Suppose the internal node vectors are:

```
v'_root = [0.1, -0.2, 0.3]
v'_n3   = [0.4,  0.1, -0.1]
v'_n2   = [-0.2, 0.3, 0.2]
v'_n1   = [0.1,  0.1, -0.3]
```

At each node, we need to determine the direction. Following the Huffman code for
"sat" = 0001:

```
At root:  go left  (code bit 0 → left)  → d_1 = +1
At n3:    go left  (code bit 0 → left)  → d_2 = +1
At n2:    go left  (code bit 0 → left)  → d_3 = +1
At n1:    go right (code bit 1 → right) → d_4 = -1
```

**Step 1: Root node.**

```
v'_root^T h = 0.1(0.3) + (-0.2)(0.3) + 0.3(0.5) = 0.03 - 0.06 + 0.15 = 0.12
σ(+0.12) = 1/(1 + exp(-0.12)) = 1/(1 + 0.8869) = 0.5300
```

**Step 2: Node n3.**

```
v'_n3^T h = 0.4(0.3) + 0.1(0.3) + (-0.1)(0.5) = 0.12 + 0.03 - 0.05 = 0.10
σ(+0.10) = 1/(1 + exp(-0.10)) = 1/(1 + 0.9048) = 0.5250
```

**Step 3: Node n2.**

```
v'_n2^T h = (-0.2)(0.3) + 0.3(0.3) + 0.2(0.5) = -0.06 + 0.09 + 0.10 = 0.13
σ(+0.13) = 1/(1 + exp(-0.13)) = 1/(1 + 0.8781) = 0.5325
```

**Step 4: Node n1 (go right, so d = -1).**

```
v'_n1^T h = 0.1(0.3) + 0.1(0.3) + (-0.3)(0.5) = 0.03 + 0.03 - 0.15 = -0.09
σ(-(-0.09)) = σ(0.09) = 1/(1 + exp(-0.09)) = 1/(1 + 0.9139) = 0.5225
```

**Final probability:**

```
P(sat | h) = 0.5300 × 0.5250 × 0.5325 × 0.5225 = 0.0774
```

**Computational savings:** 4 sigmoid evaluations instead of a softmax over V words. For
V = 1,000,000, this is ~20 operations (log₂ V) instead of 1,000,000.

### 5.5 Loss and Gradient for Hierarchical Softmax

**Loss function:**

```
J = -log P(w_O | w_I) = -Σ_{l=1}^{L(w_O)} log σ(d_l · v'_{n_l}^T h)
```

**Gradient with respect to v'_{n_l}:**

Using the identity ∂/∂x [log σ(x)] = 1 - σ(x):

```
∂J/∂v'_{n_l} = (σ(d_l · v'_{n_l}^T h) - 1) · d_l · h
```

Define the local error e_l = σ(d_l · v'_{n_l}^T h) - 1. Then:

```
v'_{n_l}^{(new)} = v'_{n_l}^{(old)} - η · e_l · d_l · h
```

**Gradient with respect to h (and hence W):**

```
∂J/∂h = Σ_{l=1}^{L(w_O)} e_l · d_l · v'_{n_l}
```

This is backpropagated to update the input embeddings exactly as in the full softmax case.

---

## 6. Negative Sampling

### 6.1 The Key Insight

Negative sampling (NEG) reformulates the problem entirely. Instead of asking:

> "What is the probability of word w_O given the context?" (a V-way classification)

We ask:

> "Is the pair (w_I, w_O) a real co-occurrence or a fake one?" (binary classification)

This transforms a single V-class softmax into **one positive** and **K negative** binary
logistic regressions, reducing the per-example cost from O(V) to O(K), where K is typically
5–20.

### 6.2 The Objective Function

For a training pair (w_I, w_O) — a center word and one of its true context words — the
negative sampling objective samples K negative words {w_1, ..., w_K} from a noise
distribution P_n(w):

```
J_NEG = -log σ(v'_{w_O}^T h) - Σ_{k=1}^{K} log σ(-v'_{w_k}^T h)
```

- **First term**: Push v'_{w_O}^T h **high** so σ → 1 → "true context word should have
  high similarity with the center word."

- **Second term**: Push each v'_{w_k}^T h **low** so σ(-score) → 1 → "noise words should
  have low similarity with the center word."

### 6.3 The Noise Distribution

The noise distribution P_n(w) determines how negative samples are drawn. Mikolov et al.
found empirically that the **unigram distribution raised to the 3/4 power** works best:

```
P_n(w) = f(w)^{3/4} / Σ_{w'∈V} f(w')^{3/4}
```

where f(w) is the frequency (count) of word w in the corpus.

### 6.4 Why the 3/4 Exponent?

The exponent 3/4 is a compromise between pure unigram (α=1, over-samples frequent words)
and uniform (α=0, over-samples rare words):

```
Word frequencies: the=1000, cat=10, sat=1

With α = 1 (unigram):
  P(the) = 1000/1011 = 0.9891
  P(cat) = 10/1011   = 0.0099
  P(sat) = 1/1011    = 0.0010

With α = 3/4:
  the: 1000^{0.75} = 177.83
  cat: 10^{0.75}   = 5.623
  sat: 1^{0.75}    = 1.000
  Sum = 184.45

  P(the) = 177.83/184.45 = 0.9641
  P(cat) = 5.623/184.45  = 0.0305
  P(sat) = 1.000/184.45  = 0.0054
```

With α = 3/4, "cat" gets sampled 3× more often and "sat" 5× more often relative to pure
unigram. The value 3/4 was found empirically by Mikolov et al.

### 6.5 Gradient Derivation — Step by Step

Define l(w) = 1 if w = w_O (positive), l(w) = 0 if w ∈ {w_1,...,w_K} (negative).

The objective for a single training pair with K negative samples:

```
J = -log σ(v'_{w_O}^T h) - Σ_{k=1}^{K} log σ(-v'_{w_k}^T h)
```

We can unify this using the label l(w) = 1 for positive, 0 for negative:

```
J = -Σ_{w ∈ {w_O} ∪ {w_1,...,w_K}} [ l(w) · log σ(v'_w^T h) + (1-l(w)) · log σ(-v'_w^T h) ]
```

This is binary cross-entropy for K+1 binary classifiers.

#### 6.5.1 Gradient with respect to v'_w

**Step 1.** For the positive word w_O (l = 1):

```
∂J/∂v'_{w_O} = ∂/∂v'_{w_O} [-log σ(v'_{w_O}^T h)]
```

**Step 2.** Using ∂/∂x [-log σ(x)] = σ(x) - 1:

```
∂J/∂v'_{w_O} = (σ(v'_{w_O}^T h) - 1) · h
```

**Step 3.** For a negative word w_k (l = 0):

```
∂J/∂v'_{w_k} = ∂/∂v'_{w_k} [-log σ(-v'_{w_k}^T h)]
```

**Step 4.** Using ∂/∂x [-log σ(-x)] = σ(x):

```
∂J/∂v'_{w_k} = σ(v'_{w_k}^T h) · h
```

**Step 5.** Unifying both cases with the label l(w):

```
∂J/∂v'_w = (σ(v'_w^T h) - l(w)) · h
```

**Update rule for output embeddings:**

```
v'_w^{(new)} = v'_w^{(old)} - η · (σ(v'_w^T h) - l(w)) · h
```

#### 6.5.2 Gradient with respect to h

**Step 6.** Accumulate gradients from all K+1 words:

```
∂J/∂h = Σ_{w ∈ {w_O, w_1,...,w_K}} (σ(v'_w^T h) - l(w)) · v'_w
```

**Step 7.** For Skip-gram (h = v_{w_t}):

```
∂J/∂v_{w_t} = ∂J/∂h = Σ_{w ∈ {w_O, w_1,...,w_K}} (σ(v'_w^T h) - l(w)) · v'_w
```

**Step 8.** For CBOW (h = (1/C) Σ v_{c_i}): ∂J/∂v_{c_i} = (1/C) · ∂J/∂h

### 6.6 Numerical Example — Negative Sampling with K=2

**Setup:** Same vocabulary, V = 5, N = 3.

Training pair (Skip-gram): center = sat (index 2), context = cat (index 1).

Negative samples drawn: w_1 = the (index 0), w_2 = mat (index 4).

```
h = v_sat = W[2,:] = [0.7, 0.8, 0.9]
```

Output vectors (columns of W'):

```
v'_cat = W'[:,1] = [0.3, 0.4, 0.2]   (positive, l=1)
v'_the = W'[:,0] = [0.2, 0.1, 0.3]   (negative, l=0)
v'_mat = W'[:,4] = [0.5, 0.1, 0.4]   (negative, l=0)
```

**Step 1: Compute dot products.**

```
v'_cat^T h = 0.3(0.7) + 0.4(0.8) + 0.2(0.9) = 0.21 + 0.32 + 0.18 = 0.71
v'_the^T h = 0.2(0.7) + 0.1(0.8) + 0.3(0.9) = 0.14 + 0.08 + 0.27 = 0.49
v'_mat^T h = 0.5(0.7) + 0.1(0.8) + 0.4(0.9) = 0.35 + 0.08 + 0.36 = 0.79
```

**Step 2: Compute sigmoid values.**

```
σ(0.71) = 1/(1 + exp(-0.71)) = 1/(1 + 0.4916) = 0.6706
σ(0.49) = 1/(1 + exp(-0.49)) = 1/(1 + 0.6126) = 0.6201
σ(0.79) = 1/(1 + exp(-0.79)) = 1/(1 + 0.4538) = 0.6881
```

**Step 3: Compute loss.**

```
J = -log σ(0.71) - log σ(-0.49) - log σ(-0.79)
  = -log(0.6706) - log(1 - 0.6201) - log(1 - 0.6881)
  = -log(0.6706) - log(0.3799) - log(0.3119)
  = 0.3994 + 0.9676 + 1.1649
  = 2.5319
```

**Step 4: Compute gradients for output vectors.**

For v'_cat (positive, l = 1):

```
∂J/∂v'_cat = (σ(0.71) - 1) · h = (0.6706 - 1) · [0.7, 0.8, 0.9]
           = -0.3294 · [0.7, 0.8, 0.9]
           = [-0.2306, -0.2635, -0.2965]
```

For v'_the (negative, l = 0):

```
∂J/∂v'_the = (σ(0.49) - 0) · h = 0.6201 · [0.7, 0.8, 0.9]
           = [0.4341, 0.4961, 0.5581]
```

For v'_mat (negative, l = 0):

```
∂J/∂v'_mat = (σ(0.79) - 0) · h = 0.6881 · [0.7, 0.8, 0.9]
           = [0.4817, 0.5505, 0.6193]
```

**Step 5: Compute gradient for input vector (h = v_sat).**

```
∂J/∂h = (0.6706 - 1)·v'_cat + (0.6201 - 0)·v'_the + (0.6881 - 0)·v'_mat

      = -0.3294·[0.3, 0.4, 0.2]
        + 0.6201·[0.2, 0.1, 0.3]
        + 0.6881·[0.5, 0.1, 0.4]

      = [-0.0988, -0.1318, -0.0659]
        + [0.1240, 0.0620, 0.1860]
        + [0.3441, 0.0688, 0.2752]

      = [0.3693, -0.0010, 0.3954]
```

**Step 6: Update with learning rate η = 0.01.**

```
v'_cat^{new} = [0.3, 0.4, 0.2] - 0.01·[-0.2306, -0.2635, -0.2965]
             = [0.3023, 0.4026, 0.2030]

v'_the^{new} = [0.2, 0.1, 0.3] - 0.01·[0.4341, 0.4961, 0.5581]
             = [0.1957, 0.0950, 0.2944]

v'_mat^{new} = [0.5, 0.1, 0.4] - 0.01·[0.4817, 0.5505, 0.6193]
             = [0.4952, 0.0945, 0.3938]

v_sat^{new}  = [0.7, 0.8, 0.9] - 0.01·[0.3693, -0.0010, 0.3954]
             = [0.6963, 0.8000, 0.8960]
```

Notice: v'_cat moved **closer** to h, while v'_the and v'_mat moved **away** — attract
true context words, repel noise words.

---

## 7. Subsampling of Frequent Words

### 7.1 The Problem with Frequent Words

In natural language, a few words ("the", "a", "is") appear extremely frequently. They cause
two problems: (1) diminishing returns — after millions of occurrences, the embedding barely
changes, yet we keep computing gradients; (2) noisy signal — "the" co-occurs with almost
everything, providing little information about its neighbors.

### 7.2 The Subsampling Formula

Mikolov et al. (2013b) introduced a probabilistic discard mechanism. Each word w_i in the
training corpus is **discarded** (skipped) with probability:

```
P(discard w_i) = 1 - √(t / f(w_i))
```

where:
- f(w_i) = frequency of word w_i (i.e., count(w_i) / total_words)
- t = threshold parameter, typically 10^{-5}

Equivalently, the probability of **keeping** the word is:

```
P(keep w_i) = √(t / f(w_i))     when f(w_i) > t
P(keep w_i) = 1                  when f(w_i) ≤ t
```

### 7.3 Concrete Example

```
Corpus size: 1 billion words
t = 10^{-5}

Word        Count          f(w)        P(keep)              P(discard)
────────────────────────────────────────────────────────────────────────
the         50,000,000     0.05        √(10⁻⁵/0.05)        1 - 0.0141
                                       = √(0.0002)          = 0.9859
                                       = 0.0141             (discard 98.6%)

cat         100,000        0.0001      √(10⁻⁵/0.0001)      1 - 0.3162
                                       = √(0.1)             = 0.6838
                                       = 0.3162             (discard 68.4%)

quixotic    100            10⁻⁷        √(10⁻⁵/10⁻⁷)        Would be negative
                                       = √(100)             → P(keep) = 1
                                       = 10 → clamped to 1  (never discard)
```

### 7.4 Why This Helps

1. **Speed**: Discarding ~99% of "the" tokens dramatically reduces training examples.
2. **Quality**: Removing frequent words effectively **widens** the context window for
   remaining words. If "the" and "on" are discarded from "the fluffy cat sat on the mat",
   the effective context "fluffy cat sat mat" lets "fluffy" and "mat" fall within each
   other's window.
3. **Balance**: Rare words get proportionally more training, improving their embeddings.

---

## 8. Properties of Word2Vec Embeddings

### 8.1 Linear Analogies

The most celebrated property of Word2Vec embeddings is that **semantic relationships are
encoded as linear offsets** in the vector space:

```
v_king - v_man + v_woman ≈ v_queen
```

This works for many relationship types:

```
Syntactic:
  v_walking - v_walk + v_swim ≈ v_swimming     (progressive tense)
  v_bigger  - v_big  + v_small ≈ v_smaller     (comparative)
  v_France  - v_Paris + v_Tokyo ≈ v_Japan       (capital-country)

Semantic:
  v_king    - v_man   + v_woman ≈ v_queen       (gender)
  v_brother - v_boy   + v_girl  ≈ v_sister      (gender)
  v_CEO     - v_company + v_university ≈ v_dean (role)
```

### 8.2 Why This Works — A Geometric Explanation

The analogy property emerges because Skip-gram with negative sampling implicitly factorizes
a word-context PMI matrix (Levy & Goldberg, 2014):

```
v'_w^T · v_c ≈ PMI(w, c) - log k
```

The relationship "gender" is a consistent statistical pattern in co-occurrence data. The
**difference in log-co-occurrence patterns** between (man, woman) is similar to the
difference between (king, queen) — both pairs differ primarily in gender-related contexts.

Geometrically, independent semantic dimensions (gender, royalty, tense, etc.) correspond to
approximately orthogonal directions in the vector space:

```
                    "gender" direction
            ┌─────────────────────────────┐
            │                             │
            ▼                             ▼
    v_man ─────────────────────── v_woman
      │                             │
      │  "royalty" direction        │  "royalty" direction
      │                             │
      ▼                             ▼
    v_king ────────────────────── v_queen
            ┌─────────────────────────────┐
            │       "gender" direction    │
            └─────────────────────────────┘
```

The vector space organizes so that independent semantic dimensions correspond to
approximately orthogonal directions — a consequence of the log-linear model structure.

### 8.3 Evaluating Analogies

The standard analogy test finds d by solving:

```
d = argmax_{w ∈ V, w ∉ {a,b,c}} cos(v_w, v_b - v_a + v_c)
```

### 8.4 Limitations of Word2Vec

Despite its revolutionary impact, Word2Vec has significant limitations:

**1. Static embeddings.** Each word gets exactly one vector, regardless of context.

```
"bank" (financial) and "bank" (river) → same vector
```

This is a fundamental limitation addressed later by contextual embeddings (ELMo, BERT).

**2. Out-of-vocabulary (OOV) words.** Word2Vec cannot handle words not seen during training.
A misspelling like "catt" gets no embedding at all. This is addressed by subword models
(FastText, Chapter 3).

**3. Window-based context.** Word2Vec only captures local co-occurrence within a fixed
window. Long-range dependencies and document-level semantics are missed.

**4. Frequency bias.** Despite subsampling, frequent words still dominate the geometry of
the space. The nearest neighbors of rare words are often frequent words rather than
semantically similar rare words.

**5. Analogy limitations.** The linear analogy property is approximate and brittle — it
works best for frequent words with clear relational patterns and fails for complex or
non-linear relationships.

**6. No compositionality.** Word2Vec provides word-level embeddings only. Representing
phrases and sentences requires additional mechanisms (simple averaging, weighted averaging,
or more sophisticated models).

### 8.5 The Legacy of Word2Vec

Word2Vec established the paradigm of pre-training embeddings on unlabeled data, demonstrated
that simple shallow models can learn rich representations, and revealed that vector
arithmetic captures semantic structure. It inspired GloVe, FastText, ELMo, and ultimately
the transformer-based models (BERT, GPT) that dominate NLP today.

The journey from Word2Vec to modern language models is one of increasing context — but the
fundamental insight that distributional patterns encode meaning as geometry remains the
foundation of it all.

---

## Summary

| Concept              | Key Formula                                                    | Complexity |
|----------------------|----------------------------------------------------------------|------------|
| CBOW forward         | h = (1/C) Σ W^T x_c; u = W'^T h; y = softmax(u)             | O(V·N)     |
| Skip-gram forward    | h = W^T x_t; u = W'^T h; y = softmax(u)                      | O(V·N)     |
| Full softmax         | P(w) = exp(u_w) / Σ exp(u_j)                                  | O(V)       |
| Hierarchical softmax | P(w) = Π σ(d_l · v'^T_{n_l} h)                                | O(log V)   |
| Negative sampling    | J = -log σ(v'^T_{w_O} h) - Σ log σ(-v'^T_{w_k} h)            | O(K)       |
| Noise distribution   | P_n(w) = f(w)^{3/4} / Z                                       | —          |
| Subsampling          | P(discard) = 1 - √(t/f(w))                                    | —          |
| Analogy              | v_b - v_a + v_c ≈ v_d                                         | —          |

---

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word
   Representations in Vector Space.* arXiv:1301.3781.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed
   Representations of Words and Phrases and their Compositionality.* NeurIPS 2013.
3. Goldberg, Y., & Levy, O. (2014). *word2vec Explained.* arXiv:1402.3722.
4. Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix
   Factorization.* NeurIPS 2014.
5. Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738.
6. Bengio, Y., et al. (2003). *A Neural Probabilistic Language Model.* JMLR.

---

*Next chapter: [Chapter 3 — GloVe: Global Vectors for Word Representation](03-glove.md)*
