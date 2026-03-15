# Chapter 3: GloVe — Global Vectors for Word Representation

## 3.1 Introduction

In 2014, Jeffrey Pennington, Richard Socher, and Christopher D. Manning at Stanford
introduced **GloVe** (Global Vectors for Word Representation), a model that elegantly
bridges two historically separate paradigms in distributional semantics:

- **Count-based methods** (LSA, HAL, COALS): Build a global co-occurrence matrix from
  the entire corpus, then apply dimensionality reduction (e.g., SVD). These methods
  capture corpus-wide statistics efficiently but struggle with analogy tasks and
  fine-grained word similarity.

- **Prediction-based methods** (Word2Vec's Skip-gram and CBOW): Train a neural network
  to predict context words from target words (or vice versa) using local context windows.
  These methods excel at analogy tasks and capture linear substructures in the embedding
  space, but they never directly operate on global statistics — they scan the corpus
  window by window.

GloVe's central thesis is that **neither family alone is optimal**. Count-based methods
have access to global information but waste it through crude dimensionality reduction.
Prediction-based methods capture rich patterns but are limited to local windows and
must rely on stochastic sampling to indirectly absorb global statistics.

GloVe resolves this tension by:

1. Computing a **global co-occurrence matrix** (like count-based methods)
2. Defining a **weighted least-squares regression** objective directly on that matrix
3. Deriving the objective from first principles about **ratios of co-occurrence
   probabilities** — the key insight that makes the model work

The result is a model that trains on aggregated global statistics (not raw text), is
computationally efficient, and produces embeddings that rival or exceed Word2Vec on
standard benchmarks.

> **Reference**: Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global
> Vectors for Word Representation*. Proceedings of EMNLP 2014, pp. 1532–1543.

---

## 3.2 The Co-occurrence Matrix

### 3.2.1 Definitions

GloVe begins with a **word-word co-occurrence matrix** $X$, where:

$$
X_{ij} = \text{number of times word } j \text{ appears in the context of word } i
$$

The "context" is defined by a **symmetric window** of size $L$ around each occurrence
of word $i$ in the corpus. If $L = 5$, we look 5 words to the left and 5 words to the
right of each occurrence of word $i$.

Optionally, GloVe applies **distance weighting**: a context word at distance $d$ from
the target contributes $1/d$ to the count rather than 1. This gives more weight to
closer context words.

From $X$, we derive:

- **Row sum** (total context count for word $i$):

$$
X_i = \sum_{k=1}^{|V|} X_{ik}
$$

- **Co-occurrence probability** (probability that word $j$ appears in the context of
  word $i$):

$$
P_{ij} = P(j \mid i) = \frac{X_{ij}}{X_i}
$$

Note that in general $X_{ij} = X_{ji}$ when using a symmetric window (the matrix is
symmetric), but $P_{ij} \neq P_{ji}$ because $X_i \neq X_j$ in general.

### 3.2.2 Step-by-Step Construction from a Sample Corpus

Consider a tiny corpus with vocabulary $V = \{\text{the, cat, sat, on, mat}\}$:

```
Corpus: "the cat sat on the mat"
```

Using a **symmetric window of size $L = 1$** (one word to each side), we scan each
token and record which words appear in its context:

| Target word | Position | Left context | Right context |
|-------------|----------|-------------|---------------|
| the         | 1        | —           | cat           |
| cat         | 2        | the         | sat           |
| sat         | 3        | cat         | on            |
| on          | 4        | sat         | the           |
| the         | 5        | on          | mat           |
| mat         | 6        | the         | —             |

Now we tally co-occurrences. For each (target, context) pair, increment $X_{ij}$:

**Step 1**: Position 1 — target = "the", context = {"cat"}
- $X_{\text{the, cat}} += 1$

**Step 2**: Position 2 — target = "cat", context = {"the", "sat"}
- $X_{\text{cat, the}} += 1$, $X_{\text{cat, sat}} += 1$

**Step 3**: Position 3 — target = "sat", context = {"cat", "on"}
- $X_{\text{sat, cat}} += 1$, $X_{\text{sat, on}} += 1$

**Step 4**: Position 4 — target = "on", context = {"sat", "the"}
- $X_{\text{on, sat}} += 1$, $X_{\text{on, the}} += 1$

**Step 5**: Position 5 — target = "the", context = {"on", "mat"}
- $X_{\text{the, on}} += 1$, $X_{\text{the, mat}} += 1$

**Step 6**: Position 6 — target = "mat", context = {"the"}
- $X_{\text{mat, the}} += 1$

Since we use a symmetric window, we symmetrize: $X_{ij} = X_{ji}$. The resulting
co-occurrence matrix:

$$
X = \begin{pmatrix}
     & \text{the} & \text{cat} & \text{sat} & \text{on} & \text{mat} \\
\text{the} & 0 & 1 & 0 & 1 & 1 \\
\text{cat} & 1 & 0 & 1 & 0 & 0 \\
\text{sat} & 0 & 1 & 0 & 1 & 0 \\
\text{on}  & 1 & 0 & 1 & 0 & 0 \\
\text{mat} & 1 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

Wait — let's be precise. Summing both directions (target→context and context→target
from the symmetric window), the actual counts after symmetrization are:

| Pair          | Count from scanning | Symmetrized $X_{ij}$ |
|---------------|--------------------|-----------------------|
| (the, cat)    | 1 + 1 = 2         | 2                     |
| (the, on)     | 1 + 1 = 2         | 2                     |
| (the, mat)    | 1 + 1 = 2         | 2                     |
| (cat, sat)    | 1 + 1 = 2         | 2                     |
| (sat, on)     | 1 + 1 = 2         | 2                     |

All other pairs have $X_{ij} = 0$.

The corrected symmetric matrix:

$$
X = \begin{pmatrix}
     & \text{the} & \text{cat} & \text{sat} & \text{on} & \text{mat} \\
\text{the} & 0 & 2 & 0 & 2 & 2 \\
\text{cat} & 2 & 0 & 2 & 0 & 0 \\
\text{sat} & 0 & 2 & 0 & 2 & 0 \\
\text{on}  & 2 & 0 & 2 & 0 & 0 \\
\text{mat} & 2 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

Row sums:

$$
X_{\text{the}} = 6, \quad X_{\text{cat}} = 4, \quad X_{\text{sat}} = 4, \quad
X_{\text{on}} = 4, \quad X_{\text{mat}} = 2
$$

Co-occurrence probabilities (selected):

$$
P(\text{cat} \mid \text{the}) = \frac{2}{6} = 0.333, \quad
P(\text{mat} \mid \text{the}) = \frac{2}{6} = 0.333
$$

$$
P(\text{the} \mid \text{cat}) = \frac{2}{4} = 0.500, \quad
P(\text{sat} \mid \text{cat}) = \frac{2}{4} = 0.500
$$

---

## 3.3 The Key Insight: Ratios of Co-occurrence Probabilities

### 3.3.1 Why Raw Probabilities Are Not Enough

Consider two words, $i = \text{ice}$ and $j = \text{steam}$. We want to learn
embeddings that capture the relationship between them. Looking at raw co-occurrence
probabilities with various probe words $k$:

| Probe word $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | Ratio $P(k \mid \text{ice}) / P(k \mid \text{steam})$ |
|-----------------|----------------------|------------------------|------------------------------------------------------|
| solid           | $3.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$  | **17.7** (large: "solid" relates to "ice")           |
| gas             | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$  | **0.085** (small: "gas" relates to "steam")          |
| water           | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$  | **1.36** (≈1: "water" relates to both)               |
| fashion         | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$  | **0.96** (≈1: "fashion" relates to neither)          |

*(Values from Pennington et al., 2014, Table 1)*

### 3.3.2 What the Ratios Tell Us

The ratio $P_{ik}/P_{jk}$ cleanly separates four cases:

1. **$k$ is related to $i$ but not $j$** → ratio $\gg 1$
   - "solid" co-occurs much more with "ice" than "steam" → ratio ≈ 17.7

2. **$k$ is related to $j$ but not $i$** → ratio $\ll 1$
   - "gas" co-occurs much more with "steam" than "ice" → ratio ≈ 0.085

3. **$k$ is related to both $i$ and $j$** → ratio $\approx 1$
   - "water" is associated with both "ice" and "steam" → ratio ≈ 1.36

4. **$k$ is related to neither $i$ nor $j$** → ratio $\approx 1$
   - "fashion" has no special relationship with either → ratio ≈ 0.96

Raw probabilities $P(k \mid \text{ice})$ and $P(k \mid \text{steam})$ individually
are noisy and hard to interpret — they depend on word frequency, corpus size, and
many confounds. But their **ratio** cancels out much of this noise and isolates the
**discriminative** semantic signal.

This is the core insight: **the ratio of co-occurrence probabilities is a more
principled carrier of meaning than the probabilities themselves**.

### 3.3.3 Why This Matters for Embeddings

If we want word vectors to encode meaning, we should design them so that some function
of the vectors **reproduces these ratios**. This is exactly what GloVe does — it
derives an objective function by requiring that the dot product of word vectors
approximates the logarithm of co-occurrence probabilities, which means that
**differences** of dot products approximate the logarithm of **ratios**.

---

## 3.4 Deriving the GloVe Objective

This section walks through the full derivation, step by step, from the ratio insight
to the final objective function. This is one of the most elegant derivations in NLP.

### 3.4.1 Setup

We want to learn:
- **Word vectors** $w_i \in \mathbb{R}^d$ for each word $i$ (as a target)
- **Context vectors** $\tilde{w}_k \in \mathbb{R}^d$ for each word $k$ (as a context)

We seek a function $F$ such that:

$$
F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}
$$

where $P_{ik} = P(k \mid i) = X_{ik}/X_i$ is the co-occurrence probability.

The derivation proceeds through a series of constraints that progressively simplify
$F$ until we arrive at a concrete, trainable objective.

### 3.4.2 The Derivation

**Step 1: Require $F$ to depend on the difference $w_i - w_j$.**

Since the ratio $P_{ik}/P_{jk}$ encodes how word $k$ discriminates between words $i$
and $j$, the function $F$ should capture the "contrast" between $i$ and $j$. In vector
space, the natural way to express contrast is through the **difference**:

$$
F(w_i - w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}
$$

This reduces the three-argument function to a two-argument function.

**Step 2: Convert the vector arguments to a scalar via the dot product.**

The left-hand side has a vector argument $(w_i - w_j) \in \mathbb{R}^d$ and another
vector $\tilde{w}_k \in \mathbb{R}^d$, but the right-hand side is a scalar. We need
to map vectors to a scalar. The simplest linear way is the **dot product**:

$$
F\!\left((w_i - w_j)^T \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}}
$$

Now $F$ is a function from $\mathbb{R} \to \mathbb{R}$.

**Step 3: Require $F$ to be a homomorphism between $(\mathbb{R}, +)$ and
$(\mathbb{R}_{>0}, \times)$.**

Expanding the dot product:

$$
(w_i - w_j)^T \tilde{w}_k = w_i^T \tilde{w}_k - w_j^T \tilde{w}_k
$$

And the right-hand side is a ratio:

$$
\frac{P_{ik}}{P_{jk}} = \frac{P_{ik}}{1} \cdot \frac{1}{P_{jk}}
$$

We want $F$ to satisfy:

$$
F(a - b) = \frac{F(a)}{F(b)}
$$

More generally, we require $F$ to be a **group homomorphism** from addition to
multiplication:

$$
F(a + b) = F(a) \cdot F(b)
$$

The unique continuous solution to this functional equation is the **exponential
function**:

$$
F(x) = \exp(x)
$$

**Step 4: Apply $F = \exp$ to get the core equation.**

Substituting back:

$$
\exp\!\left(w_i^T \tilde{w}_k - w_j^T \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}}
$$

This factors as:

$$
\frac{\exp(w_i^T \tilde{w}_k)}{\exp(w_j^T \tilde{w}_k)} = \frac{P_{ik}}{P_{jk}}
$$

For this to hold for all $i, j, k$, we need each factor to match independently:

$$
\exp(w_i^T \tilde{w}_k) = \lambda \cdot P_{ik}
$$

for some constant $\lambda$ (which may depend on $k$ but cancels in the ratio). Taking
logarithms:

$$
w_i^T \tilde{w}_k = \log(P_{ik}) + \log(\lambda)
$$

**Step 5: Expand $\log(P_{ik})$ and absorb terms into biases.**

$$
w_i^T \tilde{w}_k = \log\!\left(\frac{X_{ik}}{X_i}\right) + \text{const}
$$

$$
w_i^T \tilde{w}_k = \log(X_{ik}) - \log(X_i) + \text{const}
$$

The term $\log(X_i)$ depends only on $i$, not on $k$. We absorb it (and the constant)
into **bias terms**:

- $b_i$ absorbs $-\log(X_i)$ and any $i$-dependent constant
- $\tilde{b}_k$ absorbs any $k$-dependent constant (from $\log(\lambda)$)

**Step 6: Arrive at the final equation.**

$$
\boxed{w_i^T \tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik})}
$$

This is the **GloVe equation**: the dot product of a word vector and a context vector,
plus bias terms, should equal the logarithm of their co-occurrence count.

### 3.4.3 Summary of the Derivation

| Step | Constraint | Result |
|------|-----------|--------|
| 1 | $F$ depends on difference $w_i - w_j$ | $F(w_i - w_j, \tilde{w}_k) = P_{ik}/P_{jk}$ |
| 2 | Reduce to scalar via dot product | $F((w_i - w_j)^T \tilde{w}_k) = P_{ik}/P_{jk}$ |
| 3 | $F$ is a homomorphism $(+) \to (\times)$ | $F = \exp$ |
| 4 | Apply $\exp$ and match terms | $\exp(w_i^T \tilde{w}_k) = \lambda P_{ik}$ |
| 5 | Take log, absorb constants into biases | $w_i^T \tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik})$ |

### 3.4.4 A Note on Symmetry

The GloVe equation has an asymmetry: $w_i$ and $\tilde{w}_k$ play different roles
(target vs. context). However, the co-occurrence matrix $X$ is symmetric
($X_{ij} = X_{ji}$). The derivation above breaks this symmetry when we define
$P_{ik} = X_{ik}/X_i$ (dividing by the row sum of $i$, not $k$).

Pennington et al. note that the final model should be approximately symmetric. They
restore symmetry at the end by **averaging** the two sets of vectors (see Section 3.6).

---

## 3.5 The Weighted Least Squares Objective

### 3.5.1 From Equation to Loss Function

The GloVe equation $w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})$ should hold
for all word pairs $(i, j)$ with $X_{ij} > 0$. In practice, it will hold only
approximately. We minimize the squared error:

$$
J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

where $f(X_{ij})$ is a **weighting function** and the sum runs over all pairs with
$X_{ij} > 0$ (we skip pairs with zero co-occurrence since $\log(0)$ is undefined).

### 3.5.2 The Weighting Function

Not all co-occurrences are equally informative. Very frequent co-occurrences (like
"the" appearing with almost everything) should not dominate the objective, and very
rare co-occurrences are noisy. The weighting function $f$ addresses this:

$$
f(x) = \begin{cases}
(x / x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{if } x \geq x_{\max}
\end{cases}
$$

Or more compactly:

$$
\boxed{f(x) = \min\!\left(\left(\frac{x}{x_{\max}}\right)^\alpha,\; 1\right)}
$$

**Typical hyperparameters**: $\alpha = 3/4$, $x_{\max} = 100$.

**Properties of $f$:**

1. **$f(0) = 0$**: Zero co-occurrences contribute nothing (and we skip them anyway
   since $\log(0)$ is undefined).

2. **$f$ is non-decreasing**: More frequent co-occurrences get more weight, up to a
   cap.

3. **$f$ is capped at 1**: Co-occurrences above $x_{\max}$ all get the same weight,
   preventing very common pairs from dominating.

4. **$\alpha = 3/4 < 1$**: The sub-linear exponent means that the weight grows
   **slower than linearly** with count. This is crucial — it prevents high-frequency
   function words from overwhelming the objective.

### 3.5.3 Why $\alpha = 3/4$?

Pennington et al. found empirically that $\alpha = 3/4$ outperforms $\alpha = 1$
(linear weighting). The intuition is similar to why Word2Vec uses subsampling of
frequent words: we want to **down-weight** very common co-occurrences relative to
their raw frequency.

Numerical example of the weighting function with $x_{\max} = 100$, $\alpha = 3/4$:

| $X_{ij}$ | $x / x_{\max}$ | $(x / x_{\max})^{3/4}$ | $f(X_{ij})$ |
|-----------|----------------|------------------------|-------------|
| 1         | 0.01           | 0.0316                 | 0.032       |
| 5         | 0.05           | 0.1057                 | 0.106       |
| 10        | 0.10           | 0.1778                 | 0.178       |
| 25        | 0.25           | 0.3536                 | 0.354       |
| 50        | 0.50           | 0.5946                 | 0.595       |
| 75        | 0.75           | 0.8059                 | 0.806       |
| 100       | 1.00           | 1.0000                 | 1.000       |
| 500       | 5.00           | —                      | 1.000       |

Notice how $f(1) = 0.032$ — a pair that co-occurs once gets only 3.2% of the maximum
weight. A pair co-occurring 50 times gets about 59.5%. This is a strong compression
of the dynamic range.

### 3.5.4 Why Weighted Least Squares?

Three reasons for this particular loss:

1. **Log-space regression**: By fitting $\log(X_{ij})$ rather than $X_{ij}$ directly,
   we work in a compressed space where the dynamic range (from 1 to millions) is
   manageable.

2. **Weighting prevents degenerate solutions**: Without $f$, the model would spend
   most of its capacity fitting the most frequent pairs (like "the, of") at the
   expense of rarer but more informative pairs.

3. **Computational efficiency**: The sum is over non-zero entries of $X$ only. For a
   large corpus, $X$ is sparse — most word pairs never co-occur. The number of
   non-zero entries is $\mathcal{O}(|C|)$ where $|C|$ is the corpus size, much less
   than $|V|^2$.

### 3.5.5 Gradient Derivation

To train GloVe, we need the gradients of $J$ with respect to all parameters. Let us
define the residual for a single pair $(i, j)$:

$$
e_{ij} = w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}
$$

The contribution of pair $(i, j)$ to the loss is:

$$
J_{ij} = f(X_{ij}) \cdot e_{ij}^2
$$

**Step 1: Gradient with respect to $w_i$ (target word vector).**

$$
\frac{\partial J_{ij}}{\partial w_i} = f(X_{ij}) \cdot 2 \, e_{ij} \cdot \frac{\partial e_{ij}}{\partial w_i}
$$

Since $e_{ij} = w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}$:

$$
\frac{\partial e_{ij}}{\partial w_i} = \tilde{w}_j
$$

Therefore:

$$
\boxed{\frac{\partial J_{ij}}{\partial w_i} = 2 \, f(X_{ij}) \, e_{ij} \, \tilde{w}_j}
$$

**Step 2: Gradient with respect to $\tilde{w}_j$ (context word vector).**

By the same logic:

$$
\frac{\partial e_{ij}}{\partial \tilde{w}_j} = w_i
$$

$$
\boxed{\frac{\partial J_{ij}}{\partial \tilde{w}_j} = 2 \, f(X_{ij}) \, e_{ij} \, w_i}
$$

**Step 3: Gradient with respect to $b_i$ (target bias).**

$$
\frac{\partial e_{ij}}{\partial b_i} = 1
$$

$$
\boxed{\frac{\partial J_{ij}}{\partial b_i} = 2 \, f(X_{ij}) \, e_{ij}}
$$

**Step 4: Gradient with respect to $\tilde{b}_j$ (context bias).**

$$
\frac{\partial e_{ij}}{\partial \tilde{b}_j} = 1
$$

$$
\boxed{\frac{\partial J_{ij}}{\partial \tilde{b}_j} = 2 \, f(X_{ij}) \, e_{ij}}
$$

**Step 5: Full gradient (summing over all pairs).**

The total gradient for $w_i$ sums over all context words $j$ with $X_{ij} > 0$:

$$
\frac{\partial J}{\partial w_i} = \sum_{j: X_{ij} > 0} 2 \, f(X_{ij}) \, e_{ij} \, \tilde{w}_j
$$

Similarly for the other parameters.

### 3.5.6 Numerical Example of a Gradient Step

Suppose we have:
- $w_i = [0.5, -0.3]$, $\tilde{w}_j = [0.2, 0.8]$
- $b_i = 0.1$, $\tilde{b}_j = -0.2$
- $X_{ij} = 10$, $x_{\max} = 100$, $\alpha = 3/4$

**Step 1**: Compute the weighting.

$$
f(10) = \left(\frac{10}{100}\right)^{3/4} = (0.1)^{0.75} = 0.1778
$$

**Step 2**: Compute the dot product.

$$
w_i^T \tilde{w}_j = (0.5)(0.2) + (-0.3)(0.8) = 0.10 - 0.24 = -0.14
$$

**Step 3**: Compute the residual.

$$
e_{ij} = -0.14 + 0.1 + (-0.2) - \log(10) = -0.14 + 0.1 - 0.2 - 2.3026 = -2.5426
$$

**Step 4**: Compute the gradient for $w_i$.

$$
\frac{\partial J_{ij}}{\partial w_i} = 2 \times 0.1778 \times (-2.5426) \times [0.2, 0.8]
$$

$$
= 2 \times 0.1778 \times (-2.5426) \times [0.2, 0.8]
$$

$$
= -0.9042 \times [0.2, 0.8] = [-0.1808, -0.7234]
$$

**Step 5**: Update $w_i$ (with learning rate $\eta = 0.05$, ignoring AdaGrad for now).

$$
w_i^{\text{new}} = [0.5, -0.3] - 0.05 \times [-0.1808, -0.7234]
$$

$$
= [0.5 + 0.00904, -0.3 + 0.03617] = [0.509, -0.264]
$$

The vector moved in a direction that will reduce the residual — making the dot product
larger (more positive) to better approximate $\log(10) = 2.3026$.

---

## 3.6 Training Details

### 3.6.1 AdaGrad Optimization

GloVe uses **AdaGrad** (Adaptive Gradient) rather than vanilla SGD. AdaGrad maintains
a per-parameter sum of squared gradients and scales the learning rate inversely:

For each parameter $\theta$ (which could be any component of $w_i$, $\tilde{w}_j$,
$b_i$, or $\tilde{b}_j$):

**Initialize**: $G_\theta = 0$ (accumulated squared gradient)

**For each training pair** $(i, j)$:

1. Compute gradient $g_\theta = \partial J_{ij} / \partial \theta$
2. Accumulate: $G_\theta \leftarrow G_\theta + g_\theta^2$
3. Update: $\theta \leftarrow \theta - \frac{\eta}{\sqrt{G_\theta + \epsilon}} \, g_\theta$

where $\eta$ is the initial learning rate (typically $\eta = 0.05$) and
$\epsilon \approx 10^{-8}$ prevents division by zero.

**Why AdaGrad for GloVe?**

- **Sparse updates**: Most word pairs have zero co-occurrence. For the non-zero pairs,
  some words appear in many pairs (frequent words) and some in few (rare words).
  AdaGrad automatically gives larger effective learning rates to parameters that are
  updated infrequently (rare words) and smaller rates to frequently updated parameters
  (common words).

- **No learning rate schedule needed**: AdaGrad's adaptive rates eliminate the need to
  manually decay the learning rate.

- **Convergence**: For the convex-like landscape of GloVe's weighted least squares
  objective, AdaGrad converges reliably.

### 3.6.2 Training Procedure

The full training procedure:

1. **Build the co-occurrence matrix** $X$ by scanning the corpus with a window of
   size $L$ (typically $L = 10$, meaning 10 words to each side).

2. **Initialize** all $w_i$, $\tilde{w}_j$ randomly from $\text{Uniform}(-0.5, 0.5)$
   scaled by $1/d$, and all biases to zero.

3. **Shuffle** the list of non-zero entries $(i, j, X_{ij})$.

4. **For each epoch** (typically 50–100 epochs):
   - For each non-zero pair $(i, j)$:
     - Compute $e_{ij} = w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij})$
     - Compute $\text{weight} = f(X_{ij})$
     - Compute gradients (Section 3.5.5)
     - Update all four parameters using AdaGrad

5. **Compute final vectors**: $w_{\text{final},i} = w_i + \tilde{w}_i$

### 3.6.3 Why Sum Both Vectors? $w_{\text{final}} = w + \tilde{w}$

GloVe learns two sets of vectors: target vectors $w_i$ and context vectors
$\tilde{w}_i$. Since the co-occurrence matrix is symmetric ($X_{ij} = X_{ji}$), the
roles of "target" and "context" are interchangeable — the distinction is an artifact
of the optimization, not the data.

Both $w_i$ and $\tilde{w}_i$ contain useful information, and they are trained to
capture complementary aspects of the same word. Pennington et al. found that
**summing** them:

$$
w_{\text{final},i} = w_i + \tilde{w}_i
$$

consistently outperforms using either $w_i$ or $\tilde{w}_i$ alone. The intuition:

- $w_i$ is optimized to predict context words given word $i$ as target
- $\tilde{w}_i$ is optimized to predict target words given word $i$ as context
- Their sum captures both perspectives, providing a richer representation
- The sum also acts as a form of **ensembling**, reducing variance

In practice, the improvement from summing is modest but consistent (1–2% on analogy
tasks).

### 3.6.4 Hyperparameters and Their Effects

| Hyperparameter | Typical Value | Effect |
|---------------|--------------|--------|
| Vector dimension $d$ | 50, 100, 200, 300 | Higher $d$ captures more nuance but costs more memory and compute. Returns diminish beyond 300. |
| Window size $L$ | 5–10 | Larger windows capture more topical/associative similarity; smaller windows capture more syntactic similarity. |
| $x_{\max}$ | 100 | Cap for the weighting function. Values 50–200 work similarly. |
| $\alpha$ | 3/4 | Exponent for weighting. 3/4 consistently outperforms 1 (linear). |
| Learning rate $\eta$ | 0.05 | Initial AdaGrad learning rate. |
| Epochs | 50–100 | More epochs help, with diminishing returns after ~50. |
| Min count | 5–10 | Words appearing fewer times are discarded from vocabulary. |

**Scaling behavior**: Pennington et al. showed that GloVe performance improves
log-linearly with corpus size and vector dimension, up to a point. On the word analogy
task:

- 6B tokens, 300d → 75% accuracy
- 42B tokens, 300d → 81.9% accuracy
- 840B tokens, 300d → 83.6% accuracy (the "Common Crawl" model)

---

## 3.7 Comparison with Word2Vec

### 3.7.1 The Levy & Goldberg Connection (2014)

Shortly after GloVe's publication, Omer Levy and Yoav Goldberg published a landmark
analysis showing that Word2Vec's Skip-gram with Negative Sampling (SGNS) is
**implicitly factorizing a word-context matrix** whose entries are:

$$
M_{ij}^{\text{SGNS}} = w_i^T \tilde{w}_j = \text{PMI}(i, j) - \log k
$$

where PMI is the **Pointwise Mutual Information**:

$$
\text{PMI}(i, j) = \log \frac{P(i, j)}{P(i) \, P(j)} = \log \frac{X_{ij} \cdot |D|}{X_i \cdot X_j}
$$

and $k$ is the number of negative samples.

> **Reference**: Levy, O. & Goldberg, Y. (2014). *Neural Word Embedding as Implicit
> Matrix Factorization*. NeurIPS 2014.

### 3.7.2 Comparing the Implicit Matrices

Let's compare what each model's dot product approximates:

**GloVe**:

$$
w_i^T \tilde{w}_j \approx \log(X_{ij}) - b_i - \tilde{b}_j
$$

If we absorb the biases optimally, this is approximately:

$$
w_i^T \tilde{w}_j \approx \log(X_{ij})
$$

**Skip-gram (SGNS)**:

$$
w_i^T \tilde{w}_j \approx \text{PMI}(i, j) - \log k
$$

Expanding PMI:

$$
= \log(X_{ij}) + \log(|D|) - \log(X_i) - \log(X_j) - \log(k)
$$

The key difference: **SGNS implicitly subtracts $\log(X_i) + \log(X_j)$** (the
marginal frequencies), while **GloVe absorbs these into bias terms**. When GloVe's
biases are well-trained, they approximate $b_i \approx \log(X_i)$ and
$\tilde{b}_j \approx \log(X_j)$, making the two models factor **essentially the same
matrix**.

### 3.7.3 Side-by-Side Comparison

| Aspect | GloVe | Word2Vec (SGNS) |
|--------|-------|-----------------|
| **Training data** | Pre-computed co-occurrence matrix $X$ | Raw corpus (scanned window by window) |
| **Objective** | Weighted least squares on $\log(X_{ij})$ | Negative sampling (binary classification) |
| **Implicit matrix** | $\log(X_{ij})$ (with biases) | PMI shifted by $-\log k$ |
| **Global statistics** | Explicitly uses global counts | Implicitly captures them via SGD |
| **Training speed** | Fast: iterates over non-zero $X_{ij}$ entries | Depends on corpus size (each token scanned) |
| **Memory** | Must store $X$ (sparse, but can be large) | Streams through corpus (low memory) |
| **Parallelism** | Easily parallelizable (independent pairs) | Parallelizable but with shared parameters |
| **Rare words** | Can struggle (low counts → noisy $\log X_{ij}$) | Better with subsampling and negative sampling |
| **Preprocessing** | Requires co-occurrence matrix construction | Minimal preprocessing |
| **Analogy tasks** | Slightly better on some benchmarks | Slightly better on others |
| **Similarity tasks** | Comparable | Comparable |

### 3.7.4 When to Use Which?

**Choose GloVe when:**
- You have a fixed corpus and want reproducible results (no stochastic sampling)
- You want to leverage global statistics explicitly
- Your corpus fits in memory for co-occurrence matrix construction
- You value a clean, interpretable objective function

**Choose Word2Vec (SGNS) when:**
- Your corpus is very large or streaming (can't build full co-occurrence matrix)
- You want to incrementally update embeddings as new data arrives
- Memory is constrained (no need to store the co-occurrence matrix)
- You need embeddings for a domain with many rare words (negative sampling handles
  these more gracefully)

**In practice**: The performance difference between well-tuned GloVe and well-tuned
Word2Vec is small. Levy, Goldberg, and Dagan (2015) showed that **hyperparameter
tuning matters more than the choice of algorithm**. The same hyperparameters
(window size, dimension, subsampling) affect both models similarly.

> **Reference**: Levy, O., Goldberg, Y., & Dagan, I. (2015). *Improving Distributional
> Similarity with Lessons Learned from Word Embeddings*. TACL, 3, 211–225.

---

---

## 3.8 Computational Complexity and Practical Considerations

### 3.8.1 Complexity

- **Co-occurrence construction**: $\mathcal{O}(|C| \cdot L)$ where $|C|$ is corpus
  size and $L$ is window size.
- **Training per epoch**: $\mathcal{O}(\text{nnz}(X) \cdot d)$ where $\text{nnz}(X)$
  is the number of non-zero entries. GloVe is embarrassingly parallel — each $(i, j)$
  pair can be processed independently.
- **Space**: The sparse matrix $X$ plus $4 \cdot |V| \cdot d$ floats for vectors and
  AdaGrad accumulators.

### 3.8.2 Pre-trained GloVe Vectors

Stanford provides pre-trained vectors at [nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/):

| Corpus | Tokens | Vocab | Dimensions | File size |
|--------|--------|-------|------------|-----------|
| Wikipedia 2014 + Gigaword 5 | 6B | 400K | 50, 100, 200, 300 | 822 MB (300d) |
| Common Crawl (42B) | 42B | 1.9M | 300 | 5.0 GB |
| Common Crawl (840B) | 840B | 2.2M | 300 | 5.6 GB |
| Twitter | 27B | 1.2M | 25, 50, 100, 200 | 1.4 GB (200d) |

---

## 3.9 Limitations

1. **Static embeddings**: Each word gets one vector regardless of context. "Bank"
   (financial) and "bank" (river) share the same embedding. This motivated
   contextualized models like ELMo and BERT.

2. **Out-of-vocabulary words**: GloVe cannot produce embeddings for unseen words.
   Subword models like FastText address this.

3. **Memory for co-occurrence matrix**: For very large vocabularies, storing even the
   sparse matrix can be challenging.

4. **Corpus bias**: Embeddings reflect the biases of the training corpus.

---

## 3.10 Summary

GloVe's contribution is both theoretical and practical:

**Theoretically**, it shows that the ratio of co-occurrence probabilities is the right
quantity to model, and derives a clean objective function from first principles. The
derivation — from the ratio insight through the homomorphism argument to the weighted
least squares loss — is one of the most elegant in NLP.

**Practically**, GloVe produces high-quality embeddings that are:
- Fast to train (operates on sparse co-occurrence matrix, not raw text)
- Easily parallelizable
- Competitive with or superior to Word2Vec on standard benchmarks
- Available as high-quality pre-trained vectors

The Levy & Goldberg analysis revealed that GloVe and Word2Vec are more similar than
different — both implicitly factorize matrices related to PMI. The real lesson is that
**the distributional hypothesis is powerful**, and multiple algorithmic approaches can
extract similar structure from co-occurrence statistics.

In the next chapter, we will explore **FastText**, which extends Word2Vec with subword
information, addressing the out-of-vocabulary limitation that affects both Word2Vec and
GloVe.

---

## References

1. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for
   Word Representation. *Proceedings of the 2014 Conference on Empirical Methods in
   Natural Language Processing (EMNLP)*, 1532–1543.

2. Levy, O., & Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix
   Factorization. *Advances in Neural Information Processing Systems (NeurIPS)*, 27.

3. Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving Distributional Similarity
   with Lessons Learned from Word Embeddings. *Transactions of the Association for
   Computational Linguistics (TACL)*, 3, 211–225.

4. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed
   Representations of Words and Phrases and their Compositionality. *NeurIPS 2013*.

5. Dingwall, N., & Potts, C. (2018). Mittens: An Extension of GloVe for Learning
   Domain-Specialized Representations. *Proceedings of NAACL-HLT 2018*.


---

*Next chapter: [Chapter 4 — FastText: Subword Embeddings](04-fasttext.md)*
