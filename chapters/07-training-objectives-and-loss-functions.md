# Chapter 7: Training Objectives and Loss Functions for Embedding Learning

## 7.0 Introduction

The quality of learned embeddings is fundamentally determined by the objective function
used during training. An embedding function $f: \mathcal{X} \rightarrow \mathbb{R}^d$ maps
input data from an arbitrary space $\mathcal{X}$ into a $d$-dimensional vector space. The
central question of **metric learning** is: how do we define a loss function $\mathcal{L}$
such that, after optimization, the geometry of the embedding space reflects the semantic
structure of the data?

This chapter provides rigorous mathematical treatment of the major loss functions that
have shaped modern embedding learning.

---

## 7.1 Overview of Metric Learning

### 7.1.1 The Fundamental Goal

Given a dataset $\{x_i\}_{i=1}^{N}$ with a notion of semantic similarity, metric learning
seeks to learn an embedding function $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d$
(parameterized by $\theta$) such that:

$$
\text{sim}(x_i, x_j) \text{ is high} \implies d(f_\theta(x_i), f_\theta(x_j)) \text{ is small}
$$

$$
\text{sim}(x_i, x_j) \text{ is low} \implies d(f_\theta(x_i), f_\theta(x_j)) \text{ is large}
$$

where $d(\cdot, \cdot)$ is a distance function in the embedding space. The function $f_\theta$
is typically a neural network whose parameters $\theta$ are learned via gradient descent on
a carefully chosen loss function.

### 7.1.2 Distance and Similarity Metrics

The choice of distance metric in the embedding space is a foundational design decision.
We define the three most common metrics below.

**Euclidean Distance (L2)**

$$
d_E(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{k=1}^{d}(u_k - v_k)^2}
$$

The squared Euclidean distance $d_E^2$ is often preferred in loss functions because it
avoids the non-differentiable square root at zero:

$$
d_E^2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2^2 = \sum_{k=1}^{d}(u_k - v_k)^2
$$

Its gradient with respect to $\mathbf{u}$ is straightforward:

$$
\nabla_{\mathbf{u}} \, d_E^2(\mathbf{u}, \mathbf{v}) = 2(\mathbf{u} - \mathbf{v})
$$

**Cosine Similarity and Cosine Distance**

Cosine similarity measures the angle between two vectors, ignoring magnitude:

$$
\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|} = \frac{\sum_{k=1}^{d} u_k v_k}{\sqrt{\sum_{k=1}^{d} u_k^2} \cdot \sqrt{\sum_{k=1}^{d} v_k^2}}
$$

The corresponding cosine distance is:

$$
d_C(\mathbf{u}, \mathbf{v}) = 1 - \text{cos\_sim}(\mathbf{u}, \mathbf{v})
$$

When embeddings are L2-normalized (i.e., $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), cosine
similarity reduces to the dot product, and the squared Euclidean distance and cosine
distance become directly related:

$$
\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v} = 2 - 2\mathbf{u} \cdot \mathbf{v} = 2 \, d_C(\mathbf{u}, \mathbf{v})
$$

This equivalence is why many modern systems L2-normalize embeddings — it unifies the
Euclidean and cosine geometries.

**Manhattan Distance (L1)**

$$
d_M(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_1 = \sum_{k=1}^{d} |u_k - v_k|
$$

Manhattan distance is more robust to outliers but less commonly used in deep metric
learning due to the non-smooth absolute value. Its subgradient:

$$
\partial_{u_k} \, d_M(\mathbf{u}, \mathbf{v}) = \text{sign}(u_k - v_k)
$$

### 7.1.3 Taxonomy of Loss Functions

Loss functions for metric learning can be organized by the structure of their input:

| **Category**     | **Input Structure**          | **Examples**                        |
|------------------|------------------------------|-------------------------------------|
| Pairwise         | $(x_i, x_j, y_{ij})$        | Contrastive loss                    |
| Triplet-based    | $(x_a, x_p, x_n)$           | Triplet loss                        |
| Multi-negative   | $(x_a, x_p, \{x_{n_i}\})$   | N-pair loss, InfoNCE                |
| Classification   | $(x_i, y_i)$                | ArcFace, CosFace, SphereFace        |

We now derive each of these in full mathematical detail.

---

## 7.2 Contrastive Loss (Hadsell et al., 2006)

### 7.2.1 Formulation

The contrastive loss operates on **pairs** $(x_1, x_2)$ with a binary label
$y \in \{0, 1\}$, where $y = 0$ indicates a similar pair and $y = 1$ a dissimilar pair.

Let $\mathbf{e}_1 = f_\theta(x_1)$ and $\mathbf{e}_2 = f_\theta(x_2)$ be the embeddings,
and define the Euclidean distance:

$$
d = \|\mathbf{e}_1 - \mathbf{e}_2\|_2
$$

The contrastive loss for a single pair is:

$$
\mathcal{L}_{\text{contrastive}} = (1 - y) \cdot \frac{1}{2} d^2 + y \cdot \frac{1}{2} \big[\max(m - d, \, 0)\big]^2
$$

where $m > 0$ is the **margin** hyperparameter.

### 7.2.2 Intuition Behind Each Term

**Similar pairs** ($y = 0$): The loss reduces to $\frac{1}{2}d^2$ — a spring-like
potential pulling similar items together. Gradient is proportional to distance.

**Dissimilar pairs** ($y = 1$): The loss is $\frac{1}{2}[\max(m - d, 0)]^2$ — a hinge
penalty pushing dissimilar items apart only when $d < m$. Once $d \geq m$, no gradient
flows, preventing wasted capacity on already-separated items.

### 7.2.3 The Role of the Margin $m$

The margin $m$ defines the minimum acceptable distance between dissimilar pairs:

- **$m$ too small**: Embedding space becomes compressed, reducing discriminative power.
- **$m$ too large**: Optimization becomes difficult; similar items may remain too far apart.
- **Typical values**: $m \in [0.5, 2.0]$ for L2-normalized embeddings.

### 7.2.4 Step-by-Step Gradient Derivation

We derive $\frac{\partial \mathcal{L}}{\partial \mathbf{e}_1}$ in full detail.

**Step 1: Express the loss in terms of $d^2$.**

Let $D = d^2 = \|\mathbf{e}_1 - \mathbf{e}_2\|_2^2$ and $d = \sqrt{D}$.

**Step 2: Compute $\frac{\partial d}{\partial \mathbf{e}_1}$.**

$$
\frac{\partial d}{\partial \mathbf{e}_1} = \frac{\partial}{\partial \mathbf{e}_1} \sqrt{D}
= \frac{1}{2\sqrt{D}} \cdot \frac{\partial D}{\partial \mathbf{e}_1}
= \frac{1}{2d} \cdot 2(\mathbf{e}_1 - \mathbf{e}_2)
= \frac{\mathbf{e}_1 - \mathbf{e}_2}{d}
$$

**Step 3: Gradient for similar pairs ($y = 0$).**

$$
\mathcal{L}_{\text{sim}} = \frac{1}{2} d^2 = \frac{1}{2} D
$$

$$
\frac{\partial \mathcal{L}_{\text{sim}}}{\partial \mathbf{e}_1} = \frac{1}{2} \cdot 2(\mathbf{e}_1 - \mathbf{e}_2) = (\mathbf{e}_1 - \mathbf{e}_2)
$$

This gradient points from $\mathbf{e}_2$ toward $\mathbf{e}_1$, so gradient descent
moves $\mathbf{e}_1$ toward $\mathbf{e}_2$ — pulling similar items together.

**Step 4: Gradient for dissimilar pairs ($y = 1$).**

Define $h = \max(m - d, 0)$. Then $\mathcal{L}_{\text{dissim}} = \frac{1}{2} h^2$.

*Case A*: $d \geq m \implies h = 0 \implies \frac{\partial \mathcal{L}_{\text{dissim}}}{\partial \mathbf{e}_1} = \mathbf{0}$

*Case B*: $d < m \implies h = m - d > 0$

$$
\frac{\partial \mathcal{L}_{\text{dissim}}}{\partial \mathbf{e}_1}
= h \cdot \frac{\partial h}{\partial \mathbf{e}_1}
= (m - d) \cdot \left(-\frac{\partial d}{\partial \mathbf{e}_1}\right)
= -(m - d) \cdot \frac{\mathbf{e}_1 - \mathbf{e}_2}{d}
$$

This gradient points from $\mathbf{e}_1$ toward $\mathbf{e}_2$ (note the negative sign),
so gradient descent moves $\mathbf{e}_1$ **away** from $\mathbf{e}_2$.

**Step 5: Combined gradient.**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{e}_1} =
\begin{cases}
(\mathbf{e}_1 - \mathbf{e}_2) & \text{if } y = 0 \\[6pt]
-\dfrac{(m - d)}{d}(\mathbf{e}_1 - \mathbf{e}_2) & \text{if } y = 1 \text{ and } d < m \\[6pt]
\mathbf{0} & \text{if } y = 1 \text{ and } d \geq m
\end{cases}
$$

By symmetry, $\frac{\partial \mathcal{L}}{\partial \mathbf{e}_2} = -\frac{\partial \mathcal{L}}{\partial \mathbf{e}_1}$.

### 7.2.5 Numerical Example

Let $d = 2$, $m = 1.0$, and consider two 2D embeddings:

$$
\mathbf{e}_1 = \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix}, \quad
\mathbf{e}_2 = \begin{pmatrix} 0.2 \\ 0.1 \end{pmatrix}
$$

**Step 1: Compute distance.**

$$
\mathbf{e}_1 - \mathbf{e}_2 = \begin{pmatrix} 0.6 \\ 0.2 \end{pmatrix}
$$

$$
d = \sqrt{0.6^2 + 0.2^2} = \sqrt{0.36 + 0.04} = \sqrt{0.40} \approx 0.6325
$$

**Step 2: Similar pair ($y = 0$).**

$$
\mathcal{L} = \frac{1}{2}(0.6325)^2 = \frac{1}{2}(0.40) = 0.20
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{e}_1} = \begin{pmatrix} 0.6 \\ 0.2 \end{pmatrix}
$$

**Step 3: Dissimilar pair ($y = 1$).**

Since $d = 0.6325 < m = 1.0$, the hinge is active:

$$
\mathcal{L} = \frac{1}{2}(1.0 - 0.6325)^2 = \frac{1}{2}(0.3675)^2 = \frac{1}{2}(0.1351) \approx 0.0675
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{e}_1} = -\frac{(1.0 - 0.6325)}{0.6325} \begin{pmatrix} 0.6 \\ 0.2 \end{pmatrix}
= -0.5810 \begin{pmatrix} 0.6 \\ 0.2 \end{pmatrix}
= \begin{pmatrix} -0.3486 \\ -0.1162 \end{pmatrix}
$$

The negative sign confirms the gradient pushes $\mathbf{e}_1$ away from $\mathbf{e}_2$.

### 7.2.6 Geometric Interpretation

In the embedding space, contrastive loss creates two regimes:

- For similar pairs: a **parabolic well** centered at $d = 0$, pulling embeddings together.
- For dissimilar pairs: a **parabolic barrier** active only when $d < m$, with a "dead
  zone" for $d \geq m$ where no gradient flows.

The margin $m$ defines the decision boundary: pairs with $d < m$ are predicted similar;
pairs with $d \geq m$ are predicted dissimilar.

### 7.2.7 Limitations

1. **Pair sampling**: Requires explicit positive/negative pair construction, which scales
   as $O(N^2)$ and is dominated by uninformative easy negatives.
2. **Single negative**: Each update uses only one negative example, providing limited
   information about the global structure of the embedding space.
3. **Margin sensitivity**: Performance is sensitive to the choice of $m$, which must be
   tuned per dataset and embedding dimensionality.

---

## 7.3 Triplet Loss (Schroff et al., 2015)

### 7.3.1 Formulation

Triplet loss addresses the limitations of pairwise contrastive loss by operating on
**triplets** $(x_a, x_p, x_n)$: anchor $x_a$, positive $x_p$ (same class), and
negative $x_n$ (different class).

Let $\mathbf{a} = f_\theta(x_a)$, $\mathbf{p} = f_\theta(x_p)$, $\mathbf{n} = f_\theta(x_n)$.

Define the positive and negative distances:

$$
d_p = \|\mathbf{a} - \mathbf{p}\|_2^2, \quad d_n = \|\mathbf{a} - \mathbf{n}\|_2^2
$$

The triplet loss is:

$$
\mathcal{L}_{\text{triplet}} = \max\big(d_p - d_n + \alpha, \; 0\big)
$$

where $\alpha > 0$ is the **margin** that enforces a minimum gap between positive and
negative distances.

### 7.3.2 Intuition

The triplet loss is zero when $d_n > d_p + \alpha$. The constraint being enforced is:

$$
\|\mathbf{a} - \mathbf{p}\|_2^2 + \alpha < \|\mathbf{a} - \mathbf{n}\|_2^2
$$

### 7.3.3 Triplet Mining Strategies

The effectiveness of triplet loss depends critically on **which triplets are selected**.
Given $N$ training examples, there are $O(N^3)$ possible triplets, but most are
uninformative.

**Definition**: For a given anchor $\mathbf{a}$ and positive $\mathbf{p}$, we classify
negatives $\mathbf{n}$ into three categories:

**1. Easy Negatives**: $d_n > d_p + \alpha$

$$
\|\mathbf{a} - \mathbf{n}\|^2 > \|\mathbf{a} - \mathbf{p}\|^2 + \alpha
$$

The triplet loss is zero. These provide **no gradient signal** and are useless for
training. Unfortunately, the vast majority of randomly sampled triplets fall into this
category as training progresses.

**2. Semi-Hard Negatives**: $d_p < d_n < d_p + \alpha$

$$
\|\mathbf{a} - \mathbf{p}\|^2 < \|\mathbf{a} - \mathbf{n}\|^2 < \|\mathbf{a} - \mathbf{p}\|^2 + \alpha
$$

The negative is farther than the positive but not by the full margin. These provide
**stable, informative gradients** and are the strategy recommended by Schroff et al.

**3. Hard Negatives**: $d_n < d_p$

$$
\|\mathbf{a} - \mathbf{n}\|^2 < \|\mathbf{a} - \mathbf{p}\|^2
$$

The negative is **closer** to the anchor than the positive — a violation of the desired
ordering. These produce the largest gradients but can lead to **collapsed embeddings**
early in training.

**Mining Strategies in Practice**:

| Strategy              | Selection Rule                     | Pros                          | Cons                          |
|-----------------------|------------------------------------|-------------------------------|-------------------------------|
| Random                | Uniform sampling                   | Simple, fast                  | Mostly easy negatives         |
| Semi-hard (online)    | $d_p < d_n < d_p + \alpha$        | Stable convergence            | Requires in-batch computation |
| Hard (online)         | $\arg\min_{n} d_n$ s.t. $y_n \neq y_a$ | Maximum information      | Risk of collapse              |
| Hard (offline)        | Precompute all distances           | Global hardest negatives      | Stale embeddings, expensive   |
| Curriculum            | Easy → semi-hard → hard            | Stable early, strong late     | Requires scheduling           |

### 7.3.4 Step-by-Step Gradient Derivation

We derive the gradients of $\mathcal{L}_{\text{triplet}}$ with respect to $\mathbf{a}$,
$\mathbf{p}$, and $\mathbf{n}$.

**Step 1: Define the pre-hinge quantity.**

$$
\ell = d_p - d_n + \alpha = \|\mathbf{a} - \mathbf{p}\|^2 - \|\mathbf{a} - \mathbf{n}\|^2 + \alpha
$$

The loss is $\mathcal{L} = \max(\ell, 0)$. The gradient is zero when $\ell \leq 0$
(the hinge is inactive). When $\ell > 0$:

**Step 2: Gradient with respect to the anchor $\mathbf{a}$.**

$$
\frac{\partial \ell}{\partial \mathbf{a}}
= \frac{\partial}{\partial \mathbf{a}} \|\mathbf{a} - \mathbf{p}\|^2
- \frac{\partial}{\partial \mathbf{a}} \|\mathbf{a} - \mathbf{n}\|^2
= 2(\mathbf{a} - \mathbf{p}) - 2(\mathbf{a} - \mathbf{n})
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = 2(\mathbf{n} - \mathbf{p}) \quad \text{when } \ell > 0
$$

Interpretation: The anchor is pushed **away from the positive** and **toward the
negative**? No — gradient *descent* subtracts the gradient. The update
$\mathbf{a} \leftarrow \mathbf{a} - \eta \cdot 2(\mathbf{n} - \mathbf{p})$ moves the
anchor toward the positive and away from the negative.

**Step 3: Gradient with respect to the positive $\mathbf{p}$.**

$$
\frac{\partial \ell}{\partial \mathbf{p}}
= \frac{\partial}{\partial \mathbf{p}} \|\mathbf{a} - \mathbf{p}\|^2
= -2(\mathbf{a} - \mathbf{p})
= 2(\mathbf{p} - \mathbf{a})
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{p}} = 2(\mathbf{p} - \mathbf{a}) \quad \text{when } \ell > 0
$$

Gradient descent moves $\mathbf{p}$ toward $\mathbf{a}$.

**Step 4: Gradient with respect to the negative $\mathbf{n}$.**

Since $\ell$ contains $-\|\mathbf{a} - \mathbf{n}\|^2$:

$$
\frac{\partial}{\partial \mathbf{n}} \|\mathbf{a} - \mathbf{n}\|^2 = 2(\mathbf{n} - \mathbf{a})
$$

$$
\frac{\partial \ell}{\partial \mathbf{n}} = -2(\mathbf{n} - \mathbf{a}) = 2(\mathbf{a} - \mathbf{n})
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{n}} = 2(\mathbf{a} - \mathbf{n}) \quad \text{when } \ell > 0
$$

Gradient descent moves $\mathbf{n}$ **away from** $\mathbf{a}$.

**Step 5: Summary of gradients (when $\ell > 0$).**

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = 2(\mathbf{n} - \mathbf{p}), \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{p}} = 2(\mathbf{p} - \mathbf{a}), \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{n}} = 2(\mathbf{a} - \mathbf{n})
}
$$

All gradients are $\mathbf{0}$ when $\ell \leq 0$.

### 7.3.5 Numerical Example

Let $\alpha = 0.2$ and consider 2D embeddings:

$$
\mathbf{a} = \begin{pmatrix} 1.0 \\ 0.0 \end{pmatrix}, \quad
\mathbf{p} = \begin{pmatrix} 1.2 \\ 0.5 \end{pmatrix}, \quad
\mathbf{n} = \begin{pmatrix} 0.4 \\ -0.3 \end{pmatrix}
$$

**Step 1: Compute distances.**

$$
\mathbf{a} - \mathbf{p} = \begin{pmatrix} -0.2 \\ -0.5 \end{pmatrix}, \quad
d_p = (-0.2)^2 + (-0.5)^2 = 0.04 + 0.25 = 0.29
$$

$$
\mathbf{a} - \mathbf{n} = \begin{pmatrix} 0.6 \\ 0.3 \end{pmatrix}, \quad
d_n = 0.6^2 + 0.3^2 = 0.36 + 0.09 = 0.45
$$

**Step 2: Compute loss.**

$$
\ell = d_p - d_n + \alpha = 0.29 - 0.45 + 0.2 = 0.04
$$

Since $\ell = 0.04 > 0$, the hinge is active:

$$
\mathcal{L} = 0.04
$$

This is a **semi-hard** triplet: $d_p = 0.29 < d_n = 0.45$ (negative is farther), but
$d_n = 0.45 < d_p + \alpha = 0.49$ (not by the full margin).

**Step 3: Compute gradients.**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = 2(\mathbf{n} - \mathbf{p})
= 2\begin{pmatrix} 0.4 - 1.2 \\ -0.3 - 0.5 \end{pmatrix}
= 2\begin{pmatrix} -0.8 \\ -0.8 \end{pmatrix}
= \begin{pmatrix} -1.6 \\ -1.6 \end{pmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{p}} = 2(\mathbf{p} - \mathbf{a})
= 2\begin{pmatrix} 0.2 \\ 0.5 \end{pmatrix}
= \begin{pmatrix} 0.4 \\ 1.0 \end{pmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{n}} = 2(\mathbf{a} - \mathbf{n})
= 2\begin{pmatrix} 0.6 \\ 0.3 \end{pmatrix}
= \begin{pmatrix} 1.2 \\ 0.6 \end{pmatrix}
$$

**Step 4: Verify gradient directions.**

With learning rate $\eta = 0.1$:

- $\mathbf{a}' = (1.0, 0.0) - 0.1 \cdot (-1.6, -1.6) = (1.16, 0.16)$ → moves toward $\mathbf{p}$  ✓
- $\mathbf{p}' = (1.2, 0.5) - 0.1 \cdot (0.4, 1.0) = (1.16, 0.40)$ → moves toward $\mathbf{a}$  ✓
- $\mathbf{n}' = (0.4, -0.3) - 0.1 \cdot (1.2, 0.6) = (0.28, -0.36)$ → moves away from $\mathbf{a}$  ✓

### 7.3.6 Geometric Interpretation

Triplet loss simultaneously pulls the anchor-positive pair together and pushes the
anchor-negative pair apart. The margin $\alpha$ ensures a minimum gap between $d_p$ and
$d_n$, preventing the model from finding trivially correct but non-discriminative solutions.

---

## 7.4 N-pair Loss (Sohn, 2016)

### 7.4.1 Motivation

Triplet loss uses a single negative per update. With a batch of $B$ examples, we could
leverage $B - 1$ negatives per anchor. N-pair loss makes this generalization.

### 7.4.2 Formulation

Given an anchor $x_a$ with positive $x_p$ and $N-1$ negatives $\{x_{n_1}, \ldots, x_{n_{N-1}}\}$,
the N-pair loss is:

$$
\mathcal{L}_{\text{N-pair}} = \log\left(1 + \sum_{i=1}^{N-1} \exp\big(\mathbf{a}^T \mathbf{n}_i - \mathbf{a}^T \mathbf{p}\big)\right)
$$

where $\mathbf{a} = f_\theta(x_a)$, $\mathbf{p} = f_\theta(x_p)$, and
$\mathbf{n}_i = f_\theta(x_{n_i})$.

Equivalently, using the softplus function $\text{sp}(x) = \log(1 + e^x)$:

$$
\mathcal{L}_{\text{N-pair}} = \log\left(1 + \sum_{i=1}^{N-1} \exp\big(s_i^{-} - s^{+}\big)\right)
$$

where $s^{+} = \mathbf{a}^T \mathbf{p}$ is the positive similarity and
$s_i^{-} = \mathbf{a}^T \mathbf{n}_i$ are the negative similarities.

### 7.4.3 Connection to Softmax Cross-Entropy

The N-pair loss can be rewritten to reveal its connection to classification. Define the
logits as the set of similarities $\{s^{+}, s_1^{-}, \ldots, s_{N-1}^{-}\}$. The softmax
probability of the positive class is:

$$
P(\text{positive} \mid \mathbf{a}) = \frac{\exp(s^{+})}{\exp(s^{+}) + \sum_{i=1}^{N-1} \exp(s_i^{-})}
$$

The cross-entropy loss for this classification is:

$$
\mathcal{L}_{\text{CE}} = -\log P(\text{positive} \mid \mathbf{a})
= -\log \frac{\exp(s^{+})}{\exp(s^{+}) + \sum_{i=1}^{N-1} \exp(s_i^{-})}
$$

Factor out $\exp(s^{+})$ from the denominator:

$$
= -\log \frac{1}{1 + \sum_{i=1}^{N-1} \exp(s_i^{-} - s^{+})}
= \log\left(1 + \sum_{i=1}^{N-1} \exp(s_i^{-} - s^{+})\right)
$$

$$
\boxed{\mathcal{L}_{\text{CE}} = \mathcal{L}_{\text{N-pair}}}
$$

This is a key result: **N-pair loss is exactly softmax cross-entropy** where the "classes"
are the positive and $N-1$ negatives, and the "logits" are dot-product similarities.

### 7.4.4 Step-by-Step Gradient Derivation

**Step 1: Define shorthand.**

Let $z_i = s_i^{-} - s^{+} = \mathbf{a}^T \mathbf{n}_i - \mathbf{a}^T \mathbf{p}$ for
$i = 1, \ldots, N-1$, and let $S = 1 + \sum_{i=1}^{N-1} e^{z_i}$.

Then $\mathcal{L} = \log(S)$.

**Step 2: Gradient with respect to $\mathbf{a}$.**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = \frac{1}{S} \sum_{i=1}^{N-1} e^{z_i} \frac{\partial z_i}{\partial \mathbf{a}}
$$

Since $z_i = \mathbf{a}^T \mathbf{n}_i - \mathbf{a}^T \mathbf{p}$:

$$
\frac{\partial z_i}{\partial \mathbf{a}} = \mathbf{n}_i - \mathbf{p}
$$

Therefore:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = \frac{1}{S} \sum_{i=1}^{N-1} e^{z_i} (\mathbf{n}_i - \mathbf{p})
$$

Define the **softmax weights** $w_i = \frac{e^{z_i}}{S}$ (note $\sum_i w_i < 1$ since
$S$ includes the $+1$ term). Then:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = \sum_{i=1}^{N-1} w_i (\mathbf{n}_i - \mathbf{p})
$$

Interpretation: The gradient is a **weighted average** of directions from the positive
to each negative, where harder negatives receive larger weights — an automatic form of
hard negative mining.

**Step 3: Gradient with respect to $\mathbf{p}$.**

$$
\frac{\partial z_i}{\partial \mathbf{p}} = -\mathbf{a}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{p}} = \frac{1}{S} \sum_{i=1}^{N-1} e^{z_i} (-\mathbf{a}) = -\left(\sum_{i=1}^{N-1} w_i\right) \mathbf{a}
$$

The positive embedding is pulled toward the anchor, with strength proportional to the
total weight of active negatives.

**Step 4: Gradient with respect to $\mathbf{n}_j$.**

$$
\frac{\partial z_j}{\partial \mathbf{n}_j} = \mathbf{a}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{n}_j} = w_j \, \mathbf{a}
$$

Each negative is pushed away from the anchor, with strength proportional to its softmax
weight.

### 7.4.5 Connection to Triplet Loss

When $N = 2$ (one positive, one negative), N-pair loss reduces to:

$$
\mathcal{L} = \log\big(1 + \exp(\mathbf{a}^T \mathbf{n} - \mathbf{a}^T \mathbf{p})\big)
= \text{softplus}(\mathbf{a}^T \mathbf{n} - \mathbf{a}^T \mathbf{p})
$$

This is a **smooth approximation** of triplet loss. The softplus $\text{sp}(x) = \log(1 + e^x)$
is a smooth lower bound on $\max(x, 0)$, and the triplet margin $\alpha$ corresponds to
a bias term.

---

## 7.5 InfoNCE / NT-Xent Loss

### 7.5.1 Background and Motivation

The **InfoNCE** loss (Oord et al., 2018) and its variant **NT-Xent** (Chen et al., 2020)
have become the dominant contrastive objectives, powering SimCLR, CLIP, MoCo, and most
contemporary embedding models.

The key insight: frame contrastive learning as a **noise-contrastive estimation** problem —
given a query, identify the true positive among distractors.

### 7.5.2 Formulation

Given a batch of $N$ examples, data augmentation produces $2N$ views. For a positive pair
$(i, j)$, the NT-Xent loss for example $i$ is:

$$
\mathcal{L}_i = -\log \frac{\exp\big(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\big)}{\displaystyle\sum_{\substack{k=1 \\ k \neq i}}^{2N} \exp\big(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau\big)}
$$

where:
- $\mathbf{z}_i = g(f_\theta(x_i))$ is the projected embedding (after a projection head $g$)
- $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ is cosine similarity
- $\tau > 0$ is the **temperature** parameter
- The sum in the denominator runs over all $2N - 1$ other examples in the batch

The total loss over the batch is:

$$
\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{i=1}^{2N} \mathcal{L}_i
$$

### 7.5.3 The Temperature Parameter $\tau$

The temperature $\tau$ controls the **sharpness** of the similarity distribution:

$$
p_{k|i} = \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}{\sum_{k' \neq i} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_{k'}) / \tau)}
$$

**Low $\tau$ ($\to 0^+$)**: Distribution peaks on the hardest negative — approaches
$\arg\max$. Can cause training instability.

**High $\tau$ ($\to \infty$)**: Distribution becomes uniform — all negatives contribute
equally. Provides weak gradients.

**Optimal $\tau$ ($\approx 0.05 - 0.5$)**: Balances hard negative focus with gradient
stability. The gradient magnitude scales as $1/\tau$, so lower temperatures amplify both
signal and noise.

### 7.5.4 Connection to Mutual Information Maximization

InfoNCE derives its name from **Noise-Contrastive Estimation** of mutual information.
The key result (Oord et al., 2018) is:

$$
I(X; Y) \geq \log(N) - \mathcal{L}_{\text{InfoNCE}}
$$

where $I(X; Y)$ is the mutual information between two views and $N$ is the number of
negatives plus one.

**Derivation**: For a positive pair $(x, y^+)$ from the joint $p(x,y)$ and negatives
$\{y_k^-\}$ from the marginal $p(y)$, the InfoNCE objective is:

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{e^{f(x, y^+)}}{e^{f(x, y^+)} + \sum_{k=1}^{N-1} e^{f(x, y_k^-)}}\right]
$$

At the optimum, the critic $f^*(x,y) \propto \log\frac{p(y|x)}{p(y)}$, and substituting
yields $\mathcal{L}^* \geq \log(N) - I(X;Y)$. Rearranging gives the bound above.

Minimizing InfoNCE thus **maximizes a lower bound on mutual information**, and the bound
tightens as $N$ increases — explaining why larger batch sizes improve contrastive learning.

### 7.5.5 Step-by-Step Gradient Derivation

We derive the gradient of $\mathcal{L}_i$ with respect to the embedding $\mathbf{z}_i$.

For notational clarity, let $s_{ik} = \text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau$ and
assume L2-normalized embeddings so $\text{sim}(\mathbf{z}_i, \mathbf{z}_k) = \mathbf{z}_i^T \mathbf{z}_k$.

**Step 1: Write the loss.**

$$
\mathcal{L}_i = -s_{ij} + \log\left(\sum_{k \neq i} e^{s_{ik}}\right)
$$

where $j$ is the index of the positive pair for $i$.

**Step 2: Compute the softmax probabilities.**

Define:

$$
p_{k|i} = \frac{e^{s_{ik}}}{\sum_{k' \neq i} e^{s_{ik'}}}
$$

Then $\mathcal{L}_i = -s_{ij} + \log\left(\sum_{k \neq i} e^{s_{ik}}\right)$.

**Step 3: Gradient with respect to $s_{ik}$.**

For the positive pair ($k = j$):

$$
\frac{\partial \mathcal{L}_i}{\partial s_{ij}} = -1 + p_{j|i}
$$

For any negative ($k \neq j, k \neq i$):

$$
\frac{\partial \mathcal{L}_i}{\partial s_{ik}} = p_{k|i}
$$

**Step 4: Gradient with respect to $\mathbf{z}_i$.**

Since $s_{ik} = \mathbf{z}_i^T \mathbf{z}_k / \tau$ and embeddings are L2-normalized:

$$
\frac{\partial s_{ik}}{\partial \mathbf{z}_i} = \frac{\mathbf{z}_k}{\tau}
$$

(Note: for L2-normalized vectors, the gradient of the dot product with respect to one
vector is simply the other vector, before accounting for the normalization constraint.
The full gradient through the normalization layer adds a projection term, but we omit
this for clarity.)

Applying the chain rule:

$$
\frac{\partial \mathcal{L}_i}{\partial \mathbf{z}_i}
= \sum_{k \neq i} \frac{\partial \mathcal{L}_i}{\partial s_{ik}} \cdot \frac{\mathbf{z}_k}{\tau}
$$

$$
= \frac{1}{\tau}\left[(-1 + p_{j|i})\mathbf{z}_j + \sum_{\substack{k \neq i \\ k \neq j}} p_{k|i} \, \mathbf{z}_k\right]
$$

$$
= \frac{1}{\tau}\left[\sum_{k \neq i} p_{k|i} \, \mathbf{z}_k - \mathbf{z}_j\right]
$$

$$
\boxed{
\frac{\partial \mathcal{L}_i}{\partial \mathbf{z}_i} = \frac{1}{\tau}\left(\mathbb{E}_{k \sim p_{(\cdot|i)}}[\mathbf{z}_k] - \mathbf{z}_j\right)
}
$$

**Interpretation**: The gradient pushes $\mathbf{z}_i$ away from the probability-weighted
mean of all other embeddings and toward the positive $\mathbf{z}_j$. Hard negatives
(high $p_{k|i}$) contribute more to the repulsive force.

### 7.5.6 Numerical Example

Consider $N = 2$ examples producing $2N = 4$ views with $\tau = 0.5$ and 2D L2-normalized
embeddings:

$$
\mathbf{z}_1 = (1.0, 0.0), \; \mathbf{z}_2 = (0.96, 0.28), \;
\mathbf{z}_3 = (-0.6, 0.8), \; \mathbf{z}_4 = (-0.8, -0.6)
$$

Positive pairs: $(1,2)$ and $(3,4)$. Computing $\mathcal{L}_1$:

**Cosine similarities**: $\text{sim}_{12} = 0.96$, $\text{sim}_{13} = -0.60$, $\text{sim}_{14} = -0.80$

**Scaled logits**: $s_{12} = 1.92$, $s_{13} = -1.20$, $s_{14} = -1.60$

**Softmax**: $e^{1.92} \approx 6.821$, $e^{-1.20} \approx 0.301$, $e^{-1.60} \approx 0.202$

$$
Z = 7.324, \quad p_{2|1} = 0.931, \quad p_{3|1} = 0.041, \quad p_{4|1} = 0.028
$$

**Loss**: $\mathcal{L}_1 = -\log(0.931) \approx 0.071$ (low — model is already performing well)

**Gradient**:

$$
\frac{\partial \mathcal{L}_1}{\partial \mathbf{z}_1} = \frac{1}{\tau}\left[\sum_{k} p_{k|1} \mathbf{z}_k - \mathbf{z}_2\right]
= 2.0\left[\begin{pmatrix}0.847\\0.277\end{pmatrix} - \begin{pmatrix}0.96\\0.28\end{pmatrix}\right]
= \begin{pmatrix}-0.226\\-0.006\end{pmatrix}
$$

The small gradient confirms the model is near-optimal for this example.

### 7.5.7 Why InfoNCE Became Dominant

1. **Automatic hard negative mining**: Softmax weighting naturally upweights hard negatives.
2. **No margin hyperparameter**: Temperature $\tau$ replaces the margin, is more interpretable.
3. **Scales with batch size**: More negatives tighten the MI bound.
4. **Smooth gradients**: Log-softmax provides gradients for all batch examples.
5. **Theoretical grounding**: Connection to mutual information maximization.

---

## 7.6 Softmax-Based Angular Margin Losses

### 7.6.1 Motivation

The losses discussed so far operate on pairs or tuples. An alternative treats embedding
learning as a **classification problem** with angular margin penalties, maintaining a
learnable weight matrix $\mathbf{W} \in \mathbb{R}^{d \times C}$ ($C$ classes) and
enforcing separation in **angular space**.

The key idea: L2-normalize both embeddings $\mathbf{z} = f_\theta(x)$ and weight
vectors $\mathbf{w}_j$, so the logit for class $j$ becomes $\mathbf{w}_j^T \mathbf{z} = \cos\theta_j$
where $\theta_j$ is the angle between $\mathbf{z}$ and $\mathbf{w}_j$.

### 7.6.2 Standard Softmax Baseline

With a scale parameter $s$:

$$
\mathcal{L}_{\text{softmax}} = -\log \frac{e^{s \cos\theta_{y_i}}}{\displaystyle\sum_{j=1}^{C} e^{s \cos\theta_j}}
$$

This learns separable embeddings but does not enforce a margin between classes.

### 7.6.3 SphereFace (Liu et al., 2017) — Multiplicative Angular Margin

SphereFace introduces a **multiplicative** angular margin by replacing $\cos\theta_{y_i}$
with $\cos(m\theta_{y_i})$ where $m \geq 1$ is an integer:

$$
\mathcal{L}_{\text{SphereFace}} = -\log \frac{e^{s \cos(m\theta_{y_i})}}{e^{s \cos(m\theta_{y_i})} + \displaystyle\sum_{j \neq y_i} e^{s \cos\theta_j}}
$$

To classify correctly, the model must satisfy $\cos(m\theta_{y_i}) > \cos\theta_j$,
which (for $m = 2$) requires $\theta_{y_i} < \theta_j / 2$. In practice, a monotonically
decreasing piecewise function replaces $\cos(m\theta)$ to handle its oscillatory behavior.

### 7.6.4 CosFace (Wang et al., 2018) — Additive Cosine Margin

CosFace adds a **cosine margin** $m$ directly to the cosine similarity:

$$
\mathcal{L}_{\text{CosFace}} = -\log \frac{e^{s(\cos\theta_{y_i} - m)}}{e^{s(\cos\theta_{y_i} - m)} + \displaystyle\sum_{j \neq y_i} e^{s \cos\theta_j}}
$$

**Effect**: The decision boundary shifts from $\cos\theta_{y_i} = \cos\theta_j$ to
$\cos\theta_{y_i} - m = \cos\theta_j$, requiring $\cos\theta_{y_i} > \cos\theta_j + m$.
Simpler than SphereFace and provides a uniform margin in cosine space.

### 7.6.5 ArcFace (Deng et al., 2019) — Additive Angular Margin

ArcFace adds the margin $m$ directly to the **angle** $\theta_{y_i}$:

$$
\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \displaystyle\sum_{j \neq y_i} e^{s \cos\theta_j}}
$$

**Effect**: The decision boundary becomes $\cos(\theta_{y_i} + m) = \cos\theta_j$,
i.e., $\theta_{y_i} + m < \theta_j$. This enforces a constant **geodesic margin** on the
hypersphere — geometrically the most natural margin for normalized embeddings.

### 7.6.6 Unified Formulation

All three losses can be expressed in a unified framework:

$$
\mathcal{L} = -\log \frac{e^{s \cdot \psi(\theta_{y_i})}}{e^{s \cdot \psi(\theta_{y_i})} + \displaystyle\sum_{j \neq y_i} e^{s \cos\theta_j}}
$$

where the penalty function $\psi$ differs:

| **Method**   | $\psi(\theta_{y_i})$          | **Margin Type**            |
|--------------|-------------------------------|----------------------------|
| Softmax      | $\cos\theta_{y_i}$           | None                       |
| SphereFace   | $\cos(m \cdot \theta_{y_i})$ | Multiplicative angular     |
| CosFace      | $\cos\theta_{y_i} - m$       | Additive cosine            |
| ArcFace      | $\cos(\theta_{y_i} + m)$     | Additive angular           |

### 7.6.7 Step-by-Step ArcFace Gradient Derivation

We derive the gradient of $\mathcal{L}_{\text{ArcFace}}$ with respect to $\mathbf{z}$
(the L2-normalized embedding).

**Step 1: Define logits.**

$$
\ell_j = \begin{cases}
s \cos(\theta_{y_i} + m) & \text{if } j = y_i \\
s \cos\theta_j & \text{if } j \neq y_i
\end{cases}
$$

**Step 2: Softmax probabilities.**

$$
p_j = \frac{e^{\ell_j}}{\sum_{k=1}^{C} e^{\ell_k}}
$$

**Step 3: Cross-entropy gradient with respect to logits.**

$$
\frac{\partial \mathcal{L}}{\partial \ell_j} = p_j - \mathbf{1}[j = y_i]
$$

This is the standard softmax-cross-entropy gradient.

**Step 4: Gradient of logits with respect to $\theta_{y_i}$.**

For the target class:

$$
\frac{\partial \ell_{y_i}}{\partial \theta_{y_i}} = -s \sin(\theta_{y_i} + m)
$$

**Step 5: Gradient of $\theta_{y_i}$ with respect to $\mathbf{z}$.**

Since $\cos\theta_{y_i} = \mathbf{w}_{y_i}^T \mathbf{z}$ (both normalized):

$$
\frac{\partial \cos\theta_{y_i}}{\partial \mathbf{z}} = \mathbf{w}_{y_i} - \cos\theta_{y_i} \cdot \mathbf{z}
$$

(The second term accounts for the L2 normalization constraint.)

$$
\frac{\partial \theta_{y_i}}{\partial \mathbf{z}} = \frac{-1}{\sin\theta_{y_i}} \left(\mathbf{w}_{y_i} - \cos\theta_{y_i} \cdot \mathbf{z}\right)
$$

**Step 6: Chain rule — target class contribution.**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}}\bigg|_{y_i}
= (1 - p_{y_i}) \cdot s \cdot \frac{\sin(\theta_{y_i} + m)}{\sin\theta_{y_i}} \left(\mathbf{w}_{y_i} - \cos\theta_{y_i} \cdot \mathbf{z}\right)
$$

**Step 7: Non-target class contributions.**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}}\bigg|_{j \neq y_i}
= \sum_{j \neq y_i} p_j \cdot s \left(\mathbf{w}_j - \cos\theta_j \cdot \mathbf{z}\right)
$$

**Step 8: Total gradient.**

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mathbf{z}}
= (1 - p_{y_i}) \cdot s \cdot \frac{\sin(\theta_{y_i} + m)}{\sin\theta_{y_i}} \left(\mathbf{w}_{y_i} - \cos\theta_{y_i} \cdot \mathbf{z}\right)
+ \sum_{j \neq y_i} p_j \cdot s \left(\mathbf{w}_j - \cos\theta_j \cdot \mathbf{z}\right)
}
$$

The first term pulls $\mathbf{z}$ toward $\mathbf{w}_{y_i}$, amplified by
$\frac{\sin(\theta_{y_i} + m)}{\sin\theta_{y_i}} > 1$. The second term pushes
$\mathbf{z}$ away from non-target centers, weighted by softmax probabilities.

### 7.6.8 Geometric Interpretation on the Hypersphere

Since all embeddings and weight vectors are L2-normalized, they lie on the unit
hypersphere $\mathbb{S}^{d-1}$. The angular margin losses create **exclusion zones**
around each class center. Without margin, the decision boundary is at $\theta_1 = \theta_2$.
With ArcFace margin $m$, the boundary shifts to $\theta_1 + m = \theta_2$ (and
symmetrically $\theta_2 + m = \theta_1$), creating a gap of $2m$ between class regions
on the hypersphere.

### 7.6.9 Practical Considerations

- **Scale $s$**: Typical values $s \in [30, 64]$. Common choice: $s = 64$.
- **Margin $m$**: ArcFace: $m \in [0.3, 0.5]$ rad. CosFace: $m \in [0.2, 0.4]$.
- **When to use**: Requires class labels and a fixed class set. Excels in closed-set
  problems (face recognition). Less suitable for open-domain retrieval.

---

## 7.7 Comparative Analysis

### 7.7.1 Summary Table

| **Loss Function** | **Input**                    | **Key Formula**                                                                 | **Hyperparameters**     | **Pros**                                              | **Cons**                                              | **Primary Use Cases**                    |
|--------------------|------------------------------|---------------------------------------------------------------------------------|-------------------------|-------------------------------------------------------|-------------------------------------------------------|------------------------------------------|
| Contrastive        | Pair + label                 | $(1-y)\frac{1}{2}d^2 + y\frac{1}{2}[\max(m-d,0)]^2$                           | Margin $m$              | Simple, intuitive                                     | Only one negative per pair; margin-sensitive           | Siamese networks, signature verification |
| Triplet            | Anchor, pos, neg             | $\max(d_p - d_n + \alpha, 0)$                                                  | Margin $\alpha$         | Relative ordering; flexible                           | Mining-dependent; slow convergence                    | Face verification, image retrieval       |
| N-pair             | Anchor, pos, $N$-1 neg      | $\log(1 + \sum_i e^{s_i^- - s^+})$                                             | None (implicit)         | Multiple negatives; = softmax CE                      | Requires diverse batch                                | Fine-grained recognition                 |
| InfoNCE / NT-Xent  | Batch of augmented pairs     | $-\log\frac{e^{\text{sim}/\tau}}{\sum_k e^{\text{sim}_k/\tau}}$               | Temperature $\tau$      | Auto hard-neg mining; MI bound; smooth gradients      | Needs large batches; sensitive to $\tau$              | Self-supervised learning (SimCLR, CLIP)  |
| SphereFace         | Example + class label        | $-\log\frac{e^{s\cos(m\theta_y)}}{\cdots}$                                     | Scale $s$, margin $m$   | Angular margin on hypersphere                         | Non-monotonic $\cos(m\theta)$; complex implementation | Face recognition                         |
| CosFace            | Example + class label        | $-\log\frac{e^{s(\cos\theta_y - m)}}{\cdots}$                                  | Scale $s$, margin $m$   | Simple additive margin; stable training               | Requires class labels                                 | Face recognition, person re-ID           |
| ArcFace            | Example + class label        | $-\log\frac{e^{s\cos(\theta_y + m)}}{\cdots}$                                  | Scale $s$, margin $m$   | Geodesic margin; geometrically natural; SOTA results  | Requires class labels; fixed class set                | Face recognition, speaker verification   |

### 7.7.2 Evolution and Relationships

The loss functions form a clear evolutionary lineage:

```
Contrastive (2006) ──"relative ordering"──▶ Triplet (2015)
    ──"multiple negatives"──▶ N-pair (2016)
    ──"temperature + normalization"──▶ InfoNCE (2018/2020)
    ──"angular margins on hypersphere"──▶ ArcFace/CosFace/SphereFace (2017-2019)
```

Each step addresses a limitation: contrastive→triplet moves to relative ordering;
triplet→N-pair increases negatives from 1 to $N-1$; N-pair→InfoNCE adds temperature
scaling and MI grounding; InfoNCE→angular margins adds explicit geometric margins.

### 7.7.3 Choosing a Loss Function

- **Self-supervised pretraining**: InfoNCE/NT-Xent with large batches
- **Face recognition** with identity labels: ArcFace
- **Retrieval** with relevance labels: InfoNCE or triplet loss with semi-hard mining
- **Cross-modal retrieval** (e.g., text-image): InfoNCE (as used in CLIP)

---

## 7.8 Practical Considerations

### 7.8.1 Embedding Collapse

A degenerate solution exists for all contrastive losses: mapping all inputs to the same
point. If $f_\theta(x) = \mathbf{c}$ for all $x$, then $d = 0$ for all pairs and the
loss landscape becomes flat. Prevention strategies include L2 normalization (forces
embeddings onto the hypersphere), batch normalization (decorrelates dimensions),
stop-gradient / momentum encoders (BYOL, MoCo), and variance-covariance regularization
(VICReg).

### 7.8.2 Batch Size and Negative Sampling

The InfoNCE bound $I(X;Y) \geq \log(N) - \mathcal{L}$ tightens with more negatives $N$,
explaining why larger batches improve contrastive learning. Memory-efficient alternatives
include memory banks (MoCo), gradient accumulation, and hard negative queues.

---

## 7.9 Chapter Summary

This chapter derived the major loss functions for embedding learning: **contrastive loss**
(pair-based with margin hinge), **triplet loss** (relative ordering with mining-dependent
effectiveness), **N-pair loss** (multiple negatives equivalent to softmax cross-entropy),
**InfoNCE/NT-Xent** (temperature-scaled with mutual information grounding), and **angular
margin losses** (ArcFace, CosFace, SphereFace — classification on the hypersphere with
geometric margins). The evolution from contrastive to InfoNCE represents a progression
toward more informative gradients, stronger theoretical foundations, and greater
scalability.

---

## References

- Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality reduction by learning an
  invariant mapping. *CVPR*.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for
  face recognition and clustering. *CVPR*.
- Sohn, K. (2016). Improved deep metric learning with multi-class N-pair loss objective.
  *NeurIPS*.
- Oord, A. van den, Li, Y., & Vinyals, O. (2018). Representation learning with
  contrastive predictive coding. *arXiv:1807.03748*.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for
  contrastive learning of visual representations. *ICML*.
- Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., & Song, L. (2017). SphereFace: Deep
  hypersphere embedding for face recognition. *CVPR*.
- Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., & Liu, W. (2018).
  CosFace: Large margin cosine loss for deep face recognition. *CVPR*.
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin
  loss for deep face recognition. *CVPR*.


---

*Next chapter: [Chapter 8 — Fine-Tuning Embeddings](08-finetuning-embeddings.md)*
