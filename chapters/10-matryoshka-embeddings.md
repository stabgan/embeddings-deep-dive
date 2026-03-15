# Chapter 10: Matryoshka Embeddings — Nested Representations at Every Scale

## Table of Contents

1. [Introduction — The Fixed Dimension Problem](#101-introduction--the-fixed-dimension-problem)
2. [Core Idea — Nested Representations](#102-core-idea--nested-representations)
3. [Mathematical Formulation — Step by Step](#103-mathematical-formulation--step-by-step)
4. [Training Procedure — Step by Step](#104-training-procedure--step-by-step)
5. [Information Hierarchy — Why It Works](#105-information-hierarchy--why-it-works)
6. [Normalization for Contrastive Learning](#106-normalization-for-contrastive-learning)
7. [Adaptation to Sentence Embeddings](#107-adaptation-to-sentence-embeddings)
8. [Inference — Flexible Deployment](#108-inference--flexible-deployment)
9. [Practical Example — Complete Training Pipeline](#109-practical-example--complete-training-pipeline)
10. [Comparison with Alternatives](#1010-comparison-with-alternatives)
11. [Applications](#1011-applications)
12. [Summary](#1012-summary)

---

## 10.1 Introduction — The Fixed Dimension Problem

### 10.1.1 The Rigidity of Conventional Embeddings

Every embedding model you have encountered so far in this guide produces vectors of a
single, fixed dimensionality. BERT-base outputs 768 dimensions. ResNet-50 outputs 2048.
OpenAI's `text-embedding-3-large` outputs 3072. Once the model is trained, that number
is locked in — every input, no matter how simple or complex, gets mapped to the same
number of floating-point values.

Formally, a conventional encoder is a function:

$
F(\cdot\,; \theta_F): \mathcal{X} \rightarrow \mathbb{R}^d \tag{1}
$

where $d$ is fixed at architecture design time. The embedding of any input $x$ is:

$
\mathbf{z} = F(x; \theta_F) \in \mathbb{R}^d \tag{2}
$

This rigidity creates a fundamental tension in deployment.

### 10.1.2 The Deployment Dilemma

Different downstream tasks — and different stages within the same task — have wildly
different computational budgets:

| Scenario | Latency Budget | Memory Budget | Ideal Dimension |
|----------|---------------|---------------|-----------------|
| Real-time autocomplete | < 5 ms | Tight (mobile) | 64–128 |
| First-pass retrieval over 100M docs | < 50 ms | Moderate | 128–256 |
| Re-ranking top-1000 candidates | < 200 ms | Generous | 512–768 |
| Offline batch classification | Unlimited | Unlimited | Full (768+) |

With a fixed 768-dimensional model, you face an unpleasant choice:

1. **Use the full embedding everywhere** — pay the computational cost even when you
   don't need the precision.
2. **Train separate smaller models** — multiply your training cost, maintenance burden,
   and storage requirements by the number of deployment targets.
3. **Apply post-hoc dimensionality reduction** (PCA, random projection) — lose information
   in ways the model was never optimized for.

None of these options is satisfying. What if the model itself could produce embeddings
that are useful at *any* dimensionality?

### 10.1.3 The Matryoshka Insight

In 2022, Kusupati et al. introduced **Matryoshka Representation Learning (MRL)** at
NeurIPS, named after the Russian nesting dolls (матрёшка) where each doll contains a
smaller version of itself inside.

The key insight is deceptively simple: **train the model so that every prefix of the
embedding vector is independently useful**. The first 8 dimensions should capture the
coarsest semantic signal. The first 64 dimensions should be good enough for fast
retrieval. The first 256 dimensions should rival a dedicated 256-d model. And the full
vector should lose nothing compared to standard training.

One model. One forward pass. Every dimensionality you need.

---

## 10.2 Core Idea — Nested Representations

### 10.2.1 From Fixed to Flexible

A Matryoshka encoder uses the same architecture as a conventional encoder — the change
is entirely in the training objective. The encoder still produces a single $d$-dimensional
vector:

$
\mathbf{z} = F(x; \theta_F) \in \mathbb{R}^d \tag{3}
$

The difference is that we demand every prefix $\mathbf{z}_{1:m}$ (the first $m$ components)
to be a valid, useful representation on its own:

$
\mathbf{z}_{1:m} = [z_1, z_2, \ldots, z_m] \in \mathbb{R}^m \tag{4}
$

This is a strict nesting: $\mathbf{z}_{1:8}$ is a prefix of $\mathbf{z}_{1:64}$, which
is a prefix of $\mathbf{z}_{1:256}$, which is a prefix of the full $\mathbf{z}$. No
separate computation is needed — you just slice the vector.

### 10.2.2 The Nesting Dimensions Set

We define a set of **nesting dimensions** $\mathcal{M}$ that specifies which prefix
lengths the model is explicitly trained to optimize:

$
\mathcal{M} \subset [d], \quad |\mathcal{M}| \leq \lfloor \log_2(d) \rfloor \tag{5}
$

The constraint $|\mathcal{M}| \leq \lfloor \log_2(d) \rfloor$ keeps the training overhead
logarithmic in the full dimension. A typical choice for a 2048-dimensional model is:

$
\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\} \tag{6}
$

For a 768-dimensional sentence embedding model, a common choice is:

$
\mathcal{M} = \{32, 64, 128, 256, 512, 768\} \tag{7}
$

The powers-of-two pattern is conventional but not required. Any increasing sequence
$m_1 < m_2 < \cdots < m_k = d$ works.

### 10.2.3 The Nesting Visualized

Think of the embedding vector as a telescope with nested tubes:

```
Full embedding z ∈ ℝ^2048:
┌─────────────────────────────────────────────────────────────────┐
│ z₁ z₂ ... z₈ │ z₉ ... z₁₆ │ z₁₇ ... z₃₂ │ ... │ z₁₀₂₅ ... z₂₀₄₈ │
└─────────────────────────────────────────────────────────────────┘
 ├── z_{1:8} ──┤
 ├────── z_{1:16} ──────┤
 ├──────────── z_{1:32} ────────────┤
 ├─────────────────────────── ... ──────────────────── z_{1:2048} ┤
```

Each prefix captures progressively finer detail. The first 8 dimensions encode the
broadest semantic category. Dimensions 9–16 add nuance. Dimensions 17–32 add more.
And so on, all the way to the full representation.

### 10.2.4 What "Independently Useful" Means

For each nesting dimension $m \in \mathcal{M}$, the prefix $\mathbf{z}_{1:m}$ must be
able to solve the downstream task *on its own* — without access to the remaining
dimensions $z_{m+1}, \ldots, z_d$. Concretely, there must exist a linear classifier
$W^{(m)} \in \mathbb{R}^{L \times m}$ (where $L$ is the number of classes) such that:

$
\hat{y}^{(m)} = \arg\max_{l \in [L]} \left[ W^{(m)} \cdot \mathbf{z}_{1:m} \right]_l \tag{8}
$

achieves good accuracy. The model is trained to make this true simultaneously for *all*
$m \in \mathcal{M}$.

---

## 10.3 Mathematical Formulation — Step by Step

### 10.3.1 Setup and Notation

Let us establish the full notation before writing the objective.

- **Input space**: $\mathcal{X}$ (images, sentences, etc.)
- **Label space**: $\mathcal{Y} = \{1, 2, \ldots, L\}$ with $L$ classes
- **Training set**: $\{(x_i, y_i)\}_{i=1}^{N}$
- **Encoder**: $F(\cdot\,; \theta_F): \mathcal{X} \rightarrow \mathbb{R}^d$
- **Nesting dimensions**: $\mathcal{M} = \{m_1, m_2, \ldots, m_k\}$ with $m_k = d$
- **Per-dimension classifier**: $W^{(m)} \in \mathbb{R}^{L \times m}$ for each $m \in \mathcal{M}$
- **Importance weights**: $c_m \geq 0$ for each $m \in \mathcal{M}$
- **Loss function**: $\mathcal{L}(\cdot\,; y)$ — typically cross-entropy

### 10.3.2 The MRL Objective

The Matryoshka Representation Learning objective jointly optimizes the encoder and all
per-dimension classifiers:

$
\min_{W^{(m)}, \theta_F} \; \frac{1}{N} \sum_{i \in [N]} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}\!\left(W^{(m)} \cdot F(x_i; \theta_F)_{1:m} \;;\; y_i\right) \tag{9}
$

Let us unpack this equation piece by piece.

**Step 1 — Encode the input.** For a given input $x_i$, the encoder produces the full
embedding:

$
\mathbf{z}_i = F(x_i; \theta_F) \in \mathbb{R}^d \tag{10}
$

**Step 2 — Extract each prefix.** For each nesting dimension $m \in \mathcal{M}$, take
the first $m$ components:

$
\mathbf{z}_{i, 1:m} = [z_{i,1}, z_{i,2}, \ldots, z_{i,m}] \in \mathbb{R}^m \tag{11}
$

**Step 3 — Compute per-dimension logits.** Each prefix gets its own linear classifier:

$
\text{logits}_i^{(m)} = W^{(m)} \cdot \mathbf{z}_{i, 1:m} \in \mathbb{R}^L \tag{12}
$

where $W^{(m)} \in \mathbb{R}^{L \times m}$ maps the $m$-dimensional prefix to $L$
class scores.

**Step 4 — Compute per-dimension loss.** Apply the cross-entropy loss at each scale:

$
\mathcal{L}_m^{(i)} = \text{CrossEntropy}\!\left(\text{softmax}\!\left(\text{logits}_i^{(m)}\right), y_i\right) \tag{13}
$

Expanding the cross-entropy explicitly:

$
\mathcal{L}_m^{(i)} = -\log \frac{\exp\!\left(\left[\text{logits}_i^{(m)}\right]_{y_i}\right)}{\sum_{l=1}^{L} \exp\!\left(\left[\text{logits}_i^{(m)}\right]_l\right)} \tag{14}
$

**Step 5 — Weight and aggregate.** The total loss for sample $i$ sums across all nesting
dimensions with importance weights $c_m$:

$
\mathcal{L}_{\text{MRL}}^{(i)} = \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}_m^{(i)} \tag{15}
$

**Step 6 — Average over the dataset.** The final objective averages over all $N$ training
samples:

$
\mathcal{L}_{\text{MRL}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{MRL}}^{(i)} = \frac{1}{N} \sum_{i=1}^{N} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}_m^{(i)} \tag{16}
$

This is Equation (9) written out in full.

### 10.3.3 Importance Weights

The weights $c_m$ control how much the optimizer prioritizes each nesting dimension.
The original paper sets all weights equal:

$
c_m = 1 \quad \forall \, m \in \mathcal{M} \tag{17}
$

This works well in practice because the loss at smaller dimensions is naturally larger
(fewer dimensions make the task harder), so the gradient contribution from small-$m$
terms is already amplified. Setting $c_m = 1$ lets this natural scaling do the work.

Alternative weighting schemes are possible but rarely improve results:

| Weighting | Formula | Effect |
|-----------|---------|--------|
| Uniform | $c_m = 1$ | Default, works well |
| Inverse dimension | $c_m = d / m$ | Upweights small prefixes |
| Log-scaled | $c_m = \log(d / m)$ | Mild upweighting of small prefixes |
| Accuracy-adaptive | $c_m \propto 1 - \text{acc}_m$ | Focus on struggling dimensions |

### 10.3.4 MRL-E: The Efficient Variant

The standard MRL objective requires $|\mathcal{M}|$ separate classifier weight matrices.
For $\mathcal{M} = \{8, 16, 32, \ldots, 2048\}$ with $L = 1000$ classes (ImageNet), this
means storing:

$
\sum_{m \in \mathcal{M}} L \times m = 1000 \times (8 + 16 + 32 + \cdots + 2048) = 1000 \times 4088 \approx 4\text{M parameters} \tag{18}
$

**MRL-E** (Efficient MRL) introduces **weight tying** across dimensions. Instead of
independent classifiers, all classifiers share a single weight matrix $W \in \mathbb{R}^{L \times d}$,
and each $W^{(m)}$ is simply the first $m$ columns:

$
W^{(m)} = W_{:, 1:m} \in \mathbb{R}^{L \times m} \tag{19}
$

This reduces the classifier parameters from $\sum_{m \in \mathcal{M}} L \times m$ to just
$L \times d$:

$
\text{MRL: } 1000 \times 4088 = 4{,}088{,}000 \text{ params}
$
$
\text{MRL-E: } 1000 \times 2048 = 2{,}048{,}000 \text{ params} \tag{20}
$

A ~50% reduction in classifier memory. The MRL-E objective becomes:

$
\min_{W, \theta_F} \; \frac{1}{N} \sum_{i=1}^{N} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}\!\left(W_{:, 1:m} \cdot F(x_i; \theta_F)_{1:m} \;;\; y_i\right) \tag{21}
$

The weight tying in MRL-E has an additional benefit: it forces the classifier to use
a consistent interpretation of each dimension across all nesting levels. Dimension $j$
means the same thing whether you are using a 64-d prefix or the full 2048-d vector.

---

## 10.4 Training Procedure — Step by Step

### 10.4.1 The Complete Forward-Backward Pass

Let us trace through one complete training step for a mini-batch of size $B$.

**Step 1 — Forward pass through the encoder.**

For each sample $x_i$ in the mini-batch, compute the full embedding:

$
\mathbf{z}_i = F(x_i; \theta_F) \in \mathbb{R}^d, \quad i \in \{1, \ldots, B\} \tag{22}
$

This is a single forward pass — the encoder runs once per sample, regardless of how
many nesting dimensions we use.

**Step 2 — Multi-scale prediction.**

For each nesting dimension $m \in \mathcal{M}$, extract the prefix and compute logits:

$
\mathbf{z}_{i, 1:m} = \mathbf{z}_i[1:m] \tag{23}
$

$
\text{logits}_i^{(m)} = W^{(m)} \cdot \mathbf{z}_{i, 1:m} \in \mathbb{R}^L \tag{24}
$

This step is cheap — it is just a matrix-vector multiply for each $m$, and the prefix
extraction is a zero-cost slice operation.

**Step 3 — Per-scale loss computation.**

For each $m$ and each sample $i$:

$
\mathcal{L}_m^{(i)} = -\log \frac{\exp\!\left(\text{logits}_{i, y_i}^{(m)}\right)}{\sum_{l=1}^{L} \exp\!\left(\text{logits}_{i, l}^{(m)}\right)} \tag{25}
$

**Step 4 — Aggregate the total loss.**

$
\mathcal{L}_{\text{total}} = \frac{1}{B} \sum_{i=1}^{B} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}_m^{(i)} \tag{26}
$

**Step 5 — Backward pass.**

Compute gradients of the total loss with respect to all parameters. The critical
observation is how gradients flow to the encoder:

$
\frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta_F} = \frac{1}{B} \sum_{i=1}^{B} \sum_{m \in \mathcal{M}} c_m \cdot \frac{\partial \mathcal{L}_m^{(i)}}{\partial \theta_F} \tag{27}
$

Each nesting dimension contributes its own gradient signal to the shared encoder. This
is the mechanism that creates the information hierarchy (Section 10.5).

**Step 6 — Parameter update.**

Standard gradient descent (or Adam, etc.):

$
\theta_F \leftarrow \theta_F - \eta \cdot \nabla_{\theta_F} \mathcal{L}_{\text{total}} \tag{28}
$

$
W^{(m)} \leftarrow W^{(m)} - \eta \cdot \nabla_{W^{(m)}} \mathcal{L}_{\text{total}}, \quad \forall \, m \in \mathcal{M} \tag{29}
$

### 10.4.2 Training Overhead Analysis

How much more expensive is MRL training compared to standard training?

**Encoder cost**: Identical. The encoder runs once per sample regardless.

**Classifier cost**: For each sample, we compute $|\mathcal{M}|$ matrix-vector products
instead of one. With $|\mathcal{M}| = 9$ (for $d = 2048$), this is 9× the classifier
cost. But the classifier is tiny compared to the encoder (a single linear layer vs.
a deep transformer or ResNet), so the overhead is negligible.

**Memory cost**: We store $|\mathcal{M}|$ sets of logits and losses. Again, negligible
compared to the encoder activations.

**Quantitative overhead**: The original paper reports MRL training is approximately
**1.1× to 1.2×** the cost of standard training — a 10–20% overhead for the ability to
deploy at any dimensionality.

### 10.4.3 Pseudocode

```python
import torch
import torch.nn.functional as F

def mrl_training_step(encoder, classifiers, batch_x, batch_y, 
                       nesting_dims, weights, optimizer):
    """
    One training step for Matryoshka Representation Learning.
    
    Args:
        encoder: Neural network F(·; θ_F) mapping inputs to ℝ^d
        classifiers: dict {m: W^(m)} for each m in nesting_dims
        batch_x: Input batch [B, ...]
        batch_y: Label batch [B] with values in {0, ..., L-1}
        nesting_dims: List of nesting dimensions M = [8, 16, ..., d]
        weights: dict {m: c_m} importance weights
        optimizer: Optimizer for all parameters
    
    Returns:
        total_loss: Scalar loss value
        per_dim_losses: dict {m: loss_m} for monitoring
    """
    optimizer.zero_grad()
    
    # Step 1: Forward pass — single encoder call
    z = encoder(batch_x)  # [B, d]
    
    total_loss = 0.0
    per_dim_losses = {}
    
    for m in nesting_dims:
        # Step 2: Extract prefix (zero-cost slice)
        z_prefix = z[:, :m]  # [B, m]
        
        # Step 3: Compute logits through per-dimension classifier
        logits = classifiers[m](z_prefix)  # [B, L]
        
        # Step 4: Cross-entropy loss at this scale
        loss_m = F.cross_entropy(logits, batch_y)
        per_dim_losses[m] = loss_m.item()
        
        # Step 5: Weighted accumulation
        total_loss += weights[m] * loss_m
    
    # Step 6: Backward pass and update
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), per_dim_losses
```

---

## 10.5 Information Hierarchy — Why It Works

### 10.5.1 The Gradient Imbalance

The most elegant aspect of Matryoshka training is that it creates an information hierarchy
*without any explicit regularization*. The hierarchy emerges naturally from the structure
of the multi-scale loss. To see why, let us analyze the gradient signal received by each
dimension of the embedding.

Consider a specific dimension $z_j$ of the embedding. Which loss terms in Equation (16)
produce a gradient signal for $z_j$?

The loss $\mathcal{L}_m$ depends on the prefix $\mathbf{z}_{1:m}$, which includes $z_j$
if and only if $j \leq m$. Therefore:

$
\frac{\partial \mathcal{L}_m}{\partial z_j} \neq 0 \quad \text{if and only if} \quad j \leq m \tag{30}
$

The total gradient signal received by dimension $z_j$ is:

$
\frac{\partial \mathcal{L}_{\text{MRL}}}{\partial z_j} = \sum_{\substack{m \in \mathcal{M} \\ m \geq j}} c_m \cdot \frac{\partial \mathcal{L}_m}{\partial z_j} \tag{31}
$

### 10.5.2 Counting Gradient Sources

Let us count how many loss terms contribute gradients to each dimension, using
$\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$:

| Dimension $z_j$ | Receives gradients from $m \in$ | Number of gradient sources |
|------------------|---------------------------------|---------------------------|
| $z_1$ | $\{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ | **9** |
| $z_8$ | $\{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ | **9** |
| $z_9$ | $\{16, 32, 64, 128, 256, 512, 1024, 2048\}$ | **8** |
| $z_{33}$ | $\{64, 128, 256, 512, 1024, 2048\}$ | **6** |
| $z_{129}$ | $\{256, 512, 1024, 2048\}$ | **4** |
| $z_{513}$ | $\{1024, 2048\}$ | **2** |
| $z_{1025}$ | $\{2048\}$ | **1** |

The pattern is clear: **early dimensions receive far more gradient signal than later
dimensions**. Dimension $z_1$ gets 9× more gradient sources than dimension $z_{1025}$.

### 10.5.3 The Coarse-to-Fine Consequence

This gradient imbalance has a profound consequence for what information gets stored where.

During training, the optimizer must reduce the loss at every scale simultaneously. The
small-prefix losses ($m = 8, 16$) are the hardest to reduce because they have the fewest
dimensions to work with. The optimizer's best strategy is:

1. **Pack the most globally discriminative features into the earliest dimensions** — these
   features help reduce loss at *every* scale.
2. **Reserve later dimensions for progressively finer distinctions** — features that only
   help when you have enough capacity to use them.

The result is a natural **coarse-to-fine information hierarchy**:

```
Dimensions  1–8:    Broadest semantic categories (e.g., animal vs. vehicle vs. building)
Dimensions  9–32:   Sub-categories (e.g., dog vs. cat vs. bird)
Dimensions 33–128:  Fine-grained distinctions (e.g., golden retriever vs. labrador)
Dimensions 129–512: Subtle attributes (e.g., puppy vs. adult, indoor vs. outdoor)
Dimensions 513+:    Instance-level details (e.g., specific individual, lighting, pose)
```

This is analogous to how JPEG compression works: low-frequency components (broad shapes)
come first, high-frequency components (fine details) come later. Matryoshka embeddings
achieve the same structure, but learned end-to-end for semantic representations.

### 10.5.4 The Interpolation Property

A remarkable empirical finding is that dimensions *not* in $\mathcal{M}$ also work well.
If $\mathcal{M} = \{8, 16, 32, 64, 128, 256\}$, then the prefix $\mathbf{z}_{1:48}$
(which was never explicitly trained) still performs well — its accuracy interpolates
smoothly between the 32-d and 64-d results.

Why? Because the coarse-to-fine hierarchy is a continuous property of the learned
representation, not a discrete artifact of the training dimensions. Dimensions 33–48
naturally encode information that is "between" the 32-d and 64-d levels of granularity.

This means you can deploy at *any* dimension, not just the ones in $\mathcal{M}$.

### 10.5.5 Formal Gradient Analysis

For completeness, let us derive the gradient of the MRL loss with respect to a single
embedding dimension $z_j$ through the chain rule.

For a single sample (dropping the index $i$ for clarity), the gradient of $\mathcal{L}_m$
with respect to $z_j$ (for $j \leq m$) is:

$
\frac{\partial \mathcal{L}_m}{\partial z_j} = \frac{\partial \mathcal{L}_m}{\partial \text{logits}^{(m)}} \cdot \frac{\partial \text{logits}^{(m)}}{\partial z_j} \tag{32}
$

The logits are $\text{logits}^{(m)} = W^{(m)} \mathbf{z}_{1:m}$, so:

$
\frac{\partial \text{logits}_l^{(m)}}{\partial z_j} = W_{l,j}^{(m)} \tag{33}
$

The cross-entropy gradient with respect to logits is the classic softmax-minus-label:

$
\frac{\partial \mathcal{L}_m}{\partial \text{logits}_l^{(m)}} = p_l^{(m)} - \mathbb{1}[l = y] \tag{34}
$

where $p_l^{(m)} = \text{softmax}(\text{logits}^{(m)})_l$. Combining:

$
\frac{\partial \mathcal{L}_m}{\partial z_j} = \sum_{l=1}^{L} \left(p_l^{(m)} - \mathbb{1}[l = y]\right) \cdot W_{l,j}^{(m)} = \left(\mathbf{p}^{(m)} - \mathbf{e}_y\right)^\top W_{:,j}^{(m)} \tag{35}
$

The total gradient on $z_j$ from the MRL loss is:

$
\frac{\partial \mathcal{L}_{\text{MRL}}}{\partial z_j} = \sum_{\substack{m \in \mathcal{M} \\ m \geq j}} c_m \cdot \left(\mathbf{p}^{(m)} - \mathbf{e}_y\right)^\top W_{:,j}^{(m)} \tag{36}
$

This confirms the gradient imbalance: $z_1$ receives a sum over all $|\mathcal{M}|$ terms,
while $z_d$ receives only the single term from $m = d$.

---

## 10.6 Normalization for Contrastive Learning

### 10.6.1 Why Normalization Matters

When Matryoshka training is combined with contrastive losses (as in sentence embedding
models), normalization becomes critical. Contrastive losses like InfoNCE operate on
cosine similarities:

$
\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|} \tag{37}
$

If we simply truncate a normalized full-dimensional vector, the prefix is *not* normalized:

$
\|\mathbf{z}\| = 1 \quad \not\Rightarrow \quad \|\mathbf{z}_{1:m}\| = 1 \tag{38}
$

In fact, $\|\mathbf{z}_{1:m}\|^2 = \sum_{j=1}^{m} z_j^2 \leq 1$, with equality only
when all the energy is concentrated in the first $m$ dimensions.

### 10.6.2 Per-Prefix Normalization

The solution is to normalize each prefix independently during training:

$
\tilde{\mathbf{z}}_{1:m} = \frac{\mathbf{z}_{1:m}}{\|\mathbf{z}_{1:m}\|_2} \tag{39}
$

This ensures that the contrastive loss at each scale operates on unit vectors, which is
essential for cosine similarity to be well-behaved.

The contrastive loss at scale $m$ then becomes:

$
\mathcal{L}_m^{\text{contrastive}} = -\log \frac{\exp\!\left(\text{sim}(\tilde{\mathbf{z}}_{1:m}^{(a)}, \tilde{\mathbf{z}}_{1:m}^{(p)}) / \tau\right)}{\sum_{j} \exp\!\left(\text{sim}(\tilde{\mathbf{z}}_{1:m}^{(a)}, \tilde{\mathbf{z}}_{1:m}^{(j)}) / \tau\right)} \tag{40}
$

where $\tau$ is the temperature, $(a, p)$ is an anchor-positive pair, and the sum in the
denominator runs over all candidates (positives and negatives).

### 10.6.3 Geometric Implications

Different dimensionalities have fundamentally different geometric properties. On the
unit sphere $S^{m-1}$ in $\mathbb{R}^m$:

- **Low dimensions** ($m = 8$): The sphere has limited surface area. Random unit vectors
  have relatively high expected cosine similarity. Discrimination is harder.
- **High dimensions** ($m = 768$): The sphere is vast. Random unit vectors are nearly
  orthogonal (expected cosine similarity $\approx 0$). Discrimination is easier.

The expected cosine similarity between two random unit vectors in $\mathbb{R}^m$ is:

$
\mathbb{E}[\text{cos\_sim}(\mathbf{u}, \mathbf{v})] = 0 \tag{41}
$

But the variance decreases with dimension:

$
\text{Var}[\text{cos\_sim}(\mathbf{u}, \mathbf{v})] = \frac{1}{m} \tag{42}
$

This means that in 8 dimensions, random vectors have cosine similarities spread across
$[-0.35, 0.35]$ (one standard deviation), while in 768 dimensions, they cluster tightly
around $[-0.036, 0.036]$. The contrastive loss must work harder to separate positives
from negatives in low dimensions.

### 10.6.4 Temperature Considerations

Some practitioners use dimension-dependent temperatures $\tau_m$ to compensate for the
geometric differences:

$
\tau_m = \tau_0 \cdot \sqrt{\frac{d}{m}} \tag{43}
$

This scales the temperature inversely with the "discriminability" of the space. In
practice, a single temperature often works well enough because the model learns to
compensate, but dimension-dependent temperatures can help when the range of $\mathcal{M}$
spans a very wide range (e.g., 8 to 2048).

---

## 10.7 Adaptation to Sentence Embeddings

### 10.7.1 From Classification to Contrastive MRL

The original MRL paper focuses on classification tasks (ImageNet). For sentence
embeddings, we replace the cross-entropy classification loss with a contrastive loss.
The structure of the Matryoshka wrapper remains the same — we simply swap the inner loss.

The Matryoshka contrastive objective becomes:

$
\mathcal{L}_{\text{MRL-contrastive}} = \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}_{\text{contrastive}}^{(m)} \tag{44}
$

where $\mathcal{L}_{\text{contrastive}}^{(m)}$ is any contrastive loss (InfoNCE,
MultipleNegativesRankingLoss, etc.) applied to the normalized $m$-dimensional prefixes.

### 10.7.2 MatryoshkaLoss in sentence-transformers

The `sentence-transformers` library provides a clean `MatryoshkaLoss` wrapper that
applies any base loss at multiple dimensionalities:

```python
from sentence_transformers import SentenceTransformer, losses

# Load a base model
model = SentenceTransformer("bert-base-uncased")

# Define the base contrastive loss
base_loss = losses.MultipleNegativesRankingLoss(model)

# Wrap it with MatryoshkaLoss
matryoshka_loss = losses.MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],  # Nesting dimensions M
    matryoshka_weights=[1, 1, 1, 1, 1],         # Importance weights c_m
)
```

Under the hood, `MatryoshkaLoss` does exactly what we described:

1. The model produces full 768-d embeddings (single forward pass).
2. For each $m \in \{768, 512, 256, 128, 64\}$, it truncates to the first $m$ dimensions.
3. It normalizes each truncated embedding.
4. It computes the base loss (MNRL) at each scale.
5. It returns the weighted sum.

### 10.7.3 Training Loop Example

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Prepare training data: (anchor, positive) pairs
train_examples = [
    InputExample(texts=["How do I reset my password?", 
                         "Steps to change your account password"]),
    InputExample(texts=["What is the return policy?",
                         "Our return and refund guidelines"]),
    # ... thousands more pairs
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

# Model and loss
model = SentenceTransformer("bert-base-uncased")
base_loss = losses.MultipleNegativesRankingLoss(model)
matryoshka_loss = losses.MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
    matryoshka_weights=[1, 1, 1, 1, 1],
)

# Train
model.fit(
    train_objectives=[(train_dataloader, matryoshka_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="matryoshka-model",
)
```

### 10.7.4 Matryoshka2dLoss — Combining Dimension and Layer Reduction

`Matryoshka2dLoss` extends the idea to two axes of flexibility:

1. **Dimension axis**: Truncate the embedding to $m$ dimensions (same as standard MRL).
2. **Layer axis**: Use the output of an intermediate transformer layer instead of the
   final layer.

This creates a 2D grid of operating points:

```
              Dimensions →
         64    128    256    512    768
Layer 12  ●      ●      ●      ●      ●    ← Full model
Layer 9   ●      ●      ●      ●      ●
Layer 6   ●      ●      ●      ●      ●    ← Half model
Layer 3   ●      ●      ●      ●      ●
Layer 1   ●      ●      ●      ●      ●    ← Minimal model
```

Each ● is a valid operating point. At inference, you can exit early at layer $l$ *and*
truncate to $m$ dimensions, giving you fine-grained control over the speed-accuracy
tradeoff.

```python
from sentence_transformers import losses

base_loss = losses.MultipleNegativesRankingLoss(model)
matryoshka_2d_loss = losses.Matryoshka2dLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
    n_layers_per_step=3,  # Train on every 3rd layer
)
```

---

## 10.8 Inference — Flexible Deployment

### 10.8.1 The Simplicity of Matryoshka Inference

At inference time, Matryoshka embeddings require no special machinery. The procedure is:

1. Run the encoder once to get the full embedding $\mathbf{z} \in \mathbb{R}^d$.
2. Truncate to the desired dimension: $\mathbf{z}_{1:k} = \mathbf{z}[:k]$.
3. (If using cosine similarity) Re-normalize: $\tilde{\mathbf{z}}_{1:k} = \mathbf{z}_{1:k} / \|\mathbf{z}_{1:k}\|_2$.

That's it. No dimensionality reduction model. No separate encoder. Just a slice and
an optional normalization.

```python
import numpy as np

def matryoshka_embed(model, text, target_dim):
    """
    Embed text at any target dimensionality using a Matryoshka model.
    
    Args:
        model: Trained Matryoshka sentence-transformer
        text: Input text string
        target_dim: Desired embedding dimensionality
    
    Returns:
        Normalized embedding of shape [target_dim]
    """
    # Step 1: Full forward pass
    full_embedding = model.encode(text)  # shape: [d]
    
    # Step 2: Truncate
    truncated = full_embedding[:target_dim]  # shape: [target_dim]
    
    # Step 3: Re-normalize for cosine similarity
    norm = np.linalg.norm(truncated)
    if norm > 0:
        truncated = truncated / norm
    
    return truncated
```

### 10.8.2 Adaptive Retrieval Cascade

The most powerful deployment pattern for Matryoshka embeddings is the **adaptive retrieval
cascade**. Instead of using a single dimensionality for the entire retrieval pipeline,
use progressively higher dimensions at each stage:

```
┌─────────────────────────────────────────────────────────────┐
│                    Query: "transformer efficiency"           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Coarse Filtering (64 dimensions)                  │
│  • Compare query against 100M document embeddings           │
│  • Use approximate nearest neighbor (ANN) search            │
│  • Retrieve top-10,000 candidates                           │
│  • Cost: ~5ms (tiny vectors, fast SIMD comparisons)         │
└──────────────────────────┬──────────────────────────────────┘
                           │ 10,000 candidates
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Re-ranking (256 dimensions)                       │
│  • Re-score 10,000 candidates with 256-d embeddings         │
│  • No re-encoding needed — just read more dimensions        │
│  • Select top-100                                           │
│  • Cost: ~20ms                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ 100 candidates
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Fine Re-ranking (768 dimensions or cross-encoder) │
│  • Re-score 100 candidates with full embeddings             │
│  • Or use a cross-encoder for maximum accuracy              │
│  • Return top-10 results                                    │
│  • Cost: ~50ms                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ 10 results
                           ▼
                      Final Results
```

The key insight is that **all three stages use the same stored embedding** — you just
read different numbers of dimensions from storage. This eliminates the need to store
multiple copies of each document embedding.

### 10.8.3 Storage Layout for Cascaded Retrieval

To support cascaded retrieval efficiently, store embeddings in a column-oriented layout
where the first $k$ dimensions are contiguous:

```python
import numpy as np

class MatryoshkaIndex:
    """
    Storage-efficient index for Matryoshka embeddings supporting
    cascaded retrieval at multiple dimensionalities.
    """
    
    def __init__(self, full_dim=768):
        self.full_dim = full_dim
        self.embeddings = None  # [N, full_dim] stored contiguously
    
    def add(self, embeddings):
        """Add full-dimensional embeddings to the index."""
        # Normalize the full embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query, dim, top_k, candidate_ids=None):
        """
        Search at a specific dimensionality.
        
        Args:
            query: Query embedding [full_dim]
            dim: Number of dimensions to use
            top_k: Number of results to return
            candidate_ids: Optional subset of IDs to search within
        
        Returns:
            top_k (id, score) pairs
        """
        # Truncate query and re-normalize
        q = query[:dim]
        q = q / np.linalg.norm(q)
        
        # Select candidates
        if candidate_ids is not None:
            docs = self.embeddings[candidate_ids, :dim]
        else:
            docs = self.embeddings[:, :dim]
        
        # Re-normalize document prefixes
        doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
        docs = docs / np.maximum(doc_norms, 1e-8)
        
        # Cosine similarity via dot product (both normalized)
        scores = docs @ q
        
        # Top-k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        if candidate_ids is not None:
            top_indices = candidate_ids[top_indices]
        
        return list(zip(top_indices, scores[top_indices]))
    
    def cascade_search(self, query, stages):
        """
        Multi-stage cascaded retrieval.
        
        Args:
            query: Full-dimensional query embedding
            stages: List of (dim, top_k) tuples
        
        Returns:
            Final top-k results
        """
        candidate_ids = None
        
        for dim, top_k in stages:
            results = self.search(query, dim, top_k, candidate_ids)
            candidate_ids = np.array([r[0] for r in results])
        
        return results


# Usage example
index = MatryoshkaIndex(full_dim=768)
# index.add(document_embeddings)  # Add your document embeddings

# Cascaded retrieval: 64d → 256d → 768d
# results = index.cascade_search(
#     query_embedding,
#     stages=[(64, 10000), (256, 100), (768, 10)]
# )
```

### 10.8.4 Memory Savings

The memory savings from using lower-dimensional embeddings are linear:

$
\text{Memory}(m) = N \times m \times \text{sizeof(float32)} \tag{45}
$

For 100 million documents:

| Dimension | Memory (float32) | Memory (float16) | Relative to 768d |
|-----------|-----------------|-----------------|-------------------|
| 768 | 286 GB | 143 GB | 1.00× |
| 256 | 95 GB | 48 GB | 0.33× |
| 128 | 48 GB | 24 GB | 0.17× |
| 64 | 24 GB | 12 GB | 0.08× |

Using 64-d Matryoshka embeddings instead of full 768-d embeddings gives a **12× memory
reduction** — the difference between needing a cluster and fitting on a single machine.

---

## 10.9 Practical Example — Complete Training Pipeline

### 10.9.1 End-to-End Matryoshka Training

Here is a complete, runnable example that trains Matryoshka embeddings from scratch on
a sentence similarity task and evaluates at each dimensionality.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Tuple

# ============================================================
# Part 1: Matryoshka Loss Implementation
# ============================================================

class MatryoshkaContrastiveLoss(nn.Module):
    """
    Matryoshka wrapper around in-batch contrastive loss (InfoNCE).
    
    For each nesting dimension m, truncates embeddings to m dimensions,
    normalizes, computes InfoNCE loss, and returns the weighted sum.
    """
    
    def __init__(self, 
                 matryoshka_dims: List[int],
                 matryoshka_weights: List[float] = None,
                 temperature: float = 0.05):
        super().__init__()
        self.dims = sorted(matryoshka_dims, reverse=True)
        self.weights = matryoshka_weights or [1.0] * len(self.dims)
        self.temperature = temperature
    
    def forward(self, 
                anchor_embeds: torch.Tensor, 
                positive_embeds: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            anchor_embeds: [B, d] full-dimensional anchor embeddings
            positive_embeds: [B, d] full-dimensional positive embeddings
        
        Returns:
            total_loss: Weighted sum of per-dimension losses
            loss_dict: {dim: loss_value} for monitoring
        """
        total_loss = torch.tensor(0.0, device=anchor_embeds.device)
        loss_dict = {}
        
        for dim, weight in zip(self.dims, self.weights):
            # Truncate to first `dim` dimensions
            a = anchor_embeds[:, :dim]    # [B, dim]
            p = positive_embeds[:, :dim]  # [B, dim]
            
            # L2 normalize each prefix independently
            a = F_torch.normalize(a, p=2, dim=1)
            p = F_torch.normalize(p, p=2, dim=1)
            
            # Compute similarity matrix: [B, B]
            similarity = torch.mm(a, p.t()) / self.temperature
            
            # Labels: diagonal entries are positives
            labels = torch.arange(a.size(0), device=a.device)
            
            # InfoNCE loss (cross-entropy over similarity matrix)
            loss_m = F_torch.cross_entropy(similarity, labels)
            
            loss_dict[dim] = loss_m.item()
            total_loss = total_loss + weight * loss_m
        
        return total_loss, loss_dict
```

```python
# ============================================================
# Part 2: Simple Encoder for Demonstration
# ============================================================

class SimpleEncoder(nn.Module):
    """
    A simple MLP encoder for demonstration purposes.
    In practice, replace with a transformer (BERT, etc.).
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# Part 3: Training Loop
# ============================================================

def train_matryoshka(
    encoder: nn.Module,
    train_loader: DataLoader,
    matryoshka_dims: List[int],
    num_epochs: int = 10,
    lr: float = 1e-3,
):
    """Train an encoder with Matryoshka contrastive loss."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    
    loss_fn = MatryoshkaContrastiveLoss(
        matryoshka_dims=matryoshka_dims,
        temperature=0.05,
    )
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        encoder.train()
        epoch_losses = {dim: 0.0 for dim in matryoshka_dims}
        num_batches = 0
        
        for anchors, positives in train_loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            
            # Forward pass: encode both views
            z_anchor = encoder(anchors)
            z_positive = encoder(positives)
            
            # Matryoshka loss
            loss, loss_dict = loss_fn(z_anchor, z_positive)
            
            # Backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for dim in matryoshka_dims:
                epoch_losses[dim] += loss_dict[dim]
            num_batches += 1
        
        # Report per-dimension losses
        print(f"Epoch {epoch+1}/{num_epochs}")
        for dim in sorted(matryoshka_dims):
            avg_loss = epoch_losses[dim] / num_batches
            print(f"  dim={dim:4d}: loss={avg_loss:.4f}")
    
    return encoder
```

### 10.9.2 Evaluation at Each Dimensionality

After training, evaluate the model at each nesting dimension to see the accuracy-dimension
tradeoff:

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_matryoshka(
    encoder: nn.Module,
    eval_pairs: List[Tuple],  # (query, positive, negatives)
    matryoshka_dims: List[int],
) -> Dict[int, float]:
    """
    Evaluate retrieval accuracy at each Matryoshka dimension.
    
    For each query, check if the positive document is ranked #1
    among the positive + negatives.
    
    Returns:
        {dim: accuracy} for each dimension
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    results = {}
    
    for dim in sorted(matryoshka_dims):
        correct = 0
        total = 0
        
        for query, positive, negatives in eval_pairs:
            with torch.no_grad():
                # Encode all texts
                q_emb = encoder(query.to(device)).cpu().numpy()[:, :dim]
                p_emb = encoder(positive.to(device)).cpu().numpy()[:, :dim]
                n_embs = encoder(negatives.to(device)).cpu().numpy()[:, :dim]
            
            # Normalize
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            p_emb = p_emb / np.linalg.norm(p_emb, axis=1, keepdims=True)
            n_embs = n_embs / np.linalg.norm(n_embs, axis=1, keepdims=True)
            
            # Compute similarities
            all_docs = np.vstack([p_emb, n_embs])
            sims = cosine_similarity(q_emb, all_docs)[0]
            
            # Check if positive is ranked first
            if np.argmax(sims) == 0:
                correct += 1
            total += 1
        
        accuracy = correct / total
        results[dim] = accuracy
        print(f"  dim={dim:4d}: accuracy={accuracy:.4f}")
    
    return results
```

### 10.9.3 Typical Results

Matryoshka embeddings show a characteristic accuracy-dimension curve. Here are typical
results from the original paper on ImageNet-1K classification and from sentence embedding
benchmarks:

**ImageNet-1K (ResNet-50, $d = 2048$):**

| Dimension | MRL Accuracy | Fixed-dim Baseline | % of Full Accuracy |
|-----------|-------------|-------------------|-------------------|
| 8 | 39.0% | 22.0% | 52.3% |
| 16 | 54.1% | 44.0% | 72.6% |
| 32 | 64.3% | 58.5% | 86.3% |
| 64 | 70.1% | 66.8% | 94.1% |
| 128 | 73.0% | 71.5% | 97.9% |
| 256 | 74.2% | 73.5% | 99.6% |
| 2048 | 74.5% | 74.5% | 100.0% |

**Key observations:**

1. **No loss at full dimension**: MRL at $d = 2048$ matches the standard baseline exactly.
   The multi-scale training does not hurt full-dimensional performance.

2. **Massive gains at low dimensions**: At $d = 64$, MRL achieves 70.1% vs. 66.8% for
   a model trained specifically at 64 dimensions. MRL *beats dedicated low-dimensional
   models* because the shared encoder benefits from the multi-scale training signal.

3. **The 94–98% sweet spot**: At $d = 64$ (32× compression), MRL retains ~94% of full
   accuracy. At $d = 128$ (16× compression), it retains ~98%. This is the practical
   sweet spot for most deployment scenarios.

**Sentence Embeddings (BERT-base, $d = 768$):**

| Dimension | STS Benchmark (Spearman) | Retrieval NDCG@10 | % of Full |
|-----------|-------------------------|-------------------|-----------|
| 64 | 0.79 | 0.42 | ~93% |
| 128 | 0.82 | 0.46 | ~96% |
| 256 | 0.84 | 0.48 | ~98% |
| 512 | 0.85 | 0.49 | ~99% |
| 768 | 0.85 | 0.50 | 100% |

---

## 10.10 Comparison with Alternatives

### 10.10.1 MRL vs. PCA (Post-hoc Dimensionality Reduction)

**PCA** (Principal Component Analysis) is the most common post-hoc method for reducing
embedding dimensionality. After training a standard model, you fit PCA on the training
embeddings and project all embeddings to a lower dimension.

The PCA projection for dimension $m$ is:

$
\mathbf{z}_{\text{PCA}}^{(m)} = U_m^\top \mathbf{z} \in \mathbb{R}^m \tag{46}
$

where $U_m \in \mathbb{R}^{d \times m}$ contains the top-$m$ eigenvectors of the
embedding covariance matrix.

**Comparison:**

| Aspect | MRL | PCA |
|--------|-----|-----|
| Training cost | 1.1–1.2× standard | 1× standard + PCA fit |
| Inference cost | Slice (free) | Matrix multiply ($O(dm)$) |
| Storage | One vector per doc | One vector per doc + PCA matrix |
| Quality at low dim | Optimized end-to-end | Suboptimal (linear projection) |
| Flexibility | Any prefix length | Requires pre-computed PCA per $m$ |
| Interpolation | Smooth | Smooth |

MRL wins on every axis except training cost (marginal difference). The key advantage is
that MRL *directly optimizes* the low-dimensional representations during training, while
PCA can only find the best linear projection of representations that were optimized for
the full dimension.

### 10.10.2 MRL vs. Knowledge Distillation

**Knowledge distillation** trains a small student model to mimic a large teacher model:

$
\mathcal{L}_{\text{KD}} = \text{KL}\!\left(\sigma(z_{\text{teacher}} / T) \;\|\; \sigma(z_{\text{student}} / T)\right) \tag{47}
$

where $T$ is the distillation temperature and $\sigma$ is softmax.

To get embeddings at multiple dimensionalities, you need to train a separate student for
each target dimension — $k$ dimensions means $k$ separate training runs.

**Comparison:**

| Aspect | MRL | Knowledge Distillation |
|--------|-----|----------------------|
| Training runs | 1 | $k$ (one per dimension) |
| Models to maintain | 1 | $k$ |
| Quality | Excellent | Excellent (per model) |
| Flexibility | Any dimension | Only trained dimensions |
| Deployment complexity | Trivial | Manage $k$ models |

MRL achieves comparable quality to dedicated distilled models at each dimension, but with
a single training run and a single deployed model.

### 10.10.3 MRL vs. Fixed Low-Dimensional Training

The simplest alternative: just train a model with a smaller output dimension.

$
F_m(\cdot\,; \theta_m): \mathcal{X} \rightarrow \mathbb{R}^m \tag{48}
$

**Comparison:**

| Aspect | MRL (at dim $m$) | Fixed $m$-dim model |
|--------|-----------------|-------------------|
| Accuracy at $m$ | Equal or better | Baseline |
| Accuracy at other dims | Available | Not available |
| Training cost | 1.1× (once) | 1× (per dimension) |
| Total cost for $k$ dims | 1.1× | $k$× |

The surprising result from the original paper is that MRL at dimension $m$ often
*outperforms* a model trained specifically at dimension $m$. The multi-scale training
acts as a form of regularization that improves representations at every scale.

### 10.10.4 MRL vs. Slimmable Networks

**Slimmable Networks** (Yu et al., 2019) achieve flexible inference by training networks
that can run at different widths (number of channels). The flexibility is at the
*architecture* level — you use fewer neurons in each layer.

MRL operates at the *representation* level — the architecture is fixed, but the output
embedding can be truncated. This is a cleaner separation of concerns:

| Aspect | MRL | Slimmable Networks |
|--------|-----|-------------------|
| Flexibility level | Representation | Architecture |
| Encoder changes | None | Width-switchable layers |
| Implementation | Loss wrapper | Architecture modification |
| Applicable to | Any encoder | Specific architectures |
| Inference savings | Storage + similarity | Full compute |

MRL is simpler to implement and applies to any encoder architecture. Slimmable networks
offer deeper compute savings but require architectural changes.

---

## 10.11 Applications

### 10.11.1 Adaptive Classification

In classification, Matryoshka embeddings enable a single model to serve multiple accuracy
tiers:

$
\hat{y}^{(m)} = \arg\max_{l} \left[ W^{(m)} \cdot \mathbf{z}_{1:m} \right]_l \tag{49}
$

The original paper demonstrates **14× smaller representations** (128-d vs. 2048-d) with
less than 1% accuracy drop on ImageNet. This translates directly to:

- **14× less memory** for storing embeddings
- **14× faster** nearest-neighbor lookups
- **14× less bandwidth** for transmitting embeddings

### 10.11.2 Adaptive Retrieval

For large-scale retrieval systems, the cascade pattern from Section 10.8.2 provides
dramatic speedups. The cost of a nearest-neighbor search scales with the embedding
dimension:

$
\text{Cost}_{\text{brute-force}}(m, N) = O(N \cdot m) \tag{50}
$

$
\text{Cost}_{\text{ANN}}(m, N) = O(m \cdot \log N) \tag{51}
$

Using 64-d embeddings for the first pass instead of 768-d gives a **12× speedup** in
the similarity computation, which dominates the retrieval cost for large collections.

Combined with the cascade pattern:

$
\text{Total cost} = \underbrace{O(N \cdot m_1)}_{\text{coarse filter}} + \underbrace{O(k_1 \cdot m_2)}_{\text{re-rank stage 1}} + \underbrace{O(k_2 \cdot m_3)}_{\text{re-rank stage 2}} \tag{52}
$

where $k_1 \gg k_2$ and $m_1 \ll m_2 \ll m_3$. The expensive full-dimensional comparison
is only applied to a tiny fraction of the collection.

### 10.11.3 Multi-Tenant Serving

In a SaaS embedding service, different customers have different needs:

```
┌─────────────────────────────────────────────────────┐
│              Single Matryoshka Model                 │
│                                                     │
│  Free tier:     → 64-d embeddings   (fast, cheap)   │
│  Pro tier:      → 256-d embeddings  (balanced)      │
│  Enterprise:    → 768-d embeddings  (maximum quality)│
└─────────────────────────────────────────────────────┘
```

One model serves all tiers. No need to train, deploy, or maintain separate models for
each service level. The cost difference between tiers is purely in storage and compute
for similarity search — the encoding cost is identical.

### 10.11.4 Edge Deployment

For on-device applications (mobile, IoT), Matryoshka embeddings enable graceful
degradation based on available resources:

```python
def get_embedding_dim(device_profile):
    """Select embedding dimension based on device capabilities."""
    if device_profile == "high_end_phone":
        return 256  # Good accuracy, moderate memory
    elif device_profile == "low_end_phone":
        return 64   # Acceptable accuracy, minimal memory
    elif device_profile == "iot_sensor":
        return 16   # Coarse categorization only
    elif device_profile == "server":
        return 768  # Full accuracy
    else:
        return 128  # Safe default
```

The same model binary is deployed everywhere. Only the truncation point changes.

### 10.11.5 Progressive Loading

In bandwidth-constrained environments, Matryoshka embeddings enable **progressive
loading** — send the most important dimensions first:

1. Send $\mathbf{z}_{1:64}$ immediately → user sees coarse results in < 100ms.
2. Stream $\mathbf{z}_{65:256}$ → results refine as more data arrives.
3. Stream $\mathbf{z}_{257:768}$ → final high-quality results.

This is analogous to progressive JPEG loading, where a blurry image appears instantly
and sharpens as more data downloads. The user experience is dramatically better than
waiting for the full embedding before showing any results.

```python
async def progressive_search(query, index, websocket):
    """
    Progressive retrieval: send increasingly refined results
    as more embedding dimensions become available.
    """
    full_query = encode(query)  # Full 768-d embedding
    
    stages = [
        (64,  100, "coarse"),
        (256, 20,  "refined"),
        (768, 10,  "final"),
    ]
    
    for dim, top_k, quality in stages:
        results = index.search(full_query, dim=dim, top_k=top_k)
        await websocket.send({
            "quality": quality,
            "dimensions": dim,
            "results": results,
        })
```

### 10.11.6 Funnel Search with Matryoshka Embeddings

A concrete implementation of the cascade pattern for production systems:

```python
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class SearchResult:
    doc_id: int
    score: float
    stage: str

def funnel_search(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    stages: List[dict],
) -> List[SearchResult]:
    """
    Multi-stage funnel search using Matryoshka embeddings.
    
    Each stage uses more dimensions on fewer candidates.
    
    Args:
        query_embedding: Full-dimensional query vector [d]
        document_embeddings: All document vectors [N, d]
        stages: List of {"dim": int, "top_k": int} dicts
    
    Returns:
        Final ranked results
    
    Example:
        stages = [
            {"dim": 64,  "top_k": 10000},  # Coarse: scan all docs
            {"dim": 256, "top_k": 500},     # Medium: re-rank 10K
            {"dim": 768, "top_k": 20},      # Fine: re-rank 500
        ]
    """
    candidate_ids = np.arange(len(document_embeddings))
    
    for stage in stages:
        dim = stage["dim"]
        top_k = stage["top_k"]
        
        # Truncate and normalize query
        q = query_embedding[:dim].copy()
        q /= np.linalg.norm(q) + 1e-8
        
        # Truncate and normalize candidates
        docs = document_embeddings[candidate_ids, :dim].copy()
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        docs /= np.maximum(norms, 1e-8)
        
        # Score via dot product
        scores = docs @ q
        
        # Keep top-k
        if len(scores) > top_k:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        else:
            top_idx = np.argsort(scores)[::-1]
        
        candidate_ids = candidate_ids[top_idx]
        final_scores = scores[top_idx]
    
    return [
        SearchResult(doc_id=int(cid), score=float(s), stage=f"{dim}d")
        for cid, s in zip(candidate_ids, final_scores)
    ]
```

---

## 10.12 Summary

### 10.12.1 Key Equations

The core of Matryoshka Representation Learning is captured in a single objective:

$
\mathcal{L}_{\text{MRL}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}\!\left(W^{(m)} \cdot F(x_i; \theta_F)_{1:m} \;;\; y_i\right) \tag{53}
$

The efficient variant ties classifier weights:

$
W^{(m)} = W_{:, 1:m} \quad \text{(MRL-E)} \tag{54}
$

The information hierarchy emerges from the gradient structure:

$
\frac{\partial \mathcal{L}_{\text{MRL}}}{\partial z_j} = \sum_{\substack{m \in \mathcal{M} \\ m \geq j}} c_m \cdot \frac{\partial \mathcal{L}_m}{\partial z_j} \tag{55}
$

where early dimensions receive gradients from all scales and later dimensions from
fewer scales.

### 10.12.2 Practical Takeaways

1. **Training overhead is minimal** (~10–20%) for the ability to deploy at any dimension.

2. **No accuracy loss at full dimension** — MRL matches standard training when using all
   dimensions.

3. **The 64-d sweet spot**: For most tasks, 64 dimensions retain 93–95% of full accuracy
   while providing 12× memory savings and proportional speedups.

4. **Cascade retrieval** is the killer application: use cheap low-dimensional search for
   coarse filtering, then progressively refine with more dimensions.

5. **Implementation is trivial**: wrap your existing loss with `MatryoshkaLoss` in
   sentence-transformers, or add a loop over prefix lengths in custom code.

6. **Weight tying (MRL-E)** saves ~50% of classifier parameters with no quality loss.

7. **Always re-normalize** after truncation when using cosine similarity.

### 10.12.3 When to Use Matryoshka Embeddings

| Scenario | Use MRL? | Why |
|----------|----------|-----|
| Single fixed deployment | Optional | No flexibility needed, but no cost either |
| Multiple deployment targets | **Yes** | One model replaces many |
| Large-scale retrieval | **Yes** | Cascade search is a game-changer |
| Resource-constrained devices | **Yes** | Adapt to device capabilities |
| Multi-tenant SaaS | **Yes** | Different tiers from one model |
| Research / prototyping | **Yes** | Explore accuracy-dimension tradeoff for free |
| Maximum accuracy only | Optional | MRL matches standard at full dim |

### 10.12.4 Historical Context

Matryoshka Representation Learning was introduced by Kusupati et al. at NeurIPS 2022.
The paper demonstrated the approach on vision (ImageNet with ResNet-50) and retrieval
tasks, showing that a single MRL-trained model dominates independently trained models
at every dimensionality. The idea has since been adopted by major embedding providers —
OpenAI's `text-embedding-3-small` and `text-embedding-3-large` models support native
dimension selection, and the `sentence-transformers` library provides built-in
`MatryoshkaLoss` and `Matryoshka2dLoss` wrappers.

The elegance of MRL lies in its simplicity: a minor modification to the training loss
yields a representation with a fundamentally new capability — flexible dimensionality —
at essentially zero cost.

---

*Next chapter: [Chapter 11 — GIST Embeddings](./11-gist-embeddings.md)*
