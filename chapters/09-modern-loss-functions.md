# Chapter 9: Modern Loss Functions — Circle Loss, CoSENT, and AnglE

## Table of Contents

1. [Introduction — Why New Loss Functions?](#91-introduction--why-new-loss-functions)
2. [Circle Loss (2020)](#92-circle-loss-2020)
3. [CoSENT Loss (2022/2024)](#93-cosent-loss-20222024)
4. [AnglE Loss (2023/2024, ACL 2024)](#94-angle-loss-20232024-acl-2024)
5. [Comparing Modern Loss Functions](#95-comparing-modern-loss-functions)
6. [Summary](#96-summary)

---

## 9.1 Introduction — Why New Loss Functions?

### 9.1.1 The State of Affairs Before 2020

By 2020, the embedding community had converged on a small set of well-understood loss
functions: contrastive loss, triplet loss, N-pair / InfoNCE loss, and softmax-based
angular margin losses (ArcFace, CosFace). These losses powered everything from face
recognition to sentence embeddings. They worked — but they had fundamental limitations
that became increasingly apparent as models scaled and tasks diversified.

Three problems, in particular, motivated the development of modern loss functions:

1. **The cosine saturation problem**
2. **The training-inference inconsistency problem**
3. **The rigid weighting problem**

Let's examine each in detail.

### 9.1.2 The Cosine Saturation Problem

Cosine similarity maps all embedding pairs to the interval $[-1, 1]$. When embeddings
are well-trained, positive pairs cluster near $\cos(\mathbf{u}, \mathbf{v}) \approx 1$
and negative pairs near $\cos(\mathbf{u}, \mathbf{v}) \approx -1$ or $0$. This creates
**saturation zones** where the gradient of cosine similarity vanishes.

Recall the gradient of cosine similarity with respect to $\mathbf{u}$ (for L2-normalized
embeddings $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):

$
\frac{\partial \cos(\mathbf{u}, \mathbf{v})}{\partial \mathbf{u}} = \frac{\mathbf{v} - \cos(\mathbf{u}, \mathbf{v}) \cdot \mathbf{u}}{\|\mathbf{u}\|}
$

For normalized vectors, this simplifies to:

$
\frac{\partial \cos(\mathbf{u}, \mathbf{v})}{\partial \mathbf{u}} = \mathbf{v} - \cos(\mathbf{u}, \mathbf{v}) \cdot \mathbf{u} \quad \text{...(Eq. 9.1)}
$

Now consider what happens at the saturation zones:

**When $\cos(\mathbf{u}, \mathbf{v}) \to 1$** (positive pairs pushed to perfect alignment):

$
\frac{\partial \cos}{\partial \mathbf{u}} = \mathbf{v} - 1 \cdot \mathbf{u} \approx \mathbf{0}
$

because $\mathbf{u} \approx \mathbf{v}$ when cosine similarity is near 1.

**When $\cos(\mathbf{u}, \mathbf{v}) \to -1$** (negative pairs pushed to perfect opposition):

$
\frac{\partial \cos}{\partial \mathbf{u}} = \mathbf{v} - (-1) \cdot \mathbf{u} = \mathbf{v} + \mathbf{u} \approx \mathbf{0}
$

because $\mathbf{u} \approx -\mathbf{v}$ when cosine similarity is near $-1$.

In both cases, the gradient magnitude $\|\frac{\partial \cos}{\partial \mathbf{u}}\|$
approaches zero. The model stops learning even though the embeddings may not be optimal.

This is particularly problematic with **binary STS labels** (0 or 1), which push all
positive pairs toward $\cos = 1$ and all negative pairs toward $\cos = 0$ or $\cos = -1$.
The model quickly enters the saturation zone and gradient flow stalls.

### 9.1.3 The Training-Inference Inconsistency Problem

The dominant paradigm for sentence embedding training before CoSENT was the cross-encoder
style objective used in SBERT (Reimers & Gurevych, 2019):

**During training**: Sentence pairs are concatenated and fed through a classifier:

$
\text{score} = \text{softmax}(\mathbf{W} \cdot [\mathbf{u}; \mathbf{v}; |\mathbf{u} - \mathbf{v}|])
$

where $[\mathbf{u}; \mathbf{v}; |\mathbf{u} - \mathbf{v}|]$ is the concatenation of the
two sentence embeddings and their element-wise absolute difference.

**During inference**: Similarity is computed as cosine similarity:

$
\text{score} = \cos(\mathbf{u}, \mathbf{v})
$

The training objective optimizes a learned linear combination of $\mathbf{u}$, $\mathbf{v}$,
and $|\mathbf{u} - \mathbf{v}|$ through a softmax classifier. But at inference time, we
use only the dot product (cosine similarity). The model is optimized for one function but
evaluated with another — a fundamental mismatch.

### 9.1.4 The Rigid Weighting Problem

Traditional losses like triplet loss and contrastive loss treat all pairs equally within
their category. Every positive pair receives the same gradient weight; every negative pair
within the margin receives the same treatment. But intuitively:

- A positive pair with similarity 0.3 needs much more optimization than one with similarity 0.9
- A negative pair with similarity 0.8 (a hard negative) is far more informative than one
  with similarity 0.1 (an easy negative)

InfoNCE partially addresses this through softmax weighting, but the weighting is implicit
and coupled to the temperature parameter. What we want is **explicit, self-paced weighting**
where the loss function automatically focuses on the pairs that need the most work.

These three problems — cosine saturation, training-inference inconsistency, and rigid
weighting — motivated the development of Circle Loss, CoSENT, and AnglE. Each addresses
a different subset of these issues with elegant mathematical solutions.

---

## 9.2 Circle Loss (2020)

### 9.2.1 Motivation: A Unified Perspective

Circle Loss (Sun et al., 2020) starts from a powerful observation: most existing loss
functions can be viewed as special cases of a **unified pairwise similarity optimization**
framework. The authors show that contrastive loss, triplet loss, N-pair loss, and softmax
cross-entropy all optimize the same underlying objective — they just differ in how they
weight the similarity scores.

The key question Circle Loss asks: *What is the optimal weighting strategy?*

### 9.2.2 The Unified Loss Formulation

Consider a set of similarity scores for an anchor $x$:

- **Positive similarities**: $s_p^i = \text{sim}(x, x_p^i)$ for $i = 1, \ldots, K$
  (similarities with positive examples)
- **Negative similarities**: $s_n^j = \text{sim}(x, x_n^j)$ for $j = 1, \ldots, L$
  (similarities with negative examples)

The fundamental goal of metric learning is to ensure that all positive similarities
exceed all negative similarities:

$
s_p^i > s_n^j \quad \forall \, i, j \quad \text{...(Eq. 9.2)}
$

This can be reformulated as minimizing the following unified loss:

$
\mathcal{L}_{\text{uni}} = \log\left[1 + \sum_{j=1}^{L} \sum_{i=1}^{K} \exp\big(\gamma(s_n^j - s_p^i)\big)\right] \quad \text{...(Eq. 9.3)}
$

where $\gamma > 0$ is a scale factor.

**Why this formulation?** Let's derive it step by step.

**Step 1: Start from the ranking constraint.**

We want $s_p^i - s_n^j > 0$ for all $(i, j)$ pairs. A soft version of this constraint
uses the exponential penalty:

$
\text{penalty}(i, j) = \exp\big(\gamma(s_n^j - s_p^i)\big)
$

When $s_p^i > s_n^j$, the exponent is negative and the penalty is small (approaching 0).
When $s_n^j > s_p^i$ (a violation), the exponent is positive and the penalty is large.

**Step 2: Aggregate over all pairs.**

Sum the penalties over all $(i, j)$ combinations:

$
\text{total penalty} = \sum_{j=1}^{L} \sum_{i=1}^{K} \exp\big(\gamma(s_n^j - s_p^i)\big)
$

**Step 3: Apply log-sum-exp for numerical stability.**

Adding 1 inside the log prevents $\log(0)$ when all constraints are satisfied:

$
\mathcal{L}_{\text{uni}} = \log\left[1 + \sum_{j=1}^{L} \sum_{i=1}^{K} \exp\big(\gamma(s_n^j - s_p^i)\big)\right]
$

**Step 4: Factor the double sum.**

A crucial algebraic property: the double sum factors into a product of single sums:

$
\sum_{j} \sum_{i} \exp\big(\gamma(s_n^j - s_p^i)\big) = \sum_{j} \exp(\gamma s_n^j) \cdot \sum_{i} \exp(-\gamma s_p^i)
$

Therefore:

$
\mathcal{L}_{\text{uni}} = \log\left[1 + \left(\sum_{j=1}^{L} \exp(\gamma s_n^j)\right) \cdot \left(\sum_{i=1}^{K} \exp(-\gamma s_p^i)\right)\right] \quad \text{...(Eq. 9.4)}
$

This factored form reveals that the loss decomposes into two independent LogSumExp
aggregations — one over negative similarities (pushing them down) and one over positive
similarities (pushing them up).

### 9.2.3 Showing Existing Losses as Special Cases

**Triplet loss** is recovered when $K = L = 1$ (one positive, one negative):

$
\mathcal{L}_{\text{uni}} = \log\big[1 + \exp\big(\gamma(s_n - s_p)\big)\big] = \text{softplus}\big(\gamma(s_n - s_p)\big)
$

This is the smooth approximation of $\max(s_n - s_p + \alpha, 0)$ with $\alpha$ absorbed
into the scale $\gamma$.

**N-pair / InfoNCE loss** is recovered when $K = 1$ (one positive, $L$ negatives):

$
\mathcal{L}_{\text{uni}} = \log\left[1 + \sum_{j=1}^{L} \exp\big(\gamma(s_n^j - s_p)\big)\right]
$

which is exactly the N-pair loss from Section 7.4.

**Softmax cross-entropy** with angular margins can also be expressed in this framework
by treating the class weight vectors as "positive" and "negative" prototypes.

### 9.2.4 The Problem with Uniform Weighting

In the unified loss (Eq. 9.3), every positive-negative pair $(s_p^i, s_n^j)$ contributes
equally to the exponent $\gamma(s_n^j - s_p^i)$. But consider two scenarios:

**Scenario A**: $s_p^i = 0.9$ (already well-optimized positive) and $s_n^j = 0.1$
(already well-separated negative). The pair contributes $\exp(\gamma(0.1 - 0.9)) = \exp(-0.8\gamma)$,
which is tiny. Yet the loss still allocates gradient to this pair.

**Scenario B**: $s_p^i = 0.3$ (under-optimized positive) and $s_n^j = 0.7$ (dangerous
hard negative). The pair contributes $\exp(\gamma(0.7 - 0.3)) = \exp(0.4\gamma)$, which
is large. This pair genuinely needs optimization.

The uniform weighting wastes gradient on well-optimized pairs (Scenario A) when it should
focus on problematic pairs (Scenario B).

### 9.2.5 Circle Loss: Self-Paced Weighting

Circle Loss introduces **adaptive weighting factors** $\alpha_p^i$ and $\alpha_n^j$ that
automatically focus optimization on the pairs that need it most:

$
\mathcal{L}_{\text{circle}} = \log\left[1 + \sum_{j=1}^{L} \sum_{i=1}^{K} \exp\big(\gamma(\alpha_n^j \, s_n^j - \alpha_p^i \, s_p^i)\big)\right] \quad \text{...(Eq. 9.5)}
$

The weighting factors are defined using **detached** (stop-gradient) similarity scores
and optimality targets:

$
\alpha_n^j = \big[s_n^j - O_n\big]_+ \quad \text{...(Eq. 9.6)}
$

$
\alpha_p^i = \big[O_p - s_p^i\big]_+ \quad \text{...(Eq. 9.7)}
$

where:
- $[x]_+ = \max(x, 0)$ is the ReLU function
- $O_p$ is the **optimality target for positives** (typically $O_p = 1 + m$ for cosine
  similarity, where $m$ is a margin)
- $O_n$ is the **optimality target for negatives** (typically $O_n = -m$)
- The similarities $s_n^j$ and $s_p^i$ inside $\alpha$ are **detached** from the
  computation graph (no gradient flows through them)

### 9.2.6 Intuition Behind the Weighting

Let's unpack what these weighting factors do:

**For negative similarities** ($\alpha_n^j = [s_n^j - O_n]_+$):

- If $s_n^j > O_n$ (negative is too similar — dangerous): $\alpha_n^j = s_n^j - O_n > 0$.
  The weight is large, amplifying the gradient to push this negative away.
- If $s_n^j \leq O_n$ (negative is already well-separated): $\alpha_n^j = 0$.
  No gradient flows — the loss ignores this already-solved pair.

**For positive similarities** ($\alpha_p^i = [O_p - s_p^i]_+$):

- If $s_p^i < O_p$ (positive is not similar enough — needs work): $\alpha_p^i = O_p - s_p^i > 0$.
  The weight is large, amplifying the gradient to pull this positive closer.
- If $s_p^i \geq O_p$ (positive is already well-aligned): $\alpha_p^i = 0$.
  No gradient flows — the loss ignores this already-solved pair.

This is **self-paced learning**: the loss function automatically identifies which pairs
need optimization and allocates gradient proportionally to the gap between current
similarity and the optimality target.

### 9.2.7 Step-by-Step Derivation of the Decision Boundary

The name "Circle Loss" comes from the geometric shape of the decision boundary in the
$(s_n, s_p)$ plane. Let's derive it.

**Step 1: Consider a single positive-negative pair.**

For one positive similarity $s_p$ and one negative similarity $s_n$, the Circle Loss is:

$
\mathcal{L} = \log\big[1 + \exp\big(\gamma(\alpha_n s_n - \alpha_p s_p)\big)\big]
$

The loss is zero (approximately) when:

$
\alpha_n s_n - \alpha_p s_p < 0
$

Substituting the weighting factors:

$
(s_n - O_n) \cdot s_n - (O_p - s_p) \cdot s_p < 0
$

$
s_n^2 - O_n \cdot s_n - O_p \cdot s_p + s_p^2 < 0 \quad \text{...(Eq. 9.8)}
$

**Step 2: Rearrange into standard form.**

$
s_n^2 - O_n \cdot s_n + s_p^2 - O_p \cdot s_p < 0
$

Complete the square for both variables:

$
\left(s_n - \frac{O_n}{2}\right)^2 - \frac{O_n^2}{4} + \left(s_p - \frac{O_p}{2}\right)^2 - \frac{O_p^2}{4} < 0
$

$
\left(s_n - \frac{O_n}{2}\right)^2 + \left(s_p - \frac{O_p}{2}\right)^2 < \frac{O_n^2 + O_p^2}{4} \quad \text{...(Eq. 9.9)}
$

**Step 3: Interpret geometrically.**

Equation 9.9 is the equation of a **circle** in the $(s_n, s_p)$ plane:

- **Center**: $\left(\frac{O_n}{2}, \frac{O_p}{2}\right)$
- **Radius**: $R = \frac{\sqrt{O_n^2 + O_p^2}}{2}$

The loss is approximately zero for $(s_n, s_p)$ pairs **inside** this circle and positive
for pairs **outside** it. This is why it's called Circle Loss — the decision boundary
between "optimized" and "needs work" is a circle in similarity space.

**Step 4: Compare with the linear decision boundary of triplet loss.**

For triplet loss, the constraint is $s_p - s_n > \alpha$, which gives a linear boundary:

$
s_p = s_n + \alpha
$

This is a straight line in the $(s_n, s_p)$ plane. The circular boundary of Circle Loss
is more flexible — it naturally adapts to the difficulty of each pair.

### 9.2.8 Gradient Analysis

Let's derive the gradient of Circle Loss with respect to the similarity scores.

**Step 1: Define the inner quantity.**

Let $z = \gamma \sum_j \sum_i (\alpha_n^j s_n^j - \alpha_p^i s_p^i)$. In practice, we
analyze the gradient for a single pair. Let:

$
z_{ij} = \gamma(\alpha_n^j s_n^j - \alpha_p^i s_p^i)
$

The loss contribution from this pair passes through the log-sum-exp:

$
\mathcal{L} = \log\left[1 + \sum_{i,j} \exp(z_{ij})\right]
$

**Step 2: Gradient with respect to $s_n^j$.**

$
\frac{\partial \mathcal{L}}{\partial s_n^j} = \frac{\sum_i \exp(z_{ij}) \cdot \gamma \alpha_n^j}{1 + \sum_{i',j'} \exp(z_{i'j'})} = \gamma \alpha_n^j \sum_i \sigma_{ij} \quad \text{...(Eq. 9.10)}
$

where $\sigma_{ij} = \frac{\exp(z_{ij})}{1 + \sum_{i',j'} \exp(z_{i'j'})}$ is a
softmax-like weight.

**Step 3: Gradient with respect to $s_p^i$.**

$
\frac{\partial \mathcal{L}}{\partial s_p^i} = \frac{\sum_j \exp(z_{ij}) \cdot (-\gamma \alpha_p^i)}{1 + \sum_{i',j'} \exp(z_{i'j'})} = -\gamma \alpha_p^i \sum_j \sigma_{ij} \quad \text{...(Eq. 9.11)}
$

**Step 4: Interpret the gradient.**

The gradient for negative similarity $s_n^j$ is **positive** (scaled by $\alpha_n^j$),
meaning gradient descent will decrease $s_n^j$ — pushing the negative away.

The gradient for positive similarity $s_p^i$ is **negative** (scaled by $\alpha_p^i$),
meaning gradient descent will increase $s_p^i$ — pulling the positive closer.

The self-paced weights $\alpha_n^j$ and $\alpha_p^i$ modulate the gradient magnitude:
pairs far from their optimality targets receive stronger gradients, while already-optimized
pairs receive zero gradient. This is a form of **automatic curriculum learning**.

### 9.2.9 Hyperparameters

Circle Loss has three key hyperparameters:

| Parameter | Symbol | Typical Value | Role |
|-----------|--------|---------------|------|
| Scale factor | $\gamma$ | 64 or 256 | Controls sharpness of the loss landscape |
| Positive target | $O_p$ | $1 + m$ | Target similarity for positive pairs |
| Negative target | $O_n$ | $-m$ | Target similarity for negative pairs |
| Margin | $m$ | 0.25 | Gap between positive and negative targets |

With cosine similarity in $[-1, 1]$ and $m = 0.25$:
- $O_p = 1.25$ (positive pairs should reach cosine $\geq 1.0$, effectively 1.0 since
  cosine is bounded)
- $O_n = -0.25$ (negative pairs should have cosine $\leq -0.25$)

### 9.2.10 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization (Sun et al., 2020)
    """
    def __init__(self, gamma: float = 256, margin: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.O_p = 1 + margin    # optimality target for positives
        self.O_n = -margin        # optimality target for negatives

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sp: Positive similarity scores, shape (num_pos,)
            sn: Negative similarity scores, shape (num_neg,)
        Returns:
            Scalar loss value
        """
        # Self-paced weighting (detached — no gradient through weights)
        alpha_p = torch.clamp(self.O_p - sp.detach(), min=0)  # [O_p - s_p]+
        alpha_n = torch.clamp(sn.detach() - self.O_n, min=0)  # [s_n - O_n]+

        # Weighted similarities
        # For positives: we want to maximize sp, so we use -alpha_p * sp
        # For negatives: we want to minimize sn, so we use alpha_n * sn
        logit_p = -self.gamma * alpha_p * (sp - self.O_p)  # push sp toward O_p
        logit_n = self.gamma * alpha_n * (sn - self.O_n)   # push sn toward O_n

        # Log-sum-exp formulation
        loss = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        # Equivalent to: log[1 + sum_j sum_i exp(gamma(alpha_n*sn - alpha_p*sp))]
        # using the factored form (Eq. 9.4)

        return F.softplus(loss)


# --- Usage Example ---
# Suppose we have an anchor with 3 positive and 5 negative similarities
sp = torch.tensor([0.85, 0.72, 0.91], requires_grad=True)  # positive cosine sims
sn = torch.tensor([0.15, 0.42, 0.08, 0.55, 0.21], requires_grad=True)  # negative cosine sims

criterion = CircleLoss(gamma=256, margin=0.25)
loss = criterion(sp, sn)
print(f"Circle Loss: {loss.item():.4f}")

loss.backward()
print(f"Gradient on sp: {sp.grad}")  # negative — pulls positives closer
print(f"Gradient on sn: {sn.grad}")  # positive — pushes negatives away
```

### 9.2.11 Circle Loss with In-Batch Negatives

In practice, Circle Loss is often used with in-batch negatives, similar to MNRL. Here's
a complete implementation for sentence embeddings:

```python
class CircleLossForSentenceEmbeddings(nn.Module):
    """Circle Loss with in-batch negatives for sentence embedding training."""

    def __init__(self, gamma: float = 64, margin: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.O_p = 1 + margin
        self.O_n = -margin

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_a: (B, d) — e.g., query embeddings
            embeddings_b: (B, d) — e.g., positive passage embeddings
            Assumes embeddings_a[i] pairs with embeddings_b[i].
        """
        # L2 normalize
        a = F.normalize(embeddings_a, p=2, dim=1)
        b = F.normalize(embeddings_b, p=2, dim=1)

        # Compute B x B similarity matrix
        sim_matrix = torch.mm(a, b.t())  # (B, B)

        B = sim_matrix.size(0)

        # Diagonal = positive similarities, off-diagonal = negative similarities
        pos_mask = torch.eye(B, device=sim_matrix.device).bool()
        neg_mask = ~pos_mask

        sp = sim_matrix[pos_mask]          # (B,)
        sn = sim_matrix[neg_mask]          # (B*(B-1),)

        # Self-paced weights
        alpha_p = torch.clamp(self.O_p - sp.detach(), min=0)
        alpha_n = torch.clamp(sn.detach() - self.O_n, min=0)

        logit_p = -self.gamma * alpha_p * (sp - self.O_p)
        logit_n = self.gamma * alpha_n * (sn - self.O_n)

        loss = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        return F.softplus(loss)
```

---

## 9.3 CoSENT Loss (2022/2024)

### 9.3.1 Overview

**CoSENT** (Consistent Sentence Embedding via Similarity Ranking) was proposed by
Jianlin Su (2022) and later refined in the research community. It addresses the
training-inference inconsistency problem head-on: instead of training with a softmax
classifier on concatenated features and evaluating with cosine similarity, CoSENT
trains directly with cosine similarity using a ranking-based objective.

The core principle is simple but powerful: **if label $\text{sim}(i,j) > \text{sim}(k,l)$,
then the model should produce $\cos(\mathbf{u}_i, \mathbf{u}_j) > \cos(\mathbf{u}_k, \mathbf{u}_l)$**.

No thresholds. No classification heads. No concatenation tricks. Just ranking cosine
similarities to match the label ordering.

### 9.3.2 Problem Setup

Given a batch of sentence pairs with similarity labels:

$
\{(x_i, x_j, y_{ij})\}
$

where $y_{ij} \in \mathbb{R}$ is the similarity label (e.g., STS scores in $[0, 5]$ or
binary labels $\{0, 1\}$). We compute sentence embeddings:

$
\mathbf{u}_i = \text{Encoder}(x_i), \quad \mathbf{u}_j = \text{Encoder}(x_j)
$

and cosine similarities:

$
c_{ij} = \cos(\mathbf{u}_i, \mathbf{u}_j) = \frac{\mathbf{u}_i \cdot \mathbf{u}_j}{\|\mathbf{u}_i\| \cdot \|\mathbf{u}_j\|}
$

### 9.3.3 Binary Classification Formulation

For the simplest case with binary labels ($y_{ij} \in \{0, 1\}$), define:

- $\Omega_{\text{pos}} = \{(i, j) : y_{ij} = 1\}$ — the set of positive (similar) pairs
- $\Omega_{\text{neg}} = \{(k, l) : y_{kl} = 0\}$ — the set of negative (dissimilar) pairs

The CoSENT loss enforces that every positive pair has higher cosine similarity than every
negative pair:

$
\mathcal{L}_{\text{CoSENT}} = \log\left(1 + \sum_{\substack{(i,j) \in \Omega_{\text{pos}} \\ (k,l) \in \Omega_{\text{neg}}}} \exp\big(\lambda \cdot (c_{kl} - c_{ij})\big)\right) \quad \text{...(Eq. 9.12)}
$

where $\lambda > 0$ is a scaling hyperparameter (typically $\lambda = 20$).

### 9.3.4 Generalized Formulation for Ranked Labels

When labels are continuous (e.g., STS scores from 0 to 5), we generalize to all pairs
where the label ordering should be preserved:

$
\mathcal{L}_{\text{CoSENT}} = \log\left(1 + \sum_{\substack{y_{ij} > y_{kl}}} \exp\big(\lambda \cdot (c_{kl} - c_{ij})\big)\right) \quad \text{...(Eq. 9.13)}
$

The sum runs over all pairs of pairs $((i,j), (k,l))$ where the label for $(i,j)$ is
strictly greater than the label for $(k,l)$. The loss penalizes any case where the model's
cosine similarity ordering disagrees with the label ordering.

### 9.3.5 Step-by-Step Derivation

Let's derive CoSENT from first principles.

**Step 1: Start from the ranking constraint.**

For any two pairs $(i,j)$ and $(k,l)$ where $y_{ij} > y_{kl}$, we want:

$
\cos(\mathbf{u}_i, \mathbf{u}_j) > \cos(\mathbf{u}_k, \mathbf{u}_l)
$

Equivalently:

$
c_{ij} - c_{kl} > 0 \quad \text{...(Eq. 9.14)}
$

**Step 2: Soft margin via exponential.**

Instead of a hard constraint, we use an exponential penalty for violations:

$
\text{penalty}\big((i,j), (k,l)\big) = \exp\big(\lambda(c_{kl} - c_{ij})\big)
$

When the constraint is satisfied ($c_{ij} > c_{kl}$), the exponent $\lambda(c_{kl} - c_{ij})$
is negative and the penalty is small ($< 1$). When violated ($c_{kl} > c_{ij}$), the
exponent is positive and the penalty is large ($> 1$). The parameter $\lambda$ controls
the sharpness of the penalty.

**Step 3: Aggregate over all constrained pairs.**

Sum the penalties over all pairs where the label ordering should hold:

$
\text{total penalty} = \sum_{y_{ij} > y_{kl}} \exp\big(\lambda(c_{kl} - c_{ij})\big)
$

**Step 4: Log-sum-exp formulation.**

Apply the logarithm and add 1 for numerical stability (ensuring the loss is non-negative
and equals 0 only when all constraints are perfectly satisfied):

$
\mathcal{L}_{\text{CoSENT}} = \log\left(1 + \sum_{y_{ij} > y_{kl}} \exp\big(\lambda(c_{kl} - c_{ij})\big)\right) \quad \text{...(Eq. 9.15)}
$

**Step 5: Connection to cross-entropy.**

The CoSENT loss can be rewritten using the log-sum-exp identity. For a single constraint
pair $((i,j), (k,l))$:

$
\mathcal{L} = \log\big(1 + \exp(\lambda(c_{kl} - c_{ij}))\big) = \text{softplus}\big(\lambda(c_{kl} - c_{ij})\big)
$

This is the **binary cross-entropy** loss for a classifier that predicts whether
$c_{ij} > c_{kl}$, with logit $\lambda(c_{ij} - c_{kl})$:

$
\text{BCE} = -\log\sigma\big(\lambda(c_{ij} - c_{kl})\big) = \log\big(1 + \exp(-\lambda(c_{ij} - c_{kl}))\big) = \text{softplus}\big(\lambda(c_{kl} - c_{ij})\big)
$

where $\sigma$ is the sigmoid function. So CoSENT is equivalent to **binary cross-entropy
on pairwise ranking predictions**, aggregated over all constrained pairs via log-sum-exp.

**Step 6: The role of $\lambda$.**

The hyperparameter $\lambda$ plays a role analogous to the inverse temperature $1/\tau$
in InfoNCE:

- **Large $\lambda$** ($\lambda = 40$): Sharp penalties. Even small violations produce
  large gradients. Risk of training instability.
- **Small $\lambda$** ($\lambda = 5$): Soft penalties. The model tolerates small ranking
  violations. Smoother optimization but weaker constraint enforcement.
- **Typical $\lambda$** ($\lambda = 20$): Good balance for most STS tasks.

The gradient magnitude scales linearly with $\lambda$, so it also implicitly controls
the effective learning rate for the ranking objective.

### 9.3.6 Gradient Flow Analysis

Let's derive the gradient of CoSENT with respect to the embeddings.

**Step 1: Gradient with respect to cosine similarity $c_{ij}$.**

For a positive pair $(i,j)$ that participates in constraints against negative pairs:

$
\frac{\partial \mathcal{L}}{\partial c_{ij}} = \frac{-\lambda \sum_{(k,l): y_{kl} < y_{ij}} \exp(\lambda(c_{kl} - c_{ij}))}{1 + \sum_{\text{all}} \exp(\lambda(c_{kl} - c_{ij}))} \quad \text{...(Eq. 9.16)}
$

This is always **negative** (or zero), meaning gradient descent will **increase** $c_{ij}$
— pulling the positive pair closer together.

For a negative pair $(k,l)$ that participates in constraints against positive pairs:

$
\frac{\partial \mathcal{L}}{\partial c_{kl}} = \frac{\lambda \sum_{(i,j): y_{ij} > y_{kl}} \exp(\lambda(c_{kl} - c_{ij}))}{1 + \sum_{\text{all}} \exp(\lambda(c_{kl} - c_{ij}))} \quad \text{...(Eq. 9.17)}
$

This is always **positive**, meaning gradient descent will **decrease** $c_{kl}$ — pushing
the negative pair apart.

**Step 2: Softmax-like weighting.**

Define the softmax weights:

$
w_{(ij),(kl)} = \frac{\exp(\lambda(c_{kl} - c_{ij}))}{1 + \sum_{\text{all}} \exp(\lambda(c_{kl'} - c_{i'j'}))}
$

The gradient for each pair is weighted by $w_{(ij),(kl)}$, which is largest when the
ranking violation is most severe ($c_{kl} \gg c_{ij}$). This means CoSENT automatically
focuses on the **hardest ranking violations** — a form of automatic hard example mining.

**Step 3: Gradient of cosine similarity with respect to embeddings.**

From Equation 9.1, for L2-normalized embeddings:

$
\frac{\partial c_{ij}}{\partial \mathbf{u}_i} = \mathbf{u}_j - c_{ij} \cdot \mathbf{u}_i \quad \text{...(Eq. 9.18)}
$

This is the component of $\mathbf{u}_j$ that is **orthogonal** to $\mathbf{u}_i$. Its
magnitude is:

$
\left\|\frac{\partial c_{ij}}{\partial \mathbf{u}_i}\right\| = \sqrt{1 - c_{ij}^2} = |\sin(\theta_{ij})|
$

where $\theta_{ij}$ is the angle between $\mathbf{u}_i$ and $\mathbf{u}_j$.

**This reveals the cosine saturation problem**: when $c_{ij} \to \pm 1$ (i.e.,
$\theta_{ij} \to 0$ or $\pi$), the gradient magnitude $|\sin(\theta_{ij})| \to 0$.
The model cannot further optimize pairs that are already near-aligned or near-opposed.

**Step 4: Full gradient chain.**

Combining the chain rule:

$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_i} = \sum_{j: (i,j) \in \text{pairs}} \frac{\partial \mathcal{L}}{\partial c_{ij}} \cdot \frac{\partial c_{ij}}{\partial \mathbf{u}_i}
$

$
= \sum_{j} \frac{\partial \mathcal{L}}{\partial c_{ij}} \cdot (\mathbf{u}_j - c_{ij} \cdot \mathbf{u}_i) \quad \text{...(Eq. 9.19)}
$

The gradient for each embedding $\mathbf{u}_i$ is a weighted sum of orthogonal projections,
where the weights come from the ranking violation severity.

### 9.3.7 Advantages of CoSENT

1. **Training-inference consistency**: Both training and inference use cosine similarity.
   No mismatch between the optimization objective and the evaluation metric.

2. **No threshold required**: Unlike contrastive loss (which needs a margin $m$) or
   classification-based training (which needs a decision boundary), CoSENT only requires
   that the ranking order is correct. There's no absolute threshold to tune.

3. **Ranking-based**: CoSENT optimizes the relative ordering of similarities, not their
   absolute values. This is more aligned with how embeddings are typically used (retrieval,
   re-ranking) where relative ordering matters more than absolute scores.

4. **Handles continuous labels**: The generalized formulation (Eq. 9.13) naturally handles
   continuous similarity labels (e.g., STS scores from 0 to 5) without discretization.

5. **Automatic hard example mining**: The softmax-like weighting in the gradient
   (Section 9.3.6) automatically focuses on the hardest ranking violations.

### 9.3.8 Connection to Circle Loss

CoSENT and Circle Loss share the same mathematical backbone — both are instances of the
log-sum-exp ranking loss. The key difference:

- **Circle Loss** adds self-paced weighting ($\alpha_p$, $\alpha_n$) to modulate gradient
  magnitude based on distance from optimality targets.
- **CoSENT** uses uniform weighting but operates directly on cosine similarities with a
  ranking objective over label-ordered pairs.

In fact, CoSENT can be viewed as a special case of the unified loss (Eq. 9.3) where the
positive and negative sets are defined by label ordering rather than binary labels, and
the similarity function is explicitly cosine similarity.

### 9.3.9 Efficient Implementation

A naive implementation of CoSENT iterates over all pairs of pairs, which is $O(P^2)$
where $P$ is the number of pairs in the batch. This can be expensive. The key optimization
is to use the factored form.

For binary labels, the double sum factors:

$
\sum_{\substack{(i,j) \in \Omega_{\text{pos}} \\ (k,l) \in \Omega_{\text{neg}}}} \exp(\lambda(c_{kl} - c_{ij})) = \left(\sum_{(k,l) \in \Omega_{\text{neg}}} \exp(\lambda \cdot c_{kl})\right) \cdot \left(\sum_{(i,j) \in \Omega_{\text{pos}}} \exp(-\lambda \cdot c_{ij})\right)
$

This reduces the complexity from $O(|\Omega_{\text{pos}}| \cdot |\Omega_{\text{neg}}|)$
to $O(|\Omega_{\text{pos}}| + |\Omega_{\text{neg}}|)$.

### 9.3.10 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoSENTLoss(nn.Module):
    """
    CoSENT: Consistent Sentence Embedding via Similarity Ranking.

    Supports both binary labels and continuous similarity scores.
    """
    def __init__(self, lambda_scale: float = 20.0):
        super().__init__()
        self.lambda_scale = lambda_scale

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings_a: (B, d) — first sentence embeddings
            embeddings_b: (B, d) — second sentence embeddings
            labels: (B,) — similarity labels (binary or continuous)
        Returns:
            Scalar loss
        """
        # Compute cosine similarities for each pair
        cosine_sims = F.cosine_similarity(embeddings_a, embeddings_b, dim=1)  # (B,)

        # Scale by lambda
        cosine_sims = self.lambda_scale * cosine_sims  # (B,)

        # Build pairwise label comparison matrix
        # labels_diff[i, k] = labels[i] - labels[k]
        # We want pairs where labels[i] > labels[k]
        labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # (B, B)

        # Mask: 1 where labels[i] > labels[k] (pair i should have higher cosine than pair k)
        mask = (labels_diff > 0).float()  # (B, B)

        # Cosine difference matrix: cosine_sims[k] - cosine_sims[i]
        # We want to penalize when cosine_sims[k] > cosine_sims[i] but labels[i] > labels[k]
        cosine_diff = cosine_sims.unsqueeze(0) - cosine_sims.unsqueeze(1)  # (B, B)
        # cosine_diff[i, k] = cosine_sims[k] - cosine_sims[i]

        # Apply mask and compute loss
        # For each (i, k) where labels[i] > labels[k]:
        #   penalty = exp(lambda * (cos_k - cos_i))
        masked_exp = torch.exp(cosine_diff) * mask  # (B, B)

        # Sum all penalties and apply log(1 + sum)
        loss = torch.log(1 + masked_exp.sum())

        return loss


# --- Usage Example ---
B = 8  # batch size
d = 384  # embedding dimension

# Simulated embeddings and labels
embeddings_a = F.normalize(torch.randn(B, d), dim=1)
embeddings_b = F.normalize(torch.randn(B, d), dim=1)
labels = torch.tensor([4.2, 1.0, 3.5, 0.2, 4.8, 2.1, 0.5, 3.0])  # STS scores

criterion = CoSENTLoss(lambda_scale=20.0)
loss = criterion(embeddings_a, embeddings_b, labels)
print(f"CoSENT Loss: {loss.item():.4f}")
```

### 9.3.11 Numerical Example

Let's walk through CoSENT with a concrete batch of 4 pairs.

**Batch:**

| Pair | Sentence A | Sentence B | Label |
|------|-----------|-----------|-------|
| 1 | "A dog runs in the park" | "A puppy plays outside" | 4.2 |
| 2 | "The stock market crashed" | "A cat sleeps on a mat" | 0.3 |
| 3 | "She plays the guitar" | "A woman performs music" | 3.8 |
| 4 | "It is raining heavily" | "The weather is sunny" | 0.8 |

**Step 1: Compute cosine similarities** (from the model):

$c_1 = 0.72, \quad c_2 = 0.15, \quad c_3 = 0.68, \quad c_4 = 0.22$

**Step 2: Identify constrained pairs** (where $y_i > y_k$):

| Higher-label pair | Lower-label pair | $c_{\text{high}}$ | $c_{\text{low}}$ | Satisfied? |
|---|---|---|---|---|
| Pair 1 ($y=4.2$) | Pair 2 ($y=0.3$) | 0.72 | 0.15 | ✓ ($0.72 > 0.15$) |
| Pair 1 ($y=4.2$) | Pair 4 ($y=0.8$) | 0.72 | 0.22 | ✓ ($0.72 > 0.22$) |
| Pair 1 ($y=4.2$) | Pair 3 ($y=3.8$) | 0.72 | 0.68 | ✓ ($0.72 > 0.68$, barely) |
| Pair 3 ($y=3.8$) | Pair 2 ($y=0.3$) | 0.68 | 0.15 | ✓ ($0.68 > 0.15$) |
| Pair 3 ($y=3.8$) | Pair 4 ($y=0.8$) | 0.68 | 0.22 | ✓ ($0.68 > 0.22$) |
| Pair 4 ($y=0.8$) | Pair 2 ($y=0.3$) | 0.22 | 0.15 | ✓ ($0.22 > 0.15$) |

All constraints are satisfied, but some barely (Pair 1 vs Pair 3: margin of only 0.04).

**Step 3: Compute penalties** with $\lambda = 20$:

$
\exp(20 \cdot (0.15 - 0.72)) = \exp(-11.4) \approx 1.1 \times 10^{-5}
$
$
\exp(20 \cdot (0.22 - 0.72)) = \exp(-10.0) \approx 4.5 \times 10^{-5}
$
$
\exp(20 \cdot (0.68 - 0.72)) = \exp(-0.8) \approx 0.449 \quad \leftarrow \text{largest penalty!}
$
$
\exp(20 \cdot (0.15 - 0.68)) = \exp(-10.6) \approx 2.5 \times 10^{-5}
$
$
\exp(20 \cdot (0.22 - 0.68)) = \exp(-9.2) \approx 1.0 \times 10^{-4}
$
$
\exp(20 \cdot (0.15 - 0.22)) = \exp(-1.4) \approx 0.247
$

**Step 4: Compute loss:**

$
\mathcal{L} = \log(1 + 0.449 + 0.247 + \text{tiny terms}) \approx \log(1.696) \approx 0.528
$

The loss is dominated by the two barely-satisfied constraints (Pair 1 vs 3, and Pair 4
vs 2). The well-separated pairs contribute negligible penalty. This demonstrates CoSENT's
automatic focus on hard ranking violations.

---

## 9.4 AnglE Loss (2023/2024, ACL 2024)

### 9.4.1 Overview

**AnglE** (Angle-optimized Text Embeddings) was introduced by Li & Li (2024) and published
at ACL 2024. It directly attacks the cosine saturation problem identified in Section 9.1.2
with an elegant mathematical trick: instead of optimizing cosine similarity (which
saturates), optimize the **angle difference** between embeddings in **complex space**.

The key insight: by lifting real-valued embeddings into the complex plane and measuring
angle differences, we obtain a similarity measure whose gradient does **not** vanish at
the saturation zones of cosine similarity.

### 9.4.2 The Vanishing Gradient Problem — Revisited

Let's quantify the saturation problem more precisely. From Equation 9.1, the gradient
magnitude of cosine similarity for normalized embeddings is:

$
\left\|\frac{\partial \cos(\mathbf{u}, \mathbf{v})}{\partial \mathbf{u}}\right\| = \sqrt{1 - \cos^2(\mathbf{u}, \mathbf{v})} = |\sin(\theta)| \quad \text{...(Eq. 9.20)}
$

where $\theta$ is the angle between $\mathbf{u}$ and $\mathbf{v}$.

Plot the gradient magnitude as a function of cosine similarity:

```
Gradient magnitude |sin(θ)| vs. cosine similarity cos(θ):

|sin(θ)|
1.0 |          ****
    |        **    **
    |      **        **
    |    **            **
0.5 |  **                **
    | *                    *
    |*                      *
    *                        *
0.0 *────────────────────────*──→ cos(θ)
   -1.0  -0.5   0.0   0.5  1.0
    ↑                        ↑
  saturation              saturation
    zone                    zone
```

The gradient is maximal at $\cos(\theta) = 0$ (orthogonal embeddings) and vanishes at
$\cos(\theta) = \pm 1$ (aligned or opposed embeddings). With binary labels (0/1), the
training objective pushes positive pairs toward $\cos = 1$ and negative pairs toward
$\cos = 0$ (or $-1$), driving both into saturation zones.

### 9.4.3 The Complex Space Trick

AnglE's solution is to transform real-valued embeddings into complex numbers and measure
the angle of their quotient. Here's the step-by-step construction.

**Step 1: Split the embedding into two halves.**

Given a $d$-dimensional embedding $\mathbf{u} \in \mathbb{R}^d$ (where $d$ is even),
split it into two halves:

$
\mathbf{u} = [\mathbf{u}_{\text{re}} ; \mathbf{u}_{\text{im}}] \quad \text{where} \quad \mathbf{u}_{\text{re}}, \mathbf{u}_{\text{im}} \in \mathbb{R}^{d/2}
$

Similarly for $\mathbf{v}$:

$
\mathbf{v} = [\mathbf{v}_{\text{re}} ; \mathbf{v}_{\text{im}}]
$

**Step 2: Construct complex vectors.**

Interpret the two halves as the real and imaginary parts of a complex vector:

$
\mathbf{z} = \mathbf{u}_{\text{re}} + i \cdot \mathbf{u}_{\text{im}} \in \mathbb{C}^{d/2}
$

$
\mathbf{w} = \mathbf{v}_{\text{re}} + i \cdot \mathbf{v}_{\text{im}} \in \mathbb{C}^{d/2}
$

For notational convenience, let $\mathbf{u}_{\text{re}} = \mathbf{a}$,
$\mathbf{u}_{\text{im}} = \mathbf{b}$, $\mathbf{v}_{\text{re}} = \mathbf{c}$,
$\mathbf{v}_{\text{im}} = \mathbf{d}$. Then:

$
\mathbf{z} = \mathbf{a} + i\mathbf{b}, \quad \mathbf{w} = \mathbf{c} + i\mathbf{d}
$

**Step 3: Complex division in polar form.**

In polar coordinates, a complex number $z = r_z e^{i\phi_z}$ where $r_z = |z|$ is the
magnitude and $\phi_z$ is the phase angle. The quotient of two complex numbers is:

$
\frac{z}{w} = \frac{r_z}{r_w} e^{i(\phi_z - \phi_w)} = \gamma_{zw} \cdot e^{i \Delta\theta_{zw}} \quad \text{...(Eq. 9.21)}
$

where $\gamma_{zw} = r_z / r_w$ is the magnitude ratio and $\Delta\theta_{zw} = \phi_z - \phi_w$
is the **angle difference** — exactly what we want to measure.

**Step 4: Algebraic form of complex division.**

For element-wise operations on the vectors, the complex division $z_k / w_k$ for the
$k$-th component is:

$
\frac{z_k}{w_k} = \frac{(a_k + ib_k)}{(c_k + id_k)} = \frac{(a_k + ib_k)(c_k - id_k)}{(c_k + id_k)(c_k - id_k)}
$

$
= \frac{(a_k c_k + b_k d_k) + i(b_k c_k - a_k d_k)}{c_k^2 + d_k^2} \quad \text{...(Eq. 9.22)}
$

The real part is $\frac{a_k c_k + b_k d_k}{c_k^2 + d_k^2}$ and the imaginary part is
$\frac{b_k c_k - a_k d_k}{c_k^2 + d_k^2}$.

**Step 5: Extract the angle difference.**

To isolate the angle difference $\Delta\theta_{zw}$, we normalize by the magnitude ratio
$\gamma_{zw}$. In practice, AnglE computes the **absolute value** of the normalized
quotient, which captures the angle information:

$
\Delta\theta_{zw} = \left|\frac{z}{w} \cdot \frac{1}{\gamma_{zw}}\right| \quad \text{...(Eq. 9.23)}
$

For the full vector, this is computed element-wise and then summed (or averaged) across
dimensions.

**Step 6: Practical computation — the pairwise angle similarity.**

In practice, AnglE doesn't explicitly compute the complex division. Instead, it uses a
mathematically equivalent formulation based on normalized embeddings. For L2-normalized
$\mathbf{u}$ and $\mathbf{v}$:

$
\text{angle\_sim}(\mathbf{u}, \mathbf{v}) = |\cos(\theta) + \sin(\theta)| \quad \text{...(Eq. 9.24)}
$

where:
- $\cos(\theta) = \sum_k a_k c_k + b_k d_k$ (the standard cosine similarity between
  the full embeddings, which equals $\mathbf{u}_{\text{re}} \cdot \mathbf{v}_{\text{re}} + \mathbf{u}_{\text{im}} \cdot \mathbf{v}_{\text{im}}$)
- $\sin(\theta) = \sum_k b_k c_k - a_k d_k$ (the "sine" component, computed as
  $\mathbf{u}_{\text{im}} \cdot \mathbf{v}_{\text{re}} - \mathbf{u}_{\text{re}} \cdot \mathbf{v}_{\text{im}}$)

The absolute value $|\cos(\theta) + \sin(\theta)|$ combines both components, providing
a similarity measure that is sensitive to the angle difference in complex space.

### 9.4.4 Why the Angle Objective Avoids Saturation

The crucial property of the angle-based similarity is its gradient behavior. Let's analyze it.

For the cosine component, we already know the gradient vanishes at $\cos(\theta) = \pm 1$
(Eq. 9.20). But the sine component has a complementary gradient:

$
\frac{\partial \sin(\theta)}{\partial \theta} = \cos(\theta)
$

When $\cos(\theta) \to \pm 1$ (saturation zone for cosine), $|\cos(\theta)| \to 1$, which
means the **sine gradient is maximal**. Conversely, when $\sin(\theta) \to \pm 1$
(saturation zone for sine), $|\sin(\theta)| \to 1$, and the **cosine gradient is maximal**.

The combined objective $|\cos(\theta) + \sin(\theta)|$ has a gradient that is the sum of
both components:

$
\frac{\partial |\cos(\theta) + \sin(\theta)|}{\partial \theta} \propto -\sin(\theta) + \cos(\theta) \quad \text{...(Eq. 9.25)}
$

This gradient vanishes only when $\sin(\theta) = \cos(\theta)$, i.e., at $\theta = \pi/4 + n\pi$.
Critically, it does **not** vanish at $\theta = 0$ or $\theta = \pi$ — the exact points
where cosine similarity saturates.

```
Gradient comparison:

|gradient|
1.0 |  *   *              *   *
    | * * * *            * * * *
    |*   *   *          *   *   *
    *         *        *         *
0.5 |          *      *
    |           *    *
    |            *  *
    |             **
0.0 *──────────────*──────────────→ θ
    0    π/4   π/2   3π/4    π

    ── cos gradient (|sin θ|)
    ·· angle gradient (|-sin θ + cos θ|)
```

The angle gradient fills in the "dead zones" of the cosine gradient, ensuring continuous
learning signal across the entire similarity range.

### 9.4.5 The AnglE Loss Function

AnglE uses a **combined loss** that blends three objectives:

$
\mathcal{L}_{\text{AnglE}} = w_1 \cdot \mathcal{L}_{\cos} + w_2 \cdot \mathcal{L}_{\text{ibn}} + w_3 \cdot \mathcal{L}_{\text{angle}} \quad \text{...(Eq. 9.26)}
$

where:
- $\mathcal{L}_{\cos}$ is the **cosine contrastive loss** (standard cosine-based ranking)
- $\mathcal{L}_{\text{ibn}}$ is the **in-batch negatives loss** (InfoNCE-style)
- $\mathcal{L}_{\text{angle}}$ is the **angle loss** (the novel component)

**Default weights**: $w_1 = 1.0$, $w_2 = 1.0$, $w_3 = 0.02$

**Default temperatures**: $\tau_{\cos} = 0.05$, $\tau_{\text{ibn}} = 0.05$, $\tau_{\text{angle}} = 1.0$

The angle loss weight ($w_3 = 0.02$) is small because the angle similarity operates on
a different scale than cosine similarity. Despite the small weight, ablation studies show
it is the most critical component (Section 9.4.8).

### 9.4.6 The Angle Loss Component

The angle loss follows the same ranking structure as CoSENT, but uses angle similarity
instead of cosine similarity:

$
\mathcal{L}_{\text{angle}} = \log\left(1 + \sum_{y_{ij} > y_{kl}} \exp\left(\frac{\text{angle\_sim}_{kl} - \text{angle\_sim}_{ij}}{\tau_{\text{angle}}}\right)\right) \quad \text{...(Eq. 9.27)}
$

where $\text{angle\_sim}_{ij} = |\cos(\theta_{ij}) + \sin(\theta_{ij})|$ as defined in
Equation 9.24.

### 9.4.7 Pairwise Angle Similarity — Implementation

The implementation is elegant and efficient:

```python
import torch
import torch.nn.functional as F


def pairwise_angle_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise angle similarity between normalized embeddings.

    The embedding is split into two halves, interpreted as real and imaginary
    parts of a complex vector. The similarity combines cosine (real part of
    the dot product) and sine (imaginary part) components.

    Args:
        x: (B, d) — first set of embeddings (will be L2-normalized)
        y: (B, d) — second set of embeddings (will be L2-normalized)

    Returns:
        (B,) — angle similarity for each pair
    """
    # L2 normalize
    normalized_x = F.normalize(x, p=2, dim=1)
    normalized_y = F.normalize(y, p=2, dim=1)

    # Cosine component: standard dot product (real part)
    # cos(θ) = Σ(x_re * y_re + x_im * y_im) = Σ(x_k * y_k) = standard cosine
    cosines = torch.sum(normalized_x * normalized_y, dim=1)

    # Split into "real" and "imaginary" halves
    first_x, last_x = torch.chunk(normalized_x, 2, dim=1)  # x_re, x_im
    first_y, last_y = torch.chunk(normalized_y, 2, dim=1)  # y_re, y_im

    # Sine component: cross product (imaginary part of complex division)
    # sin(θ) = Σ(x_im * y_re - x_re * y_im)
    sines = torch.sum(last_x * first_y - first_x * last_y, dim=1)

    # Combined angle similarity
    return torch.abs(cosines + sines)
```

Let's verify this with a concrete example:

```python
# Example: 4D embeddings (split into 2D real + 2D imaginary)
x = torch.tensor([[0.5, 0.3, 0.7, 0.1]])  # [x_re, x_im] = [[0.5, 0.3], [0.7, 0.1]]
y = torch.tensor([[0.4, 0.6, 0.2, 0.5]])  # [y_re, y_im] = [[0.4, 0.6], [0.2, 0.5]]

# After normalization:
x_norm = F.normalize(x, p=2, dim=1)  # normalize full vector
y_norm = F.normalize(y, p=2, dim=1)

# Cosine: standard dot product of normalized vectors
cos_val = torch.sum(x_norm * y_norm, dim=1)

# Split
x_re, x_im = torch.chunk(x_norm, 2, dim=1)
y_re, y_im = torch.chunk(y_norm, 2, dim=1)

# Sine: x_im · y_re - x_re · y_im
sin_val = torch.sum(x_im * y_re - x_re * y_im, dim=1)

angle_sim = torch.abs(cos_val + sin_val)
print(f"Cosine: {cos_val.item():.4f}")
print(f"Sine:   {sin_val.item():.4f}")
print(f"Angle similarity: {angle_sim.item():.4f}")
```

### 9.4.8 Ablation Study Results

The AnglE paper provides detailed ablation results on STS benchmarks (average Spearman
correlation across STS12-16, STS-B, and SICK-R), demonstrating the contribution of each
loss component:

| Configuration | Avg. STS Score | Δ from Full |
|---|---|---|
| Full model ($\mathcal{L}_{\cos} + \mathcal{L}_{\text{ibn}} + \mathcal{L}_{\text{angle}}$) | **86.26** | — |
| Without angle ($\mathcal{L}_{\cos} + \mathcal{L}_{\text{ibn}}$) | 85.30 | **-0.96** |
| Without ibn ($\mathcal{L}_{\cos} + \mathcal{L}_{\text{angle}}$) | 86.00 | -0.26 |
| Without cosine ($\mathcal{L}_{\text{ibn}} + \mathcal{L}_{\text{angle}}$) | 85.89 | -0.37 |
| Only angle ($\mathcal{L}_{\text{angle}}$) | 85.12 | -1.14 |
| Only cosine ($\mathcal{L}_{\cos}$) | 84.78 | -1.48 |
| Only ibn ($\mathcal{L}_{\text{ibn}}$) | 84.95 | -1.31 |

**Key observations:**

1. **The angle objective is the most critical single addition.** Removing it causes the
   largest drop (-0.96), larger than removing ibn (-0.26) or cosine (-0.37).

2. **The in-batch negatives objective is the least critical addition** when the angle
   objective is present. This suggests that the angle objective captures much of the
   information that ibn provides.

3. **All three components together achieve the best result.** The losses are complementary:
   cosine provides direct similarity optimization, ibn provides contrastive structure from
   batch negatives, and angle provides gradient signal in saturation zones.

4. **The angle objective alone (85.12) outperforms cosine alone (84.78).** This directly
   validates the hypothesis that angle-based optimization is superior to pure cosine
   optimization.

### 9.4.9 Why the Angle Objective Is More Critical Than In-Batch Negatives

This result may seem surprising — in-batch negatives (InfoNCE) is the dominant paradigm
for contrastive learning, yet the angle objective contributes more. The explanation lies
in the gradient dynamics:

**In-batch negatives** provide more negative examples per update, improving the quality
of the contrastive signal. But the underlying similarity measure is still cosine, which
saturates. More negatives don't help if the gradient for each negative is near zero.

**The angle objective** fixes the gradient flow itself. Even with fewer comparisons, each
comparison produces a meaningful gradient because the angle similarity doesn't saturate
where cosine does. It's better to have strong gradients from fewer comparisons than weak
gradients from many comparisons.

This is analogous to the difference between having more data (ibn) versus having a better
optimizer (angle). Both help, but the optimizer improvement is more fundamental.

### 9.4.10 Pooling Strategy Comparison

The AnglE paper also evaluates different pooling strategies for extracting sentence
embeddings from the transformer:

| Pooling Strategy | Avg. STS Score |
|---|---|
| CLS token | **86.26** |
| Mean pooling | 85.97 |
| Max pooling | 85.41 |
| CLS + Mean (concatenated) | 86.08 |

CLS token pooling achieves the best result with AnglE, which differs from the common
wisdom that mean pooling is generally superior for sentence embeddings (as found in
SBERT). The authors hypothesize that the angle objective's gradient properties interact
favorably with the CLS token's role as a global sentence representation.

### 9.4.11 LLM Support with LoRA

AnglE supports training with large language models (LLMs) using LoRA (Low-Rank Adaptation).
The key modifications for LLM-based AnglE:

1. **Backbone**: Use a decoder-only LLM (e.g., LLaMA, Mistral) instead of BERT/RoBERTa
2. **LoRA**: Apply low-rank adapters to the attention layers instead of full fine-tuning
3. **Pooling**: Use the last token embedding (since decoder-only models attend left-to-right,
   the last token has seen the entire input)
4. **Prompt template**: Wrap input text in a template like
   `"Summarize sentence \"{text}\" in one word: "` to guide the LLM

The loss function remains identical — the angle objective is architecture-agnostic.

```python
# Pseudocode for LLM-based AnglE with LoRA
from peft import LoraConfig, get_peft_model

# 1. Load base LLM
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, lora_config)

# 3. Extract embeddings from last token
def get_embedding(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Last token embedding (last non-padding token)
    seq_lengths = attention_mask.sum(dim=1) - 1
    last_hidden = outputs.last_hidden_state
    embeddings = last_hidden[torch.arange(len(seq_lengths)), seq_lengths]
    return embeddings

# 4. Compute AnglE loss (same as encoder-based)
# L = w1 * L_cos + w2 * L_ibn + w3 * L_angle
```

### 9.4.12 Full AnglE Implementation

Here's a complete PyTorch implementation of the AnglE loss combining all three components:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnglELoss(nn.Module):
    """
    AnglE: Angle-optimized Text Embeddings (Li & Li, ACL 2024).

    Combines three loss components:
    1. Cosine contrastive loss (ranking-based, like CoSENT)
    2. In-batch negatives loss (InfoNCE-style)
    3. Angle loss (novel angle-based ranking in complex space)
    """
    def __init__(
        self,
        w_cos: float = 1.0,
        w_ibn: float = 1.0,
        w_angle: float = 0.02,
        tau_cos: float = 0.05,
        tau_ibn: float = 0.05,
        tau_angle: float = 1.0,
    ):
        super().__init__()
        self.w_cos = w_cos
        self.w_ibn = w_ibn
        self.w_angle = w_angle
        self.tau_cos = tau_cos
        self.tau_ibn = tau_ibn
        self.tau_angle = tau_angle

    def _pairwise_angle_sim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise angle similarity."""
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        cosines = torch.sum(x_norm * y_norm, dim=1)

        first_x, last_x = torch.chunk(x_norm, 2, dim=1)
        first_y, last_y = torch.chunk(y_norm, 2, dim=1)
        sines = torch.sum(last_x * first_y - first_x * last_y, dim=1)

        return torch.abs(cosines + sines)

    def _ranking_loss(self, sims: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Ranking loss (CoSENT-style): penalize pairs where label ordering
        disagrees with similarity ordering.
        """
        sims_scaled = sims / tau

        # Pairwise label differences: labels_diff[i,k] = labels[i] - labels[k]
        labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
        mask = (labels_diff > 0).float()

        # Similarity differences: sim_diff[i,k] = sims[k] - sims[i]
        sim_diff = sims_scaled.unsqueeze(0) - sims_scaled.unsqueeze(1)

        # Penalize when sim[k] > sim[i] but label[i] > label[k]
        masked_exp = torch.exp(sim_diff) * mask
        loss = torch.log(1 + masked_exp.sum())
        return loss

    def _ibn_loss(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """In-batch negatives loss (InfoNCE)."""
        a = F.normalize(embeddings_a, p=2, dim=1)
        b = F.normalize(embeddings_b, p=2, dim=1)

        sim_matrix = torch.mm(a, b.t()) / self.tau_ibn  # (B, B)
        B = sim_matrix.size(0)
        labels = torch.arange(B, device=sim_matrix.device)

        return F.cross_entropy(sim_matrix, labels)

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings_a: (B, d) — first sentence embeddings
            embeddings_b: (B, d) — second sentence embeddings
            labels: (B,) — similarity labels
        Returns:
            Scalar combined loss
        """
        loss = torch.tensor(0.0, device=embeddings_a.device)

        # 1. Cosine ranking loss
        if self.w_cos > 0:
            cos_sims = F.cosine_similarity(embeddings_a, embeddings_b, dim=1)
            loss_cos = self._ranking_loss(cos_sims, labels, self.tau_cos)
            loss = loss + self.w_cos * loss_cos

        # 2. In-batch negatives loss
        if self.w_ibn > 0:
            loss_ibn = self._ibn_loss(embeddings_a, embeddings_b)
            loss = loss + self.w_ibn * loss_ibn

        # 3. Angle ranking loss
        if self.w_angle > 0:
            angle_sims = self._pairwise_angle_sim(embeddings_a, embeddings_b)
            loss_angle = self._ranking_loss(angle_sims, labels, self.tau_angle)
            loss = loss + self.w_angle * loss_angle

        return loss


# --- Usage Example ---
B, d = 16, 768
embeddings_a = torch.randn(B, d, requires_grad=True)
embeddings_b = torch.randn(B, d, requires_grad=True)
labels = torch.rand(B) * 5  # STS-style scores in [0, 5]

criterion = AnglELoss()
loss = criterion(embeddings_a, embeddings_b, labels)
print(f"AnglE Loss: {loss.item():.4f}")

loss.backward()
print(f"Gradient norm (a): {embeddings_a.grad.norm().item():.6f}")
print(f"Gradient norm (b): {embeddings_b.grad.norm().item():.6f}")
```

### 9.4.13 Gradient Analysis of Angle Similarity

Let's derive the gradient of the angle similarity to formally show why it avoids saturation.

**Step 1: Define the angle similarity for a single pair.**

For normalized embeddings $\hat{\mathbf{u}}$ and $\hat{\mathbf{v}}$ (dropping the hat
for brevity), with $\mathbf{a}, \mathbf{b}$ as the first and second halves of $\mathbf{u}$,
and $\mathbf{c}, \mathbf{d}$ as the first and second halves of $\mathbf{v}$:

$
\text{angle\_sim} = |C + S|
$

where:

$
C = \mathbf{a} \cdot \mathbf{c} + \mathbf{b} \cdot \mathbf{d} = \cos(\theta) \quad \text{(cosine component)}
$

$
S = \mathbf{b} \cdot \mathbf{c} - \mathbf{a} \cdot \mathbf{d} = \sin(\theta) \quad \text{(sine component)}
$

**Step 2: Gradient with respect to the first half $\mathbf{a}$ (real part of $\mathbf{u}$).**

$
\frac{\partial C}{\partial \mathbf{a}} = \mathbf{c}, \quad \frac{\partial S}{\partial \mathbf{a}} = -\mathbf{d}
$

$
\frac{\partial (C + S)}{\partial \mathbf{a}} = \mathbf{c} - \mathbf{d} \quad \text{...(Eq. 9.28)}
$

**Step 3: Gradient with respect to the second half $\mathbf{b}$ (imaginary part of $\mathbf{u}$).**

$
\frac{\partial C}{\partial \mathbf{b}} = \mathbf{d}, \quad \frac{\partial S}{\partial \mathbf{b}} = \mathbf{c}
$

$
\frac{\partial (C + S)}{\partial \mathbf{b}} = \mathbf{d} + \mathbf{c} \quad \text{...(Eq. 9.29)}
$

**Step 4: Analyze gradient magnitude at saturation.**

When $\cos(\theta) \to 1$ (positive pair saturation), we have $\mathbf{u} \approx \mathbf{v}$,
so $\mathbf{a} \approx \mathbf{c}$ and $\mathbf{b} \approx \mathbf{d}$. The gradients become:

$
\frac{\partial (C + S)}{\partial \mathbf{a}} = \mathbf{c} - \mathbf{d} \approx \mathbf{a} - \mathbf{b}
$

$
\frac{\partial (C + S)}{\partial \mathbf{b}} = \mathbf{d} + \mathbf{c} \approx \mathbf{b} + \mathbf{a}
$

These are **not zero** (unless $\mathbf{a} = \mathbf{b}$, which is a measure-zero event
for random embeddings). The gradient of the angle similarity remains non-zero even when
cosine similarity saturates. This is the mathematical proof of AnglE's key advantage.

Compare with the cosine gradient at saturation:

$
\frac{\partial \cos(\theta)}{\partial \mathbf{u}} = \mathbf{v} - \cos(\theta) \cdot \mathbf{u} \approx \mathbf{v} - \mathbf{v} = \mathbf{0}
$

The cosine gradient vanishes, but the angle gradient persists.

### 9.4.14 Numerical Gradient Comparison

Let's verify this with concrete numbers. Consider two nearly-aligned 4D embeddings
($\cos \approx 0.99$):

$
\mathbf{u} = [0.50, 0.50, 0.50, 0.50] \quad (\text{normalized: } [0.50, 0.50, 0.50, 0.50])
$

$
\mathbf{v} = [0.51, 0.49, 0.51, 0.49] \quad (\text{normalized: } [0.51, 0.49, 0.51, 0.49])
$

**Cosine similarity**: $\cos(\mathbf{u}, \mathbf{v}) = 0.50 \times 0.51 + 0.50 \times 0.49 + 0.50 \times 0.51 + 0.50 \times 0.49 = 0.9998$

**Cosine gradient magnitude**: $\|\mathbf{v} - 0.9998 \cdot \mathbf{u}\| \approx 0.014$

**Angle similarity components**:
- $\mathbf{a} = [0.50, 0.50]$, $\mathbf{b} = [0.50, 0.50]$
- $\mathbf{c} = [0.51, 0.49]$, $\mathbf{d} = [0.51, 0.49]$
- $C = 0.50 \times 0.51 + 0.50 \times 0.49 + 0.50 \times 0.51 + 0.50 \times 0.49 = 0.9998$
- $S = 0.50 \times 0.51 + 0.50 \times 0.49 - 0.50 \times 0.51 - 0.50 \times 0.49 = 0.0$

**Angle gradient magnitude**:
- $\|\mathbf{c} - \mathbf{d}\| = \|[0.51 - 0.51, 0.49 - 0.49]\| = 0$ (degenerate case)
- $\|\mathbf{d} + \mathbf{c}\| = \|[1.02, 0.98]\| = 1.414$

The gradient through $\mathbf{b}$ has magnitude $1.414$, which is **100× larger** than
the cosine gradient ($0.014$). Even in this near-saturation regime, the angle objective
provides strong learning signal through the imaginary component.

---

## 9.5 Comparing Modern Loss Functions

### 9.5.1 Feature Comparison Table

| Feature | Triplet Loss | MNRL (InfoNCE) | Circle Loss | CoSENT | AnglE |
|---|---|---|---|---|---|
| **Year** | 2015 | 2017/2018 | 2020 | 2022 | 2024 |
| **Input format** | Triplets | Pairs (in-batch neg) | Pairs + labels | Pairs + scores | Pairs + scores |
| **Similarity metric** | Euclidean | Cosine | Any | Cosine | Cosine + Angle |
| **Margin required** | Yes ($\alpha$) | No (uses $\tau$) | Yes ($m$) | No | No |
| **Self-paced weighting** | No | Implicit (softmax) | Yes ($\alpha_p$, $\alpha_n$) | Implicit (softmax) | Implicit (softmax) |
| **Handles continuous labels** | No | No | No | Yes | Yes |
| **Training-inference consistency** | Partial | Yes | Yes | Yes | Yes |
| **Cosine saturation resistant** | N/A (uses L2) | No | Partial | No | **Yes** |
| **Batch size sensitivity** | Low | **High** | Medium | Low | Low |
| **Key hyperparameters** | $\alpha$, mining strategy | $\tau$, batch size | $\gamma$, $m$ | $\lambda$ | $w_1, w_2, w_3, \tau$ |

### 9.5.2 Gradient Behavior Comparison

The gradient behavior of each loss function reveals its optimization dynamics:

**Triplet Loss**: Binary gradient — either the full gradient (when hinge is active) or
zero (when the margin is satisfied). No intermediate weighting. This leads to
discontinuous optimization and wasted computation on easy triplets.

$
\nabla \mathcal{L}_{\text{triplet}} = \begin{cases} 2(\mathbf{n} - \mathbf{p}) & \text{if } d_p - d_n + \alpha > 0 \\ \mathbf{0} & \text{otherwise} \end{cases}
$

**MNRL (InfoNCE)**: Softmax-weighted gradient — hard negatives automatically receive
larger gradients. Smooth optimization, but gradient magnitude scales with $1/\tau$ and
vanishes when the positive dominates the softmax.

$
\nabla \mathcal{L}_{\text{MNRL}} \propto \frac{1}{\tau}\left(\sum_k p_{k|i} \mathbf{z}_k - \mathbf{z}_j\right)
$

**Circle Loss**: Self-paced gradient — pairs far from their optimality targets receive
amplified gradients, while already-optimized pairs receive zero gradient. Combines the
benefits of hard example mining with smooth optimization.

$
\nabla \mathcal{L}_{\text{circle}} \propto \gamma \cdot \alpha_n^j \cdot \sigma_{ij} \quad \text{(for negatives)}
$

**CoSENT**: Ranking-based gradient — focuses on the hardest ranking violations via
softmax weighting over pair-of-pairs. Gradient flows through cosine similarity, so it
inherits the saturation problem.

$
\nabla \mathcal{L}_{\text{CoSENT}} \propto \lambda \cdot w_{(ij),(kl)} \cdot (\mathbf{v} - \cos(\theta) \cdot \mathbf{u})
$

**AnglE**: Combines ranking-based gradient (like CoSENT) with angle-based gradient that
avoids saturation. The angle component provides gradient signal where cosine fails.

$
\nabla \mathcal{L}_{\text{AnglE}} \propto w_3 \cdot \nabla_{\text{angle}} + w_1 \cdot \nabla_{\text{cos}} + w_2 \cdot \nabla_{\text{ibn}}
$

### 9.5.3 Performance on STS Benchmarks

Reported results on STS benchmarks (average Spearman correlation, using BERT-base or
comparable backbone):

| Loss Function | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg |
|---|---|---|---|---|---|---|---|---|
| Contrastive | 64.64 | 78.90 | 69.42 | 77.02 | 73.14 | 74.28 | 66.04 | 71.92 |
| Triplet | 66.12 | 79.54 | 70.18 | 78.56 | 74.02 | 75.94 | 67.82 | 73.17 |
| MNRL (InfoNCE) | 70.42 | 82.18 | 74.56 | 81.24 | 78.36 | 79.82 | 70.14 | 76.67 |
| CoSENT | 74.18 | 84.52 | 78.24 | 83.68 | 80.14 | 83.46 | 73.28 | 79.64 |
| Circle Loss | 73.42 | 83.86 | 77.56 | 82.94 | 79.68 | 82.78 | 72.64 | 78.98 |
| AnglE | **76.82** | **86.14** | **80.48** | **85.92** | **82.36** | **85.68** | **75.52** | **81.85** |

*Note: Exact numbers vary by implementation, data, and hyperparameters. These represent
typical relative performance from published results and reproductions.*

### 9.5.4 When to Use Which Loss

**Use MNRL (InfoNCE)** when:
- You have (query, positive) pairs without explicit similarity scores
- You can afford large batch sizes (≥ 256)
- Your task is retrieval or search
- You want the simplest, most battle-tested approach

**Use CoSENT** when:
- You have pairs with continuous similarity labels (STS-style)
- Training-inference consistency is important
- You want ranking-based optimization without thresholds
- Your batch sizes are moderate (32–128)

**Use Circle Loss** when:
- You have clear positive/negative labels (not continuous scores)
- You want automatic curriculum learning (self-paced weighting)
- Your dataset has a wide range of difficulty levels
- You're working on face recognition or image retrieval

**Use AnglE** when:
- You need state-of-the-art STS performance
- Your training data has binary or near-binary labels that cause saturation
- You want to fine-tune LLMs for embeddings (with LoRA)
- You're willing to tune the three-component weight balance

**Use Triplet Loss** when:
- You have pre-mined triplets and can't restructure your pipeline
- You need the simplest possible loss for educational purposes
- (In practice, there's rarely a reason to prefer triplet loss over modern alternatives)

---

## 9.6 Summary

### 9.6.1 Key Takeaways

This chapter covered three modern loss functions that address fundamental limitations of
traditional metric learning objectives:

**Circle Loss** (Sun et al., 2020) introduced **self-paced weighting** to the unified
pairwise similarity optimization framework. By defining adaptive weights
$\alpha_p = [O_p - s_p]_+$ and $\alpha_n = [s_n - O_n]_+$, Circle Loss automatically
focuses gradient on the pairs that need the most optimization. The circular decision
boundary in $(s_n, s_p)$ space (Eq. 9.9) is more flexible than the linear boundary of
triplet loss, and the loss subsumes contrastive, triplet, and N-pair losses as special
cases.

**CoSENT** (Su, 2022) solved the **training-inference inconsistency** problem by
optimizing cosine similarity rankings directly during training — the same metric used at
inference. The ranking-based formulation (Eq. 9.13) handles continuous labels naturally,
requires no thresholds or margins, and automatically focuses on the hardest ranking
violations through softmax-like gradient weighting. Its connection to binary cross-entropy
on pairwise rankings provides theoretical grounding.

**AnglE** (Li & Li, ACL 2024) attacked the **cosine saturation problem** by introducing
angle-based similarity in complex space. By splitting embeddings into real and imaginary
components and measuring the angle of their complex quotient, AnglE obtains a similarity
measure whose gradient does not vanish at $\cos(\theta) = \pm 1$. The combined loss
$\mathcal{L} = w_1 \mathcal{L}_{\cos} + w_2 \mathcal{L}_{\text{ibn}} + w_3 \mathcal{L}_{\text{angle}}$
achieves state-of-the-art STS performance, with ablations showing the angle component
is the most critical addition despite its small weight ($w_3 = 0.02$).

### 9.6.2 The Evolution of Loss Functions

The progression from contrastive loss (2006) to AnglE (2024) reveals a clear trajectory:

```
Contrastive Loss (2006)     → Pairwise, fixed margin, binary
    ↓
Triplet Loss (2015)         → Relative ordering, but single negative
    ↓
N-pair / InfoNCE (2016-18)  → Multiple negatives, softmax weighting
    ↓
Circle Loss (2020)          → Self-paced weighting, unified framework
    ↓
CoSENT (2022)               → Training-inference consistency, ranking-based
    ↓
AnglE (2024)                → Saturation-resistant, complex space angles
```

Each generation addresses a specific limitation of its predecessors while preserving
their strengths. Modern losses are not replacements but refinements — they build on the
same mathematical foundations (log-sum-exp, softmax, ranking) with increasingly
sophisticated gradient engineering.

### 9.6.3 Looking Forward

The trend in loss function design points toward:

1. **Gradient-aware design**: Future losses will likely be designed by analyzing gradient
   flow first and deriving the loss function second (as AnglE did).

2. **Task-adaptive losses**: Rather than one loss for all tasks, we may see losses that
   automatically adapt their behavior based on the training data distribution.

3. **Integration with architecture**: The boundary between loss function and model
   architecture is blurring — AnglE's complex space trick is as much an architectural
   choice as a loss design.

4. **Theoretical unification**: Circle Loss showed that many losses are special cases of
   a unified framework. Future work may extend this unification to include CoSENT and
   AnglE, providing a single meta-loss with interpretable hyperparameters.

---

*Next chapter: [Chapter 10 — Matryoshka Embeddings](./10-matryoshka-embeddings.md)*
