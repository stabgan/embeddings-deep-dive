# Chapter 1: Origins and Foundations of Embeddings

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

The story of embeddings is the story of a single, powerful idea: that meaning can be
captured by context. This chapter traces the intellectual lineage from mid-20th-century
linguistics through the linear algebra of the 1990s to the neural revolution of the
2000s. Every modern embedding — from Word2Vec to the hidden states of GPT — descends
from the foundations laid here.

---

## 1.1 The Distributional Hypothesis (1950s–1960s)

### 1.1.1 Harris (1954): Distributional Structure

Zellig Harris, in his landmark 1954 paper *"Distributional Structure,"* proposed a
radical idea for its time: the meaning of a linguistic element is not some abstract
Platonic form but is instead determined by the distribution of other elements around it.

More precisely, Harris argued:

> If two morphemes $A$ and $B$ have identical distributions — that is, if every
> environment in which $A$ occurs is also an environment in which $B$ occurs, and
> vice versa — then $A$ and $B$ are semantically similar.

Let us formalize this. Define the **distributional profile** of a word $w$ as the set
of contexts in which $w$ appears:

$$D(w) = \{c \in \mathcal{C} \mid w \text{ occurs in context } c\}$$

where $\mathcal{C}$ is the universe of all possible contexts. Harris's hypothesis can
then be stated as:

$$D(w_1) \approx D(w_2) \implies \text{meaning}(w_1) \approx \text{meaning}(w_2)$$

This is a **similarity of distribution implies similarity of meaning** claim. It does
not define meaning directly; it defines a *proxy* for meaning that is entirely
observable from data.

#### Why This Matters

Before Harris, semantics was largely the domain of philosophy and introspection. Harris
gave us something computable: count the contexts, compare the counts, and you have a
measure of semantic similarity. Every embedding method we will study in this book is,
at its core, an operationalization of this insight.

### 1.1.2 Firth (1957): "You Shall Know a Word by the Company It Keeps"

The British linguist John Rupert Firth, working in the tradition of Malinowski's
contextual theory of meaning, crystallized the distributional idea into its most
memorable form. In his 1957 paper *"A Synopsis of Linguistic Theory, 1930–1955,"*
Firth wrote:

> "You shall know a word by the company it keeps."

Firth went further than Harris in several ways:

1. **Collocational meaning**: Firth argued that part of the meaning of a word is
   constituted by its habitual collocates. The word "dark" means something slightly
   different in "dark night" versus "dark horse" versus "dark secret." The collocate
   *is* part of the meaning.

2. **Levels of context**: Firth distinguished between the *situational context*
   (the real-world setting) and the *linguistic context* (the surrounding words).
   Modern embeddings primarily capture the latter, though multimodal embeddings
   increasingly capture the former.

3. **Paradigmatic vs. syntagmatic relations**: Words that can substitute for each
   other in the same context (paradigmatic: "cat" / "dog" in "The ___ sat on the
   mat") versus words that co-occur in sequence (syntagmatic: "strong" and "tea").
   Embeddings capture both types of relation, as we shall see.

### 1.1.3 Mathematical Formalization of Context

To move from linguistic intuition to computation, we need to formalize what "context"
means. There are several standard choices:

**Definition 1 (Symmetric Window Context).** Given a corpus
$\mathbf{w} = (w_1, w_2, \ldots, w_N)$ and a window size $k$, the context of the
word at position $i$ is:

$$C_k(w_i) = \{w_j \mid 0 < |i - j| \leq k\}$$

For example, with $k = 2$ and the sentence "The cat sat on the mat":

| Position $i$ | Word $w_i$ | Context $C_2(w_i)$          |
|:---:|:---:|:---:|
| 1             | The        | {cat, sat}                  |
| 2             | cat        | {The, sat, on}              |
| 3             | sat        | {The, cat, on, the}         |
| 4             | on         | {cat, sat, the, mat}        |
| 5             | the        | {sat, on, mat}              |
| 6             | mat        | {on, the}                   |

**Definition 2 (Document Context).** The context of a word $w$ is the document $d$
in which it appears:

$$C_{\text{doc}}(w) = \{d \in \mathcal{D} \mid w \in d\}$$

This is the basis for Latent Semantic Analysis (Section 1.4).

**Definition 3 (Dependency Context).** The context of a word $w$ is the set of words
connected to it via syntactic dependency relations:

$$C_{\text{dep}}(w_i) = \{(r, w_j) \mid (w_i, r, w_j) \in \text{DependencyParse}\}$$

where $r$ is the dependency relation label (e.g., nsubj, dobj). This produces
embeddings that capture more functional/syntactic similarity (Levy & Goldberg, 2014).

Each choice of context definition leads to a different geometry in the resulting
embedding space. Window contexts tend to capture topical and associative similarity;
dependency contexts tend to capture functional similarity.

---

## 1.2 One-Hot Encoding: The Naïve Baseline

Before we can appreciate the elegance of dense embeddings, we must understand the
representation they replaced.

### 1.2.1 Definition and Notation

Given a vocabulary $V = \{w_1, w_2, \ldots, w_{|V|}\}$ of size $|V|$, the
**one-hot encoding** of word $w_i$ is a vector $\mathbf{x}_i \in \{0, 1\}^{|V|}$
defined as:

$$(\mathbf{x}_i)_j = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}$$

Equivalently, $\mathbf{x}_i = \mathbf{e}_i$, the $i$-th standard basis vector in
$\mathbb{R}^{|V|}$.

### 1.2.2 Step-by-Step Example

Consider a tiny vocabulary:

$$V = \{\text{king}, \text{queen}, \text{man}, \text{woman}, \text{child}\}, \quad |V| = 5$$

The one-hot encodings are:

$$\mathbf{x}_{\text{king}}  = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{x}_{\text{queen}} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{x}_{\text{man}}   = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{x}_{\text{woman}} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{x}_{\text{child}} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}$$

### 1.2.3 The Three Fatal Problems

**Problem 1: Sparsity.** Each vector has exactly one non-zero entry out of $|V|$
entries. For a realistic vocabulary of $|V| = 100{,}000$ words, each vector is
99.999% zeros. This is catastrophically wasteful in terms of memory and computation.

**Problem 2: No semantic similarity.** The dot product between any two distinct
one-hot vectors is always zero:

$$\mathbf{x}_i^T \mathbf{x}_j = \sum_{k=1}^{|V|} (\mathbf{x}_i)_k (\mathbf{x}_j)_k = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

This means that "king" is exactly as similar to "queen" as it is to "banana." The
cosine similarity tells the same story:

$$\cos(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i^T \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

Every word is equidistant from every other word. The representation is semantically
flat — it encodes identity but nothing else.

**Problem 3: Curse of dimensionality.** The dimensionality of the representation
equals the vocabulary size. Any model that takes one-hot vectors as input must have
a first weight matrix $\mathbf{W} \in \mathbb{R}^{d \times |V|}$, which for
$|V| = 100{,}000$ and $d = 512$ means $\sim$51 million parameters in the first
layer alone. Worse, because each input activates only one column of $\mathbf{W}$,
the gradient signal is extremely sparse — most parameters receive no gradient on
any given training step.

### 1.2.4 The Key Insight: We Need Dense, Low-Dimensional Representations

The failures of one-hot encoding point directly to the desiderata for a good word
representation:

1. **Dense**: Most entries should be non-zero, so that every dimension carries
   information.
2. **Low-dimensional**: $d \ll |V|$, typically $d \in [50, 1024]$.
3. **Semantically meaningful**: Similar words should have similar vectors, i.e.,
   $\text{sim}(w_1, w_2) \approx \cos(\mathbf{v}_{w_1}, \mathbf{v}_{w_2})$.

The rest of this chapter is the story of how we get there.

---

## 1.3 Co-occurrence Matrices (1990s)

The distributional hypothesis tells us that context carries meaning. The most direct
way to operationalize this is to *count* contexts. This section develops the
co-occurrence matrix framework and the information-theoretic measures built on top
of it.

### 1.3.1 The Term-Document Matrix

The simplest co-occurrence structure is the **term-document matrix**
$\mathbf{X} \in \mathbb{R}^{|V| \times |D|}$, where $|V|$ is the vocabulary size
and $|D|$ is the number of documents:

$$X_{ij} = f(w_i, d_j)$$

where $f(w_i, d_j)$ is the frequency (raw count) of word $w_i$ in document $d_j$.

Each row of $\mathbf{X}$ is a vector representation of a word (across documents),
and each column is a vector representation of a document (across words). This is the
foundation of the **Vector Space Model** (Salton et al., 1975).

### 1.3.2 The Term-Term Co-occurrence Matrix

A richer representation counts how often words co-occur with *other words* rather
than with documents. The **term-term co-occurrence matrix**
$\mathbf{M} \in \mathbb{R}^{|V| \times |V|}$ is defined as:

$$M_{ij} = \#(w_i, w_j) = \sum_{t=1}^{N} \mathbf{1}[w_t = w_i \text{ and } w_j \in C_k(w_t)]$$

where $C_k(w_t)$ is the symmetric window context of size $k$ around position $t$,
and $N$ is the total number of tokens in the corpus.

Key properties of $\mathbf{M}$:

- **Symmetric** (for symmetric windows): $M_{ij} = M_{ji}$
- **Non-negative**: $M_{ij} \geq 0$
- **Sparse**: Most word pairs never co-occur
- **Size**: $|V| \times |V|$, which can be enormous (e.g., $100{,}000 \times 100{,}000$)

### 1.3.3 Window-Based Co-occurrence: Step-by-Step Example

Let us build a co-occurrence matrix from scratch. Consider the following tiny corpus
of three sentences:

> **Sentence 1:** "I like deep learning"
> **Sentence 2:** "I like NLP"
> **Sentence 3:** "I enjoy deep NLP"

**Step 1: Define the vocabulary.**

$$V = \{\text{I}, \text{like}, \text{deep}, \text{learning}, \text{NLP}, \text{enjoy}\}$$

So $|V| = 6$.

**Step 2: Choose a window size.** Let $k = 1$ (one word to the left, one to the right).

**Step 3: Enumerate all (word, context) pairs.**

From Sentence 1: "I like deep learning"
- Position 1 (I): context = {like}
- Position 2 (like): context = {I, deep}
- Position 3 (deep): context = {like, learning}
- Position 4 (learning): context = {deep}

From Sentence 2: "I like NLP"
- Position 1 (I): context = {like}
- Position 2 (like): context = {I, NLP}
- Position 3 (NLP): context = {like}

From Sentence 3: "I enjoy deep NLP"
- Position 1 (I): context = {enjoy}
- Position 2 (enjoy): context = {I, deep}
- Position 3 (deep): context = {enjoy, NLP}
- Position 4 (NLP): context = {deep}

**Step 4: Tally the counts into the matrix $\mathbf{M}$.**

|            | I | like | deep | learning | NLP | enjoy |
|:----------:|:-:|:----:|:----:|:--------:|:---:|:-----:|
| **I**      | 0 | 2    | 0    | 0        | 0   | 1     |
| **like**   | 2 | 0    | 1    | 0        | 1   | 0     |
| **deep**   | 0 | 1    | 0    | 1        | 1   | 1     |
| **learning**| 0 | 0   | 1    | 0        | 0   | 0     |
| **NLP**    | 0 | 1    | 1    | 0        | 0   | 0     |
| **enjoy**  | 1 | 0    | 1    | 0        | 0   | 0     |

**Step 5: Verify symmetry.** Check: $M_{\text{I, like}} = M_{\text{like, I}} = 2$. ✓

Each row of $\mathbf{M}$ is now a 6-dimensional vector representation of a word.
For example:

$$\mathbf{v}_{\text{like}} = [2, 0, 1, 0, 1, 0]$$
$$\mathbf{v}_{\text{enjoy}} = [1, 0, 1, 0, 0, 0]$$

We can already compute similarity. The cosine similarity between "like" and "enjoy" is:

$$\cos(\mathbf{v}_{\text{like}}, \mathbf{v}_{\text{enjoy}}) = \frac{\mathbf{v}_{\text{like}} \cdot \mathbf{v}_{\text{enjoy}}}{\|\mathbf{v}_{\text{like}}\| \|\mathbf{v}_{\text{enjoy}}\|}$$

**Numerator:**
$$\mathbf{v}_{\text{like}} \cdot \mathbf{v}_{\text{enjoy}} = (2)(1) + (0)(0) + (1)(1) + (0)(0) + (1)(0) + (0)(0) = 3$$

**Denominator:**
$$\|\mathbf{v}_{\text{like}}\| = \sqrt{4 + 0 + 1 + 0 + 1 + 0} = \sqrt{6} \approx 2.449$$
$$\|\mathbf{v}_{\text{enjoy}}\| = \sqrt{1 + 0 + 1 + 0 + 0 + 0} = \sqrt{2} \approx 1.414$$

$$\cos(\mathbf{v}_{\text{like}}, \mathbf{v}_{\text{enjoy}}) = \frac{3}{\sqrt{6} \cdot \sqrt{2}} = \frac{3}{\sqrt{12}} = \frac{3}{2\sqrt{3}} = \frac{\sqrt{3}}{2} \approx 0.866$$

This is high! The raw co-occurrence vectors already capture that "like" and "enjoy"
are semantically similar, because they share similar contexts (both co-occur with "I"
and "deep").

### 1.3.4 The Problem with Raw Counts

Raw co-occurrence counts are dominated by frequent words. Function words like "the,"
"a," "is" co-occur with almost everything, inflating their counts and drowning out
the signal from content words. We need a measure that asks: *does this word pair
co-occur more than we would expect by chance?*

This is exactly what Pointwise Mutual Information provides.

### 1.3.5 Pointwise Mutual Information (PMI): Full Derivation

**Definition.** The Pointwise Mutual Information between a target word $w$ and a
context word $c$ is:

$$\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w) \cdot P(c)}$$

**Intuition:** PMI measures how much more (or less) likely two words are to co-occur
than we would expect if they were independent. If $w$ and $c$ are independent, then
$P(w, c) = P(w) P(c)$, and $\text{PMI}(w, c) = \log_2 1 = 0$.

- $\text{PMI}(w, c) > 0$: $w$ and $c$ co-occur *more* than expected → positive association
- $\text{PMI}(w, c) = 0$: $w$ and $c$ co-occur exactly as expected → independence
- $\text{PMI}(w, c) < 0$: $w$ and $c$ co-occur *less* than expected → negative association

**Estimating probabilities from counts.** Let $\#(w, c)$ denote the co-occurrence
count from our matrix $\mathbf{M}$, and let $D = \sum_{w,c} \#(w, c)$ be the total
number of co-occurrence events. Then:

$$P(w, c) = \frac{\#(w, c)}{D}, \quad P(w) = \frac{\sum_{c'} \#(w, c')}{D} = \frac{\#(w)}{D}, \quad P(c) = \frac{\sum_{w'} \#(w', c)}{D} = \frac{\#(c)}{D}$$

Substituting into the PMI formula:

$$\text{PMI}(w, c) = \log_2 \frac{\#(w, c) / D}{(\#(w) / D)(\#(c) / D)} = \log_2 \frac{\#(w, c) \cdot D}{\#(w) \cdot \#(c)}$$

This is the practical formula: we only need the raw counts and the total.

#### Numerical Example: Computing PMI from Our Co-occurrence Matrix

Recall our co-occurrence matrix from Section 1.3.3. Let us compute PMI for selected
word pairs.

**Step 1: Compute the total count $D$.**

$$D = \sum_{i,j} M_{ij} = (0+2+0+0+0+1) + (2+0+1+0+1+0) + (0+1+0+1+1+1) + (0+0+1+0+0+0) + (0+1+1+0+0+0) + (1+0+1+0+0+0)$$
$$D = 3 + 4 + 4 + 1 + 2 + 2 = 16$$

**Step 2: Compute marginal counts $\#(w)$ for each word** (row sums):

| Word     | $\#(w)$ (row sum) |
|:--------:|:-----------------:|
| I        | 3                 |
| like     | 4                 |
| deep     | 4                 |
| learning | 1                 |
| NLP      | 2                 |
| enjoy    | 2                 |

**Step 3: Compute PMI for the pair (like, NLP).**

$$\#(\text{like}, \text{NLP}) = 1, \quad \#(\text{like}) = 4, \quad \#(\text{NLP}) = 2$$

$$\text{PMI}(\text{like}, \text{NLP}) = \log_2 \frac{1 \times 16}{4 \times 2} = \log_2 \frac{16}{8} = \log_2 2 = 1.0$$

**Step 4: Compute PMI for the pair (deep, learning).**

$$\#(\text{deep}, \text{learning}) = 1, \quad \#(\text{deep}) = 4, \quad \#(\text{learning}) = 1$$

$$\text{PMI}(\text{deep}, \text{learning}) = \log_2 \frac{1 \times 16}{4 \times 1} = \log_2 \frac{16}{4} = \log_2 4 = 2.0$$

**Step 5: Compute PMI for the pair (I, deep).**

$$\#(\text{I}, \text{deep}) = 0$$

$$\text{PMI}(\text{I}, \text{deep}) = \log_2 \frac{0 \times 16}{3 \times 4} = \log_2 0 = -\infty$$

This illustrates a well-known problem: PMI is $-\infty$ for word pairs that never
co-occur, and it is unreliable for rare events (small counts lead to extreme PMI
values).

### 1.3.6 Positive PMI (PPMI)

To address the problems with negative and infinite PMI values, we use **Positive PMI**:

$$\text{PPMI}(w, c) = \max(0, \text{PMI}(w, c))$$

**Rationale:** Negative PMI values are unreliable. A PMI of $-\infty$ simply means
the pair was not observed, which could be due to data sparsity rather than true
semantic dissimilarity. By clamping to zero, we treat "not observed" and "less than
expected" uniformly as "no evidence of association."

#### Full PPMI Matrix for Our Example

Using the counts from Section 1.3.3 with $D = 16$:

**Step 1: Compute all PMI values.**

For each cell $(w, c)$ where $\#(w, c) > 0$:

$$\text{PMI}(w, c) = \log_2 \frac{\#(w, c) \times 16}{\#(w) \times \#(c)}$$

| $(w, c)$           | $\#(w,c)$ | $\#(w)$ | $\#(c)$ | PMI                                          |
|:-------------------:|:----------:|:--------:|:--------:|:--------------------------------------------:|
| (I, like)           | 2          | 3        | 4        | $\log_2 \frac{32}{12} = \log_2 2.667 = 1.415$ |
| (I, enjoy)          | 1          | 3        | 2        | $\log_2 \frac{16}{6} = \log_2 2.667 = 1.415$  |
| (like, deep)        | 1          | 4        | 4        | $\log_2 \frac{16}{16} = \log_2 1 = 0$         |
| (like, NLP)         | 1          | 4        | 2        | $\log_2 \frac{16}{8} = \log_2 2 = 1.0$        |
| (deep, learning)    | 1          | 4        | 1        | $\log_2 \frac{16}{4} = \log_2 4 = 2.0$        |
| (deep, NLP)         | 1          | 4        | 2        | $\log_2 \frac{16}{8} = \log_2 2 = 1.0$        |
| (deep, enjoy)       | 1          | 4        | 2        | $\log_2 \frac{16}{8} = \log_2 2 = 1.0$        |

**Step 2: Apply the PPMI transformation** (clamp negatives to 0).

Since all computed PMI values above are $\geq 0$, and all unobserved pairs get
$\text{PPMI} = 0$, the PPMI matrix is:

|            | I     | like  | deep  | learning | NLP   | enjoy |
|:----------:|:-----:|:-----:|:-----:|:--------:|:-----:|:-----:|
| **I**      | 0     | 1.415 | 0     | 0        | 0     | 1.415 |
| **like**   | 1.415 | 0     | 0     | 0        | 1.0   | 0     |
| **deep**   | 0     | 0     | 0     | 2.0      | 1.0   | 1.0   |
| **learning**| 0    | 0     | 2.0   | 0        | 0     | 0     |
| **NLP**    | 0     | 1.0   | 1.0   | 0        | 0     | 0     |
| **enjoy**  | 1.415 | 0     | 1.0   | 0        | 0     | 0     |

Notice how the PPMI matrix is much more informative than the raw count matrix:

- The strong association between "deep" and "learning" ($\text{PPMI} = 2.0$) stands
  out clearly, even though the raw count was only 1.
- The association between "like" and "deep" ($\text{PPMI} = 0$) is correctly
  identified as no more than chance, despite having a raw count of 1.
- Frequent words like "I" no longer dominate: the PMI normalization accounts for
  their high base rate.

**Levy & Goldberg (2014)** showed that the Skip-gram model with negative sampling
(Word2Vec SGNS) is implicitly factorizing a shifted PPMI matrix. This deep connection
between count-based and prediction-based methods is one of the most important results
in the embedding literature. We will return to this in Chapter 2.

---

## 1.4 Latent Semantic Analysis / LSA (1990)

### 1.4.1 Motivation and History

In 1990, Scott Deerwester, Susan Dumais, George Furnas, Thomas Landauer, and Richard
Harshman published *"Indexing by Latent Semantic Analysis,"* a paper that would
become one of the most cited works in information retrieval. Their key insight was
that the term-document matrix contains *latent semantic structure* that is obscured
by the high dimensionality and noise of the raw counts.

The core problems LSA addresses:

1. **Synonymy**: Different words with the same meaning (e.g., "car" and "automobile")
   will have different rows in the term-document matrix, even though they should be
   treated as equivalent.
2. **Polysemy**: The same word with different meanings (e.g., "bank" as financial
   institution vs. river bank) will have a single row that conflates both senses.
3. **Noise**: Rare or accidental co-occurrences add noise to the matrix.

LSA's solution: apply **Singular Value Decomposition (SVD)** to the term-document
matrix and keep only the top $k$ singular values, projecting the data into a
lower-dimensional space where the latent semantic structure is revealed.

### 1.4.2 Singular Value Decomposition: The Mathematics

**Theorem (SVD).** Any real matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ can be
decomposed as:

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where:

- $\mathbf{U} \in \mathbb{R}^{m \times m}$ is an orthogonal matrix (columns are
  left singular vectors): $\mathbf{U}^T \mathbf{U} = \mathbf{I}_m$
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix of
  singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ where
  $r = \text{rank}(\mathbf{X})$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ is an orthogonal matrix (columns are
  right singular vectors): $\mathbf{V}^T \mathbf{V} = \mathbf{I}_n$

**Geometric interpretation:**

- The columns of $\mathbf{U}$ form an orthonormal basis for the column space of
  $\mathbf{X}$ (the "word space").
- The columns of $\mathbf{V}$ form an orthonormal basis for the row space of
  $\mathbf{X}$ (the "document space").
- The singular values $\sigma_i$ measure the "importance" of each dimension — how
  much variance in the data is captured by that dimension.

### 1.4.3 Step-by-Step SVD Example

Let us work through SVD on a small term-document matrix. Consider 4 words and 3
documents:

| | Doc 1 ("AI intro") | Doc 2 ("ML guide") | Doc 3 ("NLP text") |
|:---:|:---:|:---:|:---:|
| **learning** | 2 | 2 | 0 |
| **network**  | 1 | 0 | 1 |
| **language** | 0 | 1 | 2 |
| **deep**     | 1 | 1 | 1 |

So our matrix is:

$$\mathbf{X} = \begin{bmatrix} 2 & 2 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix}$$

**Step 1: Compute $\mathbf{X}^T \mathbf{X}$** (this gives us the right singular
vectors and squared singular values).

$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 2&1&0&1 \\ 2&0&1&1 \\ 0&1&2&1 \end{bmatrix} \begin{bmatrix} 2&2&0 \\ 1&0&1 \\ 0&1&2 \\ 1&1&1 \end{bmatrix}$$

Computing each entry:

$$(\mathbf{X}^T\mathbf{X})_{11} = 2^2 + 1^2 + 0^2 + 1^2 = 6$$
$$(\mathbf{X}^T\mathbf{X})_{12} = (2)(2) + (1)(0) + (0)(1) + (1)(1) = 5$$
$$(\mathbf{X}^T\mathbf{X})_{13} = (2)(0) + (1)(1) + (0)(2) + (1)(1) = 2$$
$$(\mathbf{X}^T\mathbf{X})_{22} = 2^2 + 0^2 + 1^2 + 1^2 = 6$$
$$(\mathbf{X}^T\mathbf{X})_{23} = (2)(0) + (0)(1) + (1)(2) + (1)(1) = 3$$
$$(\mathbf{X}^T\mathbf{X})_{33} = 0^2 + 1^2 + 2^2 + 1^2 = 6$$

$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 6 & 5 & 2 \\ 5 & 6 & 3 \\ 2 & 3 & 6 \end{bmatrix}$$

**Step 2: Find eigenvalues of $\mathbf{X}^T \mathbf{X}$.**

We solve $\det(\mathbf{X}^T\mathbf{X} - \lambda \mathbf{I}) = 0$:

$$\det \begin{bmatrix} 6-\lambda & 5 & 2 \\ 5 & 6-\lambda & 3 \\ 2 & 3 & 6-\lambda \end{bmatrix} = 0$$

Expanding the determinant (details of the cubic):

$$(6-\lambda)[(6-\lambda)^2 - 9] - 5[5(6-\lambda) - 6] + 2[15 - 2(6-\lambda)] = 0$$

After algebraic simplification, the characteristic polynomial is:

$$-\lambda^3 + 18\lambda^2 - 81\lambda + 54 = 0$$

or equivalently:

$$\lambda^3 - 18\lambda^2 + 81\lambda - 54 = 0$$

The eigenvalues (computed numerically) are approximately:

$$\lambda_1 \approx 13.14, \quad \lambda_2 \approx 4.12, \quad \lambda_3 \approx 0.74$$

The singular values are the square roots:

$$\sigma_1 \approx 3.625, \quad \sigma_2 \approx 2.030, \quad \sigma_3 \approx 0.860$$

**Step 3: Compute the singular value matrix.**

$$\boldsymbol{\Sigma} = \begin{bmatrix} 3.625 & 0 & 0 \\ 0 & 2.030 & 0 \\ 0 & 0 & 0.860 \\ 0 & 0 & 0 \end{bmatrix}$$

**Step 4: Compute right singular vectors $\mathbf{V}$** (eigenvectors of
$\mathbf{X}^T\mathbf{X}$, normalized).

For $\lambda_1 \approx 13.14$, solving $(\mathbf{X}^T\mathbf{X} - 13.14\mathbf{I})\mathbf{v} = 0$:

$$\mathbf{v}_1 \approx \begin{bmatrix} 0.588 \\ 0.623 \\ 0.516 \end{bmatrix}$$

For $\lambda_2 \approx 4.12$:

$$\mathbf{v}_2 \approx \begin{bmatrix} 0.515 \\ 0.104 \\ -0.851 \end{bmatrix}$$

For $\lambda_3 \approx 0.74$:

$$\mathbf{v}_3 \approx \begin{bmatrix} -0.624 \\ 0.775 \\ -0.101 \end{bmatrix}$$

**Step 5: Compute left singular vectors $\mathbf{U}$** using $\mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{X} \mathbf{v}_i$.

For $\mathbf{u}_1$:

$$\mathbf{u}_1 = \frac{1}{3.625} \begin{bmatrix} 2&2&0 \\ 1&0&1 \\ 0&1&2 \\ 1&1&1 \end{bmatrix} \begin{bmatrix} 0.588 \\ 0.623 \\ 0.516 \end{bmatrix} = \frac{1}{3.625} \begin{bmatrix} 2.422 \\ 1.104 \\ 1.655 \\ 1.727 \end{bmatrix} \approx \begin{bmatrix} 0.668 \\ 0.305 \\ 0.457 \\ 0.476 \end{bmatrix}$$

This gives us the word representations in the first latent dimension. The word
"learning" loads most heavily (0.668), followed by "deep" (0.476) and "language"
(0.457). This first dimension captures a general "technical content" factor.

### 1.4.4 Truncated SVD: Dimensionality Reduction

The key move in LSA is to keep only the top $k$ singular values and set the rest to
zero. This gives the **truncated SVD** (also called the rank-$k$ approximation):

$$\mathbf{X}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$

where:

- $\mathbf{U}_k \in \mathbb{R}^{m \times k}$ — first $k$ columns of $\mathbf{U}$
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{k \times k}$ — top-left $k \times k$ block
  of $\boldsymbol{\Sigma}$
- $\mathbf{V}_k \in \mathbb{R}^{n \times k}$ — first $k$ columns of $\mathbf{V}$

**The Eckart–Young–Mirsky theorem** guarantees that $\mathbf{X}_k$ is the best
rank-$k$ approximation to $\mathbf{X}$ in both the Frobenius norm and the spectral
norm:

$$\mathbf{X}_k = \arg\min_{\text{rank}(\mathbf{Y}) \leq k} \|\mathbf{X} - \mathbf{Y}\|_F$$

#### Continuing Our Example: Truncated SVD with $k = 2$

From our SVD above, we keep only the first two singular values:

$$\boldsymbol{\Sigma}_2 = \begin{bmatrix} 3.625 & 0 \\ 0 & 2.030 \end{bmatrix}$$

The 2-dimensional word embeddings are given by $\mathbf{U}_2 \boldsymbol{\Sigma}_2$
(or just $\mathbf{U}_2$, depending on convention). Using $\mathbf{U}_2$:

| Word       | Dim 1   | Dim 2   |
|:----------:|:-------:|:-------:|
| learning   | 0.668   | 0.434   |
| network    | 0.305   | -0.556  |
| language   | 0.457   | -0.556  |
| deep       | 0.476   | 0.138   |

Now we can visualize these in 2D! Notice:

- "network" and "language" are close together (both have similar Dim 1 and nearly
  identical Dim 2 values) — they share a "specialized topic" dimension.
- "learning" and "deep" are closer to each other than to "network" — they share
  a "general AI" dimension.
- The 2D representation captures semantic structure that was implicit in the
  original 3-document matrix.

**Information preserved:** The fraction of variance captured by the top $k$ singular
values is:

$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} = \frac{13.14 + 4.12}{13.14 + 4.12 + 0.74} = \frac{17.26}{18.00} \approx 95.9\%$$

With just 2 dimensions (down from 3), we retain 95.9% of the variance. The discarded
dimension was mostly noise.

### 1.4.5 How LSA Captures Latent Semantic Structure

The magic of LSA lies in what happens when you project into the reduced space:

1. **Synonymy resolution**: Words that never co-occur in the same document but
   co-occur with similar *other* words will be pulled together in the reduced space.
   If "car" appears in documents about "driving" and "road," and "automobile" appears
   in documents about "driving" and "highway," then in the full space they have
   different vectors, but in the reduced space the shared "driving" context pulls
   them together.

2. **Noise reduction**: Random co-occurrences that don't reflect true semantic
   structure tend to be captured by the smaller singular values. By discarding these,
   we remove noise.

3. **Implicit inference**: LSA can infer associations that are not directly observed.
   If "doctor" co-occurs with "hospital" and "hospital" co-occurs with "nurse," then
   in the reduced space "doctor" and "nurse" will be similar even if they never
   directly co-occur.

Mathematically, this works because the rank-$k$ approximation forces the model to
"explain" the observed co-occurrences using only $k$ latent factors. Words that
participate in similar latent factors end up with similar representations.

### 1.4.6 Limitations of LSA

Despite its elegance, LSA has significant limitations:

1. **Linear model**: LSA assumes that the latent structure is linear. It cannot
   capture non-linear semantic relationships.

2. **Bag-of-words assumption**: Word order is completely ignored. "Dog bites man"
   and "man bites dog" have identical representations.

3. **Scalability**: Computing the full SVD of a $|V| \times |D|$ matrix is
   $O(\min(|V|^2 |D|, |V| |D|^2))$, which is prohibitive for large corpora.
   (Iterative methods like Lanczos can compute the top-$k$ SVD more efficiently.)

4. **Static representations**: Each word gets a single vector, regardless of context.
   "Bank" (financial) and "bank" (river) share the same representation.

5. **No probabilistic interpretation**: Unlike later models (LDA, Word2Vec), LSA
   does not have a clear generative story.

6. **Sensitivity to preprocessing**: Results depend heavily on choices like
   TF-IDF weighting, stop word removal, and the value of $k$.

---

## 1.5 Random Indexing and Other Pre-Neural Approaches

Before neural networks took over, several other methods attempted to create dense
word representations efficiently. These are worth understanding both for historical
completeness and because some of their ideas resurface in modern methods.

### 1.5.1 Random Indexing (Kanerva et al., 2000; Sahlgren, 2005)

Random Indexing is an incremental, scalable alternative to LSA that avoids the
expensive SVD computation entirely.

**The algorithm:**

1. **Assign each context (word or document) a random index vector** $\mathbf{r}_c \in \mathbb{R}^d$,
   where $d$ is the desired embedding dimension (e.g., $d = 1000$). These vectors
   are sparse and random: each has a small number of randomly placed $+1$ and $-1$
   entries, with the rest being zero. Typically, $\sim 2\%$ of entries are non-zero.

2. **For each target word $w$, accumulate a context vector** by summing the random
   index vectors of all contexts in which $w$ appears:

   $$\mathbf{v}_w = \sum_{c \in \text{contexts}(w)} f(w, c) \cdot \mathbf{r}_c$$

   where $f(w, c)$ is a weighting function (e.g., raw count, PPMI).

**Why this works:** The Johnson–Lindenstrauss lemma guarantees that random projections
approximately preserve distances. If two words have similar distributional profiles
(similar rows in the co-occurrence matrix), their accumulated context vectors will
be similar, because they sum similar sets of random vectors.

**Key properties:**

- **Incremental**: New documents can be added without recomputing everything.
- **Scalable**: No matrix factorization required; $O(N \cdot d)$ where $N$ is the
  corpus size.
- **Approximate**: The resulting vectors are a random projection of the full
  co-occurrence matrix, which is an approximation to SVD.

**Formal connection to LSA:** Let $\mathbf{R} \in \mathbb{R}^{|V| \times d}$ be the
matrix of random index vectors. Then the Random Indexing word vectors are:

$$\mathbf{V}_{\text{RI}} = \mathbf{M} \mathbf{R}$$

where $\mathbf{M}$ is the co-occurrence matrix. This is a random projection of the
rows of $\mathbf{M}$ from $|V|$ dimensions to $d$ dimensions.

### 1.5.2 Hyperspace Analogue to Language (HAL) (Lund & Burgess, 1996)

HAL constructs a co-occurrence matrix using a sliding window, but with a twist:
it weights co-occurrences by distance. Words closer together in the window receive
higher weights.

For a window of size $k$, the weight of a co-occurrence at distance $d$ is:

$$w(d) = k - d + 1$$

So in a window of size 5, a word at distance 1 gets weight 5, at distance 2 gets
weight 4, and so on. HAL also distinguishes between left and right contexts,
producing an asymmetric matrix of size $|V| \times 2|V|$.

### 1.5.3 Explicit Semantic Analysis (ESA) (Gabrilovich & Markovitch, 2007)

ESA takes a different approach entirely: instead of reducing dimensionality, it
*increases* it by representing each word as a vector over Wikipedia articles. Each
dimension corresponds to a Wikipedia article, and the value is the TF-IDF weight
of the word in that article.

$$\mathbf{v}_w \in \mathbb{R}^{|\text{Wikipedia}|}$$

This produces very high-dimensional but interpretable vectors: you can look at which
Wikipedia articles a word is associated with. The cosine similarity between ESA
vectors correlates well with human judgments of semantic relatedness.

### 1.5.4 Summary: The Pre-Neural Landscape

| Method | Year | Representation | Dimensionality | Key Operation |
|:------:|:----:|:--------------:|:--------------:|:-------------:|
| LSA    | 1990 | Dense          | $k$ (50–300)   | SVD           |
| HAL    | 1996 | Dense/Sparse   | $2|V|$ or reduced | Window co-occurrence |
| Random Indexing | 2000 | Dense  | $d$ (1000+)    | Random projection |
| ESA    | 2007 | Sparse         | $|\text{Wikipedia}|$ | TF-IDF over articles |

All of these methods share a common DNA: they operationalize the distributional
hypothesis by counting contexts and then (optionally) reducing dimensionality. The
neural revolution, which we turn to next, replaces counting with *prediction* — but
as Levy & Goldberg (2014) showed, the underlying mathematics is more similar than
it first appears.

---

## 1.6 Neural Language Models — The Bridge (2003)

### 1.6.1 Bengio et al. (2003): "A Neural Probabilistic Language Model"

In 2003, Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin
published what would become one of the most influential papers in NLP: *"A Neural
Probabilistic Language Model."* This paper introduced two ideas that would reshape
the field:

1. **Learned distributed representations**: Instead of hand-designing features or
   counting co-occurrences, let a neural network *learn* the word representations
   as part of training a language model.

2. **The embedding matrix**: A lookup table $\mathbf{E} \in \mathbb{R}^{|V| \times d}$
   that maps each word in the vocabulary to a dense $d$-dimensional vector.

The paper's abstract states the goal clearly: to learn simultaneously (1) a
distributed representation for each word, and (2) the probability function for word
sequences, expressed in terms of these representations.

### 1.6.2 The Architecture

The model estimates the probability of the next word given the previous $n-1$ words
(an $n$-gram language model):

$$P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

The architecture has four components:

#### Component 1: The Embedding Matrix (Lookup Table)

The embedding matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ has one row per word
in the vocabulary. The $i$-th row, $\mathbf{E}[i] = \mathbf{e}_i \in \mathbb{R}^d$,
is the embedding of word $w_i$.

**How the lookup works:** Given a word $w_i$ represented as a one-hot vector
$\mathbf{x}_i \in \{0, 1\}^{|V|}$, the embedding is obtained by:

$$\mathbf{e}_i = \mathbf{E}^T \mathbf{x}_i$$

But since $\mathbf{x}_i$ is one-hot, this matrix multiplication simply *selects*
the $i$-th row of $\mathbf{E}$:

$$\mathbf{E}^T \mathbf{x}_i = \mathbf{E}^T \mathbf{e}_i^{\text{one-hot}} = \text{row } i \text{ of } \mathbf{E}$$

In practice, this is implemented as an array index lookup, not a matrix multiply —
but mathematically, it is equivalent. This is why the embedding layer is sometimes
called a "lookup table."

**Concrete example.** Suppose $|V| = 5$ and $d = 3$:

$$\mathbf{E} = \begin{bmatrix} 0.2 & -0.1 & 0.5 \\ 0.8 & 0.3 & -0.2 \\ -0.4 & 0.7 & 0.1 \\ 0.1 & -0.5 & 0.9 \\ 0.6 & 0.2 & -0.3 \end{bmatrix}$$

If word $w_3$ has one-hot vector $\mathbf{x}_3 = [0, 0, 1, 0, 0]^T$, then:

$$\mathbf{e}_3 = \mathbf{E}^T \mathbf{x}_3 = [-0.4, 0.7, 0.1]$$

This is simply the third row of $\mathbf{E}$.

#### Component 2: Concatenation Layer

Given a context of $n-1$ words $(w_{t-n+1}, \ldots, w_{t-1})$, we look up each
word's embedding and concatenate them into a single vector:

$$\mathbf{h}_0 = [\mathbf{e}_{t-n+1}; \mathbf{e}_{t-n+2}; \ldots; \mathbf{e}_{t-1}] \in \mathbb{R}^{(n-1) \cdot d}$$

For a trigram model ($n = 3$) with $d = 3$:

$$\mathbf{h}_0 = [\mathbf{e}_{t-2}; \mathbf{e}_{t-1}] \in \mathbb{R}^{6}$$

#### Component 3: Hidden Layer

The concatenated embedding is passed through a hidden layer with a $\tanh$
activation:

$$\mathbf{h}_1 = \tanh(\mathbf{W}_1 \mathbf{h}_0 + \mathbf{b}_1)$$

where $\mathbf{W}_1 \in \mathbb{R}^{h \times (n-1)d}$ and $\mathbf{b}_1 \in \mathbb{R}^h$,
with $h$ being the hidden layer size.

Bengio et al. also included a direct connection from the input to the output:

$$\mathbf{a} = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{W}_3 \mathbf{h}_0 + \mathbf{b}_2$$

where $\mathbf{W}_2 \in \mathbb{R}^{|V| \times h}$, $\mathbf{W}_3 \in \mathbb{R}^{|V| \times (n-1)d}$,
and $\mathbf{b}_2 \in \mathbb{R}^{|V|}$.

#### Component 4: Softmax Output Layer

The output is a probability distribution over the entire vocabulary:

$$P(w_t = w_i \mid w_{t-n+1}, \ldots, w_{t-1}) = \frac{\exp(a_i)}{\sum_{j=1}^{|V|} \exp(a_j)}$$

This is the **softmax** function, which converts the raw scores (logits) $\mathbf{a}$
into a valid probability distribution.

### 1.6.3 The Forward Pass: Step by Step

Let us trace through a complete forward pass with concrete numbers.

**Setup:**
- Vocabulary: $V = \{\text{the}, \text{cat}, \text{sat}, \text{on}, \text{mat}\}$, $|V| = 5$
- Embedding dimension: $d = 3$
- Context size: $n - 1 = 2$ (trigram model)
- Hidden layer size: $h = 4$
- Input: predict the next word after "the cat"

**Step 1: Look up embeddings.**

$$\mathbf{e}_{\text{the}} = \mathbf{E}[0] = [0.2, -0.1, 0.5]$$
$$\mathbf{e}_{\text{cat}} = \mathbf{E}[1] = [0.8, 0.3, -0.2]$$

**Step 2: Concatenate.**

$$\mathbf{h}_0 = [0.2, -0.1, 0.5, 0.8, 0.3, -0.2] \in \mathbb{R}^6$$

**Step 3: Hidden layer computation.**

Suppose (with randomly initialized weights for illustration):

$$\mathbf{W}_1 = \begin{bmatrix} 0.1 & -0.2 & 0.3 & 0.1 & -0.1 & 0.2 \\ -0.3 & 0.1 & 0.2 & -0.1 & 0.3 & 0.1 \\ 0.2 & 0.3 & -0.1 & 0.2 & -0.2 & 0.3 \\ -0.1 & 0.2 & 0.1 & 0.3 & 0.1 & -0.2 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

Computing $\mathbf{W}_1 \mathbf{h}_0$:

Row 1: $(0.1)(0.2) + (-0.2)(-0.1) + (0.3)(0.5) + (0.1)(0.8) + (-0.1)(0.3) + (0.2)(-0.2)$
$= 0.02 + 0.02 + 0.15 + 0.08 - 0.03 - 0.04 = 0.20$

Row 2: $(-0.3)(0.2) + (0.1)(-0.1) + (0.2)(0.5) + (-0.1)(0.8) + (0.3)(0.3) + (0.1)(-0.2)$
$= -0.06 - 0.01 + 0.10 - 0.08 + 0.09 - 0.02 = 0.02$

Row 3: $(0.2)(0.2) + (0.3)(-0.1) + (-0.1)(0.5) + (0.2)(0.8) + (-0.2)(0.3) + (0.3)(-0.2)$
$= 0.04 - 0.03 - 0.05 + 0.16 - 0.06 - 0.06 = 0.00$

Row 4: $(-0.1)(0.2) + (0.2)(-0.1) + (0.1)(0.5) + (0.3)(0.8) + (0.1)(0.3) + (-0.2)(-0.2)$
$= -0.02 - 0.02 + 0.05 + 0.24 + 0.03 + 0.04 = 0.32$

$$\mathbf{W}_1 \mathbf{h}_0 = [0.20, 0.02, 0.00, 0.32]$$

Applying $\tanh$:

$$\mathbf{h}_1 = \tanh([0.20, 0.02, 0.00, 0.32]) = [0.197, 0.020, 0.000, 0.309]$$

**Step 4: Output layer** (simplified, omitting the direct connection $\mathbf{W}_3$
for clarity).

Suppose:

$$\mathbf{W}_2 = \begin{bmatrix} 0.5 & -0.3 & 0.1 & 0.2 \\ -0.2 & 0.4 & 0.3 & -0.1 \\ 0.1 & 0.2 & -0.4 & 0.6 \\ 0.3 & -0.1 & 0.5 & 0.1 \\ -0.4 & 0.3 & 0.2 & -0.5 \end{bmatrix}, \quad \mathbf{b}_2 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

Computing $\mathbf{a} = \mathbf{W}_2 \mathbf{h}_1$:

$$a_{\text{the}} = (0.5)(0.197) + (-0.3)(0.020) + (0.1)(0.000) + (0.2)(0.309) = 0.099 - 0.006 + 0 + 0.062 = 0.155$$
$$a_{\text{cat}} = (-0.2)(0.197) + (0.4)(0.020) + (0.3)(0.000) + (-0.1)(0.309) = -0.039 + 0.008 + 0 - 0.031 = -0.062$$
$$a_{\text{sat}} = (0.1)(0.197) + (0.2)(0.020) + (-0.4)(0.000) + (0.6)(0.309) = 0.020 + 0.004 + 0 + 0.185 = 0.209$$
$$a_{\text{on}} = (0.3)(0.197) + (-0.1)(0.020) + (0.5)(0.000) + (0.1)(0.309) = 0.059 - 0.002 + 0 + 0.031 = 0.088$$
$$a_{\text{mat}} = (-0.4)(0.197) + (0.3)(0.020) + (0.2)(0.000) + (-0.5)(0.309) = -0.079 + 0.006 + 0 - 0.155 = -0.228$$

Logits: $\mathbf{a} = [0.155, -0.062, 0.209, 0.088, -0.228]$

**Step 5: Apply softmax.**

$$\exp(\mathbf{a}) = [\exp(0.155), \exp(-0.062), \exp(0.209), \exp(0.088), \exp(-0.228)]$$
$$= [1.168, 0.940, 1.232, 1.092, 0.796]$$

$$Z = 1.168 + 0.940 + 1.232 + 1.092 + 0.796 = 5.228$$

$$P(w_t \mid \text{the, cat}) = \frac{1}{5.228}[1.168, 0.940, 1.232, 1.092, 0.796]$$
$$= [0.223, 0.180, 0.236, 0.209, 0.152]$$

So the model predicts:

| Word     | $P(w_t \mid \text{the, cat})$ |
|:--------:|:-----------------------------:|
| the      | 0.223                         |
| cat      | 0.180                         |
| **sat**  | **0.236**                     |
| on       | 0.209                         |
| mat      | 0.152                         |

The highest probability is assigned to "sat" — which is indeed a reasonable next word
after "the cat"! (Of course, with random weights this is just luck. After training,
the probabilities would be much more peaked.)

### 1.6.4 Training: How the Embeddings Learn

The model is trained to maximize the log-likelihood of the training corpus:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{t-n+1}, \ldots, w_{t-1}; \theta)$$

where $\theta$ includes all parameters: $\mathbf{E}$, $\mathbf{W}_1$, $\mathbf{b}_1$,
$\mathbf{W}_2$, $\mathbf{W}_3$, $\mathbf{b}_2$.

The crucial insight is that $\mathbf{E}$ is part of $\theta$ — the embedding matrix
is trained jointly with the rest of the network via backpropagation. The gradients
flow back through the softmax, through the hidden layer, through the concatenation,
and into the embedding matrix.

**How gradients reach the embeddings:**

1. The loss $\mathcal{L}$ produces a gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{a}}$
   at the output layer.

2. This propagates back to $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1}$ via
   $\mathbf{W}_2$.

3. Through the $\tanh$ and $\mathbf{W}_1$, we get
   $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0}$.

4. Since $\mathbf{h}_0$ is a concatenation of embeddings, the gradient
   $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0}$ is split into gradients for
   each embedding: $\frac{\partial \mathcal{L}}{\partial \mathbf{e}_{t-n+1}}, \ldots, \frac{\partial \mathcal{L}}{\partial \mathbf{e}_{t-1}}$.

5. Each $\frac{\partial \mathcal{L}}{\partial \mathbf{e}_i}$ updates the corresponding
   row of $\mathbf{E}$.

**The sparsity of the gradient update:** On each training step, only the $n-1$ words
in the context window have their embeddings updated. All other rows of $\mathbf{E}$
receive zero gradient. This is both a feature (efficiency) and a limitation (rare
words are updated infrequently).

### 1.6.5 Why This Was Revolutionary

Bengio et al.'s contribution was not just a better language model — it was a new
*paradigm* for representing words. Here is what changed:

**1. From counting to prediction.** Previous methods (LSA, HAL, Random Indexing)
counted co-occurrences and then (optionally) reduced dimensionality. Bengio's model
*predicts* the next word, and the embeddings emerge as a byproduct of learning to
predict well. This is the shift from **count-based** to **prediction-based** methods.

**2. From two-stage to end-to-end.** In LSA, you first build the co-occurrence matrix,
then apply SVD — two separate stages. In the neural model, the representation and the
task are learned jointly. The embeddings are shaped by the prediction objective.

**3. The embedding matrix as a reusable artifact.** Once trained, the embedding matrix
$\mathbf{E}$ can be extracted and used for other tasks. This is the birth of
**transfer learning** in NLP: train on a large corpus, extract the embeddings, and
use them as features for downstream tasks (sentiment analysis, named entity
recognition, etc.).

**4. Continuous space enables generalization.** Because the embeddings are continuous
vectors, the model can generalize to word sequences it has never seen. If "the cat
sat on the mat" is in the training data, and the model learns that "dog" has a similar
embedding to "cat," then it can assign reasonable probability to "the dog sat on the
mat" even if this exact sequence was never observed.

### 1.6.6 The Computational Bottleneck

The Achilles' heel of Bengio's model was the softmax computation. Computing the
normalizing constant:

$$Z = \sum_{j=1}^{|V|} \exp(a_j)$$

requires summing over the entire vocabulary for every training example. With
$|V| = 100{,}000$, this is extremely expensive.

This bottleneck motivated a decade of research into efficient alternatives:

- **Hierarchical softmax** (Morin & Bengio, 2005): Organize the vocabulary as a
  binary tree, reducing the softmax from $O(|V|)$ to $O(\log |V|)$.
- **Noise contrastive estimation (NCE)** (Gutmann & Hyvärinen, 2010): Reformulate
  the problem as binary classification between real data and noise.
- **Negative sampling** (Mikolov et al., 2013): A simplified version of NCE that
  became the default training method for Word2Vec.

These innovations, which we will cover in Chapter 2, made it practical to train
embeddings on billion-word corpora and gave us Word2Vec, GloVe, and the modern
embedding ecosystem.

### 1.6.7 The Embedding Matrix: A Deeper Look

Let us examine the mathematical properties of the embedding matrix more carefully.

**The embedding matrix as a linear map.** $\mathbf{E} \in \mathbb{R}^{|V| \times d}$
defines a linear map from the one-hot space $\mathbb{R}^{|V|}$ to the embedding
space $\mathbb{R}^d$:

$$\phi: \mathbb{R}^{|V|} \to \mathbb{R}^d, \quad \phi(\mathbf{x}) = \mathbf{E}^T \mathbf{x}$$

This map has several important properties:

1. **Dimensionality reduction**: It maps from $|V|$ dimensions to $d$ dimensions,
   where typically $d \ll |V|$ (e.g., $d = 300$ vs. $|V| = 100{,}000$).

2. **Not orthogonality-preserving**: Unlike the one-hot space where all word vectors
   are orthogonal, the embedding space allows (and encourages) similar words to have
   non-orthogonal (similar) vectors.

3. **Learned, not fixed**: Unlike random projections (Random Indexing) or analytically
   derived projections (SVD), the embedding matrix is learned from data via gradient
   descent.

4. **The columns of $\mathbf{E}$ are latent features**: Each of the $d$ columns of
   $\mathbf{E}$ can be thought of as a latent semantic feature. The $j$-th entry of
   a word's embedding, $E_{ij}$, is the "loading" of word $w_i$ on latent feature $j$.
   Unlike LSA's singular vectors, these features are not orthogonal and are not
   ordered by importance.

**Connection to LSA:** If we train the neural language model with a linear activation
(no $\tanh$), no hidden layer, and a squared error loss on the co-occurrence matrix,
the optimal embedding matrix $\mathbf{E}$ converges to the SVD solution. The neural
model is strictly more expressive than LSA because of the non-linear hidden layer.

---

## 1.7 The Big Picture: From Distributional Counts to Neural Predictions

Let us step back and see the arc of this chapter as a single intellectual trajectory.

### The Evolution in One Table

| Era | Method | Representation | How Meaning Is Captured | Key Limitation |
|:---:|:------:|:--------------:|:-----------------------:|:--------------:|
| 1950s | Distributional Hypothesis | Conceptual | Context defines meaning | Not computational |
| 1970s | One-Hot Encoding | $\mathbf{x} \in \{0,1\}^{|V|}$ | Identity only | No similarity |
| 1990 | LSA (SVD) | $\mathbf{u} \in \mathbb{R}^k$ | Latent factors of co-occurrence | Linear, static |
| 1996 | HAL | $\mathbf{v} \in \mathbb{R}^{2|V|}$ | Distance-weighted co-occurrence | High-dimensional |
| 2000 | Random Indexing | $\mathbf{v} \in \mathbb{R}^d$ | Random projection of co-occurrence | Approximate |
| 2003 | Neural LM (Bengio) | $\mathbf{e} \in \mathbb{R}^d$ | Prediction of next word | Slow softmax |

### The Unifying Thread

Every method in this chapter implements the same core idea:

$$\text{Distributional Hypothesis} \xrightarrow{\text{operationalize}} \text{Context Representation} \xrightarrow{\text{compress}} \text{Dense Vector}$$

The methods differ in:

1. **How they define context** (window, document, dependency)
2. **How they count or predict** (raw counts, PMI, neural prediction)
3. **How they compress** (SVD, random projection, learned weights)

But the DNA is the same: meaning is context, and context can be captured in a vector.

### What Comes Next

The stage is now set for the Word2Vec revolution (Chapter 2). Mikolov et al. (2013)
will show that by simplifying Bengio's architecture — removing the hidden layer,
using negative sampling instead of softmax — we can train embeddings on billions of
words in hours rather than weeks. The resulting vectors will exhibit the famous
algebraic properties:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

But this magic is not new. It is the distributional hypothesis, formalized by Harris
in 1954, operationalized by Deerwester in 1990, neuralized by Bengio in 2003, and
scaled by Mikolov in 2013. The foundations were laid in this chapter.

---

## References

1. Harris, Z. S. (1954). Distributional structure. *Word*, 10(2-3), 146–162.

2. Firth, J. R. (1957). A synopsis of linguistic theory, 1930–1955. In *Studies in
   Linguistic Analysis*, 1–32. Blackwell.

3. Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic
   indexing. *Communications of the ACM*, 18(11), 613–620.

4. Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R.
   (1990). Indexing by latent semantic analysis. *Journal of the American Society
   for Information Science*, 41(6), 391–407.

5. Lund, K., & Burgess, C. (1996). Producing high-dimensional semantic spaces from
   lexical co-occurrence. *Behavior Research Methods, Instruments, & Computers*,
   28(2), 203–208.

6. Kanerva, P., Kristoferson, J., & Holst, A. (2000). Random indexing of text
   samples for latent semantic analysis. *Proceedings of the Annual Meeting of the
   Cognitive Science Society*, 22.

7. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural
   probabilistic language model. *Journal of Machine Learning Research*, 3,
   1137–1155.

8. Morin, F., & Bengio, Y. (2005). Hierarchical probabilistic neural network
   language model. *Proceedings of AISTATS*.

9. Sahlgren, M. (2005). An introduction to random indexing. *Proceedings of the
   Methods and Applications of Semantic Indexing Workshop at TKE*.

10. Gabrilovich, E., & Markovitch, S. (2007). Computing semantic relatedness using
    Wikipedia-based explicit semantic analysis. *Proceedings of IJCAI*, 1606–1611.

11. Gutmann, M. U., & Hyvärinen, A. (2010). Noise-contrastive estimation: A new
    estimation principle for unnormalized statistical models. *Proceedings of
    AISTATS*.

12. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation
    of word representations in vector space. *Proceedings of ICLR*.

13. Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix
    factorization. *Advances in Neural Information Processing Systems*, 27.

---

*Next chapter: [Chapter 2 — Word2Vec: The Embedding Revolution](02-word2vec.md)*
