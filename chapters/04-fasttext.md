# Chapter 4: FastText — Subword Embeddings and Beyond

---

## 4.1 Introduction

In 2017, Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov published
*"Enriching Word Vectors with Subword Information"* while working at **Facebook AI Research
(FAIR)**. The paper introduced **FastText**, an extension of the Skip-gram model from Word2Vec
that represents each word as a **bag of character n-grams** rather than a single atomic token.

### 4.1.1 The Problem with Atomic Word Vectors

Recall from Chapter 2 that Word2Vec assigns a unique vector to every word in the vocabulary.
This design has two critical limitations:

1. **Morphologically rich languages.** In languages like Turkish, Finnish, or Czech, a single
   root can generate hundreds of surface forms through inflection, derivation, and compounding.
   Word2Vec treats each form as an independent token, learning nothing about shared structure.

   | Language | Root     | Surface forms (sample)                        |
   |----------|----------|-----------------------------------------------|
   | Turkish  | gel-     | gelir, gelmiş, gelmeyecek, gelinecek, ...     |
   | Finnish  | talo     | talossa, talosta, taloon, taloihin, ...        |
   | German   | fahren   | Fahrrad, Fahrplan, abfahren, Zufahrt, ...     |

2. **Out-of-vocabulary (OOV) words.** Any word not seen during training receives no
   representation at all. This is devastating for real-world applications where new words,
   misspellings, and domain-specific jargon appear constantly.

### 4.1.2 The Core Insight

FastText's key idea is deceptively simple:

> **A word's meaning is partially encoded in its internal structure.**

The prefix "un-" signals negation. The suffix "-ness" signals a noun derived from an adjective.
The substring "electr" connects "electric", "electricity", "electron", and "electrocardiogram".
By learning representations for these subword units, FastText can:

- Share statistical strength across morphological variants
- Construct reasonable vectors for words never seen during training
- Achieve superior performance on morphologically rich languages

---

## 4.2 Character N-grams

### 4.2.1 Decomposing Words into N-grams

FastText represents each word as a bag of character n-grams. Before extracting n-grams,
special **boundary markers** `<` and `>` are added to the beginning and end of the word.
These markers serve two purposes:

1. They distinguish prefixes from suffixes (e.g., the trigram `<re` in "return" vs. `re>` in "care")
2. They allow the model to learn that certain character sequences are more common at word
   boundaries

Given a word $w$, FastText:

1. Adds boundary markers: `<w>`
2. Extracts all character n-grams of length $n_{\min}$ to $n_{\max}$
3. Also includes the **whole word** token `<w>` as a special n-gram

The default range is $n_{\min} = 3$ to $n_{\max} = 6$.

### 4.2.2 Worked Example: "where"

Let's decompose the word **"where"** with $n_{\min} = 3$ and $n_{\max} = 6$.

**Step 1:** Add boundary markers:

$$
\text{"where"} \rightarrow \text{"<where>"}
$$

**Step 2:** Extract n-grams of each length:

| Length | N-grams                                          | Count |
|--------|--------------------------------------------------|-------|
| $n=3$  | `<wh`, `whe`, `her`, `ere`, `re>`                | 5     |
| $n=4$  | `<whe`, `wher`, `here`, `ere>`                   | 4     |
| $n=5$  | `<wher`, `where`, `here>`                        | 3     |
| $n=6$  | `<where`, `where>`                               | 2     |

**Step 3:** Add the whole-word token:

$$
\text{<where>}
$$

**Total n-gram set:**

$$
\mathcal{G}_{\text{where}} = \{ \text{<wh}, \text{whe}, \text{her}, \text{ere}, \text{re>},
\text{<whe}, \text{wher}, \text{here}, \text{ere>}, \text{<wher}, \text{where},
\text{here>}, \text{<where}, \text{where>}, \text{<where>} \}
$$

That gives us $|\mathcal{G}_{\text{where}}| = 15$ n-grams (14 character n-grams + 1 whole word).

### 4.2.3 Formal Notation

For a word $w$, let $\mathcal{G}_w$ denote the set of all n-grams appearing in the
boundary-augmented form of $w$, including the whole-word token. Each n-gram $g \in \mathcal{G}_w$
is associated with a vector $\mathbf{z}_g \in \mathbb{R}^d$.

### 4.2.4 Counting N-grams

For a word of length $L$ (after adding boundary markers, the augmented length is $L + 2$),
the number of n-grams of length $n$ is:

$$
\text{count}(n) = (L + 2) - n + 1 = L - n + 3
$$

The total number of character n-grams (excluding the whole-word token) is:

$$
\text{Total} = \sum_{n=n_{\min}}^{n_{\max}} (L - n + 3)
$$

**Example:** For "where" ($L = 5$), with $n_{\min} = 3$, $n_{\max} = 6$:

$$
\text{Total} = (5 - 3 + 3) + (5 - 4 + 3) + (5 - 5 + 3) + (5 - 6 + 3) = 5 + 4 + 3 + 2 = 14
$$

Adding the whole-word token gives $14 + 1 = 15$, matching our count above. ✓

### 4.2.5 Why This Range?

The choice of $n_{\min} = 3$ and $n_{\max} = 6$ is empirically motivated:

- **Too small** ($n = 1, 2$): Individual characters and bigrams carry little semantic
  information. The letter "e" appears in almost every English word.
- **Too large** ($n > 6$): Long n-grams approach full words and lose the ability to
  generalize across morphological variants.
- **The sweet spot** ($3 \leq n \leq 6$): Captures meaningful morphemes like prefixes
  (`un-`, `re-`, `pre-`), suffixes (`-ing`, `-tion`, `-ness`), and roots.

---

## 4.3 The Modified Skip-gram Objective

### 4.3.1 Recap: Original Skip-gram Scoring

In the standard Skip-gram model (Chapter 2), the scoring function between a center word $w$
and a context word $c$ is the dot product of their embedding vectors:

$$
s(w, c) = \mathbf{u}_w^\top \mathbf{v}_c
$$

where $\mathbf{u}_w \in \mathbb{R}^d$ is the input vector for word $w$ and
$\mathbf{v}_c \in \mathbb{R}^d$ is the output (context) vector for word $c$.

### 4.3.2 FastText Scoring Function

FastText replaces the single word vector $\mathbf{u}_w$ with the **sum of its n-gram vectors**:

$$
\boxed{s(w, c) = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c}
$$

where:
- $\mathcal{G}_w$ is the set of n-grams for word $w$ (including the whole-word token)
- $\mathbf{z}_g \in \mathbb{R}^d$ is the embedding vector for n-gram $g$
- $\mathbf{v}_c \in \mathbb{R}^d$ is the context vector for word $c$

Equivalently, the **word representation** for $w$ is:

$$
\mathbf{u}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g
$$

This is the key equation of FastText: a word's vector is the sum of its subword vectors.

### 4.3.3 Numerical Example: Computing a Score

Let's compute $s(\text{"where"}, \text{"location"})$ with $d = 4$ (tiny dimension for illustration).

Suppose we have the following n-gram vectors (showing only 3-grams for brevity):

| N-gram     | $\mathbf{z}_g$                  |
|------------|---------------------------------|
| `<wh`      | $[0.2,\ 0.1,\ -0.3,\ 0.4]$    |
| `whe`      | $[0.1,\ 0.3,\ 0.2,\ -0.1]$    |
| `her`      | $[0.3,\ -0.2,\ 0.1,\ 0.2]$    |
| `ere`      | $[-0.1,\ 0.4,\ 0.0,\ 0.3]$    |
| `re>`      | $[0.0,\ 0.2,\ -0.1,\ 0.1]$    |
| `<where>`  | $[0.4,\ -0.1,\ 0.3,\ 0.0]$    |

**Step 1:** Sum the n-gram vectors to get the word representation:

$$
\mathbf{u}_{\text{where}} = \sum_{g \in \mathcal{G}_{\text{where}}} \mathbf{z}_g
$$

Computing element-wise:

$$
\mathbf{u}_{\text{where}} = \begin{bmatrix}
0.2 + 0.1 + 0.3 + (-0.1) + 0.0 + 0.4 \\
0.1 + 0.3 + (-0.2) + 0.4 + 0.2 + (-0.1) \\
(-0.3) + 0.2 + 0.1 + 0.0 + (-0.1) + 0.3 \\
0.4 + (-0.1) + 0.2 + 0.3 + 0.1 + 0.0
\end{bmatrix}
= \begin{bmatrix} 0.9 \\ 0.7 \\ 0.2 \\ 0.9 \end{bmatrix}
$$

**Step 2:** Suppose the context vector for "location" is:

$$
\mathbf{v}_{\text{location}} = [0.5,\ 0.3,\ -0.2,\ 0.6]
$$

**Step 3:** Compute the score:

$$
s(\text{where}, \text{location}) = \mathbf{u}_{\text{where}}^\top \mathbf{v}_{\text{location}}
= (0.9)(0.5) + (0.7)(0.3) + (0.2)(-0.2) + (0.9)(0.6)
$$

$$
= 0.45 + 0.21 + (-0.04) + 0.54 = \boxed{1.16}
$$

A positive score indicates that "where" and "location" are likely to co-occur — which makes
semantic sense.

### 4.3.4 Loss Function with Negative Sampling

Like Word2Vec, FastText uses **negative sampling** to make training tractable. For a given
center word $w_t$ at position $t$ in the corpus, and a context word $w_c$ within the window,
the objective is:

$$
\ell(w_t, w_c) = \log \sigma\!\left(s(w_t, w_c)\right) + \sum_{i=1}^{k} \mathbb{E}_{n_i \sim P_n} \left[\log \sigma\!\left(-s(w_t, n_i)\right)\right]
$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $k$ is the number of negative samples (typically $k = 5$ or $k = 10$)
- $P_n$ is the noise distribution (unigram distribution raised to the $3/4$ power)
- $n_i$ are the negative samples (random words unlikely to be true context)

Substituting the FastText scoring function:

$$
\ell(w_t, w_c) = \log \sigma\!\left(\sum_{g \in \mathcal{G}_{w_t}} \mathbf{z}_g^\top \mathbf{v}_{w_c}\right) + \sum_{i=1}^{k} \mathbb{E}_{n_i \sim P_n} \left[\log \sigma\!\left(-\sum_{g \in \mathcal{G}_{w_t}} \mathbf{z}_g^\top \mathbf{v}_{n_i}\right)\right]
$$

The full corpus objective is to **maximize**:

$$
J = \sum_{t=1}^{T} \sum_{w_c \in \mathcal{C}_t} \ell(w_t, w_c)
$$

where $T$ is the corpus size and $\mathcal{C}_t$ is the set of context words for position $t$.

### 4.3.5 Step-by-Step Derivation of the Gradient

To train the model, we need gradients with respect to both the n-gram vectors $\mathbf{z}_g$
and the context vectors $\mathbf{v}_c$.

**Setup:** Consider a single positive pair $(w, c)$ and one negative sample $n$. The local
objective is:

$$
\ell = \log \sigma(s^+) + \log \sigma(-s^-)
$$

where $s^+ = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c$ and
$s^- = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_n$.

**Step 1:** Recall that $\frac{d}{dx} \log \sigma(x) = 1 - \sigma(x)$ and
$\frac{d}{dx} \log \sigma(-x) = -\sigma(x)$.

**Step 2:** Gradient with respect to an n-gram vector $\mathbf{z}_g$ (for $g \in \mathcal{G}_w$):

$$
\frac{\partial \ell}{\partial \mathbf{z}_g}
= \left(1 - \sigma(s^+)\right) \mathbf{v}_c
- \sigma(s^-) \mathbf{v}_n
$$

**Step 3:** Gradient with respect to the positive context vector $\mathbf{v}_c$:

$$
\frac{\partial \ell}{\partial \mathbf{v}_c}
= \left(1 - \sigma(s^+)\right) \sum_{g \in \mathcal{G}_w} \mathbf{z}_g
= \left(1 - \sigma(s^+)\right) \mathbf{u}_w
$$

**Step 4:** Gradient with respect to the negative context vector $\mathbf{v}_n$:

$$
\frac{\partial \ell}{\partial \mathbf{v}_n}
= -\sigma(s^-) \sum_{g \in \mathcal{G}_w} \mathbf{z}_g
= -\sigma(s^-) \mathbf{u}_w
$$

**Step 5:** Update rules (with learning rate $\eta$):

$$
\mathbf{z}_g \leftarrow \mathbf{z}_g + \eta \left[\left(1 - \sigma(s^+)\right) \mathbf{v}_c - \sigma(s^-) \mathbf{v}_n\right]
$$

$$
\mathbf{v}_c \leftarrow \mathbf{v}_c + \eta \left(1 - \sigma(s^+)\right) \mathbf{u}_w
$$

$$
\mathbf{v}_n \leftarrow \mathbf{v}_n - \eta\, \sigma(s^-)\, \mathbf{u}_w
$$

Note that **every n-gram** of the center word receives the same gradient signal from the
context. This is how subword information propagates: the n-gram `her` in "where" gets
updated based on the contexts of "where", but it also gets updated when processing "here",
"there", "heritage", etc. This sharing is the mechanism by which FastText learns
morphological regularities.


---

## 4.4 Handling Out-of-Vocabulary Words

This is arguably FastText's most important practical contribution. While Word2Vec and GloVe
simply fail on unseen words, FastText can construct a meaningful representation for **any**
word — even one never encountered during training.

### 4.4.1 The Mechanism

The logic is straightforward:

1. Given an OOV word $w_{\text{oov}}$, add boundary markers: `<` $w_{\text{oov}}$ `>`
2. Extract all character n-grams: $\mathcal{G}_{w_{\text{oov}}}$
3. Look up the learned vectors for each n-gram
4. Sum them to produce the word vector:

$$
\mathbf{u}_{w_{\text{oov}}} = \sum_{g \in \mathcal{G}_{w_{\text{oov}}}} \mathbf{z}_g
$$

This works because the n-grams of the OOV word **overlap** with n-grams of words seen
during training. The model has already learned what these subword units "mean" from their
co-occurrence patterns across the training corpus.

### 4.4.2 Step-by-Step Example: Embedding an Unseen Word

Suppose the word **"unfairly"** was never seen during training, but the model has seen
"unfair", "fairly", "unfold", "fairly", "kindly", etc.

**Step 1:** Add boundary markers:

$$
\text{"unfairly"} \rightarrow \text{"<unfairly>"}
$$

**Step 2:** Extract character 3-grams (showing only trigrams for clarity):

$$
\text{<un, unf, nfa, fai, air, irl, rly, ly>}
$$

**Step 3:** Many of these n-grams were learned during training:

| N-gram | Learned from (examples)                    | Semantic signal              |
|--------|--------------------------------------------|------------------------------|
| `<un`  | "unfair", "undo", "unlock", "unhappy"      | Negation prefix              |
| `unf`  | "unfair", "unfold", "unfit"                | Negation + following char    |
| `fai`  | "unfair", "fair", "fairy", "faith"         | Fairness / related concepts  |
| `air`  | "fair", "chair", "airport", "airly"        | Various (some noise)         |
| `rly`  | "fairly", "early", "clearly", "poorly"     | Adverb-like ending           |
| `ly>`  | "fairly", "kindly", "slowly", "quickly"    | Adverb suffix                |

**Step 4:** Sum the n-gram vectors:

$$
\mathbf{u}_{\text{unfairly}} = \mathbf{z}_{\text{<un}} + \mathbf{z}_{\text{unf}} + \mathbf{z}_{\text{nfa}} + \mathbf{z}_{\text{fai}} + \mathbf{z}_{\text{air}} + \mathbf{z}_{\text{irl}} + \mathbf{z}_{\text{rly}} + \mathbf{z}_{\text{ly>}} + \cdots
$$

The resulting vector will be **close to** the vectors for "unfair" and "fairly" because they
share many n-grams. It will also carry the semantic flavor of negation (from `<un`) and
adverb formation (from `ly>`).

### 4.4.3 Numerical Walkthrough

Let's make this concrete with $d = 3$ vectors. Suppose the trained n-gram vectors are:

| N-gram | $\mathbf{z}_g$           | Primary signal    |
|--------|--------------------------|-------------------|
| `<un`  | $[−0.5,\ 0.3,\ 0.1]$   | Negation          |
| `fai`  | $[0.4,\ 0.6,\ −0.2]$   | Fairness          |
| `rly`  | $[0.1,\ −0.1,\ 0.5]$   | Adverb pattern    |
| `ly>`  | $[0.0,\ −0.2,\ 0.6]$   | Adverb suffix     |

Summing just these four key n-grams:

$$
\mathbf{u}_{\text{unfairly}} \approx \begin{bmatrix} -0.5 + 0.4 + 0.1 + 0.0 \\ 0.3 + 0.6 + (-0.1) + (-0.2) \\ 0.1 + (-0.2) + 0.5 + 0.6 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.6 \\ 1.0 \end{bmatrix}
$$

Now compare with the trained vector for "unfair":

$$
\mathbf{u}_{\text{unfair}} = \mathbf{z}_{\text{<un}} + \mathbf{z}_{\text{fai}} + \cdots \approx \begin{bmatrix} -0.1 \\ 0.9 \\ -0.1 \end{bmatrix}
$$

And the trained vector for "fairly":

$$
\mathbf{u}_{\text{fairly}} = \mathbf{z}_{\text{fai}} + \mathbf{z}_{\text{rly}} + \mathbf{z}_{\text{ly>}} + \cdots \approx \begin{bmatrix} 0.5 \\ 0.3 \\ 0.9 \end{bmatrix}
$$

Computing cosine similarities:

$$
\cos(\mathbf{u}_{\text{unfairly}}, \mathbf{u}_{\text{unfair}}) = \frac{(0.0)(-0.1) + (0.6)(0.9) + (1.0)(-0.1)}{\sqrt{0.0^2 + 0.6^2 + 1.0^2} \cdot \sqrt{0.1^2 + 0.9^2 + 0.1^2}}
$$

$$
= \frac{0 + 0.54 - 0.1}{\sqrt{1.36} \cdot \sqrt{0.83}} = \frac{0.44}{1.166 \cdot 0.911} = \frac{0.44}{1.062} \approx 0.41
$$

$$
\cos(\mathbf{u}_{\text{unfairly}}, \mathbf{u}_{\text{fairly}}) = \frac{(0.0)(0.5) + (0.6)(0.3) + (1.0)(0.9)}{\sqrt{1.36} \cdot \sqrt{1.06}}
$$

$$
= \frac{0 + 0.18 + 0.9}{1.166 \cdot 1.030} = \frac{1.08}{1.201} \approx 0.90
$$

The OOV word "unfairly" is most similar to "fairly" (sharing more n-grams), with moderate
similarity to "unfair" — a linguistically reasonable result.

### 4.4.4 Limitations of OOV Handling

While powerful, FastText's OOV mechanism has limits:

- **Completely novel character sequences** (e.g., a word in a script not seen during training)
  will have random-like representations
- **Short words** produce fewer n-grams, giving less signal to work with
- **Acronyms and codes** (e.g., "XML", "B2B") may not benefit much from subword decomposition
- The quality degrades as the OOV word shares fewer n-grams with the training vocabulary

---

## 4.5 The Hashing Trick

### 4.5.1 The Combinatorial Explosion Problem

Consider the number of possible character n-grams. With an alphabet of $A$ characters
(say $A = 26$ lowercase letters plus boundary markers, digits, etc. ≈ 40 characters):

| N-gram length | Possible n-grams ($A^n$) |
|---------------|--------------------------|
| $n = 3$       | $40^3 = 64{,}000$        |
| $n = 4$       | $40^4 = 2{,}560{,}000$   |
| $n = 5$       | $40^5 = 102{,}400{,}000$ |
| $n = 6$       | $40^6 \approx 4.1 \times 10^9$ |

Storing a separate $d$-dimensional vector for each possible n-gram across lengths 3–6 would
require billions of parameters — far more than the vocabulary-sized matrices in Word2Vec.

### 4.5.2 The Solution: Hashing N-grams to Buckets

FastText uses the **hashing trick** (also known as feature hashing) to map the unbounded
set of n-grams to a fixed-size set of **buckets**.

**Mechanism:**

1. Define a hash function $h: \text{n-grams} \rightarrow \{1, 2, \ldots, B\}$
2. Maintain a matrix $\mathbf{Z} \in \mathbb{R}^{B \times d}$ where row $b$ is the vector
   for bucket $b$
3. For an n-gram $g$, its vector is: $\mathbf{z}_g = \mathbf{Z}[h(g)]$

The default bucket size in FastText is:

$$
B = 2{,}000{,}000 \quad \text{(2 million buckets)}
$$

FastText uses the **FNV-1a** (Fowler–Noll–Vo) hash function, which is fast and provides
good distribution.

### 4.5.3 How Hashing Works in Practice

**Example:** Suppose $B = 5$ (tiny, for illustration) and we hash the n-grams of "where":

| N-gram    | $h(g) \mod 5$ | Bucket |
|-----------|----------------|--------|
| `<wh`     | $h(\text{<wh}) = 7 \mod 5$   | 2 |
| `whe`     | $h(\text{whe}) = 13 \mod 5$  | 3 |
| `her`     | $h(\text{her}) = 4 \mod 5$   | 4 |
| `ere`     | $h(\text{ere}) = 11 \mod 5$  | 1 |
| `re>`     | $h(\text{re>}) = 9 \mod 5$   | 4 |

Notice that `her` and `re>` **collide** — they map to the same bucket (4). Their vectors
will be identical: $\mathbf{z}_{\text{her}} = \mathbf{z}_{\text{re>}} = \mathbf{Z}[4]$.

### 4.5.4 The Word Vector with Hashing

The complete word representation becomes:

$$
\mathbf{u}_w = \sum_{g \in \mathcal{G}_w} \mathbf{Z}[h(g)] + \mathbf{W}[w]
$$

where:
- $\mathbf{Z} \in \mathbb{R}^{B \times d}$ is the n-gram embedding matrix (hashed)
- $\mathbf{W} \in \mathbb{R}^{V \times d}$ is the word embedding matrix (vocabulary-sized)
- The whole-word token is looked up from $\mathbf{W}$, not hashed

This separation ensures that common words retain their own dedicated vectors while sharing
subword information through the n-gram matrix.

### 4.5.5 Memory Analysis

Let's compare the memory requirements:

**Word2Vec** (vocabulary only):
$$
\text{Parameters} = V \times d
$$

For $V = 1{,}000{,}000$ and $d = 300$: $300M$ parameters.

**FastText** (vocabulary + n-gram buckets):
$$
\text{Parameters} = (V + B) \times d
$$

For $V = 1{,}000{,}000$, $B = 2{,}000{,}000$, $d = 300$: $900M$ parameters.

FastText uses roughly **3× more memory** than Word2Vec, but gains the ability to handle
any word in any language.

### 4.5.6 Trade-offs of the Hashing Trick

| Aspect              | Benefit                                    | Cost                                      |
|---------------------|--------------------------------------------|--------------------------------------------|
| Memory              | Fixed, predictable memory usage            | Some capacity wasted on empty buckets      |
| Collisions          | —                                          | Different n-grams share vectors            |
| Speed               | O(1) lookup per n-gram                     | Hash computation overhead                  |
| Scalability         | Works for any language/alphabet            | Collision rate increases with vocabulary    |
| Interpretability    | —                                          | Cannot recover which n-gram a bucket represents |

**Collision analysis:** With $B = 2M$ buckets and a typical vocabulary generating ~10M
unique n-grams, the average bucket holds ~5 n-grams. The FNV hash distributes these
relatively uniformly, so most collisions involve unrelated n-grams whose interference
averages out during training.


---

## 4.6 Supervised Text Classification with FastText

In addition to unsupervised word embeddings, the FastText library includes a remarkably
effective **supervised text classifier**, described in Joulin et al. (2017),
*"Bag of Tricks for Efficient Text Classification."*

### 4.6.1 Architecture Overview

The FastText classifier has a strikingly simple architecture:

```
Input text → N-gram features → Average embedding → Linear classifier → Softmax/Hierarchical Softmax
```

Formally, for a document with $N$ words $w_1, w_2, \ldots, w_N$:

1. **Feature extraction:** Represent each word (and word n-gram) as a vector
2. **Averaging:** Compute the document representation as the mean of feature vectors:

$$
\mathbf{d} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_{w_i}
$$

3. **Classification:** Apply a linear layer followed by softmax:

$$
P(y \mid \mathbf{d}) = \text{softmax}(\mathbf{A} \mathbf{d} + \mathbf{b})
$$

where $\mathbf{A} \in \mathbb{R}^{K \times d}$ is the weight matrix for $K$ classes.

### 4.6.2 Bag of Word N-grams

A critical trick for the classifier is using **word-level n-grams** (bigrams, trigrams)
in addition to unigrams. This captures local word order without the complexity of
recurrent or convolutional architectures.

**Example:** For the sentence "I love this movie":

| Feature type | Features                                              |
|-------------|-------------------------------------------------------|
| Unigrams    | "I", "love", "this", "movie"                          |
| Bigrams     | "I love", "love this", "this movie"                   |
| Trigrams    | "I love this", "love this movie"                      |

These word n-grams are also hashed to buckets, just like character n-grams in the
embedding model.

### 4.6.3 Hierarchical Softmax

When the number of classes $K$ is large (e.g., hundreds of thousands of categories in
tag prediction), computing the full softmax becomes expensive:

$$
\text{Standard softmax cost:} \quad O(K \cdot d)
$$

FastText uses **hierarchical softmax** based on a binary Huffman tree to reduce this to:

$$
\text{Hierarchical softmax cost:} \quad O(\log_2 K \cdot d)
$$

**How it works:**

1. Build a binary tree where each leaf is a class label
2. Frequent classes get shorter paths (Huffman coding)
3. Each internal node has a learned vector $\mathbf{v}_{\text{node}} \in \mathbb{R}^d$
4. The probability of a class is the product of sigmoid decisions along the path from
   root to leaf:

$$
P(y \mid \mathbf{d}) = \prod_{j=1}^{|\text{path}(y)|} \sigma\!\left(\text{sign}(j) \cdot \mathbf{v}_{n_j}^\top \mathbf{d}\right)
$$

where $\text{sign}(j) = +1$ if the path goes left at node $j$, and $-1$ if it goes right.

**Example:** With $K = 1{,}000{,}000$ classes:
- Standard softmax: 1,000,000 dot products per prediction
- Hierarchical softmax: $\log_2(1{,}000{,}000) \approx 20$ dot products per prediction
- **Speedup: ~50,000×**

### 4.6.4 Training Objective

The classifier minimizes the negative log-likelihood over the training set:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid \mathbf{d}_i)
$$

where $(y_i, \mathbf{d}_i)$ are the label-document pairs.

### 4.6.5 Comparison with Deep Learning Approaches

Joulin et al. (2017) showed surprising results on standard benchmarks:

| Model                  | AG News (Acc%) | Yelp Full (Acc%) | Training Time |
|------------------------|----------------|-------------------|---------------|
| Char-CNN (Zhang, 2015) | 87.2           | 60.4              | Hours (GPU)   |
| VDCNN (Conneau, 2017)  | 91.3           | 64.7              | Hours (GPU)   |
| FastText (bigrams)     | 92.5           | 63.9              | Seconds (CPU) |

Key takeaways:
- FastText achieves **comparable or better accuracy** than deep models on many tasks
- Training is **orders of magnitude faster** — seconds on CPU vs. hours on GPU
- The simplicity of the model makes it highly practical for production systems
- Deep models tend to win on tasks requiring complex reasoning or long-range dependencies

### 4.6.6 When to Use FastText Classification

FastText classification excels when:
- You need **fast training and inference** (real-time systems, rapid prototyping)
- The dataset is **large** (millions of examples)
- The number of classes is **very large** (tag prediction, product categorization)
- **Local word patterns** are sufficient (sentiment, topic classification)
- Computational resources are **limited** (CPU-only environments)

It may underperform when:
- The task requires understanding **long-range dependencies**
- **Word order** matters significantly beyond local n-grams
- The dataset is **small** (deep models with pre-training may generalize better)

---

## 4.7 Practical Impact and Legacy

### 4.7.1 Pre-trained Vectors for 157 Languages

One of FastText's most significant contributions to the NLP community is the release of
**pre-trained word vectors for 157 languages**, trained on Common Crawl and Wikipedia data.
This was a landmark moment for multilingual NLP:

- Languages with limited NLP resources (Uzbek, Yoruba, Malagasy) received high-quality
  word vectors for the first time
- Researchers could bootstrap NLP systems for low-resource languages without massive
  compute budgets
- The vectors are freely available at [fasttext.cc](https://fasttext.cc)

Each language model provides:
- 300-dimensional vectors
- Trained on Common Crawl + Wikipedia
- Available in both full model (with n-gram information) and word-vectors-only format

### 4.7.2 When FastText Beats Word2Vec

FastText consistently outperforms Word2Vec in specific scenarios:

**1. Morphologically rich languages:**

| Language | Task              | Word2Vec | FastText | Δ       |
|----------|-------------------|----------|----------|---------|
| German   | Word similarity   | 0.56     | 0.68     | +0.12   |
| Czech    | Word analogy      | 0.42     | 0.61     | +0.19   |
| Turkish  | Word similarity   | 0.38     | 0.55     | +0.17   |
| English  | Word similarity   | 0.73     | 0.75     | +0.02   |

The gains are largest for morphologically complex languages and smallest for English,
which has relatively simple morphology.

**2. Rare words:**

Words appearing fewer than 5 times in the training corpus benefit enormously from subword
information. While Word2Vec produces noisy vectors for rare words (too few training
examples), FastText leverages n-gram sharing to produce stable representations.

**3. Noisy text:**

Social media, user reviews, and informal text contain misspellings, abbreviations, and
creative word formations. FastText handles these gracefully:

| Input          | Word2Vec         | FastText                          |
|----------------|------------------|-----------------------------------|
| "amazinggg"    | OOV (no vector)  | Close to "amazing"                |
| "loooove"      | OOV (no vector)  | Close to "love"                   |
| "unfriend"     | OOV (no vector)  | Blend of "un-" + "friend"         |
| "Brexit"       | OOV (no vector)  | Blend of "Br-" + "exit" patterns  |

### 4.7.3 When Word2Vec May Be Preferred

FastText is not universally superior:

- **Memory-constrained environments:** FastText models are ~3× larger than Word2Vec
- **Languages with simple morphology** (e.g., Mandarin Chinese, where characters are
  the primary semantic units): subword information adds less value
- **When exact word identity matters:** The hashing trick means some n-gram information
  is lost to collisions
- **Very large vocabularies with short words:** The overhead of n-gram processing may
  not be justified

### 4.7.4 FastText's Place in the Embedding Timeline

```
2003  Neural LM (Bengio)
  │
2013  Word2Vec (Mikolov et al.)
  │
2014  GloVe (Pennington et al.)
  │
2017  FastText (Bojanowski et al.)  ← Subword revolution
  │
2018  ELMo (Peters et al.)         ← Contextual embeddings
  │
2018  BERT (Devlin et al.)         ← Transformer era
  │
  ↓
```

FastText represents the **culmination of static word embeddings** — the last major advance
before the field shifted to contextual representations. Its key innovations (subword
modeling, the hashing trick, efficient classification) remain relevant even in the
transformer era:

- Many transformer tokenizers (BPE, WordPiece, SentencePiece) are conceptually descended
  from the subword insight
- FastText remains the go-to solution when transformer models are too expensive or slow
- The pre-trained vectors continue to serve as features in downstream systems worldwide

---

## 4.8 Summary

| Aspect                  | Word2Vec                    | FastText                              |
|-------------------------|-----------------------------|---------------------------------------|
| Word representation     | Single vector per word      | Sum of n-gram vectors                 |
| Scoring function        | $s(w,c) = \mathbf{u}_w^\top \mathbf{v}_c$ | $s(w,c) = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c$ |
| OOV handling            | None                        | Compose from n-grams                  |
| Morphology              | Ignored                     | Captured via subword sharing           |
| Memory                  | $O(V \cdot d)$              | $O((V + B) \cdot d)$                  |
| Training speed          | Fast                        | Slightly slower (more parameters)      |
| Classification          | Not built-in                | Hierarchical softmax classifier        |
| Pre-trained languages   | Limited                     | 157 languages                          |

FastText's enduring lesson: **sometimes the simplest ideas — breaking words into pieces
and summing their vectors — produce the most robust and practical systems.**

---

## References

1. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors
   with Subword Information. *Transactions of the Association for Computational
   Linguistics*, 5, 135–146.

2. Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of Tricks for
   Efficient Text Classification. *Proceedings of the 15th Conference of the European
   Chapter of the Association for Computational Linguistics*, 2, 427–431.

3. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed
   Representations of Words and Phrases and their Compositionality. *Advances in Neural
   Information Processing Systems*, 26.

4. Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009).
   Feature Hashing for Large Scale Multitask Learning. *Proceedings of the 26th
   International Conference on Machine Learning*, 1113–1120.

---

*Next chapter: [Chapter 5 — Contextual Embeddings: ELMo, BERT, and Beyond](05-contextual-embeddings.md)*
