# Chapter 12: The Frontier & Future — State-of-the-Art Embeddings and What Comes Next

> *"The best embedding model is the one that understands what you're asking for."*

The embedding landscape has undergone a tectonic shift since the days of static word
vectors. What began as fixed lookup tables (Word2Vec, GloVe) evolved through contextual
encoders (BERT, SBERT) and has now arrived at a new frontier: instruction-tuned,
multi-modal, multi-lingual, multi-granularity embedding models that rival or surpass
purpose-built retrieval systems. This chapter surveys the state of the art as of
2024–2025, dissects the architectures and training recipes of the leading models, and
looks ahead to where the field is going.

---

## 12.1 Introduction — The Modern Embedding Landscape

### 12.1.1 From Static Vectors to Instruction-Tuned Embeddings

The evolution of text embeddings can be compressed into five paradigm shifts:

| Era | Representative Model | Key Innovation |
|:---:|:---:|:---:|
| 2013–2015 | Word2Vec, GloVe, FastText | Static word vectors via shallow objectives |
| 2018–2019 | ELMo, BERT | Contextual embeddings from deep pre-training |
| 2019–2021 | SBERT, SimCSE | Sentence embeddings via contrastive fine-tuning |
| 2022–2023 | E5, BGE, Instructor | Instruction-tuned, multi-task embeddings |
| 2024–2025 | NV-Embed, Jina v3, BGE-M3 | Decoder-only LLMs, multi-modal, multi-granularity |

Each shift did not replace the previous one — it subsumed it. Modern models like BGE-M3
produce dense, sparse, *and* multi-vector representations simultaneously, in 100+
languages, with 8192-token context windows. The gap between "embedding model" and
"retrieval system" has nearly closed.

### 12.1.2 Key Trends Defining the Frontier

1. **Longer context**: The 512-token BERT ceiling has been shattered. Models now
   routinely handle 8192 tokens (Jina v2/v3, BGE-M3, Nomic Embed) and some push
   to 32K+ tokens.

2. **Instruction-tuning**: Rather than producing a single generic embedding, models
   accept a task description ("Retrieve a passage that answers this question") and
   tailor the representation accordingly.

3. **Decoder-only architectures**: GPT-style models (Mistral, LLaMA) are being
   repurposed as embedding encoders, leveraging their massive pre-training.

4. **Multi-granularity**: A single model produces embeddings at multiple levels —
   dense vectors, sparse lexical weights, and token-level multi-vectors — enabling
   hybrid retrieval without separate systems.

5. **Matryoshka representations**: Models are trained to produce embeddings that
   remain useful when truncated to smaller dimensions (see Chapter 10), enabling
   flexible storage-accuracy trade-offs.

6. **Open-source parity**: Open models (BGE, Nomic, Jina) now match or exceed
   proprietary APIs (OpenAI, Cohere) on standard benchmarks.

### 12.1.3 The MTEB Benchmark as the Standard Yardstick

The **Massive Text Embedding Benchmark** (MTEB; Muennighoff et al., 2023) has become
the de facto evaluation framework. We cover it in detail in Section 12.12, but its
existence has shaped model development: every model in this chapter reports MTEB scores,
and leaderboard position drives adoption. This is both a strength (standardized
comparison) and a weakness (Goodhart's Law — optimizing for the benchmark rather than
real-world utility).

---

## 12.2 E5 Embeddings (Microsoft, 2022–2024)

### 12.2.1 Overview and Motivation

**E5** — short for **E**mb**E**ddings from bid**E**r**E**ctional **E**ncoder
r**E**presentations — was introduced by Wang et al. (2022) at Microsoft. The key
insight was that existing embedding models suffered from a data bottleneck: supervised
training sets (NLI, MS MARCO) were too small and too narrow to produce truly general
embeddings.

E5's solution: **weakly-supervised contrastive pre-training** on a massive, diverse
dataset of text pairs scraped from the web, followed by supervised fine-tuning.

### 12.2.2 The CCPairs Dataset

E5 introduced **CCPairs** (Curated C4 Pairs), a dataset of ~1.3 billion text pairs
harvested from Common Crawl. The pairs were extracted using heuristics:

- **(title, passage)** pairs from web pages
- **(question, answer)** pairs from forums
- **(query, clicked-document)** pairs from search logs
- **(sentence_i, sentence_{i+1})** pairs from adjacent paragraphs

The key was *consistency filtering*: pairs were scored by a cross-encoder and low-quality
pairs were discarded. This gave E5 a training set orders of magnitude larger than
MS MARCO (~500K pairs) while maintaining reasonable quality.

### 12.2.3 Two-Stage Training Pipeline

E5 follows a two-stage recipe:

**Stage 1: Contrastive Pre-training on CCPairs**

The model (initialized from a pre-trained BERT or T5 encoder) is trained with
temperature-scaled InfoNCE loss on the CCPairs dataset:

$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{q}_i, \mathbf{p}_i^+) / \tau)}{\exp(\text{sim}(\mathbf{q}_i, \mathbf{p}_i^+) / \tau) + \sum_{j=1}^{K} \exp(\text{sim}(\mathbf{q}_i, \mathbf{p}_j^-) / \tau)}$

where:
- $\mathbf{q}_i$ is the query (or first element) embedding
- $\mathbf{p}_i^+$ is the positive passage embedding
- $\mathbf{p}_j^-$ are negative passage embeddings (in-batch negatives + hard negatives)
- $\tau$ is the temperature parameter (typically 0.01–0.05)
- $\text{sim}(\cdot, \cdot)$ is cosine similarity

The temperature $\tau$ controls the sharpness of the distribution. A small $\tau$
makes the loss focus on hard negatives (pairs with high but incorrect similarity),
while a large $\tau$ treats all negatives more uniformly:

$\lim_{\tau \to 0} \mathcal{L}_{\text{InfoNCE}} \to \text{hardest-negative loss}$
$\lim_{\tau \to \infty} \mathcal{L}_{\text{InfoNCE}} \to \text{uniform weighting}$

**Stage 2: Supervised Fine-tuning**

After pre-training, the model is fine-tuned on a blend of labeled datasets:
MS MARCO, NLI (SNLI + MultiNLI), Natural Questions, HotpotQA, and others. This
stage uses the same InfoNCE loss but with higher-quality, human-annotated pairs.

### 12.2.4 E5 Model Variants

| Model | Backbone | Params | Dim | MTEB Avg |
|:---:|:---:|:---:|:---:|:---:|
| E5-small | MiniLM | 33M | 384 | 59.9 |
| E5-base | BERT-base | 110M | 768 | 61.5 |
| E5-large | BERT-large | 335M | 1024 | 64.2 |
| E5-Mistral-7B | Mistral-7B | 7.1B | 4096 | 66.6 |

### 12.2.5 E5-Mistral: Instruction-Tuned Decoder-Only Embeddings

E5-Mistral (Wang et al., 2024) was a landmark: it showed that a **decoder-only** LLM
(Mistral-7B) could be turned into a state-of-the-art embedding model. The recipe:

1. Start with Mistral-7B-v0.1 (pre-trained decoder-only LLM)
2. Use the **last token** representation as the sentence embedding (with an EOS token
   appended)
3. Fine-tune with instruction-prefixed contrastive learning

The instruction format:

```
Instruct: {task_description}\nQuery: {input_text}
```

For example:

```
Instruct: Retrieve a Wikipedia passage that answers this question\nQuery: What is the capital of France?
```

This instruction-conditioning allows a single model to produce task-specific embeddings.

### 12.2.6 Code Example: Using E5 Embeddings

```python
from sentence_transformers import SentenceTransformer

# Load E5-large-v2
model = SentenceTransformer("intfloat/e5-large-v2")

# E5 requires prefixes: "query: " for queries, "passage: " for documents
queries = [
    "query: What is the capital of France?",
    "query: How does photosynthesis work?",
]
passages = [
    "passage: Paris is the capital and most populous city of France.",
    "passage: Photosynthesis converts light energy into chemical energy in plants.",
    "passage: The Eiffel Tower is a wrought-iron lattice tower in Paris.",
]

q_embeddings = model.encode(queries, normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)

# Compute similarity matrix
import numpy as np
similarities = q_embeddings @ p_embeddings.T
print("Similarity matrix:")
print(np.round(similarities, 3))
# Expected: query 0 most similar to passage 0, query 1 to passage 1
```

```python
# E5-Mistral with instruction-tuning
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")

# Instruction-prefixed queries
queries = [
    "Instruct: Retrieve a passage that answers this question\n"
    "Query: What causes tides on Earth?",
]
passages = [
    "Tides are caused primarily by the gravitational pull of the Moon "
    "and, to a lesser extent, the Sun on Earth's oceans.",
    "The Earth rotates on its axis once every 24 hours.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
p_emb = model.encode(passages, normalize_embeddings=True)

scores = q_emb @ p_emb.T
print(f"Tide passage score:     {scores[0][0]:.4f}")
print(f"Rotation passage score: {scores[0][1]:.4f}")
# The tide passage should score significantly higher
```

### 12.2.7 Historical Significance

E5 was the **first embedding model to outperform BM25 on the BEIR benchmark in
zero-shot settings** — a milestone that demonstrated neural embeddings could finally
beat classical lexical retrieval without task-specific fine-tuning.

---

## 12.3 BGE Embeddings (BAAI, 2023–2024)

### 12.3.1 Overview

**BGE** (BAAI General Embedding) is a family of embedding models developed by the
**Beijing Academy of Artificial Intelligence** (BAAI) as part of the **FlagEmbedding**
project. BGE rapidly became one of the most popular open-source embedding families,
with models ranging from 33M to 7B parameters.

### 12.3.2 Training Pipeline: Three Stages

BGE's training follows a three-stage pipeline, each building on the previous:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Stage 1: RetroMAE  │────▶│  Stage 2: Contrastive │────▶│  Stage 3: Instruction│
│  Pre-training       │     │  Fine-tuning          │     │  Tuning             │
│                     │     │                       │     │                     │
│  • Masked auto-     │     │  • InfoNCE loss       │     │  • Task-specific    │
│    encoder on       │     │  • In-batch negatives  │     │    instructions     │
│    unlabeled text   │     │  • Hard negatives from │     │  • Multi-task       │
│  • Reconstruct      │     │    cross-encoder       │     │    training         │
│    masked tokens    │     │  • MS MARCO + NLI +    │     │  • "Represent this  │
│                     │     │    curated pairs       │     │    sentence for     │
│                     │     │                       │     │    retrieval: ..."   │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

**Stage 1: RetroMAE Pre-training**

RetroMAE (Xiao et al., 2022) is a pre-training method specifically designed for dense
retrieval. It uses an asymmetric encoder-decoder architecture:

- The **encoder** sees a lightly masked input (15% mask ratio) and produces a
  sentence embedding
- The **decoder** receives a heavily masked version (50–70% mask ratio) plus the
  encoder's sentence embedding, and must reconstruct the original text

This forces the encoder to pack maximum information into the sentence embedding,
producing representations that are better suited for retrieval than standard MLM
pre-training.

**Stage 2: Contrastive Fine-tuning**

Standard contrastive learning with InfoNCE loss, using:
- In-batch negatives (other passages in the same mini-batch)
- Hard negatives mined by a cross-encoder reranker
- Temperature $\tau = 0.02$

**Stage 3: Instruction Tuning (BGE v1.5+)**

The model is further fine-tuned with task-specific instruction prefixes, similar to
E5-Mistral but applied to encoder-only models.

### 12.3.3 BGE v1 → v1.5: Fixing the Similarity Distribution

A known issue with BGE v1 was that cosine similarity scores were clustered in a narrow
range (e.g., 0.7–0.9 for all pairs), making it hard to distinguish relevant from
irrelevant results. BGE v1.5 addressed this by:

1. Adding more diverse negative examples during training
2. Using a wider temperature range
3. Incorporating explicit hard negative mining with cross-encoder scores

The result: a more spread-out similarity distribution where relevant pairs score
significantly higher than irrelevant ones.

### 12.3.4 BGE-M3: Multi-Linguality, Multi-Functionality, Multi-Granularity

BGE-M3 (Chen et al., 2024) is the crown jewel of the BGE family. The "M3" stands for
three "multi" capabilities:

**Multi-Linguality**: Supports 100+ languages, trained on multilingual data from
OPUS, CC-100, and Wikipedia.

**Multi-Functionality**: A single forward pass produces three types of representations:

1. **Dense embedding**: The [CLS] token representation, $\mathbf{e}_d \in \mathbb{R}^{1024}$
2. **Sparse embedding**: Per-token lexical weights (like SPLADE), $\mathbf{e}_s \in \mathbb{R}^{|V|}$
3. **Multi-vector embedding**: All token representations, $\mathbf{E}_m \in \mathbb{R}^{L \times 1024}$

**Multi-Granularity**: Supports inputs from short queries to 8192-token documents.

### 12.3.5 BGE-M3 Unified Retrieval Score

The three retrieval scores are combined into a unified score:

$s_{\text{dense}}(q, d) = \frac{\mathbf{e}_d^q \cdot \mathbf{e}_d^d}{\|\mathbf{e}_d^q\| \|\mathbf{e}_d^d\|}$

$s_{\text{sparse}}(q, d) = \sum_{t \in q \cap d} w_q(t) \cdot w_d(t)$

where $w_q(t)$ and $w_d(t)$ are the learned lexical weights for token $t$.

$s_{\text{multi}}(q, d) = \frac{1}{|q|} \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \frac{\mathbf{e}_{m,i}^q \cdot \mathbf{e}_{m,j}^d}{\|\mathbf{e}_{m,i}^q\| \|\mathbf{e}_{m,j}^d\|}$

This last score is the **MaxSim** operator from ColBERT (Khattab & Zaharia, 2020):
for each query token, find the most similar document token, then average.

The final unified score is a weighted combination:

$s_{\text{unified}}(q, d) = \alpha \cdot s_{\text{dense}}(q, d) + \beta \cdot s_{\text{sparse}}(q, d) + \gamma \cdot s_{\text{multi}}(q, d)$

where $\alpha + \beta + \gamma = 1$. Typical values: $\alpha = 0.4, \beta = 0.2, \gamma = 0.4$.

### 12.3.6 Code Example: BGE-M3 Multi-Granularity Retrieval

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

queries = ["What is the capital of France?"]
passages = [
    "Paris is the capital and largest city of France.",
    "Berlin is the capital of Germany.",
    "France is a country in Western Europe.",
]

# Encode with all three representations
q_output = model.encode(
    queries,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
p_output = model.encode(
    passages,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)

# Dense retrieval score
dense_scores = q_output["dense_vecs"] @ p_output["dense_vecs"].T
print("Dense scores:", dense_scores)

# Sparse retrieval score (lexical matching)
sparse_scores = model.compute_lexical_matching_score(
    q_output["lexical_weights"][0], p_output["lexical_weights"][0]
)
print("Sparse score (q0, p0):", sparse_scores)

# Multi-vector (ColBERT) score
colbert_scores = model.colbert_score(
    q_output["colbert_vecs"][0], p_output["colbert_vecs"][0]
)
print("ColBERT score (q0, p0):", colbert_scores)

# Unified hybrid score
alpha, beta, gamma = 0.4, 0.2, 0.4
unified = alpha * dense_scores[0][0] + beta * sparse_scores + gamma * colbert_scores
print(f"Unified score (q0, p0): {unified:.4f}")
```

---

## 12.4 Nomic Embed (Nomic AI, 2024)

### 12.4.1 The Open-Source Imperative

Nomic Embed, introduced by Nussbaum et al. (2024), distinguished itself not primarily
through architecture but through **radical openness**. It was the first embedding model
to be simultaneously:

- **Open-weights**: Model weights released under Apache 2.0
- **Open-data**: Full training dataset published
- **Open-source**: Complete training code available
- **Fully reproducible**: Anyone can retrain the model from scratch

This was a deliberate response to the "open-weight but closed-everything-else" trend
in the industry, where models like OpenAI's `text-embedding-ada-002` were proprietary
black boxes and even "open" models like E5 did not release their training data.

### 12.4.2 Architecture

Nomic Embed is based on a modified BERT architecture with several key changes:

1. **Rotary Position Embeddings (RoPE)** instead of absolute position embeddings,
   enabling extrapolation to longer sequences
2. **8192 token context length** (vs. BERT's 512)
3. **Flash Attention** for efficient long-sequence processing
4. **SwiGLU activation** in the feed-forward layers (borrowed from LLaMA)

The model has ~137M parameters with a 768-dimensional embedding space.

### 12.4.3 Training Recipe

Nomic Embed's training follows a multi-stage contrastive learning pipeline on a
curated dataset of ~235M text pairs:

1. **Unsupervised contrastive pre-training**: On weakly-supervised pairs from web data
2. **Supervised contrastive fine-tuning**: On high-quality labeled datasets
3. **Matryoshka training** (v1.5): Additional training with the Matryoshka loss to
   support flexible dimensionality

The contrastive objective is the standard InfoNCE with in-batch negatives and hard
negatives mined from a teacher model.

### 12.4.4 Nomic Embed v1.5: Matryoshka Support

Nomic Embed v1.5 added **Matryoshka Representation Learning** (see Chapter 10),
allowing the 768-dimensional embedding to be truncated to smaller dimensions (512,
256, 128, 64) with graceful degradation:

| Dimension | MTEB Avg | Relative to 768d |
|:---------:|:--------:|:-----------------:|
| 768 | 62.28 | 100% |
| 512 | 61.96 | 99.5% |
| 256 | 61.54 | 98.8% |
| 128 | 60.41 | 97.0% |
| 64 | 58.15 | 93.4% |

This means you can use 64-dimensional embeddings (12× compression) and retain 93.4%
of the full performance — a massive win for storage and latency.

### 12.4.5 Performance

Despite its modest size (137M params), Nomic Embed v1.5 outperforms:
- OpenAI `text-embedding-ada-002` (unknown params, 1536d)
- OpenAI `text-embedding-3-small` (unknown params, 1536d)
- Several models 2–3× its size

This demonstrates that training data quality and training recipe matter more than
raw parameter count for embedding models.

### 12.4.6 Code Example: Nomic Embed

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Nomic uses "search_query: " and "search_document: " prefixes
queries = [
    "search_query: What are the health benefits of green tea?",
    "search_query: How to train a neural network",
]
documents = [
    "search_document: Green tea contains antioxidants called catechins "
    "that may reduce inflammation and lower the risk of heart disease.",
    "search_document: Neural networks are trained using backpropagation, "
    "which computes gradients of the loss function with respect to weights.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(documents, normalize_embeddings=True)

# Full 768d similarity
scores_768 = q_emb @ d_emb.T
print("768d scores:\n", scores_768.round(3))

# Matryoshka: truncate to 128d and re-normalize
import numpy as np

q_128 = q_emb[:, :128]
d_128 = d_emb[:, :128]
q_128 = q_128 / np.linalg.norm(q_128, axis=1, keepdims=True)
d_128 = d_128 / np.linalg.norm(d_128, axis=1, keepdims=True)

scores_128 = q_128 @ d_128.T
print("128d scores:\n", scores_128.round(3))
# Scores should be very similar to 768d
```

---

## 12.5 Jina Embeddings (Jina AI, 2023–2024)

### 12.5.1 Jina v2: Long-Context BERT

Jina Embeddings v2 (Günther et al., 2023) introduced **JinaBERT**, a BERT variant
with ALiBi (Attention with Linear Biases) position encodings that natively supports
**8192 tokens** — 16× BERT's limit.

Key innovations:
- ALiBi replaces absolute position embeddings with a linear bias added to attention
  scores, enabling length generalization
- Trained on a curated dataset of text pairs with mean pooling
- 137M parameters, 768-dimensional embeddings

### 12.5.2 Jina v3: Task-Specific LoRA Adapters

Jina Embeddings v3 (Sturua et al., 2024) represents a significant architectural
departure. Instead of a single set of weights, Jina v3 uses **task-specific LoRA
adapters** that modify the base model's behavior depending on the task.

**Architecture**: 570M parameter base model (XLM-RoBERTa backbone) with five
task-specific LoRA adapter sets:

| Adapter | Task | Use Case |
|:---:|:---:|:---:|
| `retrieval.query` | Asymmetric retrieval (query side) | Search queries |
| `retrieval.passage` | Asymmetric retrieval (doc side) | Documents to search |
| `separation` | Clustering / classification | Grouping similar texts |
| `classification` | Text classification | Sentiment, topic labels |
| `text-matching` | Symmetric similarity | STS, paraphrase detection |

### 12.5.3 LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2022) modifies a pre-trained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$
by adding a low-rank update:

$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \Delta \mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \mathbf{B} \mathbf{A} \mathbf{x}$

where:
- $\mathbf{A} \in \mathbb{R}^{r \times k}$ is the down-projection (rank $r \ll \min(d, k)$)
- $\mathbf{B} \in \mathbb{R}^{d \times r}$ is the up-projection
- $r$ is the LoRA rank (typically 4–64)

The number of trainable parameters per adapted layer is $r \times (d + k)$ instead of
$d \times k$. For $d = k = 1024$ and $r = 16$:

$\text{Full fine-tuning}: 1024 \times 1024 = 1{,}048{,}576 \text{ params}$
$\text{LoRA}: 16 \times (1024 + 1024) = 32{,}768 \text{ params} \quad (3.1\% \text{ of full})$

In Jina v3, each task adapter is a separate set of $(\mathbf{A}, \mathbf{B})$ matrices.
At inference time, the user selects which adapter to activate:

```
Base model weights: W₀  (frozen, shared across all tasks)
                    │
        ┌───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ B_r A_r │ │ B_c A_c │ │ B_s A_s │ │ B_m A_m │
   │retrieval│ │classif. │ │ separ.  │ │matching │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘

Output: h = W₀x + B_task · A_task · x
```

### 12.5.4 Matryoshka Dimension Reduction in Jina v3

Jina v3 supports Matryoshka-style truncation from 1024 dimensions down to 32:

| Dimension | Retrieval (NDCG@10) | Relative |
|:---------:|:-------------------:|:--------:|
| 1024 | 0.601 | 100% |
| 512 | 0.596 | 99.2% |
| 256 | 0.588 | 97.8% |
| 128 | 0.571 | 95.0% |
| 64 | 0.548 | 91.2% |
| 32 | 0.512 | 85.2% |

### 12.5.5 Code Example: Jina v3 with Task Adapters

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Retrieval task: asymmetric query-document matching
query = "What is quantum computing?"
documents = [
    "Quantum computing uses quantum bits (qubits) that can exist in "
    "superposition, enabling parallel computation.",
    "Classical computers use binary bits that are either 0 or 1.",
    "The weather forecast predicts rain tomorrow.",
]

# Encode query with retrieval.query adapter
q_emb = model.encode(
    [query],
    task="retrieval.query",
    normalize_embeddings=True,
)

# Encode documents with retrieval.passage adapter
d_emb = model.encode(
    documents,
    task="retrieval.passage",
    normalize_embeddings=True,
)

scores = q_emb @ d_emb.T
for i, (doc, score) in enumerate(zip(documents, scores[0])):
    print(f"  Doc {i}: {score:.4f} — {doc[:60]}...")

# Switch to classification adapter for a different task
texts = ["I love this product!", "Terrible experience.", "It was okay."]
cls_emb = model.encode(texts, task="classification", normalize_embeddings=True)
print("\nClassification embeddings shape:", cls_emb.shape)
```

---

## 12.6 Instruction-Tuned Embeddings

### 12.6.1 The Paradigm Shift: Embeddings That Understand Task Context

Traditional embedding models produce a single, fixed representation for any input text.
But the "best" embedding for a sentence depends on what you plan to *do* with it:

- For **retrieval**, you want the embedding to capture the information need
- For **classification**, you want it to emphasize category-discriminative features
- For **clustering**, you want it to capture topical similarity
- For **STS**, you want fine-grained semantic similarity

Instruction-tuned embeddings solve this by accepting a natural-language task description
alongside the input text, producing representations tailored to the downstream task.

### 12.6.2 INSTRUCTOR: "One Embedder, Any Task" (Su et al., 2022)

INSTRUCTOR was the first model to systematically explore instruction-tuned embeddings.
The key idea is simple but powerful: prepend a task-specific instruction to the input.

**Input format:**

```
Represent the {domain} {task_type} for {objective}: {input_text}
```

Examples:

```
Represent the Wikipedia document for retrieval: Paris is the capital of France...
Represent the financial query for classification: Is this stock a good buy?
Represent the scientific sentence for clustering: Mitochondria generate ATP...
```

**Architecture**: INSTRUCTOR uses a GTR (Generalizable T5-based dense Retriever)
backbone — a T5 encoder fine-tuned for embeddings. The instruction is simply
concatenated with the input and processed by the same encoder.

**Training**: INSTRUCTOR was trained on 330 diverse tasks spanning:
- Retrieval (MS MARCO, Natural Questions, HotpotQA)
- Classification (SST-2, AG News, DBpedia)
- Clustering (Reddit, StackExchange, ArXiv)
- STS (STS Benchmark, SICK)
- Pair classification (QQP, MRPC)

Each training example includes its task instruction, teaching the model to condition
its representations on the task description.

### 12.6.3 How Instruction-Tuning Changes the Embedding Space

Instruction-tuning creates a **task-conditioned** embedding space. Formally, let
$f_\theta$ be the encoder and $\mathbf{i}$ be the instruction. The embedding is:

$\mathbf{e} = f_\theta([\mathbf{i}; \mathbf{x}])$

where $[\cdot; \cdot]$ denotes concatenation. The same input $\mathbf{x}$ produces
different embeddings under different instructions:

$f_\theta([\mathbf{i}_{\text{retrieval}}; \mathbf{x}]) \neq f_\theta([\mathbf{i}_{\text{classification}}; \mathbf{x}])$

This is analogous to how attention mechanisms allow the same token to have different
representations depending on context — but at the task level rather than the token level.

### 12.6.4 Code Example: INSTRUCTOR

```python
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-large")

# Same text, different instructions → different embeddings
text = "The mitochondria is the powerhouse of the cell."

# For retrieval
retrieval_emb = model.encode(
    [["Represent the Biology sentence for retrieval:", text]]
)

# For classification
classification_emb = model.encode(
    [["Represent the Biology sentence for classification:", text]]
)

# For clustering
clustering_emb = model.encode(
    [["Represent the Biology sentence for clustering:", text]]
)

import numpy as np
from numpy.linalg import norm

def cosine(a, b):
    return (a @ b.T) / (norm(a) * norm(b))

# These will be different embeddings for the same text
print(f"Retrieval vs Classification: {cosine(retrieval_emb, classification_emb)[0][0]:.4f}")
print(f"Retrieval vs Clustering:     {cosine(retrieval_emb, clustering_emb)[0][0]:.4f}")
print(f"Classification vs Clustering:{cosine(classification_emb, clustering_emb)[0][0]:.4f}")
# Typically ~0.85-0.95 — similar but not identical
```

---

## 12.7 Decoder-Only LLMs as Embedding Models

### 12.7.1 The Architectural Shift

For years, the embedding world was dominated by **encoder-only** models (BERT, RoBERTa)
and occasionally **encoder-decoder** models (T5). The reasoning was straightforward:
encoders produce bidirectional representations where each token attends to all others,
which seems ideal for creating holistic sentence embeddings.

**Decoder-only** models (GPT, LLaMA, Mistral) use **causal attention** — each token
can only attend to previous tokens. This seems fundamentally limiting for embeddings:
the last token can see everything, but the first token sees only itself.

Yet starting in 2023–2024, decoder-only models began dominating embedding benchmarks.
Why?

### 12.7.2 Why Decoder-Only Works

Three factors explain the success:

1. **Scale of pre-training**: Decoder-only LLMs are pre-trained on trillions of tokens
   (vs. billions for BERT-scale models). This massive pre-training produces rich
   internal representations that transfer well to embedding tasks.

2. **Last-token pooling**: By using the representation of the **last token** (or an
   appended [EOS] token), we get a vector that has attended to the entire input
   sequence through the causal attention layers. The last token is effectively a
   "summary" position.

3. **Instruction-following**: Decoder-only LLMs are already trained to follow
   instructions, making instruction-tuned embeddings a natural extension.

### 12.7.3 The Causal Attention Problem and Solutions

The causal attention mask is the main architectural challenge:

```
Bidirectional (BERT):          Causal (GPT):
┌─────────────────┐            ┌─────────────────┐
│ 1  1  1  1  1   │            │ 1  0  0  0  0   │
│ 1  1  1  1  1   │            │ 1  1  0  0  0   │
│ 1  1  1  1  1   │            │ 1  1  1  0  0   │
│ 1  1  1  1  1   │            │ 1  1  1  1  0   │
│ 1  1  1  1  1   │            │ 1  1  1  1  1   │
└─────────────────┘            └─────────────────┘
  Every token sees               Token i only sees
  every other token              tokens 1..i
```

Solutions adopted by different models:

| Model | Solution |
|:---:|:---:|
| E5-Mistral | Last-token pooling with [EOS] appended |
| NV-Embed | Remove causal mask during embedding (bidirectional) |
| LLM2Vec | Remove causal mask + add bidirectional training |
| GritLM | Separate embedding and generation modes |

### 12.7.4 NV-Embed (NVIDIA, 2024)

NV-Embed (Lee et al., 2024) achieved #1 on the MTEB leaderboard by combining several
innovations on top of a Mistral-7B backbone:

**Innovation 1: Latent Attention Pooling**

Instead of mean pooling or last-token pooling, NV-Embed introduces a set of learnable
"latent" query vectors that attend to all token representations:

$\mathbf{e} = \text{LatentAttention}(\mathbf{Q}_{\text{latent}}, \mathbf{H})$

where $\mathbf{Q}_{\text{latent}} \in \mathbb{R}^{L \times d}$ are $L$ learnable query
vectors and $\mathbf{H} \in \mathbb{R}^{n \times d}$ are the token hidden states. The
attention output is then mean-pooled across the $L$ latent vectors to produce the final
embedding.

Formally:

$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{latent}} \mathbf{H}^T}{\sqrt{d}}\right) \in \mathbb{R}^{L \times n}$

$\mathbf{O} = \mathbf{A} \mathbf{H} \in \mathbb{R}^{L \times d}$

$\mathbf{e} = \frac{1}{L} \sum_{i=1}^{L} \mathbf{O}_i \in \mathbb{R}^{d}$

This is similar to the Perceiver's cross-attention mechanism and allows the model to
learn *what* to extract from the sequence, rather than relying on a fixed pooling strategy.

**Innovation 2: Bidirectional Attention During Embedding**

NV-Embed removes the causal attention mask during the contrastive training phase,
allowing full bidirectional attention. This is possible because embedding does not
require autoregressive generation — we only need a single forward pass to produce
the representation.

**Innovation 3: Two-Stage Training**

1. **Stage 1**: Contrastive instruction-tuning on retrieval datasets with in-batch
   negatives and hard negatives
2. **Stage 2**: Blended fine-tuning on non-retrieval tasks (classification, clustering,
   STS) to improve generalization without degrading retrieval performance

### 12.7.5 Code Example: Using a Decoder-Only Embedding Model

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Example with a decoder-only embedding model
model_name = "nvidia/NV-Embed-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Instruction-prefixed input
instruction = "Instruct: Retrieve relevant passages for the query\nQuery: "
queries = [instruction + "What is dark matter?"]
passages = [
    "Dark matter is a hypothetical form of matter that does not interact "
    "with the electromagnetic force but would still have gravitational effects.",
    "The Milky Way is a barred spiral galaxy with a diameter of 100,000 light-years.",
]

# Encode (model handles pooling internally)
q_inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
p_inputs = tokenizer(passages, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    q_emb = model(**q_inputs).last_hidden_state[:, -1, :]  # Last token
    p_emb = model(**p_inputs).last_hidden_state[:, -1, :]

# Normalize and compute similarity
q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
p_emb = torch.nn.functional.normalize(p_emb, dim=-1)

scores = q_emb @ p_emb.T
print("Scores:", scores)
```

---

## 12.8 Multi-Modal Embeddings

### 12.8.1 Beyond Text: Shared Embedding Spaces Across Modalities

The embedding idea generalizes naturally beyond text. If we can map words to vectors
such that similar words are nearby, why not map images, audio, and video into the
*same* vector space? A picture of a cat and the text "a fluffy cat" should have
similar embeddings.

This is the core idea behind **multi-modal embeddings**: learn a shared vector space
where items from different modalities can be directly compared via cosine similarity.

### 12.8.2 CLIP (OpenAI, 2021): Contrastive Language-Image Pre-training

CLIP (Radford et al., 2021) is the foundational multi-modal embedding model. It
learns a joint image-text embedding space using contrastive learning on 400M
image-text pairs scraped from the internet.

**Architecture:**

```
     Image                          Text
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ Vision       │              │ Text         │
│ Encoder      │              │ Encoder      │
│ (ViT or      │              │ (Transformer)│
│  ResNet)     │              │              │
└──────┬───────┘              └──────┬───────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ Linear       │              │ Linear       │
│ Projection   │              │ Projection   │
│ → d dims     │              │ → d dims     │
└──────┬───────┘              └──────┬───────┘
       │                              │
       ▼                              ▼
   v_image ∈ ℝ^d              v_text ∈ ℝ^d
       │                              │
       └──────────┬───────────────────┘
                  │
            cosine similarity
            sim(v_image, v_text)
```

**Training Objective: Symmetric Contrastive Loss**

Given a batch of $N$ image-text pairs $\{(\mathbf{v}_i, \mathbf{t}_i)\}_{i=1}^N$,
CLIP minimizes a symmetric contrastive loss:

$\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j) / \tau)}$

$\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_j) / \tau)}$

$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$

The first loss treats each image as a "query" and finds its matching text among $N$
candidates. The second loss does the reverse. Together, they align the two modalities
in a shared space.

**The temperature parameter** $\tau$ is learned (initialized to $\tau = 0.07$) and
controls the sharpness of the softmax distribution. CLIP found that learning $\tau$
was critical — a fixed temperature performed significantly worse.

### 12.8.3 Numerical Example: CLIP Contrastive Loss

Consider a mini-batch of $N = 3$ image-text pairs. After encoding and projecting,
suppose the cosine similarity matrix is:

```
              text_0   text_1   text_2
image_0  [    0.9      0.2      0.1   ]
image_1  [    0.3      0.8      0.2   ]
image_2  [    0.1      0.3      0.85  ]
```

With $\tau = 0.1$:

**Image-to-text loss for image_0:**

$\mathcal{L}_0^{\text{img}} = -\log \frac{\exp(0.9 / 0.1)}{\exp(0.9/0.1) + \exp(0.2/0.1) + \exp(0.1/0.1)}$

$= -\log \frac{\exp(9)}{\exp(9) + \exp(2) + \exp(1)}$

$= -\log \frac{8103.1}{8103.1 + 7.389 + 2.718}$

$= -\log \frac{8103.1}{8113.2} = -\log(0.99876) = 0.00124$

The loss is very small because image_0 is strongly aligned with text_0 (similarity 0.9)
and weakly aligned with the others.

**Image-to-text loss for image_1:**

$\mathcal{L}_1^{\text{img}} = -\log \frac{\exp(8)}{\exp(3) + \exp(8) + \exp(2)}$

$= -\log \frac{2981.0}{20.09 + 2981.0 + 7.389} = -\log \frac{2981.0}{3008.5} = -\log(0.9909) = 0.00914$

**Average image-to-text loss:**

$\mathcal{L}_{\text{image}} = \frac{1}{3}(0.00124 + 0.00914 + \mathcal{L}_2^{\text{img}})$

The text-to-image loss is computed symmetrically using the columns of the similarity
matrix.

### 12.8.4 ALIGN (Google, 2021) and Scaling Laws

ALIGN (Jia et al., 2021) showed that CLIP's approach scales with data: by training on
**1.8 billion** noisy image-text pairs (vs. CLIP's 400M curated pairs), ALIGN achieved
comparable or better performance with less data curation. The key insight: at sufficient
scale, noise in the training data averages out.

### 12.8.5 ImageBind (Meta, 2023): Six Modalities, One Space

ImageBind (Girdhar et al., 2023) extended the multi-modal embedding idea to **six
modalities**: images, text, audio, depth, thermal, and IMU (inertial measurement unit)
data. The key insight was that you don't need paired data for all modality combinations
— you can use images as a "binding" modality:

```
                    Image
                   /  |  \
                  /   |   \
               Text  Audio  Depth
                      |
                    Thermal
                      |
                     IMU
```

By training image-text, image-audio, image-depth, etc. pairs separately, all modalities
end up aligned in a shared space through the transitive property of the image "hub."
This means you can compute text-audio similarity even though the model was never trained
on text-audio pairs directly.

### 12.8.6 Code Example: CLIP Embeddings

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/220px-Cat_November_2010-1a.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Candidate text descriptions
texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car",
    "a landscape photograph",
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    image_emb = outputs.image_embeds  # (1, 512)
    text_emb = outputs.text_embeds    # (4, 512)

# Normalize
image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# Compute similarities
similarities = (image_emb @ text_emb.T).squeeze()
for text, sim in zip(texts, similarities):
    print(f"  {sim:.4f}  {text}")
# "a photo of a cat" should score highest
```

---

## 12.9 Long-Context Embeddings

### 12.9.1 The 512-Token Wall

BERT's absolute position embeddings are learned vectors for positions 0–511. At
position 512, there is simply no embedding — the model cannot process longer inputs.
This was acceptable in 2018 when most NLP tasks involved short texts, but modern
applications demand much more:

| Application | Typical Length |
|:---:|:---:|
| Search queries | 5–20 tokens |
| Tweets / short texts | 20–50 tokens |
| Paragraphs | 50–200 tokens |
| Full documents | 500–5000 tokens |
| Legal contracts | 5000–50000 tokens |
| Books / codebases | 50000+ tokens |

### 12.9.2 Position Encoding Solutions

Three main approaches have enabled long-context embeddings:

**ALiBi (Attention with Linear Biases; Press et al., 2022)**

ALiBi replaces learned position embeddings with a simple linear bias added to
attention scores:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + m \cdot [-(i-j)]\right) V$

where $m$ is a head-specific slope and $[-(i-j)]$ is a matrix of relative position
offsets. The bias penalizes attention to distant tokens linearly, with different
heads using different slopes (geometric sequence from $2^{-1}$ to $2^{-8/H}$ for
$H$ heads).

ALiBi requires no learned position parameters and generalizes to sequences longer
than those seen during training.

**RoPE (Rotary Position Embeddings; Su et al., 2021)**

RoPE encodes position by rotating the query and key vectors in 2D subspaces:

$\text{RoPE}(\mathbf{x}_m, m) = \begin{pmatrix} x_1 \cos(m\theta_1) - x_2 \sin(m\theta_1) \\ x_1 \sin(m\theta_1) + x_2 \cos(m\theta_1) \\ x_3 \cos(m\theta_2) - x_4 \sin(m\theta_2) \\ x_3 \sin(m\theta_2) + x_4 \cos(m\theta_2) \\ \vdots \end{pmatrix}$

where $\theta_i = 10000^{-2i/d}$ and $m$ is the position index. The key property is
that the dot product between rotated queries and keys depends only on the *relative*
position:

$\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$

RoPE is used by LLaMA, Mistral, and Nomic Embed.

**Extended Position Embeddings**

Some models simply extend BERT's position embeddings by interpolating or extrapolating
the learned embeddings to longer sequences (e.g., position interpolation in LongLoRA).

### 12.9.3 Late Chunking

**Late chunking** (Günther et al., 2024) is an inference-time technique for long
documents that avoids the information loss of traditional chunking:

```
Traditional chunking:
  Document → [chunk_1] [chunk_2] [chunk_3] → encode each independently
  Problem: chunk boundaries break context

Late chunking:
  Document → encode FULL document (up to 8192 tokens) → THEN chunk the embeddings
  Each chunk's embedding retains context from the full document
```

Formally, given a document with token representations $\mathbf{H} = [\mathbf{h}_1, \ldots, \mathbf{h}_n]$
from a long-context encoder, late chunking splits $\mathbf{H}$ into spans and
mean-pools each span:

$\mathbf{e}_{\text{chunk}_k} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{h}_i$

where $S_k$ is the set of token indices in chunk $k$. Because each $\mathbf{h}_i$
was computed with attention to the full document, the chunk embeddings carry
cross-chunk context.

### 12.9.4 ColBERT-Style Multi-Vector Representations

An alternative to single-vector embeddings for long documents is to keep **all token
embeddings** and use a late-interaction scoring mechanism:

$s(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i^T \mathbf{d}_j$

This is the **MaxSim** operator from ColBERT (Khattab & Zaharia, 2020). Each query
token finds its best-matching document token, and the scores are summed. This
preserves fine-grained token-level information that is lost in single-vector pooling.

The trade-off is storage: a 1000-token document requires storing 1000 vectors instead
of 1, increasing storage by ~1000×. Compression techniques (e.g., ColBERTv2's residual
compression) reduce this to ~20–50× overhead.

---

## 12.10 Sparse-Dense Hybrid Retrieval

### 12.10.1 The Complementary Strengths

Dense and sparse retrieval have complementary failure modes:

| Aspect | Dense (Neural) | Sparse (BM25/SPLADE) |
|:---:|:---:|:---:|
| Matching type | Semantic | Lexical |
| Handles synonyms | ✓ Well | ✗ Poorly |
| Handles rare terms | ✗ Poorly | ✓ Well |
| Handles typos | ✗ Poorly | ✓ Partially (exact match) |
| Handles entity names | ✗ Often fails | ✓ Exact match |
| Zero-shot transfer | ✓ Good | ✓ Good (BM25 is unsupervised) |
| Requires training | Yes | No (BM25) / Yes (SPLADE) |

A query like "CRISPR-Cas9 gene editing mechanism" benefits from:
- **Dense retrieval**: Understanding that "mechanism" relates to "how it works"
- **Sparse retrieval**: Exact matching on "CRISPR-Cas9" (a specific entity)

### 12.10.2 Hybrid Scoring

The simplest hybrid approach linearly combines dense and sparse scores:

$s_{\text{hybrid}}(q, d) = \alpha \cdot s_{\text{dense}}(q, d) + (1 - \alpha) \cdot s_{\text{sparse}}(q, d)$

where $\alpha \in [0, 1]$ controls the balance. Typical values of $\alpha$ range from
0.3 to 0.7, with the optimal value depending on the dataset.

**Score normalization** is critical because dense and sparse scores are on different
scales. Common approaches:

1. **Min-max normalization**: Scale each score type to $[0, 1]$
2. **Z-score normalization**: Standardize to zero mean, unit variance
3. **Rank-based fusion** (RRF): Combine rankings rather than scores

**Reciprocal Rank Fusion (RRF):**

$s_{\text{RRF}}(q, d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}$

where $\mathcal{R}$ is the set of retrieval systems, $\text{rank}_r(d)$ is the rank
of document $d$ in system $r$, and $k$ is a constant (typically 60). RRF is robust
because it depends only on rank order, not raw scores.

### 12.10.3 SPLADE: Learned Sparse Representations

SPLADE (Formal et al., 2021) learns sparse representations where each dimension
corresponds to a vocabulary token, and the weight indicates the token's importance:

$w_j = \log(1 + \text{ReLU}(\mathbf{W}_j^T \mathbf{h}_{\text{[CLS]}}))$

where $\mathbf{h}_{\text{[CLS]}}$ is the [CLS] token representation and $\mathbf{W}_j$
is the MLM head weight for token $j$. The $\log(1 + \text{ReLU}(\cdot))$ activation
ensures non-negative, sparse weights.

SPLADE representations can be indexed using inverted indices (like BM25), enabling
efficient retrieval with the same infrastructure as traditional search engines.

### 12.10.4 BGE-M3: Unified Hybrid in a Single Model

As discussed in Section 12.3, BGE-M3 produces dense, sparse, and multi-vector
representations in a single forward pass. This eliminates the need for separate
models and enables end-to-end training of the hybrid scoring function.

The training loss for BGE-M3 combines all three retrieval objectives:

$\mathcal{L} = \mathcal{L}_{\text{dense}} + \mathcal{L}_{\text{sparse}} + \mathcal{L}_{\text{multi-vec}}$

Each component uses the InfoNCE loss with its respective scoring function. The
gradients flow through all three representation types simultaneously, encouraging
them to be complementary rather than redundant.

### 12.10.5 Code Example: Hybrid Retrieval with Score Fusion

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Corpus
documents = [
    "CRISPR-Cas9 is a molecular tool for editing DNA sequences precisely.",
    "Gene therapy involves modifying genes to treat or prevent disease.",
    "The CRISPR mechanism uses guide RNA to target specific DNA locations.",
    "Machine learning models can predict protein structures from sequences.",
]

query = "How does CRISPR-Cas9 work?"

# --- Sparse retrieval (BM25) ---
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
sparse_scores = bm25.get_scores(query.lower().split())

# --- Dense retrieval ---
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
q_emb = model.encode([query], normalize_embeddings=True)
d_emb = model.encode(documents, normalize_embeddings=True)
dense_scores = (q_emb @ d_emb.T).flatten()

# --- Normalize scores to [0, 1] ---
def min_max_norm(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

sparse_norm = min_max_norm(sparse_scores)
dense_norm = min_max_norm(dense_scores)

# --- Hybrid fusion ---
alpha = 0.5
hybrid_scores = alpha * dense_norm + (1 - alpha) * sparse_norm

print("Document Rankings:")
print(f"{'Doc':>4} {'Dense':>8} {'Sparse':>8} {'Hybrid':>8}  Text")
print("-" * 72)
for i in np.argsort(-hybrid_scores):
    print(f"{i:>4} {dense_norm[i]:>8.3f} {sparse_norm[i]:>8.3f} "
          f"{hybrid_scores[i]:>8.3f}  {documents[i][:50]}...")
```

---

## 12.11 Embedding Quantization and Efficiency

### 12.11.1 The Storage Problem at Scale

At production scale, embedding storage becomes a dominant cost:

| Embeddings | Dimensions | Precision | Storage |
|:---:|:---:|:---:|:---:|
| 1M | 1024 | float32 | 4.0 GB |
| 10M | 1024 | float32 | 40.0 GB |
| 100M | 1024 | float32 | 400.0 GB |
| 1B | 1024 | float32 | 4.0 TB |

For billion-scale applications (web search, e-commerce), this is prohibitive. Quantization
reduces storage by representing each dimension with fewer bits.

### 12.11.2 Scalar Quantization (int8)

The simplest approach: map each float32 value to an 8-bit integer.

$q(x) = \text{round}\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \times 255\right)$

This gives **4× compression** (32 bits → 8 bits) with minimal quality loss (typically
<1% degradation on retrieval benchmarks).

**Dequantization** (approximate reconstruction):

$\hat{x} = q(x) \times \frac{x_{\max} - x_{\min}}{255} + x_{\min}$

### 12.11.3 Binary Quantization

The most aggressive approach: each dimension is represented by a single bit.

$b(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$

This gives **32× compression** (32 bits → 1 bit). Similarity is computed using
Hamming distance (number of differing bits), which can be computed extremely fast
using hardware popcount instructions:

$\text{sim}_{\text{binary}}(\mathbf{b}_q, \mathbf{b}_d) = d - 2 \cdot \text{hamming}(\mathbf{b}_q, \mathbf{b}_d)$

where $d$ is the number of dimensions.

**Quality impact**: Binary quantization typically degrades retrieval quality by 5–15%,
making it suitable for first-stage candidate retrieval (followed by rescoring with
full-precision embeddings).

### 12.11.4 Product Quantization (PQ)

Product quantization (Jégou et al., 2011) splits the embedding vector into $M$
sub-vectors and quantizes each independently using a small codebook:

$\mathbf{e} = [\mathbf{e}^1, \mathbf{e}^2, \ldots, \mathbf{e}^M]$

Each sub-vector $\mathbf{e}^m \in \mathbb{R}^{d/M}$ is replaced by the index of its
nearest centroid in a codebook $\mathcal{C}^m = \{\mathbf{c}_1^m, \ldots, \mathbf{c}_K^m\}$
of size $K$ (typically $K = 256$):

$\text{PQ}(\mathbf{e}^m) = \arg\min_{k} \|\mathbf{e}^m - \mathbf{c}_k^m\|^2$

Storage per vector: $M \times \lceil\log_2 K\rceil$ bits. For $M = 64$ sub-vectors
and $K = 256$ centroids: $64 \times 8 = 512$ bits = 64 bytes, compared to
$1024 \times 4 = 4096$ bytes for float32 — a **64× compression**.

### 12.11.5 Matryoshka + Quantization: Compound Savings

Matryoshka embeddings (Chapter 10) and quantization are orthogonal techniques that
can be combined for dramatic compression:

| Configuration | Dims | Precision | Bytes/vector | Compression vs. 1024d float32 |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | 1024 | float32 | 4096 | 1× |
| Matryoshka only | 256 | float32 | 1024 | 4× |
| int8 only | 1024 | int8 | 1024 | 4× |
| Binary only | 1024 | binary | 128 | 32× |
| Matryoshka + int8 | 256 | int8 | 256 | 16× |
| Matryoshka + binary | 256 | binary | 32 | 128× |

A 128× compression means 1 billion embeddings fit in ~32 GB instead of ~4 TB.

### 12.11.6 Code Example: Quantization in Practice

```python
import numpy as np

# Generate sample embeddings (simulating model output)
np.random.seed(42)
embeddings = np.random.randn(10000, 1024).astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# --- Scalar (int8) Quantization ---
def quantize_int8(emb):
    """Quantize float32 embeddings to int8."""
    mn = emb.min(axis=0)
    mx = emb.max(axis=0)
    scale = (mx - mn) / 255.0
    quantized = np.round((emb - mn) / scale).astype(np.uint8)
    return quantized, mn, scale

def dequantize_int8(quantized, mn, scale):
    """Dequantize int8 back to float32 (approximate)."""
    return quantized.astype(np.float32) * scale + mn

q_int8, mn, scale = quantize_int8(embeddings)
reconstructed = dequantize_int8(q_int8, mn, scale)

# Measure reconstruction error
mse = np.mean((embeddings - reconstructed) ** 2)
print(f"int8 MSE: {mse:.6f}")
print(f"int8 storage: {q_int8.nbytes / 1e6:.1f} MB vs "
      f"float32: {embeddings.nbytes / 1e6:.1f} MB "
      f"({embeddings.nbytes / q_int8.nbytes:.0f}× compression)")

# --- Binary Quantization ---
def quantize_binary(emb):
    """Quantize to binary (1 bit per dimension)."""
    return np.packbits((emb > 0).astype(np.uint8), axis=1)

def hamming_distance(a, b):
    """Compute Hamming distance between packed binary arrays."""
    xor = np.bitwise_xor(a, b)
    return np.sum(np.unpackbits(xor, axis=1), axis=1)

q_binary = quantize_binary(embeddings)
print(f"\nBinary storage: {q_binary.nbytes / 1e6:.1f} MB "
      f"({embeddings.nbytes / q_binary.nbytes:.0f}× compression)")

# Compare: find nearest neighbor with float32 vs binary
query = embeddings[0:1]
q_bin = quantize_binary(query)

# Float32 nearest neighbor
float_sims = (query @ embeddings[1:].T).flatten()
float_nn = np.argmax(float_sims)

# Binary nearest neighbor (using Hamming distance)
ham_dists = hamming_distance(
    np.repeat(q_bin, len(q_binary) - 1, axis=0), q_binary[1:]
)
binary_nn = np.argmin(ham_dists)

print(f"\nFloat32 nearest neighbor: index {float_nn}")
print(f"Binary nearest neighbor:  index {binary_nn}")
print(f"Match: {float_nn == binary_nn}")
```

---

## 12.12 The MTEB Benchmark

### 12.12.1 Overview

The **Massive Text Embedding Benchmark** (MTEB; Muennighoff et al., 2023) is the
standard evaluation framework for text embedding models. It provides a unified
evaluation across 8 task categories, 58+ datasets, and 112+ languages.

MTEB was created to address a fragmented evaluation landscape where different papers
reported results on different subsets of tasks, making fair comparison impossible.

### 12.12.2 Task Categories

| Category | # Datasets | Metric | What It Measures |
|:---:|:---:|:---:|:---:|
| Classification | 12 | Accuracy | Linear probe on frozen embeddings |
| Clustering | 11 | V-measure | k-means on frozen embeddings |
| Pair Classification | 3 | AP (avg precision) | Binary: are two texts related? |
| Reranking | 4 | MAP | Reorder candidate documents |
| Retrieval | 15 | NDCG@10 | Find relevant docs from corpus |
| STS | 10 | Spearman corr. | Predict human similarity scores |
| Summarization | 1 | Spearman corr. | Machine vs. human summary similarity |
| Bitext Mining | 2+ | F1 | Find translation pairs across languages |

### 12.12.3 How MTEB Scoring Works

Each task category produces a single score (the average across datasets in that
category). The overall MTEB score is the average across all categories:

$\text{MTEB}_{\text{avg}} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \text{score}_t$

where $\mathcal{T}$ is the set of task categories. This gives equal weight to each
category regardless of the number of datasets within it.

### 12.12.4 Leaderboard Snapshot (as of early 2025)

| Rank | Model | Params | Dim | MTEB Avg | Architecture |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | NV-Embed-v2 | 7.1B | 4096 | 72.31 | Decoder (Mistral) |
| 2 | SFR-Embedding-2 | 7.1B | 4096 | 71.13 | Decoder (Mistral) |
| 3 | Conan-Embedding-v2 | 7.1B | 4096 | 70.28 | Decoder |
| 4 | voyage-3-large | Unknown | 2048 | 69.23 | Proprietary |
| 5 | E5-Mistral-7B | 7.1B | 4096 | 66.63 | Decoder (Mistral) |
| — | BGE-large-en-v1.5 | 335M | 1024 | 64.23 | Encoder (BERT) |
| — | Nomic-Embed-v1.5 | 137M | 768 | 62.28 | Encoder (BERT) |
| — | all-MiniLM-L6-v2 | 22M | 384 | 56.26 | Encoder (MiniLM) |

The trend is clear: decoder-only models with 7B+ parameters dominate the top of the
leaderboard, but smaller encoder models remain competitive when accounting for
efficiency (score per FLOP).

### 12.12.5 Limitations of MTEB

Despite its value, MTEB has significant limitations:

1. **English-centric**: The original MTEB is heavily English-focused. MTEB-multilingual
   extensions exist but are less comprehensive.

2. **Static evaluation**: Benchmarks are fixed snapshots. Models can overfit to the
   test sets through data contamination or benchmark-specific tuning.

3. **Average hides trade-offs**: A model that excels at retrieval but fails at
   clustering may have the same average score as a balanced model. The "best" model
   depends on your use case.

4. **Missing real-world tasks**: MTEB doesn't cover important applications like
   RAG quality, code search, or multi-turn conversation retrieval.

5. **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good
   measure." Models are increasingly optimized specifically for MTEB, which may not
   translate to real-world performance.

### 12.12.6 Code Example: Evaluating on MTEB

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Evaluate on a specific task
evaluation = MTEB(tasks=["STS12", "STS13", "STS14"])
results = evaluation.run(model, output_folder="results/bge-small")

# Evaluate on a full category
evaluation = MTEB(task_types=["Retrieval"])
results = evaluation.run(model, output_folder="results/bge-small-retrieval")

# Print results
for task_result in results:
    print(f"{task_result.task_name}: {task_result.get_score():.4f}")
```

---

## 12.13 Future Directions

### 12.13.1 Training-Free KV-Based Embeddings

An emerging research direction explores extracting embeddings directly from the
**key-value caches** of pre-trained LLMs without any additional training:

The idea: when an LLM processes a text, its internal key and value matrices at each
layer contain rich representations of the input. By aggregating these KV states
(e.g., via mean pooling across layers and positions), we can obtain embeddings that
capture the model's "understanding" of the text.

$\mathbf{e}_{\text{KV}} = \frac{1}{L} \sum_{l=1}^{L} \frac{1}{n} \sum_{i=1}^{n} [\mathbf{k}_i^{(l)}; \mathbf{v}_i^{(l)}]$

where $\mathbf{k}_i^{(l)}$ and $\mathbf{v}_i^{(l)}$ are the key and value vectors
for token $i$ at layer $l$, and $[\cdot; \cdot]$ denotes concatenation.

This approach is appealing because it requires zero additional training — any
pre-trained LLM can be used as an embedding model immediately. Early results show
competitive performance on STS tasks, though retrieval performance still lags behind
dedicated embedding models.

### 12.13.2 Reasoning-State Embeddings

As LLMs develop stronger reasoning capabilities (chain-of-thought, tree-of-thought),
a natural question arises: can we embed the *reasoning process* itself?

Reasoning-state embeddings would capture not just what a text says, but the
intermediate logical steps needed to understand it. For example, the embedding of
"If it rains, the ground gets wet. It rained yesterday." would encode not just the
surface text but the implicit conclusion "the ground was wet yesterday."

This connects to the broader trend of **inference-time compute** — using more
computation at inference time (rather than just at training time) to produce better
representations.

### 12.13.3 Dedicated Embedding LLMs

Models like **Conan-Embedding-v2** represent a new category: LLMs designed and trained
*specifically* for embedding tasks from the ground up, rather than being adapted from
general-purpose LLMs.

The hypothesis is that the pre-training objective matters: a model pre-trained with
embedding-aware objectives (contrastive learning, retrieval-oriented tasks) from the
start will produce better embeddings than one pre-trained for text generation and
then adapted.

### 12.13.4 Task-Adaptive LoRA Adapters at Scale

Jina v3's approach of task-specific LoRA adapters (Section 12.5) points toward a
future where embedding models ship with a *library* of adapters:

```
Base Model (frozen)
    │
    ├── retrieval.query adapter
    ├── retrieval.passage adapter
    ├── classification.sentiment adapter
    ├── classification.topic adapter
    ├── clustering.news adapter
    ├── clustering.scientific adapter
    ├── code.search adapter
    ├── code.similarity adapter
    ├── legal.retrieval adapter
    ├── medical.retrieval adapter
    └── ... (community-contributed adapters)
```

Each adapter adds minimal parameters (~1–3% of the base model) but can dramatically
improve performance on its target task. Users would select or combine adapters based
on their specific use case, and the community could contribute domain-specific adapters
without retraining the base model.

### 12.13.5 Unified Retrieval: Dense + Sparse + Multi-Vector

BGE-M3 demonstrated that a single model can produce dense, sparse, and multi-vector
representations simultaneously. The future likely extends this to:

1. **Learned fusion weights**: Instead of manually tuning $\alpha, \beta, \gamma$,
   the model learns task-specific fusion weights
2. **Adaptive granularity**: The model decides whether to use dense, sparse, or
   multi-vector retrieval based on the query
3. **End-to-end training with the index**: Training the embedding model jointly with
   the retrieval index (e.g., differentiable approximate nearest neighbor search)

### 12.13.6 Embedding Models as First-Class AI Products

Embedding models are increasingly treated as standalone products rather than
components of larger systems:

- **Cohere Embed**: Proprietary embedding API with built-in search optimization
- **Voyage AI**: Specialized embedding models for code, legal, and financial domains
- **OpenAI Embeddings**: text-embedding-3-small/large with native Matryoshka support
- **Google Gecko**: Compact embedding model distilled from larger LLMs

The trend is toward **domain-specialized** embedding products: a medical embedding
model trained on PubMed, a legal model trained on case law, a code model trained on
GitHub. General-purpose models will remain important, but the highest-value
applications will use specialized models.

### 12.13.7 Multimodal Embedding Unification

The future points toward truly unified embedding spaces that handle arbitrary
modalities:

- Text, images, audio, video, code, structured data, graphs
- Cross-modal retrieval: "find me a code snippet that implements this diagram"
- Temporal embeddings: representations that evolve over time
- Embodied embeddings: representations grounded in physical interaction

---

## 12.14 Summary — The Evolution at a Glance

### 12.14.1 Timeline: From Distributional Semantics to Universal Embeddings

| Year | Model / Milestone | Key Innovation | Dims | Context |
|:---:|:---:|:---:|:---:|:---:|
| 1954 | Harris | Distributional hypothesis | — | — |
| 1957 | Firth | "You shall know a word by the company it keeps" | — | — |
| 1990 | LSA (Deerwester) | SVD on term-document matrix | 100–300 | Document |
| 2003 | Bengio NNLM | Neural language model embeddings | 30–100 | 5 words |
| 2013 | Word2Vec | Shallow log-linear models, negative sampling | 100–300 | 5–10 words |
| 2014 | GloVe | Global co-occurrence matrix factorization | 100–300 | Document |
| 2016 | FastText | Subword n-gram embeddings | 100–300 | 5–10 words |
| 2018 | ELMo | Contextual embeddings from biLSTM | 1024 | Sentence |
| 2018 | BERT | Bidirectional Transformer pre-training | 768 | 512 tokens |
| 2019 | SBERT | Siamese BERT for sentence embeddings | 768 | 512 tokens |
| 2021 | SimCSE | Simple contrastive sentence embeddings | 768 | 512 tokens |
| 2021 | CLIP | Joint image-text embedding space | 512 | 77 tokens |
| 2022 | E5 | Weakly-supervised contrastive pre-training | 1024 | 512 tokens |
| 2022 | INSTRUCTOR | Instruction-tuned embeddings | 768 | 512 tokens |
| 2022 | Matryoshka | Flexible-dimension embeddings | Any | 512 tokens |
| 2023 | BGE v1/v1.5 | RetroMAE + contrastive + instruction tuning | 1024 | 512 tokens |
| 2023 | GTE | General Text Embeddings (Alibaba) | 1024 | 8192 tokens |
| 2023 | Jina v2 | 8192-token BERT with ALiBi | 768 | 8192 tokens |
| 2024 | BGE-M3 | Multi-lingual, multi-functional, multi-granularity | 1024 | 8192 tokens |
| 2024 | Nomic Embed v1.5 | Fully open + Matryoshka | 768 | 8192 tokens |
| 2024 | E5-Mistral | Decoder-only LLM as embedding model | 4096 | 32768 tokens |
| 2024 | NV-Embed v2 | Latent attention pooling, MTEB #1 | 4096 | 32768 tokens |
| 2024 | Jina v3 | Task-specific LoRA adapters | 1024 | 8192 tokens |
| 2024 | GritLM | Unified generation + embedding model | 4096 | 32768 tokens |
| 2025 | Conan-Embed-v2 | Dedicated embedding LLM | 4096 | 32768 tokens |

### 12.14.2 Key Paradigm Shifts

Looking across this timeline, five paradigm shifts stand out:

**Shift 1: Static → Contextual (2018)**
ELMo and BERT showed that word meaning depends on context. The same word gets
different embeddings in different sentences. This was the death of the "one vector
per word" paradigm.

**Shift 2: Token → Sentence (2019)**
SBERT showed that BERT's [CLS] token is a poor sentence embedding and that
contrastive fine-tuning with siamese networks produces dramatically better sentence
representations. This enabled practical semantic search.

**Shift 3: Generic → Instruction-Tuned (2022)**
INSTRUCTOR and E5 showed that embeddings should be task-aware. The same text should
have different representations depending on whether you're doing retrieval,
classification, or clustering.

**Shift 4: Encoder → Decoder (2024)**
E5-Mistral and NV-Embed showed that decoder-only LLMs, despite their causal attention
mask, produce superior embeddings when properly adapted. The massive scale of
decoder-only pre-training outweighs the architectural disadvantage.

**Shift 5: Single-Vector → Multi-Granularity (2024)**
BGE-M3 showed that a single model can produce dense, sparse, and multi-vector
representations simultaneously, enabling hybrid retrieval without separate systems.

### 12.14.3 What to Use in 2025/2026

The "best" model depends on your constraints. Here is a practical decision framework:

```
START
  │
  ├─ Need multilingual support?
  │   ├─ Yes → BGE-M3 (100+ languages, 8192 tokens)
  │   └─ No ↓
  │
  ├─ Need maximum quality (cost is secondary)?
  │   ├─ Yes → NV-Embed-v2 or SFR-Embedding-2 (7B params, GPU required)
  │   └─ No ↓
  │
  ├─ Need task-specific adapters?
  │   ├─ Yes → Jina v3 (LoRA adapters for retrieval/classification/clustering)
  │   └─ No ↓
  │
  ├─ Need hybrid dense+sparse retrieval?
  │   ├─ Yes → BGE-M3 (unified dense + sparse + multi-vector)
  │   └─ No ↓
  │
  ├─ Need full reproducibility / open-source?
  │   ├─ Yes → Nomic Embed v1.5 (Apache 2.0, open data, open code)
  │   └─ No ↓
  │
  ├─ Need to run on CPU / edge devices?
  │   ├─ Yes → all-MiniLM-L6-v2 (22M params) or BGE-small-en-v1.5 (33M)
  │   └─ No ↓
  │
  └─ General-purpose, good balance?
      └─ BGE-large-en-v1.5 or E5-large-v2 (335M params, strong all-around)
```

### 12.14.4 The Embedding Landscape Is Not Converging — It's Diverging

A final observation: unlike language models (where scaling laws push toward ever-larger
models), the embedding landscape is *diverging* into specialized niches:

- **Tiny models** (22–33M params) for edge deployment and real-time applications
- **Medium models** (100–350M params) for the best quality-per-dollar
- **Large models** (7B+ params) for maximum quality when cost is secondary
- **Domain-specific models** for legal, medical, code, and financial applications
- **Multi-modal models** for cross-modal retrieval
- **Hybrid models** for unified dense + sparse retrieval

There is no single "best" embedding model, and there likely never will be. The art
lies in matching the model to the task, the data, and the constraints.

---

## References

1. Wang, L., et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv:2212.03533*.
2. Wang, L., et al. (2024). "Improving Text Embeddings with Large Language Models." *arXiv:2401.00368*.
3. Xiao, S., et al. (2023). "C-Pack: Packaged Resources To Advance General Chinese Embedding." *arXiv:2309.07597*.
4. Chen, J., et al. (2024). "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." *arXiv:2402.03216*.
5. Nussbaum, Z., et al. (2024). "Nomic Embed: Training a Reproducible Long Context Text Embedder." *arXiv:2402.01613*.
6. Günther, M., et al. (2023). "Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents." *arXiv:2310.19923*.
7. Sturua, S., et al. (2024). "jina-embeddings-v3: Multilingual Embeddings With Task LoRA." *arXiv:2409.10173*.
8. Su, H., et al. (2022). "One Embedder, Any Task: Instruction-Finetuned Text Embeddings." *arXiv:2212.09741*.
9. Lee, C., et al. (2024). "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." *arXiv:2405.17428*.
10. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*.
11. Girdhar, R., et al. (2023). "ImageBind: One Embedding Space To Bind Them All." *CVPR 2023*.
12. Khattab, O. & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." *SIGIR 2020*.
13. Formal, T., et al. (2021). "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." *SIGIR 2021*.
14. Muennighoff, N., et al. (2023). "MTEB: Massive Text Embedding Benchmark." *EACL 2023*.
15. Hu, E., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
16. Jégou, H., et al. (2011). "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*.
17. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *ICLR 2022*.
18. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
19. Xiao, S., et al. (2022). "RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder." *EMNLP 2022*.


---

*← [Back to Table of Contents](../README.md)*
