# Embeddings: A Deep Dive — From Foundations to Frontier

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

A comprehensive, open-source guide to embeddings in NLP and machine learning — 22,000+ lines across 12 chapters, covering everything from 1950s distributional semantics to 2025's instruction-tuned multimodal models. Every mathematical concept is derived from first principles, every loss function implemented in PyTorch, and every technique illustrated with runnable code and ASCII architecture diagrams.

📐 50+ mathematical derivations · 🐍 40+ runnable Python examples · 🎨 30+ ASCII diagrams · 📄 12 chapters · 📚 22,000+ lines

---

## Table of Contents

- [Chapter Overview](#chapter-overview)
- [Detailed Chapter Summaries](#detailed-chapter-summaries)
- [Getting Started](#getting-started)
- [Reading Paths](#reading-paths)
- [How to Use This Guide](#how-to-use-this-guide)
  - [Code Examples](#code-examples)
  - [Diagrams](#diagrams)
  - [Mathematical Notation](#mathematical-notation)
  - [References](#references)
- [Prerequisites](#prerequisites)
- [Who Is This For?](#who-is-this-for)
- [Contributing](#contributing)
- [Reporting Issues & Requesting Features](#reporting-issues--requesting-features)
- [Errata & Corrections](#errata--corrections)
- [How to Cite](#how-to-cite)
- [License](#license)
- [Support the Project](#support-the-project)

---

## Chapter Overview

| # | Chapter | Lines | Key Topics | Contains |
|---|---------|------:|------------|----------|
| 01 | [Origins & Foundations](chapters/01-origins-and-foundations.md) | 1,695 | Distributional hypothesis, one-hot encoding, co-occurrence matrices, PMI, SVD, LSA | 📐🐍🎨 |
| 02 | [Word2Vec](chapters/02-word2vec.md) | 1,386 | CBOW, Skip-gram, negative sampling, hierarchical softmax, HNSW | 📐🐍🎨 |
| 03 | [GloVe](chapters/03-glove.md) | 1,386 | Co-occurrence statistics, weighted least squares, log-bilinear model, analogy tasks | 📐🐍🎨 |
| 04 | [FastText](chapters/04-fasttext.md) | 1,122 | Subword embeddings, character n-grams, OOV handling, morphologically rich languages | 📐🐍🎨 |
| 05 | [Contextual Embeddings](chapters/05-contextual-embeddings.md) | 1,642 | ELMo, BERT, GPT embeddings, attention mechanisms, anisotropy problem | 📐🐍🎨 |
| 06 | [Sentence Embeddings & SBERT](chapters/06-sentence-embeddings-sbert.md) | 1,228 | Siamese networks, Sentence-BERT, pooling strategies, cross-encoder vs bi-encoder | 📐🐍🎨 |
| 07 | [Training Objectives & Loss Functions](chapters/07-training-objectives-and-loss-functions.md) | 1,888 | Contrastive loss, triplet loss, InfoNCE, N-pair loss, ArcFace, softmax loss | 📐🐍🎨 |
| 08 | [Fine-Tuning Embeddings](chapters/08-finetuning-embeddings.md) | 2,343 | MNRL, in-batch negatives, hard negative mining, knowledge distillation, SimCSE | 📐🐍🎨 |
| 09 | [Modern Loss Functions](chapters/09-modern-loss-functions.md) | 2,769 | Circle Loss, CoSENT Loss, AnglE Loss, cosine saturation, angular optimization | 📐🐍🎨 |
| 10 | [Matryoshka Embeddings](chapters/10-matryoshka-embeddings.md) | 2,159 | Matryoshka Representation Learning, nested dimensions, adaptive retrieval | 📐🐍🎨 |
| 11 | [GIST Embeddings](chapters/11-gist-embeddings.md) | 1,860 | Guided In-sample Selection of Training Negatives, false negative problem, guide models | 📐🐍🎨 |
| 12 | [The Frontier & Future](chapters/12-frontier-and-future.md) | 2,411 | E5, BGE, Jina, Nomic, Qwen3-Embedding, ColPali, binary quantization, multimodal | 📐🐍🎨 |

> 📐 = Math derivations · 🐍 = Runnable code · 🎨 = ASCII diagrams

---

## Detailed Chapter Summaries

### Chapter 1: Origins & Foundations

Traces the intellectual lineage of embeddings from Harris's distributional hypothesis (1954) and Firth's contextual theory of meaning through the linear algebra era of the 1990s. Covers one-hot encoding, bag-of-words, TF-IDF, pointwise mutual information (PMI), co-occurrence matrices, and dimensionality reduction via SVD/LSA. Includes Python implementations of each technique from scratch, showing how sparse symbolic representations evolved into dense vector spaces.

### Chapter 2: Word2Vec

A complete treatment of Mikolov et al.'s Word2Vec (2013), the model that launched the modern embedding era. Derives both the CBOW and Skip-gram architectures from first principles, then works through the computational bottleneck of the full softmax and its two solutions: hierarchical softmax and negative sampling. Includes the full gradient derivation for Skip-gram with negative sampling (SGNS), subsampling of frequent words, and the famous analogy arithmetic (king - man + woman ≈ queen).

### Chapter 3: GloVe

Covers Pennington et al.'s GloVe (2014), which bridges count-based and prediction-based methods. Derives the weighted least-squares objective from the key insight about ratios of co-occurrence probabilities. Includes a from-scratch GloVe training implementation in PyTorch, analysis of the weighting function, and comparison with Word2Vec on analogy and similarity benchmarks.

### Chapter 4: FastText

Explains Bojanowski et al.'s FastText (2017), which extends Skip-gram by representing words as bags of character n-grams. Covers the subword decomposition algorithm, how OOV words get representations by summing their n-gram vectors, and why this matters for morphologically rich languages (Turkish, Finnish, German). Includes n-gram extraction code and OOV handling demonstrations.

### Chapter 5: Contextual Embeddings

The paradigm shift from static to contextual representations. Covers ELMo's bidirectional LSTM approach, BERT's masked language modeling, and GPT-style autoregressive embeddings. Explains the anisotropy problem (why naive BERT embeddings cluster in a narrow cone) and the transition from token-level to sentence-level representations. Includes architecture diagrams and comparison code showing how the same word gets different vectors in different contexts.

### Chapter 6: Sentence Embeddings & SBERT

Covers Reimers & Gurevych's Sentence-BERT (2019), which made BERT practical for similarity search by training siamese/triplet networks. Explains why raw BERT is too slow for pairwise comparison (N² forward passes), how SBERT reduces this to N passes + cosine similarity, and the choice between cross-encoder (accurate but slow) and bi-encoder (fast but less accurate) architectures. Covers pooling strategies (CLS, mean, max) with implementation code.

### Chapter 7: Training Objectives & Loss Functions

The mathematical heart of the guide. Provides rigorous derivations of the six foundational loss functions for metric learning: contrastive loss, triplet loss, N-pair loss, InfoNCE/NT-Xent, softmax-based losses, and angular margin losses (ArcFace, CosFace). Each loss is derived from first principles, analyzed for gradient behavior, and implemented in PyTorch. Includes comparison of convergence properties and guidance on when to use which loss.

### Chapter 8: Fine-Tuning Embeddings

A practitioner's deep dive into adapting pre-trained models for domain-specific tasks. Covers Multiple Negatives Ranking Loss (MNRL), in-batch negative strategies, hard negative mining with cross-encoders, knowledge distillation from cross-encoders to bi-encoders, data augmentation techniques, and SimCSE. Includes a complete end-to-end fine-tuning recipe using sentence-transformers v3, evaluation code computing nDCG@10 and recall@k, and a "Common Pitfalls" section with practical advice.

### Chapter 9: Modern Loss Functions

Covers three post-2020 loss functions that address fundamental limitations of classical approaches. Circle Loss (2020) introduces flexible optimization with separate similarity weighting for positive and negative pairs. CoSENT Loss (2022) solves the cosine saturation problem by operating on similarity differences rather than absolute values. AnglE Loss (2024, ACL) works in complex space to avoid gradient vanishing. Each is derived mathematically and implemented from scratch, with a comparison training script.

### Chapter 10: Matryoshka Embeddings

Explains Matryoshka Representation Learning (Kusupati et al., 2022), which trains embeddings where the first $d'$ dimensions of a $d$-dimensional vector form a valid $d'$-dimensional embedding. Derives the multi-scale loss function, explains why information naturally organizes hierarchically in the prefix dimensions, and provides a complete training pipeline. Covers practical deployment scenarios: use 64 dimensions for fast filtering, 256 for re-ranking, full 768 for final scoring.

### Chapter 11: GIST Embeddings

Addresses the false negative problem in contrastive learning — when in-batch negatives are actually semantically similar to the query. GIST (Guided In-sample Selection of Training Negatives) uses a separate "guide model" (typically a cross-encoder) to identify and mask false negatives during training. Derives the masking strategy mathematically, covers margin-based and threshold-based approaches, and includes implementation with sentence-transformers' `GISTEmbedLoss` and `CachedGISTEmbedLoss`.

### Chapter 12: The Frontier & Future

Surveys the state of the art as of 2024–2025. Covers instruction-tuned embeddings (E5, BGE, Instructor), decoder-only LLM embeddings (NV-Embed, Qwen3-Embedding), late interaction models (ColBERT, ColPali, ColQwen2), multimodal embeddings (Qwen3-VL-Embedding), multi-granularity models (BGE-M3), binary quantization for 32× compression, and the convergence of embedding models with retrieval systems. Includes code examples for the latest APIs and a forward-looking analysis of where the field is heading.

---

## Getting Started

Clone the repository and start reading:

```bash
git clone https://github.com/stabgan/embeddings-deep-dive.git
cd embeddings-deep-dive
```

Each chapter is a standalone Markdown file in the `chapters/` directory. Open them in any Markdown viewer, or read directly on GitHub where LaTeX math renders natively.

For the code examples, you'll need Python 3.8+ and a few common libraries:

```bash
pip install torch numpy scipy scikit-learn gensim sentence-transformers transformers
```

---

## Reading Paths

Not everyone needs to read all 22,000 lines. Here are three suggested paths:

### 🟢 "I'm new to embeddings"

Read linearly from Chapter 1 through Chapter 12. Each chapter builds on the previous one, introducing concepts in the order they were historically developed.

```
Ch 1 → Ch 2 → Ch 3 → Ch 4 → Ch 5 → Ch 6 → Ch 7 → Ch 8 → Ch 9 → Ch 10 → Ch 11 → Ch 12
```

### 🟡 "I need to fine-tune an embedding model for production"

Jump straight to the practical chapters. Skim Chapter 6 for SBERT basics, then focus on the training and fine-tuning pipeline:

```
Ch 6 (skim) → Ch 7 → Ch 8 → Ch 9 → Ch 10 → Ch 12
```

### 🔴 "I'm researching modern embedding techniques"

Start with the latest developments and work backward as needed:

```
Ch 12 → Ch 9 → Ch 10 → Ch 11 → Ch 7 (for foundations)
```

---

## How to Use This Guide

### Code Examples

All code examples are written in Python using PyTorch and are designed to be copy-paste runnable. They follow a consistent pattern:

- **Quick-start snippets** at the top of each chapter show the simplest way to use the technique (e.g., loading a pre-trained model in 5 lines)
- **From-scratch implementations** derive the algorithm step by step, prioritizing clarity over optimization
- **Production recipes** in later chapters show real-world training pipelines with proper evaluation

Most examples require only `torch`, `numpy`, and `sentence-transformers`. Chapter-specific dependencies are noted inline.

### Diagrams

Architecture diagrams and data flow illustrations use ASCII art for maximum portability — they render correctly on GitHub, in terminals, in any text editor, and in print. For best results, view them in a monospace font. Example:

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Input   │────▶│ Encoder │────▶│ Embedding│
│  Text    │     │ (BERT)  │     │  768-d   │
└─────────┘     └─────────┘     └─────────┘
```

### Mathematical Notation

Mathematical derivations use standard LaTeX notation rendered by GitHub's native math support. Conventions used throughout:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{u}, \mathbf{v}$ | Embedding vectors |
| $f_\theta(\cdot)$ | Encoder function with parameters $\theta$ |
| $\text{sim}(\cdot, \cdot)$ | Similarity function (usually cosine) |
| $d(\cdot, \cdot)$ | Distance function |
| $\mathcal{L}$ | Loss function |
| $\tau$ | Temperature parameter |
| $B$ | Batch size |

Every derivation is broken into numbered steps. If a step uses a result from a previous chapter, a cross-reference is provided.

### References

Academic papers are cited inline using the standard format: Author (Year). Full references appear at the end of each chapter. Key papers are linked to their arXiv or conference proceedings pages where available.

---

## Prerequisites

This guide assumes familiarity with the following. If you need a refresher, the linked resources are free and excellent:

| Topic | What You Need | Recommended Resource |
|-------|---------------|---------------------|
| Linear Algebra | Vectors, matrices, dot products, eigenvalues, SVD | [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) |
| Calculus | Gradients, chain rule, partial derivatives | [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| Probability | Distributions, likelihood, Bayes' theorem, softmax | [StatQuest: Probability Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) |
| Neural Networks | Forward/backward pass, SGD, loss functions | [Andrej Karpathy: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) |
| Python | NumPy, basic PyTorch tensors | [Official PyTorch Tutorials](https://pytorch.org/tutorials/) |

Chapters 1–4 are accessible with just linear algebra and basic calculus. Chapters 5+ assume familiarity with neural networks and backpropagation.

---

## Who Is This For?

- **ML engineers** building semantic search, RAG pipelines, or recommendation systems who need to understand what's happening inside their embedding models
- **NLP researchers** looking for a unified reference that connects classical distributional semantics to modern contrastive learning
- **Graduate students** studying representation learning who want derivations that don't skip steps
- **Practitioners** evaluating embedding models (which loss function? which architecture? how many dimensions?) who need the theory behind the benchmarks
- **Anyone** who has used `model.encode("hello world")` and wondered what those 768 numbers actually mean

---

## Contributing

Contributions are welcome. Here's how:

1. **Fork** the repository
2. **Create a branch** for your changes (`git checkout -b fix/typo-chapter-3`)
3. **Make your changes** following the style guidelines below
4. **Submit a pull request** with a clear description of what you changed and why

### Style Guidelines

- **Math**: Use LaTeX notation (`$...$` for inline, `$$...$$` for display). Number equations when referenced later.
- **Code**: Python 3.8+, PyTorch preferred. Include comments explaining each step. Keep examples self-contained and runnable.
- **Diagrams**: ASCII art using box-drawing characters (`┌ ─ ┐ │ └ ┘ ▶ ▼`). Ensure they render in monospace at 80-column width.
- **Prose**: Write for a technical audience but don't assume prior knowledge of the specific topic. Define terms on first use.

### What We're Looking For

- Corrections to mathematical derivations or code
- Additional code examples or alternative implementations
- Translations to other languages
- New diagrams that clarify complex concepts
- Updates reflecting new papers or models

---

## Reporting Issues & Requesting Features

- **Found a bug in the code?** Open a [GitHub Issue](https://github.com/stabgan/embeddings-deep-dive/issues) with the chapter number, the code block, and the error message.
- **Found a math error?** Open an issue with the equation number and your proposed correction.
- **Want a new topic covered?** Open an issue with the tag `feature-request` describing what you'd like to see and why it fits the guide's scope.
- **Have a question about the content?** Use [GitHub Discussions](https://github.com/stabgan/embeddings-deep-dive/discussions) for Q&A.

---

## Errata & Corrections

Despite careful review, errors may exist in a 22,000-line technical document. If you find one:

1. Check the [Issues](https://github.com/stabgan/embeddings-deep-dive/issues) page to see if it's already reported
2. If not, open a new issue with the label `errata`
3. Include: chapter number, section, the error, and (if possible) the correction

Confirmed corrections are merged promptly and credited in the commit message.

---

## How to Cite

If you use this guide in your research or teaching, please cite it:

```bibtex
@misc{embeddings-deep-dive,
  title   = {Embeddings: A Deep Dive — From Foundations to Frontier},
  author  = {stabgan},
  year    = {2025},
  url     = {https://github.com/stabgan/embeddings-deep-dive},
  note    = {Open-source technical guide, 22,000+ lines}
}
```

Or in prose: *"Embeddings: A Deep Dive — From Foundations to Frontier"* (stabgan, 2025). Available at https://github.com/stabgan/embeddings-deep-dive.

---

## License

MIT License — use freely for learning, teaching, and building. See [LICENSE](LICENSE) for details.

---

## Support the Project

If this guide helped you, consider:

- ⭐ **Starring** the repository — it helps others discover it
- 🔀 **Sharing** it with colleagues, students, or on social media
- 🛠️ **Contributing** a fix, example, or translation
- 💬 **Opening a discussion** to share how you're using it

---

<p align="center">
  <i>Built with math, code, and too much coffee.</i>
</p>
