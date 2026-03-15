# Embeddings: A Deep Dive — From Foundations to Frontier

A comprehensive, chronological guide to embeddings in NLP and machine learning. Every mathematical concept is explained step by step, every loss function derived from first principles, and every fine-tuning technique illustrated with examples.

## Table of Contents

| Chapter | Title | Key Topics |
|---------|-------|------------|
| 01 | [Origins & Foundations](chapters/01-origins-and-foundations.md) | Distributional semantics, one-hot encoding, co-occurrence matrices, LSA, word vectors before deep learning |
| 02 | [Word2Vec](chapters/02-word2vec.md) | CBOW, Skip-gram, negative sampling, hierarchical softmax, full math derivation |
| 03 | [GloVe](chapters/03-glove.md) | Co-occurrence statistics, weighted least squares, log-bilinear model, full math |
| 04 | [FastText](chapters/04-fasttext.md) | Subword embeddings, character n-grams, OOV handling |
| 05 | [Contextual Embeddings](chapters/05-contextual-embeddings.md) | ELMo, BERT, GPT embeddings, attention-based representations |
| 06 | [Sentence Embeddings & SBERT](chapters/06-sentence-embeddings-sbert.md) | Siamese networks, sentence-BERT, pooling strategies |
| 07 | [Training Objectives & Classical Loss Functions](chapters/07-training-objectives-and-loss-functions.md) | Contrastive loss, triplet loss, softmax loss, cross-entropy for embeddings |
| 08 | [Fine-Tuning Embeddings — A Deep Dive](chapters/08-finetuning-embeddings.md) | MNRL, in-batch negatives, hard negative mining, knowledge distillation, full pipeline |
| 09 | [Modern Loss Functions](chapters/09-modern-loss-functions.md) | CoSENT loss, AnglE loss, circle loss, and their mathematical derivations |
| 10 | [Matryoshka Embeddings](chapters/10-matryoshka-embeddings.md) | Matryoshka Representation Learning, nested dimensions, multi-granularity training |
| 11 | [GIST Embeddings](chapters/11-gist-embeddings.md) | Guided In-sample Selection of Training Negatives, guide models, hard negative selection |
| 12 | [The Frontier & Future](chapters/12-frontier-and-future.md) | E5, BGE, Nomic, Jina, instruction-tuned embeddings, multi-modal embeddings |

## How to Read This Guide

Each chapter is self-contained but builds on previous ones. Mathematical derivations use standard notation and are broken into numbered steps. Code examples use Python with PyTorch.

## Prerequisites

- Linear algebra (vectors, matrices, dot products)
- Basic calculus (gradients, chain rule)
- Probability theory (distributions, likelihood)
- Familiarity with neural networks

## License

MIT License — use freely for learning and teaching.
