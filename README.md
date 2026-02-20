# Sentiment Analysis Using LSTM Networks

## Research Overview

Documents a supervised natural language processing study investigating binary sentiment classification using a Long Short-Term Memory (LSTM) neural network implemented in PyTorch.

The objective of this research is to evaluate whether a compact recurrent neural architecture can effectively model short-form review text and generalize across multiple domains (Amazon, Yelp, IMDB).

The broader motivation includes:

* Understanding sequence modeling performance on small text corpora
* Evaluating embedding dimensionality tradeoffs
* Assessing generalization across heterogeneous review sources

---

## Dataset Description

Three labeled sentiment datasets were combined into a unified corpus:

* `amazon_cells_labelled.txt`
* `yelp_labelled.txt`
* `imdb_labelled.txt`

Each dataset consists of short review sentences labeled as:

* 0 → Negative sentiment
* 1 → Positive sentiment

Total dataset size:

* **3,000 labeled sentences**

Stratified splitting preserved class balance:

* Training: 2,400 samples (80%)
* Validation: 300 samples (10%)
* Test: 300 samples (10%)

---

## Text Preprocessing Pipeline

All preprocessing steps were implemented manually to ensure reproducibility and full control over the tokenization process.

### 1. Text Normalization

* Converted text to lowercase
* Removed non-word characters using regex tokenization

### 2. Vocabulary Construction

* Unique token count: **5,183 words**
* Tokens mapped to integer indices
* Vocabulary size including padding index: 5,184

### 3. Sequence Encoding

* Each sentence converted to integer sequence
* Maximum sequence length determined using the 95th percentile of review length distribution

  * `max_seq_len = 26`
* Sequences padded to uniform length

Final tensor shape:

* (3000, 26)

This design balances information retention with computational efficiency.

---

## Model Architecture

A compact LSTM-based classifier was implemented with the following structure:

* Embedding Layer:

  * `Embedding(5184, 8)`
* LSTM Layer:

  * Input size: 8
  * Hidden size: 64
  * Dropout: 0.3
  * Batch-first configuration
* Fully Connected Layer:

  * `Linear(64 → 1)`
* Sigmoid activation for probability output

Total trainable parameters: **60,481**

The model uses the final hidden state of the LSTM as the learned sentence representation for classification.

---

## Optimization Strategy

Loss Function:

* Binary Cross Entropy (BCELoss)

Optimizer:

* Adam
* Learning rate: 0.001

Training Configuration:

* Maximum 10 epochs
* Early stopping with patience = 2 (based on validation loss)

The best-performing model weights were checkpointed and restored prior to final evaluation.

---

## Experimental Results

Training terminated early at epoch 7 due to validation convergence.

### Test Performance

* Accuracy: **0.677**
* F1 Score: **0.676**

Results indicate moderate performance given:

* Small dataset size
* Limited embedding dimensionality
* Compact recurrent architecture

Performance reflects the tradeoff between model capacity and overfitting control.

---

## Model Persistence

The trained model weights are stored as:

`sentiment_lstm_model.pth`

Model artifacts are excluded from version control and can be regenerated using the training script.

---

## Research Contributions Demonstrated

* End-to-end NLP preprocessing without high-level wrappers
* Manual vocabulary construction and sequence padding
* LSTM-based sequence modeling in PyTorch
* Early stopping implementation for overfitting mitigation
* Multi-metric evaluation (Accuracy and F1)
* Controlled experimentation across heterogeneous text domains

---

## Future Work

Potential extensions include:

* Increasing embedding dimensionality
* Incorporating pretrained embeddings (GloVe, FastText)
* Replacing LSTM with GRU or Transformer encoder
* Domain-specific fine-tuning experiments
* Attention mechanism integration for interpretability

---

This repository represents a controlled investigation into recurrent neural network performance for short-form sentiment classification, emphasizing reproducibility, transparent preprocessing, and empirical evaluation.
