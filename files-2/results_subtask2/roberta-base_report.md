# Experiment Report: roberta-base

**Date:** 2026-02-20 18:28:09

## Technique Description


This experiment uses **roberta-base**, a transformer-based model fine-tuned for multi-label classification.

**Model Characteristics:**
- Architecture: Transformer encoder
- Pre-training: Masked language modeling
- Maximum sequence length: 128 tokens
- Fine-tuning approach: Full model fine-tuning with multi-label classification head

**Multi-Label Configuration:**
- Problem type: multi_label_classification
- Loss function: Binary Cross-Entropy with Logits
- Output: Sigmoid activation per label
- Threshold: 0.5 for binary decision

**Training Configuration:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3 (with early stopping)
- Optimizer: AdamW
- Weight decay: 0.01


## Results

### Overall Multi-Label Metrics
- **Subset Accuracy (Exact Match):** 0.6875
- **Hamming Loss:** 0.0813
- **F1 Score (Micro):** 0.5578
- **F1 Score (Macro):** 0.1769
- **F1 Score (Weighted):** 0.5035
- **F1 Score (Samples):** 0.2040
- **Jaccard Score (Micro):** 0.3868
- **Jaccard Score (Macro):** 0.1283

### Per-Label Performance

#### political
- Precision: 0.7358
- Recall: 0.6724
- F1 Score: 0.7027
- Support: 58

#### racial/ethnic
- Precision: 0.2500
- Recall: 0.1429
- F1 Score: 0.1818
- Support: 14

#### religious
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
- Support: 5

#### gender/sexual
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
- Support: 3

#### other
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
- Support: 6

## Key Observations


- The model achieves a micro F1 score of 0.5578 (overall performance).
- The macro F1 score is 0.1769 (average across labels).
- Subset accuracy (exact match): 0.6875
- Hamming loss (fraction of wrong labels): 0.0813

**Per-Label Analysis:**
- Best performing label: **political** (F1: 0.7027)
- Most challenging label: **religious** (F1: 0.0000)
- F1 score range: 0.0000 - 0.7027
- Average per-label F1: 0.1769

**Multi-Label Characteristics:**
- The model shows imbalanced performance across labels.
- Strong exact match performance indicates the model often predicts all labels correctly.


## Conclusion

This experiment achieved:
- Micro F1 score of 0.5578 (overall performance across all labels)
- Macro F1 score of 0.1769 (average performance per label)
- Subset accuracy of 0.6875 (exact match ratio)

The high subset accuracy indicates that the model frequently predicts the exact label combination correctly.

