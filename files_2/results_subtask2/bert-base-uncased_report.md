# Experiment Report: bert-base-uncased

**Date:** 2026-02-20 18:23:18

## Technique Description


This experiment uses **bert-base-uncased**, a transformer-based model fine-tuned for multi-label classification.

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
- **Subset Accuracy (Exact Match):** 0.6937
- **Hamming Loss:** 0.0775
- **F1 Score (Micro):** 0.5571
- **F1 Score (Macro):** 0.2879
- **F1 Score (Weighted):** 0.5257
- **F1 Score (Samples):** 0.1800
- **Jaccard Score (Micro):** 0.3861
- **Jaccard Score (Macro):** 0.2045

### Per-Label Performance

#### political
- Precision: 0.7955
- Recall: 0.6034
- F1 Score: 0.6863
- Support: 58

#### racial/ethnic
- Precision: 0.2500
- Recall: 0.1429
- F1 Score: 0.1818
- Support: 14

#### religious
- Precision: 1.0000
- Recall: 0.4000
- F1 Score: 0.5714
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


- The model achieves a micro F1 score of 0.5571 (overall performance).
- The macro F1 score is 0.2879 (average across labels).
- Subset accuracy (exact match): 0.6937
- Hamming loss (fraction of wrong labels): 0.0775

**Per-Label Analysis:**
- Best performing label: **political** (F1: 0.6863)
- Most challenging label: **gender/sexual** (F1: 0.0000)
- F1 score range: 0.0000 - 0.6863
- Average per-label F1: 0.2879

**Multi-Label Characteristics:**
- The model shows imbalanced performance across labels.
- Strong exact match performance indicates the model often predicts all labels correctly.


## Conclusion

This experiment achieved:
- Micro F1 score of 0.5571 (overall performance across all labels)
- Macro F1 score of 0.2879 (average performance per label)
- Subset accuracy of 0.6937 (exact match ratio)

The high subset accuracy indicates that the model frequently predicts the exact label combination correctly.

