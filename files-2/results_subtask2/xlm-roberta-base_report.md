# Experiment Report: xlm-roberta-base

**Date:** 2026-02-20 18:20:00

## Technique Description


This experiment uses **xlm-roberta-base**, a transformer-based model fine-tuned for multi-label classification.

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
- **Subset Accuracy (Exact Match):** 0.6562
- **Hamming Loss:** 0.0788
- **F1 Score (Micro):** 0.5772
- **F1 Score (Macro):** 0.1421
- **F1 Score (Weighted):** 0.4793
- **F1 Score (Samples):** 0.2240
- **Jaccard Score (Micro):** 0.4057
- **Jaccard Score (Macro):** 0.1103

### Per-Label Performance

#### political
- Precision: 0.6825
- Recall: 0.7414
- F1 Score: 0.7107
- Support: 58

#### racial/ethnic
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
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


- The model achieves a micro F1 score of 0.5772 (overall performance).
- The macro F1 score is 0.1421 (average across labels).
- Subset accuracy (exact match): 0.6562
- Hamming loss (fraction of wrong labels): 0.0788

**Per-Label Analysis:**
- Best performing label: **political** (F1: 0.7107)
- Most challenging label: **racial/ethnic** (F1: 0.0000)
- F1 score range: 0.0000 - 0.7107
- Average per-label F1: 0.1421

**Multi-Label Characteristics:**
- The model shows imbalanced performance across labels.
- Strong exact match performance indicates the model often predicts all labels correctly.


## Conclusion

This experiment achieved:
- Micro F1 score of 0.5772 (overall performance across all labels)
- Macro F1 score of 0.1421 (average performance per label)
- Subset accuracy of 0.6562 (exact match ratio)

The high subset accuracy indicates that the model frequently predicts the exact label combination correctly.

