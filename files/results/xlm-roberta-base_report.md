# Experiment Report: xlm-roberta-base

**Date:** 2026-02-18 17:46:04

## Technique Description


This experiment uses **xlm-roberta-base**, a transformer-based model pre-trained on large text corpora.

**Model Characteristics:**
- Architecture: Transformer encoder
- Pre-training: Masked language modeling
- Maximum sequence length: 128 tokens
- Fine-tuning approach: Full model fine-tuning

**Training Configuration:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3 (with early stopping)
- Optimizer: AdamW
- Weight decay: 0.01


## Results

### Overall Performance
- **Accuracy:** 0.7688
- **Macro F1 Score:** 0.7525
- **Weighted F1 Score:** 0.7691

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8200
- Recall: 0.8119
- F1 Score: 0.8159
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6833
- Recall: 0.6949
- F1 Score: 0.6891
- Support: 59

## Key Observations


- The model shows imbalanced performance across classes.
- Class 0 (non-polarized) F1: 0.8159
- Class 1 (polarized) F1: 0.6891
- The macro F1 score of 0.7525 indicates strong performance.


## Conclusion

This experiment achieved a macro F1 score of 0.7525, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

