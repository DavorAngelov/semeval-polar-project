# Experiment Report: roberta-base

**Date:** 2026-02-18 17:53:58

## Technique Description


This experiment uses **roberta-base**, a transformer-based model pre-trained on large text corpora.

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
- **Accuracy:** 0.8125
- **Macro F1 Score:** 0.7986
- **Weighted F1 Score:** 0.8125

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8515
- Recall: 0.8515
- F1 Score: 0.8515
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.7458
- Recall: 0.7458
- F1 Score: 0.7458
- Support: 59

## Key Observations


- The model shows imbalanced performance across classes.
- Class 0 (non-polarized) F1: 0.8515
- Class 1 (polarized) F1: 0.7458
- The macro F1 score of 0.7986 indicates strong performance.


## Conclusion

This experiment achieved a macro F1 score of 0.7986, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

