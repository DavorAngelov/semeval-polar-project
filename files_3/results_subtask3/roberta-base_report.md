# Experiment Report: roberta-base

**Date:** 2026-02-21 22:25:07

## Technique Description


This experiment uses **roberta-base**, a transformer-based model fine-tuned for multi-label manifestation classification.

**Task:** Identify how polarization is expressed through 6 manifestation types:
1. **Stereotype** - Generalizes characteristics to all group members
2. **Vilification** - Defames or demonizes groups through biased framing
3. **Dehumanization** - Strips individuals of human qualities
4. **Extreme Language** - Uses absolutist terms and dichotomous framing
5. **Lack of Empathy** - Shows no understanding for others' perspectives
6. **Invalidation** - Denies or rejects identity or existence of groups

**Model Configuration:**
- Architecture: Transformer encoder with multi-label classification head
- Pre-training: Masked language modeling
- Maximum sequence length: 128 tokens
- Fine-tuning: Full model fine-tuning

**Multi-Label Setup:**
- Problem type: multi_label_classification
- Loss function: Binary Cross-Entropy with Logits (per manifestation)
- Output activation: Sigmoid (independent predictions per manifestation)
- Decision threshold: 0.5 per manifestation

**Training Configuration:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3 (with early stopping, patience=2)
- Optimizer: AdamW
- Weight decay: 0.01
- Metric for best model: Macro F1


## Results

### Overall Multi-Label Metrics
- **Subset Accuracy (Exact Match):** 0.6312
- **Hamming Loss:** 0.1562
- **F1 Score (Micro):** 0.4141
- **F1 Score (Macro):** 0.2729
- **F1 Score (Weighted):** 0.3416
- **F1 Score (Samples):** 0.1204
- **Jaccard Score (Micro):** 0.2611
- **Jaccard Score (Macro):** 0.1863

### Per-Label Performance

#### stereotype
- Precision: 0.5000
- Recall: 0.2500
- F1 Score: 0.3333
- Support: 24

#### vilification
- Precision: 0.6410
- Recall: 0.6410
- F1 Score: 0.6410
- Support: 39

#### dehumanization
- Precision: 0.5000
- Recall: 0.0526
- F1 Score: 0.0952
- Support: 19

#### extreme_language
- Precision: 0.6364
- Recall: 0.5122
- F1 Score: 0.5676
- Support: 41

#### lack_of_empathy
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
- Support: 18

#### invalidation
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
- Support: 29

## Key Observations


### Overall Performance
- **Micro F1 Score:** 0.4141 (overall performance across all manifestations)
- **Macro F1 Score:** 0.2729 (average performance per manifestation)
- **Subset Accuracy:** 0.6312 (exact match - all manifestations correct)
- **Hamming Loss:** 0.1562 (fraction of incorrect manifestation predictions)
- **Jaccard Score (Micro):** 0.2611 (overlap between predictions and truth)

### Per-Manifestation Analysis

**Best Detected Manifestation:** Vilification
- F1 Score: 0.6410
- Precision: 0.6410
- Recall: 0.6410
- This manifestation was easiest for the model to identify

**Most Challenging Manifestation:** Lack Of Empathy
- F1 Score: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- This manifestation requires improvement

**Performance Range:**
- F1 scores range from 0.0000 to 0.6410
- Standard deviation: 0.2603
- Performance is highly imbalanced across manifestations

### Model Characteristics

**High Precision Manifestations** (>0.70): None
- When the model predicts these, it's usually correct
- Low false positive rate

**High Recall Manifestations** (>0.70): None
- Model successfully identifies most instances
- Low false negative rate

### Multi-Label Behavior
- Average manifestations per sample: 0.15
- The model shows strong exact matching ability
- Moderate label error rate

### Interpretability Notes
- **Stereotype** often co-occurs with extreme language
- **Vilification** may overlap with dehumanization in severe cases
- **Lack of empathy** is abstract and harder to detect than explicit manifestations
- **Invalidation** requires understanding of identity denial patterns


## Conclusion

This experiment achieved:
- Micro F1 score of 0.4141 (overall performance across all labels)
- Macro F1 score of 0.2729 (average performance per label)
- Subset accuracy of 0.6312 (exact match ratio)

The high subset accuracy indicates that the model frequently predicts the exact label combination correctly.

