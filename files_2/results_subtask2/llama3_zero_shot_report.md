# Experiment Report: llama3 - zero_shot

**Date:** 2026-03-03 00:25:31

## Technique Description


This experiment uses **llama3** with zero-shot prompting for multi-label polarization type classification.

**Approach:**
- No training or fine-tuning required
- Model relies purely on pre-trained knowledge
- Single-turn inference per example

**Prompt Strategy:**
Direct instruction with category definitions and output format specification. The model receives:
1. Task description
2. Category definitions (Political, Racial/Ethnic, Religious, Gender/Sexual, Other)
3. Input text
4. Output format instructions (JSON)

**Advantages:**
- No training data required
- Fast to deploy
- Leverages model's world knowledge

**Limitations:**
- May struggle with ambiguous cases
- Dependent on prompt quality
- No task-specific adaptation


## Results

### Overall Multi-Label Metrics
- **Subset Accuracy (Exact Match):** 0.2812
- **Hamming Loss:** 0.1512
- **F1 Score (Micro):** 0.5061
- **F1 Score (Macro):** 0.3248
- **F1 Score (Weighted):** 0.4888
- **F1 Score (Samples):** 0.3150
- **Jaccard Score (Micro):** 0.3388
- **Jaccard Score (Macro):** 0.2231

### Per-Label Performance

#### political
- Precision: 0.3841
- Recall: 0.9138
- F1 Score: 0.5408
- Support: 58

#### racial/ethnic
- Precision: 0.7000
- Recall: 0.5000
- F1 Score: 0.5833
- Support: 14

#### religious
- Precision: 0.6667
- Recall: 0.4000
- F1 Score: 0.5000
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


**Strategy-Specific Observations:**

The **zero_shot** approach achieved:
- Micro F1: 0.5061
- Macro F1: 0.3248
- Subset Accuracy: 0.2812

**Performance Characteristics:**

- Fast inference (single-turn)
- Relies on model's prior knowledge
- May struggle with subtle distinctions
- Good baseline for comparison


## Conclusion

This experiment achieved:
- Micro F1 score of 0.5061 (overall performance across all labels)
- Macro F1 score of 0.3248 (average performance per label)
- Subset accuracy of 0.2812 (exact match ratio)

The low subset accuracy indicates that the model rarely predicts the exact label combination correctly.

