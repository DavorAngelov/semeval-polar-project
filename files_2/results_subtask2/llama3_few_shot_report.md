# Experiment Report: llama3 - few_shot

**Date:** 2026-03-03 00:27:40

## Technique Description


This experiment uses **llama3** with few-shot in-context learning.

**Approach:**
- Provides 4 annotated examples in the prompt
- Examples demonstrate different polarization types
- Model learns from examples without parameter updates

**Prompt Strategy:**
Each prompt includes:
1. Task description
2. 4 diverse examples with explanations
3. Target text to classify
4. Output format

**Example Selection:**
- Single-label examples (Political, Racial/Ethnic, Religious)
- Multi-label example (Political + Other)
- Demonstrates reasoning process

**Advantages:**
- Better than zero-shot for nuanced tasks
- Examples guide model behavior
- Can handle edge cases shown in examples

**Limitations:**
- Longer prompts (higher token cost)
- Example selection impacts performance
- Still no gradient-based learning


## Results

### Overall Multi-Label Metrics
- **Subset Accuracy (Exact Match):** 0.1875
- **Hamming Loss:** 0.2338
- **F1 Score (Micro):** 0.4138
- **F1 Score (Macro):** 0.2647
- **F1 Score (Weighted):** 0.4546
- **F1 Score (Samples):** 0.2790
- **Jaccard Score (Micro):** 0.2609
- **Jaccard Score (Macro):** 0.1662

### Per-Label Performance

#### political
- Precision: 0.4050
- Recall: 0.8448
- F1 Score: 0.5475
- Support: 58

#### racial/ethnic
- Precision: 0.2667
- Recall: 0.8571
- F1 Score: 0.4068
- Support: 14

#### religious
- Precision: 0.1000
- Recall: 0.6000
- F1 Score: 0.1714
- Support: 5

#### gender/sexual
- Precision: 0.0833
- Recall: 0.3333
- F1 Score: 0.1333
- Support: 3

#### other
- Precision: 0.0400
- Recall: 0.1667
- F1 Score: 0.0645
- Support: 6

## Key Observations


**Strategy-Specific Observations:**

The **few_shot** approach achieved:
- Micro F1: 0.4138
- Macro F1: 0.2647
- Subset Accuracy: 0.1875

**Performance Characteristics:**

- Better guided by examples
- Learns from demonstrations
- More consistent with shown patterns
- Improvement over zero-shot expected


## Conclusion

This experiment achieved:
- Micro F1 score of 0.4138 (overall performance across all labels)
- Macro F1 score of 0.2647 (average performance per label)
- Subset accuracy of 0.1875 (exact match ratio)

The low subset accuracy indicates that the model rarely predicts the exact label combination correctly.

