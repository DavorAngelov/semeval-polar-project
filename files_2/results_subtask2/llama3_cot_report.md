# Experiment Report: llama3 - cot

**Date:** 2026-03-03 00:44:21

## Technique Description


This experiment uses **llama3** with Chain-of-Thought (CoT) prompting.

**Approach:**
- Prompts model to reason step-by-step
- Encourages explicit intermediate reasoning
- Mirrors human analytical process

**Reasoning Steps:**
1. Identify key entities mentioned
2. Analyze tone and intent
3. Map entities to categories
4. Provide final classification

**Advantages:**
- Improves reasoning on complex cases
- Provides interpretable decision process
- Reduces impulsive/incorrect classifications

**Limitations:**
- Longer generation time
- Requires parsing reasoning from final answer
- May over-explain simple cases


## Results

### Overall Multi-Label Metrics
- **Subset Accuracy (Exact Match):** 0.1313
- **Hamming Loss:** 0.2587
- **F1 Score (Micro):** 0.3858
- **F1 Score (Macro):** 0.2956
- **F1 Score (Weighted):** 0.4706
- **F1 Score (Samples):** 0.2492
- **Jaccard Score (Micro):** 0.2390
- **Jaccard Score (Macro):** 0.1929

### Per-Label Performance

#### political
- Precision: 0.3923
- Recall: 0.8793
- F1 Score: 0.5426
- Support: 58

#### racial/ethnic
- Precision: 0.3750
- Recall: 0.6429
- F1 Score: 0.4737
- Support: 14

#### religious
- Precision: 0.4000
- Recall: 0.4000
- F1 Score: 0.4000
- Support: 5

#### gender/sexual
- Precision: 0.0000
- Recall: 0.0000
- F1 Score: 0.0000
- Support: 3

#### other
- Precision: 0.0330
- Recall: 0.5000
- F1 Score: 0.0619
- Support: 6

## Key Observations


**Strategy-Specific Observations:**

The **cot** approach achieved:
- Micro F1: 0.3858
- Macro F1: 0.2956
- Subset Accuracy: 0.1313

**Performance Characteristics:**

- Shows reasoning process
- Better for complex cases
- May improve recall on minority labels
- Interpretable decision path


## Conclusion

This experiment achieved:
- Micro F1 score of 0.3858 (overall performance across all labels)
- Macro F1 score of 0.2956 (average performance per label)
- Subset accuracy of 0.1313 (exact match ratio)

The low subset accuracy indicates that the model rarely predicts the exact label combination correctly.

