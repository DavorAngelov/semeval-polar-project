# Experiment Report: Statistical Features

**Date:** 2026-02-18 17:54:49

## Technique Description


This experiment adds **statistical text features** to TF-IDF:

**Statistical Features:**
1. Text length (characters)
2. Word count
3. Average word length
4. Exclamation mark count
5. Question mark count
6. Uppercase letter ratio
7. Punctuation ratio

**Rationale:**
Polarizing content may have distinctive stylistic patterns (e.g., excessive punctuation, capitalization).


## Results

### Overall Performance
- **Accuracy:** 0.7250
- **Macro F1 Score:** 0.7163
- **Weighted F1 Score:** 0.7293

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8276
- Recall: 0.7129
- F1 Score: 0.7660
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6027
- Recall: 0.7458
- F1 Score: 0.6667
- Support: 59

## Key Observations


- Statistical features capture writing style and emotional intensity.
- Features like uppercase ratio and punctuation can indicate strong opinions.
- Macro F1: 0.7163
- Statistical features provide complementary information


## Conclusion

This experiment achieved a macro F1 score of 0.7163, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

