# Experiment Report: Sentiment Features

**Date:** 2026-02-18 17:54:50

## Technique Description


This experiment adds **sentiment-based features** to TF-IDF:

**Sentiment Features:**
1. Positive word count
2. Negative word count
3. Polarizing word count (absolute terms like "always", "never")
4. Sentiment ratio (positive - negative)

**Hypothesis:**
Polarizing content may use more extreme sentiment language and absolute statements.


## Results

### Overall Performance
- **Accuracy:** 0.7375
- **Macro F1 Score:** 0.7266
- **Weighted F1 Score:** 0.7409

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8242
- Recall: 0.7426
- F1 Score: 0.7812
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6232
- Recall: 0.7288
- F1 Score: 0.6719
- Support: 59

## Key Observations


- Sentiment features capture emotional tone and extreme language.
- Polarizing words (always, never, must) may be indicative of polarization.
- Macro F1: 0.7266
- Sentiment features show potential for polarization detection


## Conclusion

This experiment achieved a macro F1 score of 0.7266, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

