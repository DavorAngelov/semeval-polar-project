# Experiment Report: Character N-grams

**Date:** 2026-02-18 17:54:40

## Technique Description


This experiment uses **Character N-grams** for feature extraction:

**Features:**
- Character-level n-grams (3-5 characters)
- Max features: 10,000
- TF-IDF weighting

**Advantages:**
- Captures misspellings and informal language
- Language-agnostic patterns
- Robust to word variations

**Example:**
Text: "hello" -> ["hel", "ell", "llo", "hell", "ello", "hello"]


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


- Character n-grams capture sub-word patterns useful for social media text.
- Particularly effective for handling typos and informal spellings.
- Macro F1: 0.7266
- Shows promise for noisy text data


## Conclusion

This experiment achieved a macro F1 score of 0.7266, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

