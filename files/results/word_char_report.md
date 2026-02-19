# Experiment Report: Word + Character Features

**Date:** 2026-02-18 17:54:46

## Technique Description


This experiment **combines word and character n-grams**:

**Features:**
1. **Word n-grams**: Unigrams and bigrams (10,000 features)
2. **Character n-grams**: 3-4 character sequences (5,000 features)
3. Total: 15,000 combined features

**Combination Strategy:**
- Horizontal stacking of feature matrices
- Both feature types contribute to final representation

**Rationale:**
Word features capture semantic meaning, while character features handle spelling variations.


## Results

### Overall Performance
- **Accuracy:** 0.7562
- **Macro F1 Score:** 0.7440
- **Weighted F1 Score:** 0.7587

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8298
- Recall: 0.7723
- F1 Score: 0.8000
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6515
- Recall: 0.7288
- F1 Score: 0.6880
- Support: 59

## Key Observations


- Combining word and character features leverages both semantic and sub-word patterns.
- Total feature space: 15,000 dimensions
- Macro F1: 0.7440
- Significant improvement over character-only features


## Conclusion

This experiment achieved a macro F1 score of 0.7440, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

