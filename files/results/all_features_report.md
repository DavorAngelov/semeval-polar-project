# Experiment Report: All Features Combined

**Date:** 2026-02-18 17:54:52

## Technique Description


This experiment **combines all feature types**:

**Feature Components:**
1. Word TF-IDF (1-2 grams, 8,000 features)
2. Character TF-IDF (3-4 grams, 3,000 features)
3. Statistical features (7 dimensions)
4. Sentiment features (4 dimensions)

**Total Feature Space:**
- Approximately 11,011 dimensions
- Comprehensive representation of text

**Advantage:**
Captures multiple aspects: semantics, style, patterns, and sentiment.


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


- Comprehensive feature engineering combining multiple signal types.
- Rich representation with ~11k features.
- Macro F1: 0.7440
- Combined features achieve best performance
- Risk of overfitting with high-dimensional features should be monitored.


## Conclusion

This experiment achieved a macro F1 score of 0.7440, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

