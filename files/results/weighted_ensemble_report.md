# Experiment Report: Weighted Ensemble

**Date:** 2026-02-18 17:54:36

## Technique Description


This experiment uses a **Weighted Ensemble** with manual weight tuning:

**Models and Weights:**
1. **Logistic Regression** (40%) - High-dimensional TF-IDF (50k features, 1-3 grams)
2. **Random Forest** (30%) - 200 trees with balanced class weights
3. **Gradient Boosting** (30%) - 150 estimators

**Combination Strategy:**
- Weighted average of probability predictions
- Weights are set based on expected model strengths
- Linear models get higher weight for interpretability

**Feature Engineering:**
- Different feature sets for each model
- Varying n-gram ranges to capture different patterns


## Results

### Overall Performance
- **Accuracy:** 0.7500
- **Macro F1 Score:** 0.7275
- **Weighted F1 Score:** 0.7480

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.7905
- Recall: 0.8218
- F1 Score: 0.8058
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6727
- Recall: 0.6271
- F1 Score: 0.6491
- Support: 59

## Key Observations


- Manual weighting allows expert knowledge to guide ensemble.
- Logistic Regression receives higher weight (40%) for its stability.
- Macro F1 score: 0.7275
- The weighted approach provides competitive performance.
- Weights can be further optimized using validation data.


## Conclusion

This experiment achieved a macro F1 score of 0.7275, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

