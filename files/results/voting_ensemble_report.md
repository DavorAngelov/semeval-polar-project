# Experiment Report: Voting Ensemble

**Date:** 2026-02-18 17:54:08

## Technique Description


This experiment uses a **Soft Voting Ensemble** that combines predictions from multiple classifiers:

**Base Classifiers:**
1. **Logistic Regression** with TF-IDF (10,000 features, 1-2 grams)
2. **Random Forest** (100 trees) with TF-IDF (5,000 features)
3. **Gradient Boosting** (100 estimators) with TF-IDF (5,000 features)

**Voting Strategy:**
- Soft voting: Averages predicted probabilities from all models
- Each model votes based on its confidence
- Final prediction is the class with highest average probability

**Rationale:**
Ensemble methods reduce overfitting and improve generalization by combining diverse models.


## Results

### Overall Performance
- **Accuracy:** 0.7063
- **Macro F1 Score:** 0.6730
- **Weighted F1 Score:** 0.7004

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.7455
- Recall: 0.8119
- F1 Score: 0.7773
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6200
- Recall: 0.5254
- F1 Score: 0.5688
- Support: 59

## Key Observations


- The ensemble combines the strengths of linear (LogisticRegression), tree-based (RandomForest), and boosting (GradientBoosting) approaches.
- Soft voting leverages probability estimates for more nuanced predictions.
- Macro F1 score: 0.6730
- The ensemble shows improved robustness to individual models.
- Class balance: Some imbalance detected


## Conclusion

This experiment achieved a macro F1 score of 0.6730, which falls below the 0.70 threshold commonly considered strong performance for binary classification tasks.

