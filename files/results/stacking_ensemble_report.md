# Experiment Report: Stacking Ensemble

**Date:** 2026-02-18 17:54:30

## Technique Description


This experiment uses a **Stacking Ensemble** with a meta-learner:

**Base Learners (Level 0):**
1. **Logistic Regression** with TF-IDF
2. **Random Forest** with TF-IDF
3. **Support Vector Machine (SVM)** with TF-IDF

**Meta-Learner (Level 1):**
- Logistic Regression trained on base learner predictions
- Uses 5-fold cross-validation to generate meta-features

**How It Works:**
1. Base learners make predictions on training data (via cross-validation)
2. Meta-learner learns to optimally combine these predictions
3. Final model uses both base predictions and meta-learner

**Advantage:**
The meta-learner can learn which base models to trust for different types of inputs.


## Results

### Overall Performance
- **Accuracy:** 0.7188
- **Macro F1 Score:** 0.6896
- **Weighted F1 Score:** 0.7146

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.7593
- Recall: 0.8119
- F1 Score: 0.7847
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6346
- Recall: 0.5593
- F1 Score: 0.5946
- Support: 59

## Key Observations


- Stacking allows the meta-learner to adaptively weight base models.
- The meta-learner can identify when specific models are more reliable.
- Macro F1 score: 0.6896
- The stacking approach shows improved performance over simple voting.
- Cross-validation during training helps prevent overfitting.


## Conclusion

This experiment achieved a macro F1 score of 0.6896, which falls below the 0.70 threshold commonly considered strong performance for binary classification tasks.

