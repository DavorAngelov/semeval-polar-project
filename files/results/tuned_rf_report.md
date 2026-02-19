# Experiment Report: Tuned Random Forest

**Date:** 2026-02-18 17:55:18

## Technique Description


This experiment uses **Random Search** to optimize Random Forest hyperparameters:

**Search Space:**
- TF-IDF max features: [3000, 5000, 8000, 10000]
- N-gram range: [(1,1), (1,2)]
- Number of trees: [50, 100, 150, 200]
- Max depth: [None, 10, 20, 30, 40]
- Min samples split: [2, 5, 10]
- Min samples leaf: [1, 2, 4]
- Class weight: [balanced, balanced_subsample, None]

**Best Parameters Found:**
```
{'tfidf__ngram_range': (1, 1), 'tfidf__max_features': 5000, 'clf__n_estimators': 150, 'clf__min_samples_split': 10, 'clf__min_samples_leaf': 2, 'clf__max_depth': 10, 'clf__class_weight': 'balanced'}
```

**Search Method:**
- Random sampling of 20 combinations
- 3-fold cross-validation
- Scoring metric: Macro F1

**Advantage:**
Random search is more efficient than grid search for large parameter spaces.


## Results

### Overall Performance
- **Accuracy:** 0.7312
- **Macro F1 Score:** 0.7177
- **Weighted F1 Score:** 0.7340

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8085
- Recall: 0.7525
- F1 Score: 0.7795
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6212
- Recall: 0.6949
- F1 Score: 0.6560
- Support: 59

## Key Observations


- Random search efficiently explored the parameter space.
- Tested 20 random combinations vs. thousands for complete grid
- Macro F1 on dev set: 0.7177
- Best n_estimators: 150
- Best max_depth: 10
- Shallower trees prevent overfitting


## Conclusion

This experiment achieved a macro F1 score of 0.7177, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

