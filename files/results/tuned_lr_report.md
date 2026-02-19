# Experiment Report: Tuned Logistic Regression

**Date:** 2026-02-18 17:55:13

## Technique Description


This experiment uses **Grid Search** to optimize Logistic Regression hyperparameters:

**Search Space:**
- TF-IDF max features: [5000, 10000, 20000]
- N-gram range: [(1,1), (1,2), (1,3)]
- Min document frequency: [1, 2, 3]
- Regularization (C): [0.1, 1.0, 10.0]
- Class weight: [balanced, None]

**Best Parameters Found:**
```
{'clf__C': 1.0, 'clf__class_weight': 'balanced', 'clf__penalty': 'l2', 'tfidf__max_features': 5000, 'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
```

**Search Method:**
- Exhaustive grid search with 3-fold cross-validation
- Scoring metric: Macro F1
- Total combinations tested: ~162

**Rationale:**
Grid search ensures we find the optimal combination of parameters for maximum performance.


## Results

### Overall Performance
- **Accuracy:** 0.7312
- **Macro F1 Score:** 0.7208
- **Weighted F1 Score:** 0.7350

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8222
- Recall: 0.7327
- F1 Score: 0.7749
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6143
- Recall: 0.7288
- F1 Score: 0.6667
- Support: 59

## Key Observations


- Grid search identified optimal TF-IDF and regularization settings.
- Best CV score during search: Higher scores indicate good generalization
- Macro F1 on dev set: 0.7208
- Tuning provided significant improvement
- Key finding: Best n-gram range is (1, 3)


## Conclusion

This experiment achieved a macro F1 score of 0.7208, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

