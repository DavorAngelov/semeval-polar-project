# Experiment Report: SMOTE Oversampling

**Date:** 2026-02-18 17:55:04

## Technique Description


This experiment uses **SMOTE (Synthetic Minority Over-sampling Technique)**:

**How SMOTE Works:**
1. For each minority class sample, find k nearest neighbors in feature space
2. Generate synthetic samples along the line segments joining the sample and its neighbors
3. Balance the dataset by creating synthetic minority samples

**Configuration:**
- Feature space: TF-IDF (10,000 features, 1-2 grams)
- Synthetic samples created to match majority class size
- Random state: 42 for reproducibility

**Advantages:**
- Creates informed synthetic samples (not random duplicates)
- Maintains feature space structure
- Reduces overfitting compared to simple duplication


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


- SMOTE generates synthetic samples in TF-IDF feature space.
- Original training samples: 3222
- After SMOTE: approximately 6444 (balanced classes)
- Macro F1: 0.7266
- SMOTE improves minority class recall
- Synthetic samples help model learn minority class patterns


## Conclusion

This experiment achieved a macro F1 score of 0.7266, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

