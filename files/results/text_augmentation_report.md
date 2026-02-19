# Experiment Report: Text Augmentation

**Date:** 2026-02-18 17:55:07

## Technique Description


This experiment uses **Text Augmentation** to expand the minority class:

**Augmentation Techniques:**
1. **Synonym Replacement**: Replace words with synonyms
2. **Random Deletion**: Randomly remove words (10% probability)
3. **Random Swap**: Swap positions of random word pairs

**Process:**
- Identify minority class samples
- Generate 2 augmented versions per sample
- Apply random augmentation technique for variation
- Combine with original data

**Advantage:**
Creates natural text variations that maintain semantic meaning while adding diversity.


## Results

### Overall Performance
- **Accuracy:** 0.7312
- **Macro F1 Score:** 0.7161
- **Weighted F1 Score:** 0.7333

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: 0.8021
- Recall: 0.7624
- F1 Score: 0.7817
- Support: 101

#### Class 1 (Polarized)
- Precision: 0.6250
- Recall: 0.6780
- F1 Score: 0.6504
- Support: 59

## Key Observations


- Text augmentation creates linguistic variations of minority samples.
- Augmentation factor: 2x (doubles minority class size)
- Macro F1: 0.7161
- Text augmentation provides natural sample diversity
- More sophisticated augmentation (e.g., back-translation, paraphrasing) could improve results


## Conclusion

This experiment achieved a macro F1 score of 0.7161, which exceeds the 0.70 threshold commonly considered strong performance for binary classification tasks.

