# Polarization Detection - Master Report

**Generated:** 2026-02-18 18:06:17

## Executive Summary

Total experiments conducted: **22**

### Top 5 Models by Macro F1

| Model                    | Accuracy | Macro F1 | Weighted F1 | Class 0 F1 | Class 1 F1 | Timestamp           |
| ------------------------ | -------- | -------- | ----------- | ---------- | ---------- | ------------------- |
| roberta-base             | 0.8125   | 0.7986   | 0.8125      | 0.8515     | 0.7458     | 2026-02-18 17:53:58 |
| distilbert-base-uncased  | 0.8063   | 0.7824   | 0.8013      | 0.8545     | 0.7103     | 2026-02-18 17:51:05 |
| bert-base-uncased        | 0.7812   | 0.7565   | 0.7769      | 0.8341     | 0.6789     | 2026-02-18 17:49:15 |
| xlm-roberta-base         | 0.7688   | 0.7525   | 0.7691      | 0.8159     | 0.6891     | 2026-02-18 17:46:04 |
| Word + Character N-grams | 0.7562   | 0.7440   | 0.7587      | 0.8000     | 0.6880     | 2026-02-18 17:54:46 |

### Top 3 Models

**Best Model:** roberta-base

- Macro F1: 0.7986
- Accuracy: 0.8125

**Second Best:** distilbert-base-uncased

- Macro F1: 0.7824
- Accuracy: 0.8063

**Third Best:** bert-base-uncased

- Macro F1: 0.7565
- Accuracy: 0.7812

## All Models Ranked by Performance

| Model                         | Accuracy | Macro F1 | Weighted F1 | Class 0 F1 | Class 1 F1 | Timestamp           |
| ----------------------------- | -------- | -------- | ----------- | ---------- | ---------- | ------------------- |
| roberta-base                  | 0.8125   | 0.7986   | 0.8125      | 0.8515     | 0.7458     | 2026-02-18 17:53:58 |
| distilbert-base-uncased       | 0.8063   | 0.7824   | 0.8013      | 0.8545     | 0.7103     | 2026-02-18 17:51:05 |
| bert-base-uncased             | 0.7812   | 0.7565   | 0.7769      | 0.8341     | 0.6789     | 2026-02-18 17:49:15 |
| xlm-roberta-base              | 0.7688   | 0.7525   | 0.7691      | 0.8159     | 0.6891     | 2026-02-18 17:46:04 |
| Word + Character N-grams      | 0.7562   | 0.7440   | 0.7587      | 0.8000     | 0.6880     | 2026-02-18 17:54:46 |
| All Features Combined         | 0.7562   | 0.7440   | 0.7587      | 0.8000     | 0.6880     | 2026-02-18 17:54:52 |
| Weighted Ensemble             | 0.7500   | 0.7275   | 0.7480      | 0.8058     | 0.6491     | 2026-02-18 17:54:36 |
| TF-IDF + Sentiment Features   | 0.7375   | 0.7266   | 0.7409      | 0.7812     | 0.6719     | 2026-02-18 17:54:50 |
| Character N-grams (3-5)       | 0.7375   | 0.7266   | 0.7409      | 0.7812     | 0.6719     | 2026-02-18 17:54:40 |
| SMOTE Oversampling            | 0.7375   | 0.7266   | 0.7409      | 0.7812     | 0.6719     | 2026-02-18 17:55:04 |
| Random Oversampling           | 0.7375   | 0.7266   | 0.7409      | 0.7812     | 0.6719     | 2026-02-18 17:55:06 |
| Tuned Logistic Regression     | 0.7312   | 0.7208   | 0.7350      | 0.7749     | 0.6667     | 2026-02-18 17:55:13 |
| Tuned Random Forest           | 0.7312   | 0.7177   | 0.7340      | 0.7795     | 0.6560     | 2026-02-18 17:55:18 |
| TF-IDF + Statistical Features | 0.7250   | 0.7163   | 0.7293      | 0.7660     | 0.6667     | 2026-02-18 17:54:49 |
| Text Augmentation             | 0.7312   | 0.7161   | 0.7333      | 0.7817     | 0.6504     | 2026-02-18 17:55:07 |
| Class Weight Balanced         | 0.7250   | 0.7150   | 0.7290      | 0.7684     | 0.6615     | 2026-02-18 17:55:03 |
| Random Undersampling          | 0.7188   | 0.7105   | 0.7233      | 0.7594     | 0.6617     | 2026-02-18 17:55:07 |
| Tuned SVM                     | 0.7250   | 0.7086   | 0.7267      | 0.7778     | 0.6393     | 2026-02-18 17:55:25 |
| Stacking Ensemble             | 0.7188   | 0.6896   | 0.7146      | 0.7847     | 0.5946     | 2026-02-18 17:54:30 |
| Baseline (Imbalanced)         | 0.7312   | 0.6890   | 0.7191      | 0.8037     | 0.5743     | 2026-02-18 17:55:00 |
| Tuned Gradient Boosting       | 0.7250   | 0.6764   | 0.7093      | 0.8018     | 0.5510     | 2026-02-18 17:55:31 |
| Voting Ensemble (Soft)        | 0.7063   | 0.6730   | 0.7004      | 0.7773     | 0.5688     | 2026-02-18 17:54:08 |

## Key Insights

### Best Performing Techniques

1. **roberta-base**
   - Achieved highest macro F1 of 0.7986
   - Class 0 F1: 0.8515
   - Class 1 F1: 0.7458
   - Balance score: 0.1057

### Performance Distribution

- Models with Macro F1 > 0.70: 18
- Models with Macro F1 > 0.65: 22
- Models with Macro F1 > 0.60: 22

### Class Balance Analysis

Average class F1 difference: 0.1379

Models with good class balance (F1 difference < 0.05):
0

## Recommendations

### For Production Deployment

**Recommended Model:** roberta-base

**Rationale:**

- Highest macro F1 score
- Acceptable performance across classes
- Strong generalization capability

---
