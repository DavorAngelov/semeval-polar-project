"""
Experiment 2: Ensemble Methods
Combining multiple models for improved performance.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


from preprocessing import load_and_preprocess_data
from evaluation_utils import (
    evaluate_model, 
    plot_confusion_matrix, 
    save_results,
    create_experiment_report
)


def create_tfidf_feature_extractor(max_features=10000, ngram_range=(1, 2)):
    """TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95
    )


def train_voting_ensemble(train_df, dev_df):
    """
    Train a voting ensemble combining multiple classifiers.
  
    Returns tuple: (model, predictions)        
    """
    print("\n" + "="*60)
    print("Training Voting Ensemble")
    print("="*60 + "\n")
    
    # base estimators
    estimators = [
        ('lr', Pipeline([
            ('tfidf', create_tfidf_feature_extractor(max_features=10000)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])),
        ('rf', Pipeline([
            ('tfidf', create_tfidf_feature_extractor(max_features=5000)),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])),
        ('gb', Pipeline([
            ('tfidf', create_tfidf_feature_extractor(max_features=5000)),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]))
    ]
    
    #  voting classifier
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    
    
    print("Training individual models and ensemble...")
    voting_clf.fit(train_df['text'], train_df['polarization'])
    
    
    dev_preds = voting_clf.predict(dev_df['text'])
    
    return voting_clf, dev_preds


def train_stacking_ensemble(train_df, dev_df):
    """
    Train a stacking ensemble with a meta-learner.
        
    Returns tuple: (model, predictions)
    """
    print("\n" + "="*60)
    print("Training Stacking Ensemble")
    print("="*60 + "\n")
    
    #  base estimators
    estimators = [
        ('lr', Pipeline([
            ('tfidf', create_tfidf_feature_extractor(max_features=10000)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])),
        ('rf', Pipeline([
            ('tfidf', create_tfidf_feature_extractor(max_features=5000)),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])),
        ('svm', Pipeline([
            ('tfidf', create_tfidf_feature_extractor(max_features=5000)),
            ('clf', SVC(probability=True, class_weight='balanced', random_state=42))
        ]))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(max_iter=1000)
    
    #  stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5
    )
    
    
    print("Training base models and meta-learner...")
    stacking_clf.fit(train_df['text'], train_df['polarization'])
    
   
    dev_preds = stacking_clf.predict(dev_df['text'])
    
    return stacking_clf, dev_preds


def train_weighted_ensemble(train_df, dev_df):
    """
    Train individual models and combine with learned weights.
        
    Returns tuple: (models_dict, predictions)
    """
    print("\n" + "="*60)
    print("Training Weighted Ensemble")
    print("="*60 + "\n")
    
    models = {}
    predictions = {}
    
    # Model 1: Logistic Regression with high-dimensional features
    print("Training Logistic Regression...")
    lr_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 3))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    lr_model.fit(train_df['text'], train_df['polarization'])
    predictions['lr'] = lr_model.predict_proba(dev_df['text'])
    models['lr'] = lr_model
    
    # Model 2: Random Forest
    print("Training Random Forest...")
    rf_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    rf_model.fit(train_df['text'], train_df['polarization'])
    predictions['rf'] = rf_model.predict_proba(dev_df['text'])
    models['rf'] = rf_model
    
    # Model 3: Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
        ('clf', GradientBoostingClassifier(n_estimators=150, random_state=42))
    ])
    gb_model.fit(train_df['text'], train_df['polarization'])
    predictions['gb'] = gb_model.predict_proba(dev_df['text'])
    models['gb'] = gb_model
    
    # Weighted average (can tune these weights based on individual model performance)
    weights = {'lr': 0.4, 'rf': 0.3, 'gb': 0.3}
    
    ensemble_probs = sum(weights[name] * pred for name, pred in predictions.items())
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    models['ensemble'] = {'models': models, 'weights': weights}
    
    return models, ensemble_preds


def main():
    
    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask1/train/eng.csv',
        '../data/test_phase/subtask1/dev/eng.csv',
        '../data/test_phase/subtask1/test/eng.csv'
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}\n")
    
    all_results = []
    
    # Experiment 1: Voting Ensemble
    print("\n" + "EXPERIMENT 2.1: VOTING ENSEMBLE")
    voting_model, voting_preds = train_voting_ensemble(train_df, dev_df)
    
    results = evaluate_model(
        dev_df['polarization'], 
        voting_preds, 
        model_name="Voting Ensemble (Soft)"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        voting_preds, 
        model_name="Voting Ensemble",
        save_path="./results/voting_ensemble_confusion_matrix.png"
    )
    
    save_results(results, "./results/voting_ensemble_results.json")
    
    technique_desc = """
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
"""
    
    observations = f"""
- The ensemble combines the strengths of linear (LogisticRegression), tree-based (RandomForest), and boosting (GradientBoosting) approaches.
- Soft voting leverages probability estimates for more nuanced predictions.
- Macro F1 score: {results['macro_f1']:.4f}
- The ensemble shows {'improved robustness' if results['macro_f1'] > 0.65 else 'comparable performance'} to individual models.
- Class balance: {'Well-balanced' if abs(results['per_class_metrics']['class_0']['f1'] - results['per_class_metrics']['class_1']['f1']) < 0.05 else 'Some imbalance detected'}
"""
    
    report = create_experiment_report(
        "Voting Ensemble", 
        technique_desc, 
        results, 
        observations
    )
    
    with open("./results/voting_ensemble_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 2: Stacking Ensemble
    print("\n" + "EXPERIMENT 2.2: STACKING ENSEMBLE")
    stacking_model, stacking_preds = train_stacking_ensemble(train_df, dev_df)
    
    results = evaluate_model(
        dev_df['polarization'], 
        stacking_preds, 
        model_name="Stacking Ensemble"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        stacking_preds, 
        model_name="Stacking Ensemble",
        save_path="./results/stacking_ensemble_confusion_matrix.png"
    )
    
    save_results(results, "./results/stacking_ensemble_results.json")
    
    technique_desc = """
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
"""
    
    observations = f"""
- Stacking allows the meta-learner to adaptively weight base models.
- The meta-learner can identify when specific models are more reliable.
- Macro F1 score: {results['macro_f1']:.4f}
- {'The stacking approach shows improved performance over simple voting.' if results['macro_f1'] > 0.67 else 'Performance is comparable to voting ensemble.'}
- Cross-validation during training helps prevent overfitting.
"""
    
    report = create_experiment_report(
        "Stacking Ensemble", 
        technique_desc, 
        results, 
        observations
    )
    
    with open("./results/stacking_ensemble_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 3: Weighted Ensemble
    print("\n" + "EXPERIMENT 2.3: WEIGHTED ENSEMBLE")
    weighted_models, weighted_preds = train_weighted_ensemble(train_df, dev_df)
    
    results = evaluate_model(
        dev_df['polarization'], 
        weighted_preds, 
        model_name="Weighted Ensemble"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        weighted_preds, 
        model_name="Weighted Ensemble",
        save_path="./results/weighted_ensemble_confusion_matrix.png"
    )
    
    save_results(results, "./results/weighted_ensemble_results.json")
    
    technique_desc = """
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
"""
    
    observations = f"""
- Manual weighting allows expert knowledge to guide ensemble.
- Logistic Regression receives higher weight (40%) for its stability.
- Macro F1 score: {results['macro_f1']:.4f}
- The weighted approach provides {'competitive' if results['macro_f1'] > 0.65 else 'baseline'} performance.
- Weights can be further optimized using validation data.
"""
    
    report = create_experiment_report(
        "Weighted Ensemble", 
        technique_desc, 
        results, 
        observations
    )
    
    with open("./results/weighted_ensemble_report.md", 'w') as f:
        f.write(report)
    
    # Compare all ensemble methods
    print("\n" + "="*60)
    print("ENSEMBLE METHODS COMPARISON")
    print("="*60)
    
    from evaluation_utils import compare_models
    comparison_df = compare_models(all_results)
    comparison_df.to_csv("./results/ensemble_comparison.csv", index=False)
    
    
    print("Results saved in ./results/ directory")


if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    main()
