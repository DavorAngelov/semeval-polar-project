"""
Experiment 5: Hyperparameter Tuning
Systematic optimization of model hyperparameters using grid search and random search.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer


from preprocessing import load_and_preprocess_data
from evaluation_utils import (
    evaluate_model, 
    plot_confusion_matrix, 
    save_results,
    create_experiment_report
)


def grid_search_logistic_regression(train_df, dev_df):
    """
    Grid search for Logistic regression hyperparameters
    """
    print("\n" + "="*60)
    print("Grid Search: Logistic Regression")
    print("="*60 + "\n")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    param_grid = {
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__class_weight': ['balanced', None],
        'clf__penalty': ['l2']
    }
    

    scorer = make_scorer(f1_score, average='macro')
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring=scorer,
        n_jobs=-1,
        verbose=2
    )
    
    print(f"Testing {len(param_grid['tfidf__max_features']) * len(param_grid['tfidf__ngram_range']) * len(param_grid['tfidf__min_df']) * len(param_grid['clf__C']) * len(param_grid['clf__class_weight'])} combinations...")
    
    grid_search.fit(train_df['text'], train_df['polarization'])
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predict on dev set 
    preds = grid_search.predict(dev_df['text'])
    
    return grid_search.best_estimator_, preds, grid_search.best_params_


def random_search_random_forest(train_df, dev_df):
    """
    Random search for Random Forest hyperparameters.
    """
    print("\n" + "="*60)
    print("Random Search: Random Forest")
    print("="*60 + "\n")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    param_distributions = {
        'tfidf__max_features': [3000, 5000, 8000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': [50, 100, 150, 200],
        'clf__max_depth': [None, 10, 20, 30, 40],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    scorer = make_scorer(f1_score, average='macro')
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=20,  #  20 random combinations
        cv=3,
        scoring=scorer,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print(f"Testing 20 random combinations from parameter space...")
    
    random_search.fit(train_df['text'], train_df['polarization'])
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    # Predict on dev set
    preds = random_search.predict(dev_df['text'])
    
    return random_search.best_estimator_, preds, random_search.best_params_


def grid_search_svm(train_df, dev_df):
    """
    Grid search for SVM hyperparameters.
    """
    print("\n" + "="*60)
    print("Grid Search: Support Vector Machine")
    print("="*60 + "\n")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC())
    ])
    
    param_grid = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__kernel': ['linear', 'rbf'],
        'clf__class_weight': ['balanced', None],
    }
    
    scorer = make_scorer(f1_score, average='macro')
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring=scorer,
        n_jobs=-1,
        verbose=2
    )
    
    print(f"Testing parameter combinations...")
    
    grid_search.fit(train_df['text'], train_df['polarization'])
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predict on dev set
    preds = grid_search.predict(dev_df['text'])
    
    return grid_search.best_estimator_, preds, grid_search.best_params_


def random_search_gradient_boosting(train_df, dev_df):
    """
    Random search for Gradient Boosting hyperparameters.
    """
    print("\n" + "="*60)
    print("Random Search: Gradient Boosting")
    print("="*60 + "\n")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    
    param_distributions = {
        'tfidf__max_features': [5000, 8000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': [50, 100, 150],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__subsample': [0.8, 0.9, 1.0]
    }
    
    scorer = make_scorer(f1_score, average='macro')
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=15,
        cv=3,
        scoring=scorer,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print(f"Testing 15 random combinations...")
    
    random_search.fit(train_df['text'], train_df['polarization'])
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    # Predict on dev set
    preds = random_search.predict(dev_df['text'])
    
    return random_search.best_estimator_, preds, random_search.best_params_


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
    best_params_summary = {}
    
    # Experiment 5.1: Logistic Regression Grid Search
    print("\n EXPERIMENT 5.1: LOGISTIC REGRESSION TUNING")
    model, preds, best_params = grid_search_logistic_regression(train_df, dev_df)
    best_params_summary['Logistic Regression'] = best_params
    
    results = evaluate_model(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned Logistic Regression"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned Logistic Regression",
        save_path="./results/tuned_lr_confusion.png"
    )
    
    save_results(results, "./results/tuned_lr_results.json")
    
    technique_desc = f"""
This experiment uses **Grid Search** to optimize Logistic Regression hyperparameters:

**Search Space:**
- TF-IDF max features: [5000, 10000, 20000]
- N-gram range: [(1,1), (1,2), (1,3)]
- Min document frequency: [1, 2, 3]
- Regularization (C): [0.1, 1.0, 10.0]
- Class weight: [balanced, None]

**Best Parameters Found:**
```
{best_params}
```

**Search Method:**
- Exhaustive grid search with 3-fold cross-validation
- Scoring metric: Macro F1
- Total combinations tested: ~162

**Rationale:**
Grid search ensures we find the optimal combination of parameters for maximum performance.
"""
    
    observations = f"""
- Grid search identified optimal TF-IDF and regularization settings.
- Best CV score during search: Higher scores indicate good generalization
- Macro F1 on dev set: {results['macro_f1']:.4f}
- {'Tuning provided significant improvement' if results['macro_f1'] > 0.70 else 'Further tuning may be needed'}
- Key finding: {f"Best n-gram range is {best_params.get('tfidf__ngram_range', 'N/A')}"}
"""
    
    report = create_experiment_report(
        "Tuned Logistic Regression", 
        technique_desc, 
        results, 
        observations
    )
    
    with open("./results/tuned_lr_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 5.2: Random Forest Random Search
    print("\nEXPERIMENT 5.2: RANDOM FOREST TUNING")
    model, preds, best_params = random_search_random_forest(train_df, dev_df)
    best_params_summary['Random Forest'] = best_params
    
    results = evaluate_model(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned Random Forest"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned Random Forest",
        save_path="./results/tuned_rf_confusion.png"
    )
    
    save_results(results, "./results/tuned_rf_results.json")
    
    technique_desc = f"""
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
{best_params}
```

**Search Method:**
- Random sampling of 20 combinations
- 3-fold cross-validation
- Scoring metric: Macro F1

**Advantage:**
Random search is more efficient than grid search for large parameter spaces.
"""
    
    observations = f"""
- Random search efficiently explored the parameter space.
- Tested 20 random combinations vs. thousands for complete grid
- Macro F1 on dev set: {results['macro_f1']:.4f}
- Best n_estimators: {best_params.get('clf__n_estimators', 'N/A')}
- Best max_depth: {best_params.get('clf__max_depth', 'N/A')}
- {'Random Forest benefits from deeper trees' if best_params.get('clf__max_depth') and best_params.get('clf__max_depth') > 20 else 'Shallower trees prevent overfitting'}
"""
    
    report = create_experiment_report(
        "Tuned Random Forest", 
        technique_desc, 
        results, 
        observations
    )
    
    with open("./results/tuned_rf_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 5.3: SVM Grid Search
    print("\nEXPERIMENT 5.3: SVM TUNING")
    model, preds, best_params = grid_search_svm(train_df, dev_df)
    best_params_summary['SVM'] = best_params
    
    results = evaluate_model(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned SVM"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned SVM",
        save_path="./results/tuned_svm_confusion.png"
    )
    
    save_results(results, "./results/tuned_svm_results.json")
    
    # Experiment 5.4: Gradient Boosting Random Search
    print("\nEXPERIMENT 5.4: GRADIENT BOOSTING TUNING")
    model, preds, best_params = random_search_gradient_boosting(train_df, dev_df)
    best_params_summary['Gradient Boosting'] = best_params
    
    results = evaluate_model(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned Gradient Boosting"
    )
    all_results.append(results)
    
    plot_confusion_matrix(
        dev_df['polarization'], 
        preds, 
        model_name="Tuned Gradient Boosting",
        save_path="./results/tuned_gb_confusion.png"
    )
    
    save_results(results, "./results/tuned_gb_results.json")
    
    # Compare all tuned models
    print("\n" + "="*60)
    print("TUNED MODELS COMPARISON")
    print("="*60)
    
    from evaluation_utils import compare_models
    comparison_df = compare_models(all_results)
    comparison_df.to_csv("./results/hyperparameter_tuning_comparison.csv", index=False)
    
    # Save best parameters summary
    print("\n" + "="*60)
    print("BEST PARAMETERS SUMMARY")
    print("="*60)
    for model_name, params in best_params_summary.items():
        print(f"\n{model_name}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    # Save to file
    import json
    with open("./results/best_hyperparameters.json", 'w') as f:
        json.dump(best_params_summary, f, indent=4)
    
    
    print("Results saved in ./results/ directory")
    print("Best parameters saved in ./results/best_hyperparameters.json")


if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    main()
