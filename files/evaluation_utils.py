"""
Evaluation and reporting utilities for polarization detection experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    evaluation of a models predictions
        
    
    returns dict: dictionary containing all metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": {
            "class_0": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0])
            },
            "class_1": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1])
            }
        }
    }
    
    # Print classification report
    print(f"\n{'='*60}")
    print(f"{model_name} - Classification Report")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return results


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()


def save_results(results, filepath):
    """
    Save results to a JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def compare_models(results_list):
    """
    Compare multiple models and create a comparison table
        
    Returns comparison dataframe
    """
    import pandas as pd
    
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            "Model": result["model_name"],
            "Accuracy": result["accuracy"],
            "Macro F1": result["macro_f1"],
            "Weighted F1": result["weighted_f1"],
            "Class 0 F1": result["per_class_metrics"]["class_0"]["f1"],
            "Class 1 F1": result["per_class_metrics"]["class_1"]["f1"]
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Macro F1", ascending=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def create_experiment_report(model_name, technique_description, results, observations):
    """
    Create a markdown report for an experiment
    
    """
    report = f"""# Experiment Report: {model_name}

**Date:** {results['timestamp']}

## Technique Description

{technique_description}

## Results

### Overall Performance
- **Accuracy:** {results['accuracy']:.4f}
- **Macro F1 Score:** {results['macro_f1']:.4f}
- **Weighted F1 Score:** {results['weighted_f1']:.4f}

### Per-Class Performance

#### Class 0 (Non-Polarized)
- Precision: {results['per_class_metrics']['class_0']['precision']:.4f}
- Recall: {results['per_class_metrics']['class_0']['recall']:.4f}
- F1 Score: {results['per_class_metrics']['class_0']['f1']:.4f}
- Support: {results['per_class_metrics']['class_0']['support']}

#### Class 1 (Polarized)
- Precision: {results['per_class_metrics']['class_1']['precision']:.4f}
- Recall: {results['per_class_metrics']['class_1']['recall']:.4f}
- F1 Score: {results['per_class_metrics']['class_1']['f1']:.4f}
- Support: {results['per_class_metrics']['class_1']['support']}

## Key Observations

{observations}

## Conclusion

This experiment achieved a macro F1 score of {results['macro_f1']:.4f}, which {'exceeds' if results['macro_f1'] > 0.70 else 'falls below'} the 0.70 threshold commonly considered strong performance for binary classification tasks.

"""
    return report


