"""
Evaluation and reporting utilities for multi-label polarization type detection.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, f1_score, hamming_loss, jaccard_score,
    accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
)
import json
from datetime import datetime


def evaluate_multilabel_model(y_true, y_pred, label_names, model_name="Model"):
    """
    evaluation of a models predictions
        
    
    returns dict: dictionary containing all metrics
    """
    
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    
    # Subset accuracy 
    subset_accuracy = accuracy_score(y_true, y_pred)
    
    # Hamming loss 
    hamming = hamming_loss(y_true, y_pred)
    
    # Jaccard score 
    jaccard_micro = jaccard_score(y_true, y_pred, average='micro')
    jaccard_macro = jaccard_score(y_true, y_pred, average='macro')
    jaccard_samples = jaccard_score(y_true, y_pred, average='samples')
    
    # F1 scores
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_samples = f1_score(y_true, y_pred, average='samples')
    
    # Per label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_label_metrics = {}
    for i, label in enumerate(label_names):
        per_label_metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subset_accuracy": float(subset_accuracy),
        "hamming_loss": float(hamming),
        "jaccard_micro": float(jaccard_micro),
        "jaccard_macro": float(jaccard_macro),
        "jaccard_samples": float(jaccard_samples),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_samples": float(f1_samples),
        "per_label_metrics": per_label_metrics
    }
    
    
    print(f"\n{'='*70}")
    print(f"{model_name} - Multi-Label Classification Report")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Subset Accuracy (Exact Match): {subset_accuracy:.4f}")
    print(f"  Hamming Loss:                   {hamming:.4f}")
    print(f"  Jaccard Score (Micro):          {jaccard_micro:.4f}")
    print(f"  Jaccard Score (Macro):          {jaccard_macro:.4f}")
    print(f"  F1 Score (Micro):               {f1_micro:.4f}")
    print(f"  F1 Score (Macro):               {f1_macro:.4f}")
    print(f"  F1 Score (Weighted):            {f1_weighted:.4f}")
    print(f"  F1 Score (Samples):             {f1_samples:.4f}")
    
    print(f"\nPer-Label Metrics:")
    print(f"{'Label':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    for label in label_names:
        metrics = per_label_metrics[label]
        print(f"{label:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['support']:<10}")
    
    print(f"{'='*70}\n")
    
    return results


def plot_multilabel_confusion_matrices(y_true, y_pred, label_names, model_name="Model", save_path=None):
    
    
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    
    cm_list = multilabel_confusion_matrix(y_true, y_pred)
    
    
    n_labels = len(label_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (cm, label) in enumerate(zip(cm_list, label_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        axes[idx].set_title(f'{label}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide the last subplot if odd number of labels
    if n_labels < 6:
        axes[5].axis('off')
    
    plt.suptitle(f'{model_name} - Confusion Matrices per Label', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()


def plot_label_performance(results_list, save_path=None):
    """
    Plot comparison of model performance across different labels

    """
    # Extract data for plotting
    models = [r['model_name'] for r in results_list]
    label_names = list(results_list[0]['per_label_metrics'].keys())
    
    # Prepare data
    data = {label: [] for label in label_names}
    for result in results_list:
        for label in label_names:
            data[label].append(result['per_label_metrics'][label]['f1'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.15
    
    for i, label in enumerate(label_names):
        offset = width * (i - len(label_names) / 2 + 0.5)
        ax.bar(x + offset, data[label], width, label=label)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Label F1 Scores Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Labels')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()


def save_results(results, filepath):

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def compare_models(results_list):
    """
    Compare multiple models and create a comparison table
    """
    import pandas as pd
    
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            "Model": result["model_name"],
            "Subset Accuracy": f"{result['subset_accuracy']:.4f}",
            "Hamming Loss": f"{result['hamming_loss']:.4f}",
            "F1 Micro": f"{result['f1_micro']:.4f}",
            "F1 Macro": f"{result['f1_macro']:.4f}",
            "F1 Weighted": f"{result['f1_weighted']:.4f}",
            "Jaccard Micro": f"{result['jaccard_micro']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON - MULTI-LABEL CLASSIFICATION")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df


def create_experiment_report(model_name, technique_description, results, observations):
    """
    markdown report for experiment
    """
    
    label_f1s = [metrics['f1'] for metrics in results['per_label_metrics'].values()]
    avg_label_f1 = np.mean(label_f1s)
    
    report = f"""# Experiment Report: {model_name}

**Date:** {results['timestamp']}

## Technique Description

{technique_description}

## Results

### Overall Multi-Label Metrics
- **Subset Accuracy (Exact Match):** {results['subset_accuracy']:.4f}
- **Hamming Loss:** {results['hamming_loss']:.4f}
- **F1 Score (Micro):** {results['f1_micro']:.4f}
- **F1 Score (Macro):** {results['f1_macro']:.4f}
- **F1 Score (Weighted):** {results['f1_weighted']:.4f}
- **F1 Score (Samples):** {results['f1_samples']:.4f}
- **Jaccard Score (Micro):** {results['jaccard_micro']:.4f}
- **Jaccard Score (Macro):** {results['jaccard_macro']:.4f}

### Per-Label Performance

"""
    
    for label, metrics in results['per_label_metrics'].items():
        report += f"""#### {label}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}
- Support: {metrics['support']}

"""
    
    report += f"""## Key Observations

{observations}

## Conclusion

This experiment achieved:
- Micro F1 score of {results['f1_micro']:.4f} (overall performance across all labels)
- Macro F1 score of {results['f1_macro']:.4f} (average performance per label)
- Subset accuracy of {results['subset_accuracy']:.4f} (exact match ratio)

The {'high' if results['subset_accuracy'] > 0.5 else 'moderate' if results['subset_accuracy'] > 0.3 else 'low'} subset accuracy indicates that the model {'frequently' if results['subset_accuracy'] > 0.5 else 'sometimes' if results['subset_accuracy'] > 0.3 else 'rarely'} predicts the exact label combination correctly.

"""
    return report
