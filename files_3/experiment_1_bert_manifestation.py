"""
Experiment 1: BERT-based Model Comparison for Manifestation Identification
"""
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import f1_score
import os


from preprocessing_subtask3 import (
    load_and_preprocess_data, get_label_columns, analyze_dataset,
    print_manifestation_guide, get_label_descriptions
)


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'files_2'))

from evaluation_utils_subtask2 import (
    evaluate_multilabel_model, 
    plot_multilabel_confusion_matrices, 
    save_results,
    create_experiment_report
)

def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


def train_bert_manifestation(
    model_name,
    train_df,
    dev_df,
    label_cols,
    max_length=128,
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
    output_dir=None
):
    """
    Train a BERT-based model for multi-label manifestation classification
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name} for Manifestation Identification")
    print(f"{'='*70}\n")
    
    num_labels = len(label_cols)
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
   
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    
    train_labels = train_df[label_cols].values.astype(np.float32)
    dev_labels = dev_df[label_cols].values.astype(np.float32)
    
    
    train_dataset = Dataset.from_dict({
        'text': train_df['text'].tolist(),
        'labels': train_labels.tolist()
    })
    dev_dataset = Dataset.from_dict({
        'text': dev_df['text'].tolist(),
        'labels': dev_labels.tolist()
    })
    
  
    train_tok = train_dataset.map(tokenize, batched=True)
    dev_tok = dev_dataset.map(tokenize, batched=True)
    
    
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
   
    if output_dir is None:
        output_dir = f"./models_subtask3/{model_name.replace('/', '_')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    
    trainer.train()
    
    
    predictions = trainer.predict(dev_tok)
    logits = predictions.predictions
    
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    
    return trainer, preds


def main():
    
    print_manifestation_guide()
    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask3/train/eng.csv',
        '../data/test_phase/subtask3/dev/eng.csv',
        '../data/test_phase/subtask3/test/eng.csv'
    )
    
    label_cols = get_label_columns()
    label_descriptions = get_label_descriptions()
    
    print(f"  Train size: {len(train_df)}")
    print(f"  Dev size: {len(dev_df)}")
    print(f"  Test size: {len(test_df)}")
    print(f"  Number of manifestation types: {len(label_cols)}")
    print(f"  Manifestations: {', '.join(label_cols)}")
    
    
    analyze_dataset(train_df, "Training Set", label_cols)
    analyze_dataset(dev_df, "Development Set", label_cols)
    
  
    models_to_test = [
        "xlm-roberta-base",  # Multilingual
        "bert-base-uncased",  # English-focused
        "distilbert-base-uncased",  # Lighter, faster
        "roberta-base",  # RoBERTa variant
    ]
    
    all_results = []
    
    for model_name in models_to_test:
        try:
         
            trainer, dev_preds = train_bert_manifestation(
                model_name=model_name,
                train_df=train_df,
                dev_df=dev_df,
                label_cols=label_cols,
                max_length=128,
                batch_size=16,
                epochs=3,
                learning_rate=2e-5
            )
            
           
            results = evaluate_multilabel_model(
                dev_df[label_cols], 
                dev_preds,
                label_cols,
                model_name=model_name
            )
            all_results.append(results)
            
       
            plot_multilabel_confusion_matrices(
                dev_df[label_cols], 
                dev_preds,
                label_cols,
                model_name=model_name,
                save_path=f"./results_subtask3/{model_name.replace('/', '_')}_confusion.png"
            )
            
         
            save_results(
                results, 
                f"./results_subtask3/{model_name.replace('/', '_')}_results.json"
            )
            
            # Create report
            technique_desc = f"""
This experiment uses **{model_name}**, a transformer-based model fine-tuned for multi-label manifestation classification.

**Task:** Identify how polarization is expressed through 6 manifestation types:
1. **Stereotype** - Generalizes characteristics to all group members
2. **Vilification** - Defames or demonizes groups through biased framing
3. **Dehumanization** - Strips individuals of human qualities
4. **Extreme Language** - Uses absolutist terms and dichotomous framing
5. **Lack of Empathy** - Shows no understanding for others' perspectives
6. **Invalidation** - Denies or rejects identity or existence of groups

**Model Configuration:**
- Architecture: Transformer encoder with multi-label classification head
- Pre-training: Masked language modeling
- Maximum sequence length: 128 tokens
- Fine-tuning: Full model fine-tuning

**Multi-Label Setup:**
- Problem type: multi_label_classification
- Loss function: Binary Cross-Entropy with Logits (per manifestation)
- Output activation: Sigmoid (independent predictions per manifestation)
- Decision threshold: 0.5 per manifestation

**Training Configuration:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3 (with early stopping, patience=2)
- Optimizer: AdamW
- Weight decay: 0.01
- Metric for best model: Macro F1
"""
            
            
            label_f1s = [metrics['f1'] for metrics in results['per_label_metrics'].values()]
            best_manifestation = max(results['per_label_metrics'].items(), key=lambda x: x[1]['f1'])
            worst_manifestation = min(results['per_label_metrics'].items(), key=lambda x: x[1]['f1'])
            
          
            high_precision = [k for k, v in results['per_label_metrics'].items() if v['precision'] > 0.7]
            high_recall = [k for k, v in results['per_label_metrics'].items() if v['recall'] > 0.7]
            
            observations = f"""
### Overall Performance
- **Micro F1 Score:** {results['f1_micro']:.4f} (overall performance across all manifestations)
- **Macro F1 Score:** {results['f1_macro']:.4f} (average performance per manifestation)
- **Subset Accuracy:** {results['subset_accuracy']:.4f} (exact match - all manifestations correct)
- **Hamming Loss:** {results['hamming_loss']:.4f} (fraction of incorrect manifestation predictions)
- **Jaccard Score (Micro):** {results['jaccard_micro']:.4f} (overlap between predictions and truth)

### Per-Manifestation Analysis

**Best Detected Manifestation:** {best_manifestation[0].replace('_', ' ').title()}
- F1 Score: {best_manifestation[1]['f1']:.4f}
- Precision: {best_manifestation[1]['precision']:.4f}
- Recall: {best_manifestation[1]['recall']:.4f}
- This manifestation was easiest for the model to identify

**Most Challenging Manifestation:** {worst_manifestation[0].replace('_', ' ').title()}
- F1 Score: {worst_manifestation[1]['f1']:.4f}
- Precision: {worst_manifestation[1]['precision']:.4f}
- Recall: {worst_manifestation[1]['recall']:.4f}
- This manifestation requires improvement

**Performance Range:**
- F1 scores range from {min(label_f1s):.4f} to {max(label_f1s):.4f}
- Standard deviation: {np.std(label_f1s):.4f}
- Performance is {'well-balanced' if np.std(label_f1s) < 0.1 else 'somewhat imbalanced' if np.std(label_f1s) < 0.15 else 'highly imbalanced'} across manifestations

### Model Characteristics

**High Precision Manifestations** (>0.70): {', '.join([m.replace('_', ' ').title() for m in high_precision]) if high_precision else 'None'}
- When the model predicts these, it's usually correct
- Low false positive rate

**High Recall Manifestations** (>0.70): {', '.join([m.replace('_', ' ').title() for m in high_recall]) if high_recall else 'None'}
- Model successfully identifies most instances
- Low false negative rate

### Multi-Label Behavior
- Average manifestations per sample: {results['per_label_metrics'][label_cols[0]]['support'] / len(dev_df):.2f}
- The model shows {'strong' if results['subset_accuracy'] > 0.4 else 'moderate' if results['subset_accuracy'] > 0.25 else 'weak'} exact matching ability
- {'Good' if results['hamming_loss'] < 0.15 else 'Moderate' if results['hamming_loss'] < 0.25 else 'High'} label error rate

### Interpretability Notes
- **Stereotype** often co-occurs with extreme language
- **Vilification** may overlap with dehumanization in severe cases
- **Lack of empathy** is abstract and harder to detect than explicit manifestations
- **Invalidation** requires understanding of identity denial patterns
"""
            
            report = create_experiment_report(
                model_name, 
                technique_desc, 
                results, 
                observations
            )
            
          
            with open(f"./results_subtask3/{model_name.replace('/', '_')}_report.md", 'w') as f:
                f.write(report)
            
            print(f"✓ Completed: {model_name}\n")
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue
    
   
    if all_results:
        from evaluation_utils_subtask2 import compare_models
        comparison_df = compare_models(all_results)
        comparison_df.to_csv("./results_subtask3/bert_models_comparison.csv", index=False)
        print("Comparison saved to ./results_subtask3/bert_models_comparison.csv")
        
        
        print("\n" + "="*80)
        print("KEY INSIGHTS ACROSS ALL MODELS")
        print("="*80)
        
        
        best_model = max(all_results, key=lambda x: x['f1_micro'])
        print(f"\nBest Model (Micro F1): {best_model['model_name']}")
        print(f"   - Micro F1: {best_model['f1_micro']:.4f}")
        print(f"   - Macro F1: {best_model['f1_macro']:.4f}")
        print(f"   - Subset Accuracy: {best_model['subset_accuracy']:.4f}")
        
        #  most challenging manifestation across all models
        all_manifestation_f1s = {label: [] for label in label_cols}
        for result in all_results:
            for label in label_cols:
                all_manifestation_f1s[label].append(result['per_label_metrics'][label]['f1'])
        
        avg_manifestation_f1s = {label: np.mean(f1s) for label, f1s in all_manifestation_f1s.items()}
        easiest = max(avg_manifestation_f1s.items(), key=lambda x: x[1])
        hardest = min(avg_manifestation_f1s.items(), key=lambda x: x[1])
        
        print(f"\nManifestation Difficulty (averaged across all models):")
        print(f"   Easiest: {easiest[0].replace('_', ' ').title()} (Avg F1: {easiest[1]:.4f})")
        print(f"   Hardest: {hardest[0].replace('_', ' ').title()} (Avg F1: {hardest[1]:.4f})")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    os.makedirs("./results_subtask3", exist_ok=True)
    os.makedirs("./models_subtask3", exist_ok=True)
    main()
