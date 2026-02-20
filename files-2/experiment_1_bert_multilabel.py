"""
BERT-based Model Comparison for Multi-Label Classification
Comparing different pre-trained BERT models for polarization type detection.
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

from preprocessing_subtask2 import load_and_preprocess_data, get_label_columns, analyze_dataset
from evaluation_utils_subtask2 import (
    evaluate_multilabel_model, 
    plot_multilabel_confusion_matrices, 
    save_results,
    create_experiment_report
)


def compute_metrics(eval_pred):
    """Compute metrics for multi label evaluation"""
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


def train_bert_multilabel(
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
    Train a bert-based model for multi-label classification.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} for Multi-Label Classification")
    print(f"{'='*60}\n")
    
    num_labels = len(label_cols)
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # Tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    # Prepare labels
    train_labels = train_df[label_cols].values.astype(np.float32)
    dev_labels = dev_df[label_cols].values.astype(np.float32)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_df['text'].tolist(),
        'labels': train_labels.tolist()
    })
    dev_dataset = Dataset.from_dict({
        'text': dev_df['text'].tolist(),
        'labels': dev_labels.tolist()
    })
    
    # Tokenize
    train_tok = train_dataset.map(tokenize, batched=True)
    dev_tok = dev_dataset.map(tokenize, batched=True)
    
    # Set format
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    
    if output_dir is None:
        output_dir = f"./models_subtask2/{model_name.replace('/', '_')}"
    
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
    
    #  sigmoid and threshold
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    
    return trainer, preds


def main():
    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask2/train/eng.csv',
        '../data/test_phase/subtask2/dev/eng.csv',
        '../data/test_phase/subtask2/test/eng.csv'
    )
    
    label_cols = get_label_columns()
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Number of labels: {len(label_cols)}")
    print(f"Labels: {label_cols}")
    
    
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
            
            trainer, dev_preds = train_bert_multilabel(
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
                save_path=f"./results_subtask2/{model_name.replace('/', '_')}_confusion.png"
            )
            
            
            save_results(
                results, 
                f"./results_subtask2/{model_name.replace('/', '_')}_results.json"
            )
            
            #  report
            technique_desc = f"""
This experiment uses **{model_name}**, a transformer-based model fine-tuned for multi-label classification.

**Model Characteristics:**
- Architecture: Transformer encoder
- Pre-training: Masked language modeling
- Maximum sequence length: 128 tokens
- Fine-tuning approach: Full model fine-tuning with multi-label classification head

**Multi-Label Configuration:**
- Problem type: multi_label_classification
- Loss function: Binary Cross-Entropy with Logits
- Output: Sigmoid activation per label
- Threshold: 0.5 for binary decision

**Training Configuration:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3 (with early stopping)
- Optimizer: AdamW
- Weight decay: 0.01
"""
            
            
            label_f1s = [metrics['f1'] for metrics in results['per_label_metrics'].values()]
            best_label = max(results['per_label_metrics'].items(), key=lambda x: x[1]['f1'])
            worst_label = min(results['per_label_metrics'].items(), key=lambda x: x[1]['f1'])
            
            observations = f"""
- The model achieves a micro F1 score of {results['f1_micro']:.4f} (overall performance).
- The macro F1 score is {results['f1_macro']:.4f} (average across labels).
- Subset accuracy (exact match): {results['subset_accuracy']:.4f}
- Hamming loss (fraction of wrong labels): {results['hamming_loss']:.4f}

**Per-Label Analysis:**
- Best performing label: **{best_label[0]}** (F1: {best_label[1]['f1']:.4f})
- Most challenging label: **{worst_label[0]}** (F1: {worst_label[1]['f1']:.4f})
- F1 score range: {min(label_f1s):.4f} - {max(label_f1s):.4f}
- Average per-label F1: {np.mean(label_f1s):.4f}

**Multi-Label Characteristics:**
- The model shows {'balanced' if max(label_f1s) - min(label_f1s) < 0.1 else 'imbalanced'} performance across labels.
- {'Strong exact match performance' if results['subset_accuracy'] > 0.5 else 'Moderate exact match performance' if results['subset_accuracy'] > 0.3 else 'Low exact match performance'} indicates the model {'often' if results['subset_accuracy'] > 0.5 else 'sometimes' if results['subset_accuracy'] > 0.3 else 'rarely'} predicts all labels correctly.
"""
            
            report = create_experiment_report(
                model_name, 
                technique_desc, 
                results, 
                observations
            )
            
           
            with open(f"./results_subtask2/{model_name.replace('/', '_')}_report.md", 'w') as f:
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
        comparison_df.to_csv("./results_subtask2/bert_models_comparison.csv", index=False)
        print("Comparison saved to ./results_subtask2/bert_models_comparison.csv")


if __name__ == "__main__":
    os.makedirs("./results_subtask2", exist_ok=True)
    os.makedirs("./models_subtask2", exist_ok=True)
    main()
