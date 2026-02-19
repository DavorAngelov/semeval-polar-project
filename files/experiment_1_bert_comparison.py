"""
Experiment 1: BERT-based Model Comparison
Comparing different pre-trained BERT models for polarization detection.
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


from preprocessing import load_and_preprocess_data
from evaluation_utils import (
    evaluate_model, 
    plot_confusion_matrix, 
    save_results,
    create_experiment_report
)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def train_bert_model(
    model_name,
    train_df,
    dev_df,
    max_length=128,
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
    output_dir=None
):
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'polarization']])
    dev_dataset = Dataset.from_pandas(dev_df[['text', 'polarization']])
    
    # Tokenize
    train_tok = train_dataset.map(tokenize, batched=True)
    dev_tok = dev_dataset.map(tokenize, batched=True)
    
    # Rename columns
    train_tok = train_tok.rename_column("polarization", "labels")
    dev_tok = dev_tok.rename_column("polarization", "labels")
    
    # Set format
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    
    if output_dir is None:
        output_dir = f"./models/{model_name.replace('/', '_')}"
    
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
        metric_for_best_model="macro_f1",
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
    preds = np.argmax(predictions.predictions, axis=1)
    
    return trainer, preds


def main():
    
    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask1/train/eng.csv',
        '../data/test_phase/subtask1/dev/eng.csv',
        '../data/test_phase/subtask1/test/eng.csv'
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}\n")
    
    # Models to compare
    models_to_test = [
        "xlm-roberta-base",  # Multilingual
        "bert-base-uncased",  # English-focused
        "distilbert-base-uncased",  # Lighter, faster
        "roberta-base",  # RoBERTa variant
    ]
    
    all_results = []
    
    for model_name in models_to_test:
        try:
            
            trainer, dev_preds = train_bert_model(
                model_name=model_name,
                train_df=train_df,
                dev_df=dev_df,
                max_length=128,
                batch_size=16,
                epochs=3,
                learning_rate=2e-5
            )
            
            
            results = evaluate_model(
                dev_df['polarization'], 
                dev_preds, 
                model_name=model_name
            )
            all_results.append(results)
            
            
            plot_confusion_matrix(
                dev_df['polarization'], 
                dev_preds, 
                model_name=model_name,
                save_path=f"./results/{model_name.replace('/', '_')}_confusion_matrix.png"
            )
            
            
            save_results(
                results, 
                f"./results/{model_name.replace('/', '_')}_results.json"
            )
            
            # Create report
            technique_desc = f"""
This experiment uses **{model_name}**, a transformer-based model pre-trained on large text corpora.

**Model Characteristics:**
- Architecture: Transformer encoder
- Pre-training: Masked language modeling
- Maximum sequence length: 128 tokens
- Fine-tuning approach: Full model fine-tuning

**Training Configuration:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3 (with early stopping)
- Optimizer: AdamW
- Weight decay: 0.01
"""
            
            observations = f"""
- The model shows {'balanced' if abs(results['per_class_metrics']['class_0']['f1'] - results['per_class_metrics']['class_1']['f1']) < 0.05 else 'imbalanced'} performance across classes.
- Class 0 (non-polarized) F1: {results['per_class_metrics']['class_0']['f1']:.4f}
- Class 1 (polarized) F1: {results['per_class_metrics']['class_1']['f1']:.4f}
- The macro F1 score of {results['macro_f1']:.4f} indicates {'strong' if results['macro_f1'] > 0.70 else 'moderate'} performance.
"""
            
            report = create_experiment_report(
                model_name, 
                technique_desc, 
                results, 
                observations
            )
            
            # Save report
            with open(f"./results/{model_name.replace('/', '_')}_report.md", 'w') as f:
                f.write(report)
            
            print(f"Completed: {model_name}\n")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}\n")
            continue
    
    # Compare all models
    if all_results:
        from evaluation_utils import compare_models
        comparison_df = compare_models(all_results)
        comparison_df.to_csv("./results/bert_models_comparison.csv", index=False)
        print("Comparison saved to ./results/bert_models_comparison.csv")


if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    main()
