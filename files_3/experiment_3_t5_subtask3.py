"""
Experiment: T5-based Multi-Label Classification for Polarization Manifestation Detection

"""

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False



try:
    from preprocessing_subtask3 import load_and_preprocess_data, get_label_columns
    from evaluation_utils_subtask2 import (
        evaluate_multilabel_model,
        plot_multilabel_confusion_matrices,
        save_results,
        create_experiment_report,
    )
    PROJECT_UTILS_AVAILABLE = True
except ImportError:
    PROJECT_UTILS_AVAILABLE = False
 



LABEL_NAMES = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]
LABEL_SEP  = ", "
NONE_TOKEN = "none"



# encode / decode labels


def labels_to_string(label_vector) -> str:
    """[1,1,0,0,0,0] - 'stereotype, vilification'"""
    active = [LABEL_NAMES[i] for i, v in enumerate(label_vector) if v == 1]
    return LABEL_SEP.join(active) if active else NONE_TOKEN


def string_to_labels(generated: str) -> np.ndarray:
    """'stereotype, vilification' - [1,1,0,0,0,0]"""
    generated = generated.lower().strip()
    result = np.zeros(len(LABEL_NAMES), dtype=int)
    if generated == NONE_TOKEN or generated == "":
        return result
    for i, label in enumerate(LABEL_NAMES):
        
        label_surface = label.replace("_", " ")
        if label in generated or label_surface in generated:
            result[i] = 1
    return result




def build_hf_dataset(df: pd.DataFrame, tokenizer, max_input_length=256, max_target_length=64):
    
    label_cols = get_label_columns() if PROJECT_UTILS_AVAILABLE else LABEL_NAMES

    records = []
    for _, row in df.iterrows():
        label_vec  = [int(row[col]) for col in label_cols]
        target_str = labels_to_string(label_vec)
        records.append({
            "input_text":  f"classify polarization manifestation: {row['text']}",
            "target_text": target_str,
        })

    hf_ds = Dataset.from_list(records)

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenised = hf_ds.map(tokenize_fn, batched=True, remove_columns=["input_text", "target_text"])
    return tokenised



def make_compute_metrics(tokenizer):
    """returns a compute_metrics function compatible with seq2seqtrainer"""
    from sklearn.metrics import f1_score

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_ids      = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_matrix = np.array([string_to_labels(p) for p in decoded_preds])
        true_matrix = np.array([string_to_labels(t) for t in decoded_labels])

        f1_micro = f1_score(true_matrix, pred_matrix, average="micro", zero_division=0)
        f1_macro = f1_score(true_matrix, pred_matrix, average="macro", zero_division=0)

        return {"f1_micro": f1_micro, "f1_macro": f1_macro}

    return compute_metrics




class T5ManifestationClassifier:
    """
    Fine-tunes a T5 model on polarization manifestation
    classificatio and provides a predict interface
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        output_dir: str = "./t5_checkpoints_subtask3",
        max_input_length: int = 256,
        max_target_length: int = 64,
        num_train_epochs: int = 5,
        train_batch_size: int = 16,
        eval_batch_size: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        fp16: bool = False,
        label_smoothing: float = 0.1,
    ):
        self.model_name        = model_name
        self.output_dir        = output_dir
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length
        self.num_train_epochs  = num_train_epochs
        self.train_batch_size  = train_batch_size
        self.eval_batch_size   = eval_batch_size
        self.learning_rate     = learning_rate
        self.weight_decay      = weight_decay
        self.warmup_ratio      = warmup_ratio
        self.fp16              = fp16
        self.label_smoothing   = label_smoothing

        self.tokenizer  = None
        self.model      = None
        self.trainer    = None
        self._is_fitted = False

    def _load_model(self):
        print(f"Loading tokeniser model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        print(f"  params {sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self, train_df: pd.DataFrame, dev_df: pd.DataFrame):
        """Fine-tune on train_df, evaluate on dev_df."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("torch not installed.")

        self._load_model()

        train_ds = build_hf_dataset(train_df, self.tokenizer,
                                    self.max_input_length, self.max_target_length)
        dev_ds   = build_hf_dataset(dev_df,   self.tokenizer,
                                    self.max_input_length, self.max_target_length)

        safe_name = self.model_name.replace("/", "_")
        ckpt_dir  = os.path.join(self.output_dir, safe_name)

      
        import transformers as _tf
        _new_api  = tuple(int(x) for x in _tf.__version__.split(".")[:2]) >= (4, 45)
        _eval_key = "eval_strategy" if _new_api else "evaluation_strategy"

        training_args = Seq2SeqTrainingArguments(
            output_dir=ckpt_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            fp16=self.fp16,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            **{_eval_key: "epoch"},
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            label_smoothing_factor=self.label_smoothing,
            logging_steps=50,
            report_to="none",
            dataloader_num_workers=0,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, label_pad_token_id=-100
        )


        import inspect as _inspect
        _trainer_params = _inspect.signature(Seq2SeqTrainer.__init__).parameters
        _tok_kwarg      = "processing_class" if "processing_class" in _trainer_params else "tokenizer"

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            **{_tok_kwarg: self.tokenizer},
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print(f"\n{self.model_name} …")
        self.trainer.train()
        self._is_fitted = True


    def predict(self, text: str) -> np.ndarray:
        """Predict a single text - binary label array of shape (6,)"""
        if not self._is_fitted:
            raise RuntimeError("call fit before predict")

        input_text = f"classify polarization manifestation: {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return string_to_labels(decoded)

    def predict_batch(self, texts: list, verbose: bool = True) -> np.ndarray:
        """Predict a list of texts - array of shape (N, 6)"""
        predictions = []
        iterator = tqdm(texts, desc=f"Predicting [{self.model_name}]") if verbose else texts
        for text in iterator:
            predictions.append(self.predict(text))
        return np.array(predictions)




T5_MODELS = {
    "t5_base":       "google-t5/t5-base",
    "t5_large":      "google-t5/t5-large",
    "flan_t5_base":  "google/flan-t5-base",
    "flan_t5_large": "google/flan-t5-large",
}



def run_t5_experiment(
    train_df: pd.DataFrame,
    dev_df:   pd.DataFrame,
    model_keys: list = None,
    results_dir: str = "./results_subtask3",
    **trainer_kwargs,
) -> dict:
  
    os.makedirs(results_dir, exist_ok=True)
    label_cols = get_label_columns() if PROJECT_UTILS_AVAILABLE else LABEL_NAMES

    if model_keys is None:
        model_keys = list(T5_MODELS.keys())

    all_results = {}

    for key in model_keys:
        model_name = T5_MODELS[key]
        print(f"\n{'='*70}")
        print(f"T5 Subtask3 : {key}  ({model_name})")
        print(f"{'='*70}\n")

        try:
            classifier = T5ManifestationClassifier(
                model_name=model_name,
                output_dir=os.path.join(results_dir, "checkpoints"),
                **trainer_kwargs,
            )

            classifier.fit(train_df, dev_df)

            predictions = classifier.predict_batch(dev_df["text"].tolist(), verbose=True)
            true_labels = dev_df[label_cols].values

            results = evaluate_multilabel_model(
                true_labels, predictions, label_cols, model_name=key
            )

            all_results[key] = {"results": results, "predictions": predictions}

            plot_multilabel_confusion_matrices(
                true_labels, predictions, label_cols,
                model_name=f"T5 Subtask3 – {key}",
                save_path=os.path.join(results_dir, f"{key}_confusion.png"),
            )

            save_results(results, os.path.join(results_dir, f"{key}_results.json"))

            report = create_experiment_report(
                f"T5 Subtask3 – {key}",
                _technique_description(key, model_name),
                results,
                _observations(key, results),
            )
            with open(os.path.join(results_dir, f"{key}_report.md"), "w") as f:
                f.write(report)

          

        except Exception as exc:
            import traceback
           
            traceback.print_exc()
            continue


    if all_results:
   
        print("T5 SUBTASK 3 MODEL COMPARISON")
  
        rows = []
        for k, v in all_results.items():
            r = v["results"]
            rows.append({
                "Model":        k,
                "Micro F1":     f"{r['f1_micro']:.4f}",
                "Macro F1":     f"{r['f1_macro']:.4f}",
                "Subset Acc":   f"{r['subset_accuracy']:.4f}",
                "Hamming Loss": f"{r['hamming_loss']:.4f}",
            })
        cmp_df = pd.DataFrame(rows)
        print("\n" + cmp_df.to_string(index=False))
        cmp_df.to_csv(os.path.join(results_dir, "t5_subtask3_model_comparison.csv"), index=False)

    return all_results



def _technique_description(key: str, model_name: str) -> str:
    desc = f"""
This experiment fine-tunes **{model_name}** for multi-label polarization manifestation
classification (Subtask 3) using a **text-to-text (Seq2Seq)** framing.

**Labels:** Stereotype, Vilification, Dehumanization, Extreme Language, Lack of Empathy, Invalidation

**Input format:**
```
classify polarization manifestation: <raw_text>
```

**Target format:**
```
stereotype, vilification    (comma-separated active labels, or "none")
```

**Training details:**
- Seq2SeqTrainer with `predict_with_generate=True`
- Beam search decoding (num_beams=4)
- Early stopping (patience=2) on Micro F1
- Label smoothing (0.1) to handle class imbalance

**Subtask 3 specifics:**
- 6 labels vs 5 in Subtask 2 — slightly harder output space
- Labels are more semantically overlapping (e.g. Vilification vs Dehumanization)
- English-only data (Italian and Russian excluded)
"""
    if "flan" in key:
        desc += """
**Flan-T5 advantage:**
Flan-T5's instruction tuning gives it prior knowledge of classification-style
tasks, which is especially valuable here given the subtle label distinctions.
"""
    return desc


def _observations(key: str, results: dict) -> str:
    return f"""
**Model:** {key}

Achieved:
- Micro F1: {results['f1_micro']:.4f}
- Macro F1: {results['f1_macro']:.4f}
- Subset Accuracy: {results['subset_accuracy']:.4f}

**Notes:**
- Subtask 3 is harder than Subtask 2 due to more overlapping label semantics.
- Dehumanization and Invalidation are expected to be the rarest and hardest labels.
- Flan-T5 variants handle subtle distinctions better due to instruction pre-training.
"""



def main():

    print("T5-BASED POLARIZATION MANIFESTATION CLASSIFICATION")

 
    train_df, dev_df, test_df = load_and_preprocess_data(
        "../data/test_phase/subtask3/train/eng.csv",
        "../data/test_phase/subtask3/dev/eng.csv",
        "../data/test_phase/subtask3/test/eng.csv",
    )
    print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")

    if not TRANSFORMERS_AVAILABLE:
        return

    model_keys = ["flan_t5_base", "t5_base"]   

    results = run_t5_experiment(
        train_df,
        dev_df,
        model_keys=model_keys,
        num_train_epochs=5,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),
    )

   


if __name__ == "__main__":
    os.makedirs("./results_subtask3", exist_ok=True)
    main()
