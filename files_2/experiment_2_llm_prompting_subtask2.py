"""
Experiment 2: LLM-based Multi-Label Classification with Chain-of-Thought Prompting
Using Llama and other LLMs for polarization type classification.
"""
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import time

# Ollama for local LLM inference
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available. Install with: pip install ollama")


try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


try:
    from preprocessing_subtask2 import load_and_preprocess_data, get_label_columns
    from evaluation_utils_subtask2 import (
        evaluate_multilabel_model, 
        plot_multilabel_confusion_matrices, 
        save_results,
        create_experiment_report
    )
except ImportError:
    print("sda")


class LLMPolarizationClassifier:
    """LLM-based classifier with various prompting strategies."""
    
    def __init__(self, model_name="llama3", use_ollama=True):
        """
        Initialize LLM classifier
        """
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.label_names = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
        
        if not use_ollama and TRANSFORMERS_AVAILABLE:
            print(f"Loading {model_name} ")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def create_zero_shot_prompt(self, text):
        """zero-shot prompt."""
        prompt = f"""You are an expert in analyzing social media content for polarization.

Task: Identify the TARGET(S) of polarization in the following text. A text can have MULTIPLE targets.

Categories:
1. Political - Division between political parties/ideologies
2. Racial/Ethnic - Division based on race or ethnicity
3. Religious - Division based on religious beliefs
4. Gender/Sexual - Division based on gender or sexual orientation
5. Other - Division targeting other groups (economy, technology, media, etc.)

Text: "{text}"

Instructions:
- Respond with ONLY a JSON object
- Format: {{"political": 0 or 1, "racial/ethnic": 0 or 1, "religious": 0 or 1, "gender/sexual": 0 or 1, "other": 0 or 1}}
- Use 1 if the category is a target, 0 if not
- A text can have multiple 1s or all 0s

Response:"""
        return prompt
    
    def create_cot_prompt(self, text):
        """Create a chain-of-thought prompt."""
        prompt = f"""You are an expert in analyzing social media content for polarization.

Task: Identify the TARGET(S) of polarization in the following text using step-by-step reasoning.

Categories:
1. Political - Division between political parties/ideologies
2. Racial/Ethnic - Division based on race or ethnicity  
3. Religious - Division based on religious beliefs
4. Gender/Sexual - Division based on gender or sexual orientation
5. Other - Division targeting other groups (economy, technology, media, etc.)

Text: "{text}"

Think step-by-step:

Step 1 - Identify key entities mentioned:
[List the groups, identities, or entities mentioned]

Step 2 - Analyze the tone and intent:
[Is the text divisive, attacking, or marginalizing any group?]

Step 3 - Map to categories:
[For each category, explain if it applies and why]

Step 4 - Final classification:
[Provide final answer as JSON]

Format: {{"political": 0 or 1, "racial/ethnic": 0 or 1, "religious": 0 or 1, "gender/sexual": 0 or 1, "other": 0 or 1}}

Response:"""
        return prompt
    
    def create_few_shot_prompt(self, text):
        """Create a few-shot prompt with examples."""
        prompt = f"""You are an expert in analyzing social media content for polarization.

Task: Identify the TARGET(S) of polarization. A text can have MULTIPLE targets.

Examples:

Example 1:
Text: "Republicans are destroying this country with their policies"
Answer: {{"political": 1, "racial/ethnic": 0, "religious": 0, "gender/sexual": 0, "other": 0}}

Example 2:
Text: "These immigrants are taking our jobs and bringing crime"
Answer: {{"political": 0, "racial/ethnic": 1, "religious": 0, "gender/sexual": 0, "other": 0}}

Example 3:
Text: "Religious fanatics are trying to control everyone's lives"
Answer: {{"political": 0, "racial/ethnic": 0, "religious": 1, "gender/sexual": 0, "other": 0}}

Example 4:
Text: "The liberal media is brainwashing people with their agenda"
Answer: {{"political": 1, "racial/ethnic": 0, "religious": 0, "gender/sexual": 0, "other": 1}}

Now classify this text. Respond with ONLY the JSON object, nothing else:

Text: "{text}"

Answer:"""
        return prompt
    
    
    
    def query_ollama(self, prompt):
        """Query Ollama API"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,  
                    'top_p': 0.9,
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return None
    
    def query_huggingface(self, prompt, max_length=512):
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response = response[len(prompt):].strip()
            return response
        except Exception as e:
            print(f"Error querying HuggingFace: {e}")
            return None
    
    def parse_response(self, response):
        """Parse LLM response to extract predictions."""
        if response is None:
            return np.array([0, 0, 0, 0, 0], dtype=int)
        
     
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                # Clean up common JSON issues
                json_str = json_str.replace("'", '"') 
                json_str = json_str.replace('True', '1').replace('False', '0')  # Handle booleans
                
                predictions = json.loads(json_str)
                
                
                result = np.array([
                    int(bool(predictions.get('political', 0))),
                    int(bool(predictions.get('racial/ethnic', 0))),
                    int(bool(predictions.get('religious', 0))),
                    int(bool(predictions.get('gender/sexual', 0))),
                    int(bool(predictions.get('other', 0)))
                ], dtype=int)
                
                return result
        except Exception as e:
            
            if np.random.random() < 0.1:  
                print(f"Parse error: {e}")
                print(f"Response preview: {response[:150]}...")
        
       
        response_lower = response.lower()
        predictions = []
        
        for label in self.label_names:
            # Look for various patterns
            found = False
            
            # Pattern 1: "label": 1
            if f'"{label}": 1' in response_lower or f"'{label}': 1" in response_lower:
                found = True
            # Pattern 2: label: 1 (without quotes)
            elif f'{label}: 1' in response_lower:
                found = True
            # Pattern 3: "label": true
            elif f'"{label}": true' in response_lower or f"'{label}': true" in response_lower:
                found = True
            # Pattern 4: YES for this category
            elif f'{label}: yes' in response_lower or f'{label} - yes' in response_lower:
                found = True
            
            predictions.append(1 if found else 0)
        
       
        if len(predictions) == 5:
            return np.array(predictions, dtype=int)
        else:
            return np.array([0, 0, 0, 0, 0], dtype=int)
    
    def predict(self, text, prompt_strategy='cot'):
        """
        Predict polarization types for a single text
        
        Returns:
            numpy array of predictions [political, racial/ethnic, religious, gender/sexual, other]
        """
        # Create prompt based on strategy
        if prompt_strategy == 'zero_shot':
            prompt = self.create_zero_shot_prompt(text)
        elif prompt_strategy == 'few_shot':
            prompt = self.create_few_shot_prompt(text)
        elif prompt_strategy == 'cot':
            prompt = self.create_cot_prompt(text)
        else:
            raise ValueError(f"Unknown strategy: {prompt_strategy}")
        
        # Query model
        if self.use_ollama:
            response = self.query_ollama(prompt)
        else:
            response = self.query_huggingface(prompt)
        
        # Parse response
        predictions = self.parse_response(response)
        
        # Validate output is numpy array of shape (5,) with dtype int
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions, dtype=int)
        
        if predictions.shape != (5,):
            print(f"⚠️ Invalid prediction shape: {predictions.shape}, resetting to zeros")
            predictions = np.array([0, 0, 0, 0, 0], dtype=int)
        
        # Ensure binary values
        predictions = np.clip(predictions, 0, 1).astype(int)
        
        return predictions
    
    def predict_batch(self, texts, prompt_strategy='cot', verbose=True):
        """
        Predict for multiple texts
        """
        predictions = []
        
        iterator = tqdm(texts, desc=f"Classifying with {prompt_strategy}") if verbose else texts
        
        for text in iterator:
            pred = self.predict(text, prompt_strategy)
            predictions.append(pred)
            
           
            if self.use_ollama:
                time.sleep(0.1) 
        
        return np.array(predictions)


def run_llm_experiment(
    train_df, 
    dev_df, 
    model_name="llama3",
    prompt_strategies=['zero_shot', 'few_shot', 'cot'],
    use_ollama=True
):
    """
    Run LLM experiments with different prompting strategies
    """
    label_cols = get_label_columns()
    
    # Initialize classifier
    print(f"\n{'='*70}")
    print(f"Initializing {model_name} classifier")
    print(f"{'='*70}\n")
    
    classifier = LLMPolarizationClassifier(model_name=model_name, use_ollama=use_ollama)
    
    all_results = {}
    
    for strategy in prompt_strategies:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy.upper()}")
        print(f"{'='*70}\n")
        
        try:
            # Predict
            predictions = classifier.predict_batch(
                dev_df['text'].tolist(), 
                prompt_strategy=strategy,
                verbose=True
            )
            
        
            results = evaluate_multilabel_model(
                dev_df[label_cols].values,
                predictions,
                label_cols,
                model_name=f"{model_name}_{strategy}"
            )
            
            all_results[strategy] = {
                'results': results,
                'predictions': predictions
            }
            
            
            plot_multilabel_confusion_matrices(
                dev_df[label_cols].values,
                predictions,
                label_cols,
                model_name=f"{model_name} - {strategy}",
                save_path=f"./results_subtask2/{model_name.replace('/', '_')}_{strategy}_confusion.png"
            )
            
           
            save_results(
                results,
                f"./results_subtask2/{model_name.replace('/', '_')}_{strategy}_results.json"
            )
            
         
            technique_desc = create_technique_description(model_name, strategy)
            observations = create_observations(results, strategy)
            
            report = create_experiment_report(
                f"{model_name} - {strategy}",
                technique_desc,
                results,
                observations
            )
            
            with open(f"./results_subtask2/{model_name.replace('/', '_')}_{strategy}_report.md", 'w') as f:
                f.write(report)
            
            print(f"✓ Completed: {strategy}\n")
            
        except Exception as e:
            print(f"✗ Error with {strategy}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def create_technique_description(model_name, strategy):
    """Create technique description for report."""
    descriptions = {
        'zero_shot': f"""
This experiment uses **{model_name}** with zero-shot prompting for multi-label polarization type classification.

**Approach:**
- No training or fine-tuning required
- Model relies purely on pre-trained knowledge
- Single-turn inference per example

**Prompt Strategy:**
Direct instruction with category definitions and output format specification. The model receives:
1. Task description
2. Category definitions (Political, Racial/Ethnic, Religious, Gender/Sexual, Other)
3. Input text
4. Output format instructions (JSON)

**Advantages:**
- No training data required
- Fast to deploy
- Leverages model's world knowledge

**Limitations:**
- May struggle with ambiguous cases
- Dependent on prompt quality
- No task-specific adaptation
""",
        'few_shot': f"""
This experiment uses **{model_name}** with few-shot in-context learning.

**Approach:**
- Provides 4 annotated examples in the prompt
- Examples demonstrate different polarization types
- Model learns from examples without parameter updates

**Prompt Strategy:**
Each prompt includes:
1. Task description
2. 4 diverse examples with explanations
3. Target text to classify
4. Output format

**Example Selection:**
- Single-label examples (Political, Racial/Ethnic, Religious)
- Multi-label example (Political + Other)
- Demonstrates reasoning process

**Advantages:**
- Better than zero-shot for nuanced tasks
- Examples guide model behavior
- Can handle edge cases shown in examples

**Limitations:**
- Longer prompts (higher token cost)
- Example selection impacts performance
- Still no gradient-based learning
""",
        'cot': f"""
This experiment uses **{model_name}** with Chain-of-Thought (CoT) prompting.

**Approach:**
- Prompts model to reason step-by-step
- Encourages explicit intermediate reasoning
- Mirrors human analytical process

**Reasoning Steps:**
1. Identify key entities mentioned
2. Analyze tone and intent
3. Map entities to categories
4. Provide final classification

**Advantages:**
- Improves reasoning on complex cases
- Provides interpretable decision process
- Reduces impulsive/incorrect classifications

**Limitations:**
- Longer generation time
- Requires parsing reasoning from final answer
- May over-explain simple cases
""",
        'structured_cot': f"""
This experiment uses **{model_name}** with Structured Chain-of-Thought prompting.

**Approach:**
- Highly structured reasoning template
- Explicit steps with required format
- Forces systematic analysis

**Structured Steps:**
1. **Entity Extraction**: List all groups/identities
2. **Tone Analysis**: Identify if divisive/attacking/marginalizing
3. **Category Mapping**: For EACH category, explicit YES/NO with justification
4. **Final Answer**: JSON output only

**Advantages:**
- Most structured and consistent
- Forces complete analysis of all categories
- Reduces omission errors
- Highly interpretable

**Limitations:**
- Longest prompts
- Most tokens generated
- Potential for verbose responses
"""
    }
    
    return descriptions.get(strategy, f"Using {model_name} with {strategy} strategy")


def create_observations(results, strategy):
    """Create observations for report."""
    obs = f"""
**Strategy-Specific Observations:**

The **{strategy}** approach achieved:
- Micro F1: {results['f1_micro']:.4f}
- Macro F1: {results['f1_macro']:.4f}
- Subset Accuracy: {results['subset_accuracy']:.4f}

**Performance Characteristics:**
"""
    
    if strategy == 'zero_shot':
        obs += """
- Fast inference (single-turn)
- Relies on model's prior knowledge
- May struggle with subtle distinctions
- Good baseline for comparison
"""
    elif strategy == 'few_shot':
        obs += """
- Better guided by examples
- Learns from demonstrations
- More consistent with shown patterns
- Improvement over zero-shot expected
"""
    elif strategy == 'cot':
        obs += """
- Shows reasoning process
- Better for complex cases
- May improve recall on minority labels
- Interpretable decision path
"""
    elif strategy == 'structured_cot':
        obs += """
- Most systematic approach
- Forces analysis of all categories
- Reduces omission errors
- Highest interpretability
- Best for difficult multi-label cases
"""
    
    return obs


def main():
    """Main execution function."""
    
    print("="*70)
    print("LLM-BASED POLARIZATION TYPE CLASSIFICATION")
    print("Experiment 2: Prompting Strategies with Llama")
    print("="*70)
    
    
    print("\nLoading data...")
    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask2/train/eng.csv',
        '../data/test_phase/subtask2/dev/eng.csv',
        '../data/test_phase/subtask2/test/eng.csv'
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")
    
   
    if not OLLAMA_AVAILABLE:
      
        return
    
   
    model_name = "llama3" 
    
   
    strategies = [
        'zero_shot',
        'few_shot', 
        'cot'
    ]
    
    results = run_llm_experiment(
        train_df,
        dev_df,
        model_name=model_name,
        prompt_strategies=strategies,
        use_ollama=True
    )
    
    # Compare strategies
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    
    comparison_data = []
    for strategy, data in results.items():
        res = data['results']
        comparison_data.append({
            'Strategy': strategy,
            'Micro F1': f"{res['f1_micro']:.4f}",
            'Macro F1': f"{res['f1_macro']:.4f}",
            'Subset Acc': f"{res['subset_accuracy']:.4f}",
            'Hamming Loss': f"{res['hamming_loss']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    comparison_df.to_csv(f"./results_subtask2/{model_name}_strategy_comparison.csv", index=False)
    

    print(f"Results saved in ./results_subtask2/")


if __name__ == "__main__":
    os.makedirs("./results_subtask2", exist_ok=True)
    main()
