"""
This script runs all experiments sequentially and generates a comparison report.
"""
import subprocess
import json
import pandas as pd
from datetime import datetime
import os


def run_experiment(script_name, experiment_name):
    """
    Run an experiment script and capture output
    """
    print("\n" + "="*80)
    print(f"Running: {experiment_name}")
    print("="*80)
    
    try:
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"{experiment_name} completed successfully!")
            return True
        else:
            print(f"{experiment_name} failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{experiment_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Error running {experiment_name}: {str(e)}")
        return False


def load_all_results():
    """Load all result JSON files from the results directory."""
    results = []
    results_dir = "./results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        return results
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return results


def create_master_report(results):
    """master report comparing all experiments"""
    
    if not results:
        print("No results found to create master report!")
        return
    
    # Sort by macro F1
    results_sorted = sorted(results, key=lambda x: x['macro_f1'], reverse=True)
    
    # Create comparison dataframe
    comparison_data = []
    for result in results_sorted:
        comparison_data.append({
            "Model": result["model_name"],
            "Accuracy": f"{result['accuracy']:.4f}",
            "Macro F1": f"{result['macro_f1']:.4f}",
            "Weighted F1": f"{result['weighted_f1']:.4f}",
            "Class 0 F1": f"{result['per_class_metrics']['class_0']['f1']:.4f}",
            "Class 1 F1": f"{result['per_class_metrics']['class_1']['f1']:.4f}",
            "Timestamp": result['timestamp']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Generate markdown report
    report = f"""# Polarization Detection - Master Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

Total experiments conducted: **{len(results)}**

### Top 5 Models by Macro F1

{df.head(5).to_markdown(index=False)}

### Top 3 Models

**Best Model:** {results_sorted[0]['model_name']}
- Macro F1: {results_sorted[0]['macro_f1']:.4f}
- Accuracy: {results_sorted[0]['accuracy']:.4f}

**Second Best:** {results_sorted[1]['model_name'] if len(results_sorted) > 1 else 'N/A'}
- Macro F1: {results_sorted[1]['macro_f1']:.4f if len(results_sorted) > 1 else 'N/A'}
- Accuracy: {results_sorted[1]['accuracy']:.4f if len(results_sorted) > 1 else 'N/A'}

**Third Best:** {results_sorted[2]['model_name'] if len(results_sorted) > 2 else 'N/A'}
- Macro F1: {results_sorted[2]['macro_f1']:.4f if len(results_sorted) > 2 else 'N/A'}
- Accuracy: {results_sorted[2]['accuracy']:.4f if len(results_sorted) > 2 else 'N/A'}

## All Models Ranked by Performance

{df.to_markdown(index=False)}

## Key insights

### Best Performing Techniques

1. **{results_sorted[0]['model_name']}**
   - Achieved highest macro F1 of {results_sorted[0]['macro_f1']:.4f}
   - Class 0 F1: {results_sorted[0]['per_class_metrics']['class_0']['f1']:.4f}
   - Class 1 F1: {results_sorted[0]['per_class_metrics']['class_1']['f1']:.4f}
   - Balance score: {abs(results_sorted[0]['per_class_metrics']['class_0']['f1'] - results_sorted[0]['per_class_metrics']['class_1']['f1']):.4f}

### Performance Distribution

- Models with Macro F1 > 0.70: {sum(1 for r in results if r['macro_f1'] > 0.70)}
- Models with Macro F1 > 0.65: {sum(1 for r in results if r['macro_f1'] > 0.65)}
- Models with Macro F1 > 0.60: {sum(1 for r in results if r['macro_f1'] > 0.60)}

### Class Balance Analysis

Average class F1 difference: {np.mean([abs(r['per_class_metrics']['class_0']['f1'] - r['per_class_metrics']['class_1']['f1']) for r in results]):.4f}

Models with good class balance (F1 difference < 0.05):
{sum(1 for r in results if abs(r['per_class_metrics']['class_0']['f1'] - r['per_class_metrics']['class_1']['f1']) < 0.05)}

## Recommendations

### For Production Deployment

**Recommended Model:** {results_sorted[0]['model_name']}

**Rationale:**
- Highest macro F1 score
- {'Well-balanced' if abs(results_sorted[0]['per_class_metrics']['class_0']['f1'] - results_sorted[0]['per_class_metrics']['class_1']['f1']) < 0.05 else 'Acceptable'} performance across classes
- {'Strong generalization capability' if results_sorted[0]['macro_f1'] > 0.70 else 'Moderate performance'}

"""
    
    # Save report
    with open("./results/MASTER_REPORT.md", 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("MASTER REPORT GENERATED")
    print("="*80)
    print(f"Location: ./results/MASTER_REPORT.md")
    print("\nTop 3 Models:")
    for i, result in enumerate(results_sorted[:3], 1):
        print(f"{i}. {result['model_name']}: Macro F1 = {result['macro_f1']:.4f}")
    
    return report


def main():
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    os.makedirs("./results", exist_ok=True)
    
    
    experiments = [
        ("experiment_1_bert_comparison.py", "BERT Model Comparison"),
        ("experiment_2_ensembles.py", "Ensemble Methods"),
        ("experiment_3_feature_engineering.py", "Feature Engineering"),
        ("experiment_4_data_augmentation.py", "Data Augmentation & Class Balancing"),
        ("experiment_5_hyperparameter_tuning.py", "Hyperparameter Tuning"),
    ]
    
    
    successful_experiments = []
    failed_experiments = []
    
    # Run each experiment
    for script_name, experiment_name in experiments:
        if os.path.exists(script_name):
            success = run_experiment(script_name, experiment_name)
            if success:
                successful_experiments.append(experiment_name)
            else:
                failed_experiments.append(experiment_name)
        else:
            print(f"⚠️  Script not found: {script_name}")
            failed_experiments.append(experiment_name)
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUITE SUMMARY")
    print("="*80)
    print(f"\n Successful: {len(successful_experiments)}/{len(experiments)}")
    for exp in successful_experiments:
        print(f"   - {exp}")
    
    if failed_experiments:
        print(f"\n Failed: {len(failed_experiments)}/{len(experiments)}")
        for exp in failed_experiments:
            print(f"   - {exp}")
    
    
    print("\n" + "="*80)
    print("COMPILING RESULTS")
    print("="*80)
    
    all_results = load_all_results()
    print(f"\nLoaded {len(all_results)} result files")
    
    if all_results:
        # Create master report
        create_master_report(all_results)
        
        # Create overall comparison CSV
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                "Model": result["model_name"],
                "Accuracy": result["accuracy"],
                "Macro_F1": result["macro_f1"],
                "Weighted_F1": result["weighted_f1"],
                "Class_0_F1": result["per_class_metrics"]["class_0"]["f1"],
                "Class_1_F1": result["per_class_metrics"]["class_1"]["f1"],
                "Timestamp": result["timestamp"]
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Macro_F1", ascending=False)
        df.to_csv("./results/ALL_EXPERIMENTS_COMPARISON.csv", index=False)
        print("Overall comparison saved to: ./results/ALL_EXPERIMENTS_COMPARISON.csv")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import numpy as np  
    main()
