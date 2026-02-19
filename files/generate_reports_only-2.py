"""
 Report Generator
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_all_results(results_dir="./results"):
    """Load all result json files from the results directory."""
    results = []
    
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
                    print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return results


def df_to_markdown(df):
    """Convert dataframe to markdown table without tabulate dependency."""
    headers = '| ' + ' | '.join(df.columns) + ' |'
    separator = '|' + '|'.join(['---' for _ in df.columns]) + '|'
    rows = []
    for _, row in df.iterrows():
        rows.append('| ' + ' | '.join(str(v) for v in row.values) + ' |')
    return '\n'.join([headers, separator] + rows)


def create_master_report(results, results_dir="./results"):
    """master report comparing all experiments"""
    
    if not results:
        print("No results found to create master report!")
        return
    
    print(f"\nProcessing {len(results)} result files...")
    
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
    
    #  markdown report
    report = f"""# Polarization Detection - Master Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

Total experiments conducted: **{len(results)}**

### Top 5 Models by Macro F1

{df_to_markdown(df.head(5))}

### Top 3 Models

**Best Model:** {results_sorted[0]['model_name']}
- Macro F1: {results_sorted[0]['macro_f1']:.4f}
- Accuracy: {results_sorted[0]['accuracy']:.4f}

**Second Best:** {results_sorted[1]['model_name'] if len(results_sorted) > 1 else 'N/A'}
- Macro F1: {f"{results_sorted[1]['macro_f1']:.4f}" if len(results_sorted) > 1 else 'N/A'}
- Accuracy: {f"{results_sorted[1]['accuracy']:.4f}" if len(results_sorted) > 1 else 'N/A'}

**Third Best:** {results_sorted[2]['model_name'] if len(results_sorted) > 2 else 'N/A'}
- Macro F1: {f"{results_sorted[2]['macro_f1']:.4f}" if len(results_sorted) > 2 else 'N/A'}
- Accuracy: {f"{results_sorted[2]['accuracy']:.4f}" if len(results_sorted) > 2 else 'N/A'}

## All Models Ranked by Performance

{df_to_markdown(df)}

## Key Insights

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

---

"""
    
    # Save report
    report_path = os.path.join(results_dir, "MASTER_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("MASTER REPORT GENERATED")
    print("="*80)
    print(f"Location: {report_path}")
    print(f"\nTop 3 Models:")
    for i, result in enumerate(results_sorted[:3], 1):
        print(f"{i}. {result['model_name']}: Macro F1 = {result['macro_f1']:.4f}")
    
    return report, df


def create_comparison_csv(results, results_dir="./results"):
    """Create overall comparison CSV."""
    comparison_data = []
    for result in results:
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
    
    csv_path = os.path.join(results_dir, "ALL_EXPERIMENTS_COMPARISON.csv")
    df.to_csv(csv_path, index=False)
    print(f"Overall comparison saved to: {csv_path}")
    
    return df


def main():
    
    
    print("="*80)
    print("GENERATING REPORTS FROM EXISTING RESULTS")
    print("="*80)
    print()
    
    
    
    all_results = load_all_results("./results")
    
    if not all_results:
        print("\n No result files found in ./results/ directory!")
        return
    
    print(f"\nLoaded {len(all_results)} result files")
    
    # Create master report
    print("\nGenerating master report...")
    report, df = create_master_report(all_results)
    
    # Create comparison CSV
    print("\nGenerating comparison CSV...")
    create_comparison_csv(all_results)
    
    print("\n" + "="*80)
    print("ALL REPORTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - ./results/MASTER_REPORT.md")
    print("  - ./results/ALL_EXPERIMENTS_COMPARISON.csv")


if __name__ == "__main__":
    main()
