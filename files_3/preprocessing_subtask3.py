"""
Shared preprocessing utilities for polarization manifestation detection (Subtask 3 - Multi-label).
"""
import re
import pandas as pd
import numpy as np


def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing mentions (@username)
    - Removing special characters (keeping basic punctuation)
    """
    # lowercase
    text = text.lower()
    # remove mentions
    text = re.sub(r"@\w+", "", text)
    # remove special characters (keep basic)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    return text


def load_and_preprocess_data(train_path, dev_path, test_path):
    """
    Load and preprocess all datasets
    """
    # Load data
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    
    # Apply preprocessing
    train_df['text'] = train_df['text'].apply(preprocess_text)
    dev_df['text'] = dev_df['text'].apply(preprocess_text)
    test_df['text'] = test_df['text'].apply(preprocess_text)
    
    return train_df, dev_df, test_df


def get_label_columns():
    """
    list of label columns for multi-label classification
    """
    return ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']


def get_label_descriptions():
    """
     descriptions of each manifestation type
    
    Returns dict: dictionary mapping label names to descriptions
    """
    return {
        'stereotype': 'Generalizes characteristics to all group members, ignoring individual differences',
        'vilification': 'Defames or demonizes a group through exaggeration, misrepresentation, or biased framing',
        'dehumanization': 'Strips individuals of human qualities by comparing to animals, objects, or denying humanity',
        'extreme_language': 'Uses absolutist terms (always, never, worst, best) or dichotomous framing (us vs them)',
        'lack_of_empathy': 'Shows no understanding for others\' perspectives, marginalizes alternative viewpoints',
        'invalidation': 'Denies or rejects the identity or existence of certain people or groups'
    }


def get_label_distribution(df, label_cols=None):
    """
    Get label distribution from a dataframe for multi-label classification
        
    Returns Label counts and percentages
    """
    if label_cols is None:
        label_cols = get_label_columns()
    
    counts = df[label_cols].sum()
    percentages = (counts / len(df) * 100).round(2)
    
    result = pd.DataFrame({
        'Manifestation': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values
    })
    
    return result


def get_label_combinations(df, label_cols=None):
    """
    Get distribution of manifestation combinations in the dataser
    """
    if label_cols is None:
        label_cols = get_label_columns()
    
    # string representation of each label combination
    df_copy = df.copy()
    df_copy['combination'] = df_copy[label_cols].apply(
        lambda row: ' + '.join([label for label, val in zip(label_cols, row) if val == 1]), 
        axis=1
    )
    df_copy['combination'] = df_copy['combination'].replace('', 'No Manifestations')
    
    combination_counts = df_copy['combination'].value_counts()
    
    return pd.DataFrame({
        'Combination': combination_counts.index,
        'Count': combination_counts.values,
        'Percentage': (combination_counts.values / len(df) * 100).round(2)
    })


def get_label_correlations(df, label_cols=None):
    """
    Calculate correlations between different manifestation types
        
    Returns Correlation matrix
    """
    if label_cols is None:
        label_cols = get_label_columns()
    
    return df[label_cols].corr()


def analyze_dataset(df, dataset_name="Dataset", label_cols=None):
    """
    Print comprehensive dataset analysis for multi-label manifestation classification
    """
    if label_cols is None:
        label_cols = get_label_columns()
    
    print(f"\n{'='*80}")
    print(f"{dataset_name} Analysis - Polarization Manifestations")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    
    
    print("\nManifestation Type Distribution:")
    label_dist = get_label_distribution(df, label_cols)
    print(label_dist.to_string(index=False))
    
    # Multi label statistics
    df_labels = df[label_cols]
    labels_per_sample = df_labels.sum(axis=1)
    
    print("\nManifestations per Sample:")
    print(f"  Average: {labels_per_sample.mean():.2f}")
    print(f"  Min: {labels_per_sample.min()}")
    print(f"  Max: {labels_per_sample.max()}")
    print(f"  Samples with 0 manifestations: {(labels_per_sample == 0).sum()} ({(labels_per_sample == 0).sum()/len(df)*100:.1f}%)")
    print(f"  Samples with 1 manifestation: {(labels_per_sample == 1).sum()} ({(labels_per_sample == 1).sum()/len(df)*100:.1f}%)")
    print(f"  Samples with 2+ manifestations: {(labels_per_sample >= 2).sum()} ({(labels_per_sample >= 2).sum()/len(df)*100:.1f}%)")
    
    
    print("\nTop 10 Manifestation Combinations:")
    combinations = get_label_combinations(df, label_cols)
    print(combinations.head(10).to_string(index=False))
    
    
    print("\nManifestation Correlations (top pairs):")
    corr_matrix = get_label_correlations(df, label_cols)
    
    
    correlations = []
    for i in range(len(label_cols)):
        for j in range(i+1, len(label_cols)):
            correlations.append({
                'Pair': f"{label_cols[i]} <-> {label_cols[j]}",
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    print(corr_df.head(5).to_string(index=False))
    
    print(f"{'='*80}\n")


def get_manifestation_examples():
   
    return {
        'stereotype': [
            "Keywords: 'all', 'every', 'typical', 'always'",
            "Pattern: Broad generalizations about groups",
            "Example indicators: Group-wide attributions ignoring individuals"
        ],
        'vilification': [
            "Keywords: 'evil', 'dangerous', 'threat', 'destroy'",
            "Pattern: Demonization through exaggeration",
            "Example indicators: Fear-inducing language, biased framing"
        ],
        'dehumanization': [
            "Keywords: Animal comparisons, object metaphors",
            "Pattern: Denying human qualities",
            "Example indicators: 'vermin', 'animals', 'things', 'it'"
        ],
        'extreme_language': [
            "Keywords: 'always', 'never', 'worst', 'best', 'all', 'none'",
            "Pattern: Absolutist or dichotomous framing",
            "Example indicators: 'us vs them', 'right vs wrong', extreme modifiers"
        ],
        'lack_of_empathy': [
            "Keywords: Dismissive language, mockery",
            "Pattern: Refusing to acknowledge other perspectives",
            "Example indicators: Marginalization of viewpoints, lack of understanding"
        ],
        'invalidation': [
            "Keywords: 'not real', 'doesn't exist', 'fake', 'illegitimate'",
            "Pattern: Denying identity or existence",
            "Example indicators: Dismissing legitimacy or right to exist"
        ]
    }


def print_manifestation_guide():
    descriptions = get_label_descriptions()
    examples = get_manifestation_examples()
    
    print("\n" + "="*80)
    print("POLARIZATION MANIFESTATION TYPES - REFERENCE GUIDE")
    print("="*80)
    
    for i, (label, desc) in enumerate(descriptions.items(), 1):
        print(f"\n{i}. {label.upper().replace('_', ' ')}")
        print(f"   Description: {desc}")
        print(f"   Characteristics:")
        for char in examples[label]:
            print(f"     • {char}")
    
    print("\n" + "="*80 + "\n")
