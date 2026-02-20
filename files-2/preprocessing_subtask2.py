"""
Shared preprocessing utilities for polarization type detection (Subtask 2 - Multi-label).
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
    return ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']


def get_label_distribution(df, label_cols=None):
    """
    Get label distribution from a dataframe for multi-label classification.
        
    Returns pd.DataFrame: label counts and percentages
    """
    if label_cols is None:
        label_cols = get_label_columns()
    
    counts = df[label_cols].sum()
    percentages = (counts / len(df) * 100).round(2)
    
    result = pd.DataFrame({
        'Label': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values
    })
    
    return result


def get_label_combinations(df, label_cols=None):
    """
    Get distribution of label combinations in the dataset
        
    Returns pd.DataFrame: Combination counts
    """
    if label_cols is None:
        label_cols = get_label_columns()
    
    # string representation of each label combination
    df_copy = df.copy()
    df_copy['combination'] = df_copy[label_cols].apply(
        lambda row: '-'.join([label for label, val in zip(label_cols, row) if val == 1]), 
        axis=1
    )
    df_copy['combination'] = df_copy['combination'].replace('', 'No Labels')
    
    combination_counts = df_copy['combination'].value_counts()
    
    return pd.DataFrame({
        'Combination': combination_counts.index,
        'Count': combination_counts.values
    })


def analyze_dataset(df, dataset_name="Dataset", label_cols=None):
    
    if label_cols is None:
        label_cols = get_label_columns()
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Analysis")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    
    
    print("\nLabel Distribution:")
    label_dist = get_label_distribution(df, label_cols)
    print(label_dist.to_string(index=False))
    
    # Multi label statistics
    df_labels = df[label_cols]
    labels_per_sample = df_labels.sum(axis=1)
    
    print("\nLabels per Sample:")
    print(f"  Average: {labels_per_sample.mean():.2f}")
    print(f"  Min: {labels_per_sample.min()}")
    print(f"  Max: {labels_per_sample.max()}")
    print(f"  Samples with 0 labels: {(labels_per_sample == 0).sum()}")
    print(f"  Samples with 1 label: {(labels_per_sample == 1).sum()}")
    print(f"  Samples with 2+ labels: {(labels_per_sample >= 2).sum()}")
    
    # Top label combinations
    print("\nTop 10 Label Combinations:")
    combinations = get_label_combinations(df, label_cols)
    print(combinations.head(10).to_string(index=False))
    print(f"{'='*60}\n")
