"""
Shared preprocessing utilities for polarization detection task.
"""
import re
import pandas as pd


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


def get_class_distribution(df, label_col='polarization'):
    """
    Get class distribution from a dataframe.
    """
    return df[label_col].value_counts()
