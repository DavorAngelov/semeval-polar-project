"""
Experiment 3: Advanced Feature Engineering
Testing different feature extraction and engineering techniques.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack
import re

# Import shared utilities
from preprocessing import load_and_preprocess_data
from evaluation_utils import (
    evaluate_model, 
    plot_confusion_matrix, 
    save_results,
    create_experiment_report
)


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """Extract statistical features from text."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            stats = {
                'length': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0,
            }
            features.append(list(stats.values()))
        return np.array(features)


class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract sentiment-based features."""
    
    def __init__(self):
        # Simple sentiment word lists
        self.positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                                   'love', 'best', 'happy', 'perfect', 'brilliant'])
        self.negative_words = set(['bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 
                                   'disgusting', 'poor', 'disappointing', 'useless'])
        self.polarizing_words = set(['always', 'never', 'everyone', 'nobody', 'all', 'none',
                                     'must', 'should', 'absolutely', 'completely', 'totally'])
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            words = text.lower().split()
            stats = {
                'positive_count': sum(1 for w in words if w in self.positive_words),
                'negative_count': sum(1 for w in words if w in self.negative_words),
                'polarizing_count': sum(1 for w in words if w in self.polarizing_words),
                'sentiment_ratio': (sum(1 for w in words if w in self.positive_words) - 
                                   sum(1 for w in words if w in self.negative_words)) / len(words) if words else 0,
            }
            features.append(list(stats.values()))
        return np.array(features)


def experiment_char_ngrams(train_df, dev_df):
    """Character n-grams feature extraction."""
    print("\n" + "="*60)
    print("Experiment 3.1: Character N-grams")
    print("="*60 + "\n")
    
    model = Pipeline([
        ('char_tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=10000
        )),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model.fit(train_df['text'], train_df['polarization'])
    preds = model.predict(dev_df['text'])
    
    return model, preds


def experiment_word_char_combined(train_df, dev_df):
    """Combined word and character n-grams."""
    print("\n" + "="*60)
    print("Experiment 3.2: Combined Word + Character N-grams")
    print("="*60 + "\n")
    
    # Combine word and character features
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 4),
        max_features=5000
    )
    
   
    word_features_train = word_vectorizer.fit_transform(train_df['text'])
    char_features_train = char_vectorizer.fit_transform(train_df['text'])
    
    word_features_dev = word_vectorizer.transform(dev_df['text'])
    char_features_dev = char_vectorizer.transform(dev_df['text'])
    
    # Combine features
    X_train = hstack([word_features_train, char_features_train])
    X_dev = hstack([word_features_dev, char_features_dev])
    
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, train_df['polarization'])
    
    preds = clf.predict(X_dev)
    
    return {'word_vec': word_vectorizer, 'char_vec': char_vectorizer, 'clf': clf}, preds


def experiment_statistical_features(train_df, dev_df):
    """Statistical text features combined with TF-IDF."""
    print("\n" + "="*60)
    print("Experiment 3.3: TF-IDF + Statistical Features")
    print("="*60 + "\n")
    
    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_train = tfidf.fit_transform(train_df['text'])
    tfidf_dev = tfidf.transform(dev_df['text'])
    
    # Statistical features
    stats_extractor = TextStatsExtractor()
    stats_train = stats_extractor.transform(train_df['text'])
    stats_dev = stats_extractor.transform(dev_df['text'])
    
    # Combine
    X_train = hstack([tfidf_train, stats_train])
    X_dev = hstack([tfidf_dev, stats_dev])
    
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, train_df['polarization'])
    
    preds = clf.predict(X_dev)
    
    return {'tfidf': tfidf, 'stats': stats_extractor, 'clf': clf}, preds


def experiment_sentiment_features(train_df, dev_df):
    """Sentiment-based features combined with TF-IDF"""
    print("\n" + "="*60)
    print("Experiment 3.4: TF-IDF + Sentiment Features")
    print("="*60 + "\n")
    
    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_train = tfidf.fit_transform(train_df['text'])
    tfidf_dev = tfidf.transform(dev_df['text'])
    
    # Sentiment features
    sentiment_extractor = SentimentFeatureExtractor()
    sentiment_train = sentiment_extractor.transform(train_df['text'])
    sentiment_dev = sentiment_extractor.transform(dev_df['text'])
    
    # Combine
    X_train = hstack([tfidf_train, sentiment_train])
    X_dev = hstack([tfidf_dev, sentiment_dev])
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, train_df['polarization'])
    
    preds = clf.predict(X_dev)
    
    return {'tfidf': tfidf, 'sentiment': sentiment_extractor, 'clf': clf}, preds


def experiment_all_features_combined(train_df, dev_df):
    """Combine all feature types."""
    print("\n" + "="*60)
    print("Experiment 3.5: All Features Combined")
    print("="*60 + "\n")
    
    # Word TF-IDF
    word_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    word_train = word_tfidf.fit_transform(train_df['text'])
    word_dev = word_tfidf.transform(dev_df['text'])
    
    # Character TF-IDF
    char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=3000)
    char_train = char_tfidf.fit_transform(train_df['text'])
    char_dev = char_tfidf.transform(dev_df['text'])
    
    # Statistical features
    stats = TextStatsExtractor()
    stats_train = stats.transform(train_df['text'])
    stats_dev = stats.transform(dev_df['text'])
    
    # Sentiment features
    sentiment = SentimentFeatureExtractor()
    sentiment_train = sentiment.transform(train_df['text'])
    sentiment_dev = sentiment.transform(dev_df['text'])
    
    # Combine all r
    X_train = hstack([word_train, char_train, stats_train, sentiment_train])
    X_dev = hstack([word_dev, char_dev, stats_dev, sentiment_dev])
    
    print(f"Combined feature dimensions: {X_train.shape[1]}")
    
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, train_df['polarization'])
    
    preds = clf.predict(X_dev)
    
    return {
        'word_tfidf': word_tfidf,
        'char_tfidf': char_tfidf,
        'stats': stats,
        'sentiment': sentiment,
        'clf': clf
    }, preds


def main():
    
    
    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask1/train/eng.csv',
        '../data/test_phase/subtask1/dev/eng.csv',
        '../data/test_phase/subtask1/test/eng.csv'
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}\n")
    
    all_results = []
    
    # Experiment 3.1: Character N-grams
    model, preds = experiment_char_ngrams(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Character N-grams (3-5)")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Character N-grams",
                         save_path="./results/char_ngrams_confusion.png")
    save_results(results, "./results/char_ngrams_results.json")
    
    technique_desc = """
This experiment uses **Character N-grams** for feature extraction:

**Features:**
- Character-level n-grams (3-5 characters)
- Max features: 10,000
- TF-IDF weighting

**Advantages:**
- Captures misspellings and informal language
- Language-agnostic patterns
- Robust to word variations

**Example:**
Text: "hello" -> ["hel", "ell", "llo", "hell", "ello", "hello"]
"""
    
    observations = f"""
- Character n-grams capture sub-word patterns useful for social media text.
- Particularly effective for handling typos and informal spellings.
- Macro F1: {results['macro_f1']:.4f}
- {'Shows promise for noisy text data' if results['macro_f1'] > 0.60 else 'May need combination with word features'}
"""
    
    report = create_experiment_report("Character N-grams", technique_desc, results, observations)
    with open("./results/char_ngrams_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 3.2: Combined Word + Character
    model, preds = experiment_word_char_combined(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Word + Character N-grams")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Word + Character N-grams",
                         save_path="./results/word_char_confusion.png")
    save_results(results, "./results/word_char_results.json")
    
    technique_desc = """
This experiment **combines word and character n-grams**:

**Features:**
1. **Word n-grams**: Unigrams and bigrams (10,000 features)
2. **Character n-grams**: 3-4 character sequences (5,000 features)
3. Total: 15,000 combined features

**Combination Strategy:**
- Horizontal stacking of feature matrices
- Both feature types contribute to final representation

**Rationale:**
Word features capture semantic meaning, while character features handle spelling variations.
"""
    
    observations = f"""
- Combining word and character features leverages both semantic and sub-word patterns.
- Total feature space: 15,000 dimensions
- Macro F1: {results['macro_f1']:.4f}
- {'Significant improvement over character-only features' if results['macro_f1'] > all_results[0]['macro_f1'] else 'Comparable to single feature type'}
"""
    
    report = create_experiment_report("Word + Character Features", technique_desc, results, observations)
    with open("./results/word_char_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 3.3: Statistical Features
    model, preds = experiment_statistical_features(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "TF-IDF + Statistical Features")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Statistical Features",
                         save_path="./results/statistical_confusion.png")
    save_results(results, "./results/statistical_results.json")
    
    technique_desc = """
This experiment adds **statistical text features** to TF-IDF:

**Statistical Features:**
1. Text length (characters)
2. Word count
3. Average word length
4. Exclamation mark count
5. Question mark count
6. Uppercase letter ratio
7. Punctuation ratio

**Rationale:**
Polarizing content may have distinctive stylistic patterns (e.g., excessive punctuation, capitalization).
"""
    
    observations = f"""
- Statistical features capture writing style and emotional intensity.
- Features like uppercase ratio and punctuation can indicate strong opinions.
- Macro F1: {results['macro_f1']:.4f}
- {'Statistical features provide complementary information' if results['macro_f1'] > 0.65 else 'Impact varies by dataset'}
"""
    
    report = create_experiment_report("Statistical Features", technique_desc, results, observations)
    with open("./results/statistical_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 3.4: Sentiment Features
    model, preds = experiment_sentiment_features(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "TF-IDF + Sentiment Features")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Sentiment Features",
                         save_path="./results/sentiment_confusion.png")
    save_results(results, "./results/sentiment_results.json")
    
    technique_desc = """
This experiment adds **sentiment-based features** to TF-IDF:

**Sentiment Features:**
1. Positive word count
2. Negative word count
3. Polarizing word count (absolute terms like "always", "never")
4. Sentiment ratio (positive - negative)

**Hypothesis:**
Polarizing content may use more extreme sentiment language and absolute statements.
"""
    
    observations = f"""
- Sentiment features capture emotional tone and extreme language.
- Polarizing words (always, never, must) may be indicative of polarization.
- Macro F1: {results['macro_f1']:.4f}
- {'Sentiment features show potential for polarization detection' if results['macro_f1'] > 0.65 else 'May need expanded word lists'}
"""
    
    report = create_experiment_report("Sentiment Features", technique_desc, results, observations)
    with open("./results/sentiment_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 3.5: All Combined
    model, preds = experiment_all_features_combined(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "All Features Combined")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "All Features Combined",
                         save_path="./results/all_features_confusion.png")
    save_results(results, "./results/all_features_results.json")
    
    technique_desc = """
This experiment **combines all feature types**:

**Feature Components:**
1. Word TF-IDF (1-2 grams, 8,000 features)
2. Character TF-IDF (3-4 grams, 3,000 features)
3. Statistical features (7 dimensions)
4. Sentiment features (4 dimensions)

**Total Feature Space:**
- Approximately 11,011 dimensions
- Comprehensive representation of text

**Advantage:**
Captures multiple aspects: semantics, style, patterns, and sentiment.
"""
    
    observations = f"""
- Comprehensive feature engineering combining multiple signal types.
- Rich representation with ~11k features.
- Macro F1: {results['macro_f1']:.4f}
- {'Combined features achieve best performance' if results['macro_f1'] == max(r['macro_f1'] for r in all_results) else 'Some feature types may be redundant'}
- Risk of overfitting with high-dimensional features should be monitored.
"""
    
    report = create_experiment_report("All Features Combined", technique_desc, results, observations)
    with open("./results/all_features_report.md", 'w') as f:
        f.write(report)
    
    # Compare all feature engineering approaches
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPARISON")
    print("="*60)
    
    from evaluation_utils import compare_models
    comparison_df = compare_models(all_results)
    comparison_df.to_csv("./results/feature_engineering_comparison.csv", index=False)
    
    print("Results saved in ./results/ directory")


if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    main()
