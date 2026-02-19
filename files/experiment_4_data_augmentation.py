"""
Experiment 4: Data Augmentation and Class Balancing
Testing techniques to handle class imbalance and expand training data.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import random


from preprocessing import load_and_preprocess_data, get_class_distribution
from evaluation_utils import (
    evaluate_model, 
    plot_confusion_matrix, 
    save_results,
    create_experiment_report
)


class TextAugmenter:
    """Simple text augmentation techniques."""
    
    @staticmethod
    def synonym_replacement(text, n=1):
        """Replace n words with synonyms."""
        
        synonym_dict = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'like': ['enjoy', 'love', 'prefer'],
            'think': ['believe', 'feel', 'consider'],
            'very': ['extremely', 'really', 'quite'],
        }
        
        words = text.split()
        for _ in range(n):
            replaceable_words = [w for w in words if w.lower() in synonym_dict]
            if replaceable_words:
                word = random.choice(replaceable_words)
                idx = words.index(word)
                words[idx] = random.choice(synonym_dict[word.lower()])
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text, p=0.1):
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    @staticmethod
    def random_swap(text, n=1):
        """Randomly swap two words n times"""
        words = text.split()
        for _ in range(n):
            if len(words) < 2:
                break
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    @staticmethod
    def back_translation_simulation(text):
        """Simulate back-translation by simple word reordering"""
        words = text.split()
        # Randomly reorder some phrases
        if len(words) > 3 and random.random() > 0.5:
            mid = len(words) // 2
            words = words[mid:] + words[:mid]
        return ' '.join(words)


def augment_minority_class(train_df, target_col='polarization', augmentation_factor=2):
    """
    Augment the minority class using text augmentation.
    
    Args:
        train_df: Training dataframe
        target_col: Name of target column
        augmentation_factor: How many augmented samples per original sample
        
    Returns:
        Augmented dataframe
    """
    print("Augmenting minority class...")
    
    # Find minority class
    class_counts = train_df[target_col].value_counts()
    minority_class = class_counts.idxmin()
    
    print(f"Original class distribution:")
    print(class_counts)
    
    # Get minority samples
    minority_samples = train_df[train_df[target_col] == minority_class].copy()
    
    augmenter = TextAugmenter()
    augmented_samples = []
    
    for _ in range(augmentation_factor):
        for idx, row in minority_samples.iterrows():
            #  random augmentation
            aug_choice = random.choice(['synonym', 'deletion', 'swap'])
            
            if aug_choice == 'synonym':
                aug_text = augmenter.synonym_replacement(row['text'], n=2)
            elif aug_choice == 'deletion':
                aug_text = augmenter.random_deletion(row['text'], p=0.1)
            else:
                aug_text = augmenter.random_swap(row['text'], n=1)
            
            augmented_samples.append({
                'id': f"aug_{idx}_{_}",
                'text': aug_text,
                target_col: row[target_col]
            })
    
    #  original and augmented data
    aug_df = pd.DataFrame(augmented_samples)
    combined_df = pd.concat([train_df, aug_df], ignore_index=True)
    
    print(f"\nAugmented class distribution:")
    print(combined_df[target_col].value_counts())
    print(f"Added {len(aug_df)} augmented samples")
    
    return combined_df


def experiment_baseline_imbalanced(train_df, dev_df):
    """Baseline with no balancing."""
    print("\n" + "="*60)
    print("Experiment 4.1: Baseline (No Balancing)")
    print("="*60 + "\n")
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    model.fit(train_df['text'], train_df['polarization'])
    preds = model.predict(dev_df['text'])
    
    return model, preds


def experiment_class_weight(train_df, dev_df):
    """ class weights to handle imbalance."""
    print("\n" + "="*60)
    print("Experiment 4.2: Class Weight Balancing")
    print("="*60 + "\n")
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model.fit(train_df['text'], train_df['polarization'])
    preds = model.predict(dev_df['text'])
    
    return model, preds


def experiment_smote(train_df, dev_df):
    """ SMOTE for oversampling."""
    print("\n" + "="*60)
    print("Experiment 4.3: SMOTE Oversampling")
    print("="*60 + "\n")
    
    # Extract features
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['polarization']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original samples: {X_train.shape[0]}")
    print(f"Resampled samples: {X_resampled.shape[0]}")
    
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_resampled, y_resampled)
    
    
    X_dev = vectorizer.transform(dev_df['text'])
    preds = clf.predict(X_dev)
    
    return {'vectorizer': vectorizer, 'clf': clf}, preds


def experiment_random_oversampling(train_df, dev_df):
    """Use random oversampling."""
    print("\n" + "="*60)
    print("Experiment 4.4: Random Oversampling")
    print("="*60 + "\n")
    
    # Extract features
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['polarization']
    
    # Apply random oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    print(f"Original samples: {X_train.shape[0]}")
    print(f"Resampled samples: {X_resampled.shape[0]}")
    
   
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_resampled, y_resampled)
    
   
    X_dev = vectorizer.transform(dev_df['text'])
    preds = clf.predict(X_dev)
    
    return {'vectorizer': vectorizer, 'clf': clf}, preds


def experiment_undersampling(train_df, dev_df):
    """Use random undersampling."""
    print("\n" + "="*60)
    print("Experiment 4.5: Random Undersampling")
    print("="*60 + "\n")
    
    # Extract features
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['polarization']
    
    # Apply random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"Original samples: {X_train.shape[0]}")
    print(f"Resampled samples: {X_resampled.shape[0]}")
    
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_resampled, y_resampled)
    
    
    X_dev = vectorizer.transform(dev_df['text'])
    preds = clf.predict(X_dev)
    
    return {'vectorizer': vectorizer, 'clf': clf}, preds


def experiment_text_augmentation(train_df, dev_df):
    """Use text augmentation for minority class."""
    print("\n" + "="*60)
    print("Experiment 4.6: Text Augmentation")
    print("="*60 + "\n")
    
    # Augment data
    augmented_df = augment_minority_class(train_df, augmentation_factor=2)
    
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model.fit(augmented_df['text'], augmented_df['polarization'])
    preds = model.predict(dev_df['text'])
    
    return model, preds


def main():

    train_df, dev_df, test_df = load_and_preprocess_data(
        '../data/test_phase/subtask1/train/eng.csv',
        '../data/test_phase/subtask1/dev/eng.csv',
        '../data/test_phase/subtask1/test/eng.csv'
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")
    
    
    print("\nOriginal class distribution:")
    print(get_class_distribution(train_df))
    
    all_results = []
    
    # Experiment 4.1: Baseline (no balancing)
    model, preds = experiment_baseline_imbalanced(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Baseline (Imbalanced)")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Baseline Imbalanced",
                         save_path="./results/baseline_imbalanced_confusion.png")
    save_results(results, "./results/baseline_imbalanced_results.json")
    
    # Experiment 4.2: Class weights
    model, preds = experiment_class_weight(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Class Weight Balanced")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Class Weight Balanced",
                         save_path="./results/class_weight_confusion.png")
    save_results(results, "./results/class_weight_results.json")
    
    # Experiment 4.3: SMOTE
    model, preds = experiment_smote(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "SMOTE Oversampling")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "SMOTE",
                         save_path="./results/smote_confusion.png")
    save_results(results, "./results/smote_results.json")
    
    technique_desc = """
This experiment uses **SMOTE (Synthetic Minority Over-sampling Technique)**:

**How SMOTE Works:**
1. For each minority class sample, find k nearest neighbors in feature space
2. Generate synthetic samples along the line segments joining the sample and its neighbors
3. Balance the dataset by creating synthetic minority samples

**Configuration:**
- Feature space: TF-IDF (10,000 features, 1-2 grams)
- Synthetic samples created to match majority class size
- Random state: 42 for reproducibility

**Advantages:**
- Creates informed synthetic samples (not random duplicates)
- Maintains feature space structure
- Reduces overfitting compared to simple duplication
"""
    
    observations = f"""
- SMOTE generates synthetic samples in TF-IDF feature space.
- Original training samples: {len(train_df)}
- After SMOTE: approximately {len(train_df) * 2} (balanced classes)
- Macro F1: {results['macro_f1']:.4f}
- {'SMOTE improves minority class recall' if results['per_class_metrics']['class_1']['recall'] > 0.65 else 'May need parameter tuning'}
- Synthetic samples help model learn minority class patterns
"""
    
    report = create_experiment_report("SMOTE Oversampling", technique_desc, results, observations)
    with open("./results/smote_report.md", 'w') as f:
        f.write(report)
    
    # Experiment 4.4: Random Oversampling
    model, preds = experiment_random_oversampling(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Random Oversampling")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Random Oversampling",
                         save_path="./results/random_oversample_confusion.png")
    save_results(results, "./results/random_oversample_results.json")
    
    # Experiment 4.5: Undersampling
    model, preds = experiment_undersampling(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Random Undersampling")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Random Undersampling",
                         save_path="./results/undersampling_confusion.png")
    save_results(results, "./results/undersampling_results.json")
    
    # Experiment 4.6: Text Augmentation
    model, preds = experiment_text_augmentation(train_df, dev_df)
    results = evaluate_model(dev_df['polarization'], preds, "Text Augmentation")
    all_results.append(results)
    
    plot_confusion_matrix(dev_df['polarization'], preds, "Text Augmentation",
                         save_path="./results/text_augmentation_confusion.png")
    save_results(results, "./results/text_augmentation_results.json")
    
    technique_desc = """
This experiment uses **Text Augmentation** to expand the minority class:

**Augmentation Techniques:**
1. **Synonym Replacement**: Replace words with synonyms
2. **Random Deletion**: Randomly remove words (10% probability)
3. **Random Swap**: Swap positions of random word pairs

**Process:**
- Identify minority class samples
- Generate 2 augmented versions per sample
- Apply random augmentation technique for variation
- Combine with original data

**Advantage:**
Creates natural text variations that maintain semantic meaning while adding diversity.
"""
    
    observations = f"""
- Text augmentation creates linguistic variations of minority samples.
- Augmentation factor: 2x (doubles minority class size)
- Macro F1: {results['macro_f1']:.4f}
- {'Text augmentation provides natural sample diversity' if results['macro_f1'] > 0.65 else 'Simple augmentations may not capture complex patterns'}
- More sophisticated augmentation (e.g., back-translation, paraphrasing) could improve results
"""
    
    report = create_experiment_report("Text Augmentation", technique_desc, results, observations)
    with open("./results/text_augmentation_report.md", 'w') as f:
        f.write(report)
    
    #  all approaches
    print("\n" + "="*60)
    print("CLASS BALANCING TECHNIQUES COMPARISON")
    print("="*60)
    
    from evaluation_utils import compare_models
    comparison_df = compare_models(all_results)
    comparison_df.to_csv("./results/class_balancing_comparison.csv", index=False)
    
    print("Results saved in ./results/ directory")


if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    main()
