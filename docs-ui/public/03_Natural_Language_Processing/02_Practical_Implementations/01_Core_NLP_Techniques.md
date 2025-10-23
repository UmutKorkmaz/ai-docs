---
title: "Natural Language Processing - Core NLP Techniques:"
description: "## 1. Text Preprocessing and Feature Engineering. Comprehensive guide covering transformer models, NLP techniques, algorithm, gradient descent, classificatio..."
keywords: "transformer models, NLP techniques, algorithm, transformer models, NLP techniques, algorithm, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Core NLP Techniques: Practical Implementations

## 1. Text Preprocessing and Feature Engineering

### 1.1 Text Cleaning and Normalization

```python
import re
import string
import unicodedata
from typing import List, Dict, Optional, Set
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from collections import Counter
import pandas as pd

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for NLP applications
    """

    def __init__(self, language: str = 'english'):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        self.stop_words = set(stopwords.words(language))

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing special characters and normalizing unicode
        """
        # Convert to lowercase
        text = text.lower()

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        # Remove accents
        text = ''.join([c for c in text if not unicodedata.combining(c)])

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_special_characters(self, text: str, keep_punctuation: bool = False) -> str:
        """
        Remove special characters, optionally keeping punctuation
        """
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:\'"-]'
        else:
            # Keep only letters and numbers
            pattern = r'[^a-zA-Z0-9\s]'

        return re.sub(pattern, '', text)

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK's word_tokenize
        """
        return word_tokenize(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences
        """
        return sent_tokenize(text)

    def remove_stopwords(self, tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
        """
        Remove stopwords from token list
        """
        stop_words = self.stop_words.copy()
        if custom_stopwords:
            stop_words.update(custom_stopwords)

        return [token for token in tokens if token not in stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens using Porter stemmer
        """
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens using WordNet lemmatizer
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_numbers(self, tokens: List[str]) -> List[str]:
        """
        Remove numeric tokens
        """
        return [token for token in tokens if not token.isdigit()]

    def remove_short_tokens(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """
        Remove tokens shorter than min_length
        """
        return [token for token in tokens if len(token) >= min_length]

    def preprocess_pipeline(self, text: str, options: Dict[str, bool] = None) -> List[str]:
        """
        Complete preprocessing pipeline with configurable options
        """
        if options is None:
            options = {
                'normalize': True,
                'remove_special_chars': True,
                'remove_stopwords': True,
                'lemmatize': True,
                'remove_numbers': True,
                'min_length': 2
            }

        # Normalize text
        if options.get('normalize', True):
            text = self.normalize_text(text)

        # Remove special characters
        if options.get('remove_special_chars', True):
            text = self.remove_special_characters(text)

        # Tokenize
        tokens = self.tokenize_words(text)

        # Remove stopwords
        if options.get('remove_stopwords', True):
            tokens = self.remove_stopwords(tokens)

        # Lemmatize or stem
        if options.get('lemmatize', True):
            tokens = self.lemmatize_tokens(tokens)

        # Remove numbers
        if options.get('remove_numbers', True):
            tokens = self.remove_numbers(tokens)

        # Remove short tokens
        min_length = options.get('min_length', 2)
        if min_length > 0:
            tokens = self.remove_short_tokens(tokens, min_length)

        return tokens

    def extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Extract n-grams from token list
        """
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def build_vocabulary(self, documents: List[List[str]], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from preprocessed documents
        """
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc)

        # Filter by minimum frequency
        vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.items())
                     if count >= min_freq}

        return vocabulary

    def documents_to_bow(self, documents: List[List[str]], vocabulary: Dict[str, int]) -> np.ndarray:
        """
        Convert documents to bag-of-words representation
        """
        num_docs = len(documents)
        vocab_size = len(vocabulary)

        bow_matrix = np.zeros((num_docs, vocab_size))

        for doc_idx, doc in enumerate(documents):
            for word in doc:
                if word in vocabulary:
                    word_idx = vocabulary[word]
                    bow_matrix[doc_idx, word_idx] += 1

        return bow_matrix
```

### 1.2 Advanced Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import spacy

class FeatureExtractor:
    """
    Advanced feature extraction for NLP
    """

    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        """
        Initialize with spaCy model for advanced NLP features
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Spacy model {spacy_model} not found. Using basic features only.")
            self.nlp = None

    def tfidf_features(self, documents: List[str], max_features: int = 10000,
                      ngram_range: tuple = (1, 2)) -> np.ndarray:
        """
        Extract TF-IDF features
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )

        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix.toarray(), vectorizer

    def bag_of_words_features(self, documents: List[str], max_features: int = 10000) -> np.ndarray:
        """
        Extract bag-of-words features
        """
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )

        bow_matrix = vectorizer.fit_transform(documents)
        return bow_matrix.toarray(), vectorizer

    def extract_pos_features(self, text: str) -> Dict[str, int]:
        """
        Extract part-of-speech features using spaCy
        """
        if self.nlp is None:
            return {}

        doc = self.nlp(text)
        pos_counts = {}

        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        return pos_counts

    def extract_dependency_features(self, text: str) -> List[Dict]:
        """
        Extract dependency parsing features
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        dependencies = []

        for token in doc:
            dep_info = {
                'text': token.text,
                'pos': token.pos_,
                'dep': token.dep_,
                'head_text': token.head.text,
                'head_pos': token.head.pos_
            }
            dependencies.append(dep_info)

        return dependencies

    def extract_named_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities using spaCy
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            }
            entities.append(entity_info)

        return entities

    def extract_syntactic_features(self, text: str) -> Dict:
        """
        Extract syntactic features
        """
        if self.nlp is None:
            return {}

        doc = self.nlp(text)

        features = {
            'n_sentences': len(list(doc.sents)),
            'n_tokens': len(doc),
            'avg_sentence_length': len(doc) / max(len(list(doc.sents)), 1),
            'n_nouns': len([token for token in doc if token.pos_ == 'NOUN']),
            'n_verbs': len([token for token in doc if token.pos_ == 'VERB']),
            'n_adjectives': len([token for token in doc if token.pos_ == 'ADJ']),
            'n_adverbs': len([token for token in doc if token.pos_ == 'ADV']),
            'lexical_diversity': len(set([token.text.lower() for token in doc])) / len(doc)
        }

        return features

    def sentiment_features(self, text: str) -> Dict:
        """
        Extract sentiment-based features
        """
        if self.nlp is None:
            return {}

        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_stop and not token.is_punct]

        # Simple positive/negative word counting
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'worst']

        positive_count = sum(1 for token in tokens if token.text.lower() in positive_words)
        negative_count = sum(1 for token in tokens if token.text.lower() in negative_words)

        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_ratio': (positive_count - negative_count) / max(len(tokens), 1),
            'polarity': (positive_count - negative_count) / max(positive_count + negative_count, 1)
        }

    def topic_modeling_features(self, documents: List[str], n_topics: int = 10) -> Tuple[np.ndarray, object]:
        """
        Extract topic modeling features using LSA
        """
        # TF-IDF vectorization
        tfidf_matrix, vectorizer = self.tfidf_features(documents)

        # Apply LSA (Latent Semantic Analysis)
        lsa = TruncatedSVD(n_components=n_topics, random_state=42)
        lsa_features = lsa.fit_transform(tfidf_matrix)

        # Normalize features
        normalizer = Normalizer(copy=False)
        lsa_features = normalizer.fit_transform(lsa_features)

        return lsa_features, (vectorizer, lsa, normalizer)

    def readability_features(self, text: str) -> Dict:
        """
        Extract readability features
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Calculate syllables (simplified)
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count = 1
            return count

        syllable_counts = [count_syllables(word) for word in words if word.isalpha()]

        features = {
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_syllables_per_word': np.mean(syllable_counts) if syllable_counts else 0,
            'flesch_reading_ease': self._flesch_reading_ease(text),
            'flesch_kincaid_grade': self._flesch_kincaid_grade(text),
            'automated_readability_index': self._automated_readability_index(text)
        }

        return features

    def _flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        total_syllables = sum(self._count_syllables(word) for word in words if word.isalpha())
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables_per_word = total_syllables / max(len([w for w in words if w.isalpha()]), 1)

        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))

    def _flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        total_syllables = sum(self._count_syllables(word) for word in words if word.isalpha())
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables_per_word = total_syllables / max(len([w for w in words if w.isalpha()]), 1)

        score = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        return max(0, score)

    def _automated_readability_index(self, text: str) -> float:
        """
        Calculate Automated Readability Index
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        characters = sum(len(word) for word in words)

        avg_chars_per_word = characters / max(len(words), 1)
        avg_words_per_sentence = len(words) / max(len(sentences), 1)

        score = (4.71 * avg_chars_per_word) + (0.5 * avg_words_per_sentence) - 21.43
        return max(0, score)

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simplified)
        """
        word = word.lower()
        if word.endswith('e'):
            word = word[:-1]

        vowels = 'aeiouy'
        count = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels and not prev_char_was_vowel:
                count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        return max(1, count)
```

## 2. Traditional Machine Learning for NLP

### 2.1 Text Classification

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

class TextClassifier:
    """
    Comprehensive text classification with multiple algorithms
    """

    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='linear', random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB()
        }

        self.feature_extractors = {
            'tfidf': TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
            'bow': CountVectorizer(max_features=10000)
        }

        self.trained_models = {}
        self.feature_extractor = None
        self.best_model = None
        self.best_score = 0

    def prepare_data(self, texts: List[str], labels: List[str],
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare training and testing data
        """
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train: List[str], y_train: List[str],
                    feature_type: str = 'tfidf') -> Dict:
        """
        Train multiple models and compare performance
        """
        results = {}

        if feature_type not in self.feature_extractors:
            raise ValueError(f"Feature type {feature_type} not supported")

        self.feature_extractor = self.feature_extractors[feature_type]

        # Extract features
        X_train_features = self.feature_extractor.fit_transform(X_train)

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")

            # Train model
            model.fit(X_train_features, y_train)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_features, y_train, cv=5)
            mean_cv_score = np.mean(cv_scores)

            # Store trained model
            self.trained_models[model_name] = model

            results[model_name] = {
                'cv_scores': cv_scores,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': np.std(cv_scores)
            }

            # Update best model
            if mean_cv_score > self.best_score:
                self.best_score = mean_cv_score
                self.best_model = model_name

        return results

    def evaluate_models(self, X_test: List[str], y_test: List[str]) -> Dict:
        """
        Evaluate trained models on test data
        """
        if self.feature_extractor is None:
            raise ValueError("No feature extractor fitted. Train models first.")

        X_test_features = self.feature_extractor.transform(X_test)
        evaluation_results = {}

        for model_name, model in self.trained_models.items():
            # Predictions
            y_pred = model.predict(X_test_features)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }

        return evaluation_results

    def hyperparameter_tuning(self, X_train: List[str], y_train: List[str],
                            model_name: str = 'logistic_regression') -> object:
        """
        Perform hyperparameter tuning for a specific model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")

        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }

        if model_name not in param_grids:
            raise ValueError(f"Parameter grid for {model_name} not defined")

        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.feature_extractors['tfidf']),
            ('classifier', self.models[model_name])
        ])

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        return grid_search

    def predict(self, text: str) -> str:
        """
        Make prediction on new text using best model
        """
        if self.best_model is None:
            raise ValueError("No trained model available")

        features = self.feature_extractor.transform([text])
        model = self.trained_models[self.best_model]
        prediction = model.predict(features)[0]

        return prediction

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importance for interpretable models
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]

        if hasattr(model, 'coef_'):
            # Linear models
            feature_names = self.feature_extractor.get_feature_names_out()
            coefficients = model.coef_[0]

            # Get top features
            top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
            feature_importance = [
                (feature_names[i], coefficients[i]) for i in top_indices
            ]

            return feature_importance
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            feature_names = self.feature_extractor.get_feature_names_out()
            importances = model.feature_importances_

            top_indices = np.argsort(importances)[-top_n:][::-1]
            feature_importance = [
                (feature_names[i], importances[i]) for i in top_indices
            ]

            return feature_importance
        else:
            return []

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str],
                            model_name: str, save_path: str = None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

### 2.2 Named Entity Recognition

```python
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    """
    Dataset for Named Entity Recognition
    """

    def __init__(self, sentences: List[List[str]], tags: List[List[str]],
                 word_to_idx: Dict[str, int], tag_to_idx: Dict[str, int]):
        self.sentences = sentences
        self.tags = tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_sequence = self.tags[idx]

        # Convert to indices
        word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                        for word in sentence]
        tag_indices = [self.tag_to_idx[tag] for tag in tag_sequence]

        return torch.tensor(word_indices), torch.tensor(tag_indices)

class BiLSTMNER(nn.Module):
    """
    BiLSTM model for Named Entity Recognition
    """

    def __init__(self, vocab_size: int, tag_size: int, embedding_dim: int = 100,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super(BiLSTMNER, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding layer
        embedded = self.embedding(x)

        # LSTM layer
        lstm_out, _ = self.lstm(embedded)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Classification layer
        logits = self.classifier(lstm_out)

        return logits

class NERTrainer:
    """
    Trainer for NER models
    """

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch_idx, (words, tags) in enumerate(dataloader):
            words, tags = words.to(self.device), tags.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(words)

            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.shape[-1]),
                                tags.view(-1))

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        true_tags = []

        with torch.no_grad():
            for words, tags in dataloader:
                words, tags = words.to(self.device), tags.to(self.device)

                logits = self.model(words)
                loss = self.criterion(logits.view(-1, logits.shape[-1]),
                                    tags.view(-1))

                total_loss += loss.item()

                # Get predictions
                pred_tags = torch.argmax(logits, dim=-1)
                predictions.extend(pred_tags.cpu().numpy().flatten())
                true_tags.extend(tags.cpu().numpy().flatten())

        # Remove padding tokens
        mask = np.array(true_tags) != -100
        predictions = np.array(predictions)[mask]
        true_tags = np.array(true_tags)[mask]

        # Calculate metrics
        accuracy = accuracy_score(true_tags, predictions)
        f1 = f1_score(true_tags, predictions, average='weighted')

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1_score': f1
        }

class NERPipeline:
    """
    Complete NER pipeline
    """

    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.tag_to_idx = {'<PAD>': 0}
        self.idx_to_tag = {0: '<PAD>'}
        self.model = None
        self.trainer = None

    def prepare_data(self, annotated_sentences: List[Tuple[List[str], List[str]]]):
        """
        Prepare data from annotated sentences
        """
        sentences = [sentence for sentence, _ in annotated_sentences]
        tags = [tag_seq for _, tag_seq in annotated_sentences]

        # Build vocabularies
        all_words = [word for sentence in sentences for word in sentence]
        all_tags = [tag for tag_seq in tags for tag in tag_seq]

        # Word vocabulary
        word_counts = Counter(all_words)
        for word, count in word_counts.most_common(10000):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)

        # Tag vocabulary
        unique_tags = set(all_tags)
        for tag in unique_tags:
            if tag not in self.tag_to_idx:
                self.tag_to_idx[tag] = len(self.tag_to_idx)
                self.idx_to_tag[len(self.idx_to_tag)] = tag

        return sentences, tags

    def train_model(self, sentences: List[List[str]], tags: List[List[str]],
                   batch_size: int = 32, epochs: int = 10, validation_split: float = 0.2):
        """
        Train NER model
        """
        # Split data
        train_sentences, val_sentences, train_tags, val_tags = train_test_split(
            sentences, tags, test_size=validation_split, random_state=42
        )

        # Create datasets
        train_dataset = NERDataset(train_sentences, train_tags,
                                  self.word_to_idx, self.tag_to_idx)
        val_dataset = NERDataset(val_sentences, val_tags,
                                self.word_to_idx, self.tag_to_idx)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        vocab_size = len(self.word_to_idx)
        tag_size = len(self.tag_to_idx)

        self.model = BiLSTMNER(vocab_size, tag_size)
        self.trainer = NERTrainer(self.model)

        # Training loop
        best_f1 = 0

        for epoch in range(epochs):
            train_loss = self.trainer.train_epoch(train_loader)
            val_metrics = self.trainer.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            print("-" * 50)

            # Save best model
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                torch.save(self.model.state_dict(), 'best_ner_model.pth')

    def predict(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        Predict NER tags for a sentence
        """
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        # Convert sentence to indices
        word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                       for word in sentence]

        # Create tensor
        word_tensor = torch.tensor([word_indices]).to(self.trainer.device)

        # Predict
        with torch.no_grad():
            logits = self.model(word_tensor)
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # Convert indices to tags
        entities = []
        for word, pred_idx in zip(sentence, predictions):
            tag = self.idx_to_tag.get(pred_idx, 'O')
            entities.append((word, tag))

        return entities

    def evaluate_spacy_ner(self, texts: List[str]) -> List[List[Dict]]:
        """
        Evaluate spaCy NER on given texts
        """
        if not self.nlp:
            return []

        results = []

        for text in texts:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'confidence', None)
                }
                entities.append(entity_info)

            results.append(entities)

        return results
```

### 2.3 Sentiment Analysis

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis with multiple approaches
    """

    def __init__(self):
        self.lexicon_sentiment = self._load_sentiment_lexicon()
        self.transformer_models = {}

    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """
        Load or create sentiment lexicon
        """
        # Simplified sentiment lexicon
        positive_words = {
            'excellent': 3.0, 'amazing': 2.8, 'wonderful': 2.7, 'fantastic': 2.6,
            'great': 2.5, 'good': 2.0, 'nice': 1.5, 'pleased': 1.8,
            'happy': 1.7, 'love': 2.9, 'perfect': 3.0, 'awesome': 2.7
        }

        negative_words = {
            'terrible': -3.0, 'awful': -2.8, 'horrible': -2.7, 'disgusting': -2.6,
            'bad': -2.0, 'poor': -2.2, 'worst': -3.0, 'hate': -2.9,
            'disappointed': -2.3, 'annoying': -2.1, 'frustrating': -2.4
        }

        return {**positive_words, **negative_words}

    def lexicon_based_sentiment(self, text: str) -> Dict[str, float]:
        """
        Lexicon-based sentiment analysis
        """
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize_words(text.lower())

        sentiment_scores = []
        intensifiers = {'very': 1.5, 'extremely': 2.0, 'quite': 1.3, 'really': 1.4}

        for i, token in enumerate(tokens):
            base_score = self.lexicon_sentiment.get(token, 0)

            # Check for intensifiers
            if i > 0 and tokens[i-1] in intensifiers:
                base_score *= intensifiers[tokens[i-1]]

            if base_score != 0:
                sentiment_scores.append(base_score)

        if not sentiment_scores:
            return {'sentiment': 0.0, 'confidence': 0.0}

        avg_sentiment = np.mean(sentiment_scores)
        confidence = min(abs(avg_sentiment) / 3.0, 1.0)

        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'positive_count': sum(1 for s in sentiment_scores if s > 0),
            'negative_count': sum(1 for s in sentiment_scores if s < 0)
        }

    def load_transformer_model(self, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'):
        """
        Load transformer-based sentiment model
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.transformer_models[model_name] = {
                'tokenizer': tokenizer,
                'model': model
            }
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False

    def transformer_sentiment(self, text: str, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest') -> Dict:
        """
        Sentiment analysis using transformer models
        """
        if model_name not in self.transformer_models:
            if not self.load_transformer_model(model_name):
                return self.lexicon_based_sentiment(text)

        model_info = self.transformer_models[model_name]
        tokenizer = model_info['tokenizer']
        model = model_info['model']

        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)

        # Get sentiment label and confidence
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_label = sentiment_labels[predictions.item()]
        confidence = probs.max().item()

        return {
            'sentiment': predicted_label,
            'confidence': confidence,
            'probabilities': {
                label: prob.item() for label, prob in zip(sentiment_labels, probs[0])
            }
        }

    def aspect_based_sentiment(self, text: str, aspects: List[str]) -> Dict[str, Dict]:
        """
        Aspect-based sentiment analysis
        """
        sentences = sent_tokenize(text)
        aspect_sentiments = {aspect: {'sentiment': 0.0, 'confidence': 0.0, 'evidence': []}
                           for aspect in aspects}

        for sentence in sentences:
            sentence_lower = sentence.lower()

            for aspect in aspects:
                if aspect.lower() in sentence_lower:
                    # Get sentiment for this sentence
                    sentiment = self.lexicon_based_sentiment(sentence)

                    # Add to aspect sentiment
                    if sentiment['confidence'] > 0.3:
                        aspect_sentiments[aspect]['sentiment'] += sentiment['sentiment']
                        aspect_sentiments[aspect]['confidence'] = max(
                            aspect_sentiments[aspect]['confidence'], sentiment['confidence']
                        )
                        aspect_sentiments[aspect]['evidence'].append(sentence)

        # Normalize sentiment scores
        for aspect in aspect_sentiments:
            if aspect_sentiments[aspect]['sentiment'] != 0:
                aspect_sentiments[aspect]['sentiment'] = np.clip(
                    aspect_sentiments[aspect]['sentiment'], -3.0, 3.0
                )

        return aspect_sentiments

    def emotion_detection(self, text: str) -> Dict[str, float]:
        """
        Emotion detection using lexicon approach
        """
        emotion_lexicon = {
            'joy': ['happy', 'joy', 'excited', 'delighted', 'cheerful', 'pleased'],
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'down', 'blue'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'repulsed', 'appalled']
        }

        tokens = TextPreprocessor().tokenize_words(text.lower())
        emotion_scores = {emotion: 0 for emotion in emotion_lexicon}

        for token in tokens:
            for emotion, words in emotion_lexicon.items():
                if token in words:
                    emotion_scores[emotion] += 1

        # Normalize scores
        total_emotions = sum(emotion_scores.values())
        if total_emotions > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_emotions

        return emotion_scores

    def comprehensive_sentiment_analysis(self, text: str) -> Dict:
        """
        Comprehensive sentiment analysis combining multiple approaches
        """
        results = {
            'lexicon_sentiment': self.lexicon_based_sentiment(text),
            'transformer_sentiment': self.transformer_sentiment(text),
            'emotions': self.emotion_detection(text),
            'text_features': FeatureExtractor().extract_syntactic_features(text)
        }

        # Aggregate results
        lexicon_score = results['lexicon_sentiment']['sentiment']
        transformer_sentiment = results['transformer_sentiment']['sentiment']

        # Convert transformer sentiment to numeric score
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        transformer_score = sentiment_mapping.get(transformer_sentiment, 0)

        # Combined sentiment score
        combined_sentiment = 0.4 * lexicon_score + 0.6 * transformer_score

        results['combined_sentiment'] = {
            'score': combined_sentiment,
            'label': 'positive' if combined_sentiment > 0.1 else 'negative' if combined_sentiment < -0.1 else 'neutral',
            'confidence': max(results['lexicon_sentiment']['confidence'],
                            results['transformer_sentiment']['confidence'])
        }

        return results
```

## 3. Advanced NLP Implementations

### 3.1 Machine Translation

```python
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    """
    Dataset for machine translation
    """

    def __init__(self, source_sentences: List[str], target_sentences: List[str],
                 source_tokenizer, target_tokenizer, max_length: int = 512):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source = self.source_sentences[idx]
        target = self.target_sentences[idx]

        # Tokenize
        source_tokens = self.source_tokenizer(
            source, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        target_tokens = self.target_tokenizer(
            target, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        return {
            'source_input_ids': source_tokens['input_ids'].squeeze(),
            'source_attention_mask': source_tokens['attention_mask'].squeeze(),
            'target_input_ids': target_tokens['input_ids'].squeeze(),
            'target_attention_mask': target_tokens['attention_mask'].squeeze()
        }

class Seq2SeqTranslator(nn.Module):
    """
    Sequence-to-Sequence translator using encoder-decoder architecture
    """

    def __init__(self, source_vocab_size: int, target_vocab_size: int,
                 embedding_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.1):
        super(Seq2SeqTranslator, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_embedding = nn.Embedding(source_vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                   dropout=dropout if num_layers > 1 else 0,
                                   bidirectional=True)

        # Decoder
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers,
                                   dropout=dropout if num_layers > 1 else 0)

        # Attention
        self.attention = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim * 2 + hidden_dim, embedding_dim)

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_input_ids: torch.Tensor, target_input_ids: torch.Tensor,
                source_attention_mask: torch.Tensor, target_attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = source_input_ids.size(0)
        max_len = target_input_ids.size(1)

        # Encoder
        embedded_source = self.dropout(self.encoder_embedding(source_input_ids))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_source)

        # Initialize decoder hidden state
        hidden = hidden[:self.decoder_lstm.num_layers]
        cell = cell[:self.decoder_lstm.num_layers]

        # Decoder
        outputs = []
        for t in range(max_len):
            # Get current input
            current_input = target_input_ids[:, t:t+1]

            # Embed current input
            embedded = self.dropout(self.decoder_embedding(current_input))

            # Attention
            query = hidden[-1].unsqueeze(1)  # Last layer, last time step
            attention_scores = torch.bmm(encoder_outputs.transpose(1, 2), query.transpose(1, 2))
            attention_weights = F.softmax(attention_scores, dim=1)

            # Apply attention
            context = torch.bmm(encoder_outputs, attention_weights)
            context = context.transpose(1, 2).squeeze(1)

            # Combine embedded input and context
            lstm_input = torch.cat([embedded.squeeze(1), context], dim=1)

            # Decoder LSTM
            output, (hidden, cell) = self.decoder_lstm(
                lstm_input.unsqueeze(1), (hidden, cell)
            )

            # Output layer
            output = self.output_layer(output.squeeze(1))
            outputs.append(output)

        return torch.stack(outputs, dim=1)

class TranslationTrainer:
    """
    Trainer for machine translation models
    """

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            # Move to device
            source_input_ids = batch['source_input_ids'].to(self.device)
            target_input_ids = batch['target_input_ids'].to(self.device)
            source_attention_mask = batch['source_attention_mask'].to(self.device)
            target_attention_mask = batch['target_attention_mask'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(source_input_ids, target_input_ids,
                               source_attention_mask, target_attention_mask)

            # Calculate loss (ignore padding)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)),
                                target_input_ids.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []

        with torch.no_grad():
            for batch in dataloader:
                source_input_ids = batch['source_input_ids'].to(self.device)
                target_input_ids = batch['target_input_ids'].to(self.device)
                source_attention_mask = batch['source_attention_mask'].to(self.device)
                target_attention_mask = batch['target_attention_mask'].to(self.device)

                outputs = self.model(source_input_ids, target_input_ids,
                                   source_attention_mask, target_attention_mask)

                loss = self.criterion(outputs.view(-1, outputs.size(-1)),
                                    target_input_ids.view(-1))
                total_loss += loss.item()

                # Get predictions (for BLEU calculation)
                pred_tokens = torch.argmax(outputs, dim=-1)
                predictions.extend(pred_tokens.cpu().numpy())
                references.extend(target_input_ids.cpu().numpy())

        # Calculate BLEU score (simplified)
        bleu_score = self._calculate_bleu(predictions, references)

        return {
            'loss': total_loss / len(dataloader),
            'bleu_score': bleu_score
        }

    def _calculate_bleu(self, predictions: List[np.ndarray], references: List[np.ndarray]) -> float:
        """
        Simplified BLEU score calculation
        """
        # This is a simplified version - in practice, use nltk.bleu_score
        bleu_scores = []

        for pred, ref in zip(predictions, references):
            # Remove padding
            pred = pred[pred != 0]
            ref = ref[ref != 0]

            if len(pred) == 0 or len(ref) == 0:
                continue

            # Calculate unigram precision
            pred_unigrams = set(pred)
            ref_unigrams = set(ref)
            common_unigrams = pred_unigrams.intersection(ref_unigrams)

            precision = len(common_unigrams) / max(len(pred_unigrams), 1)
            brevity_penalty = min(1, len(pred) / len(ref))

            bleu = brevity_penalty * precision
            bleu_scores.append(bleu)

        return np.mean(bleu_scores) if bleu_scores else 0.0

    def translate(self, source_sentence: str, source_tokenizer, target_tokenizer,
                  max_length: int = 100) -> str:
        """
        Translate a single sentence
        """
        self.model.eval()

        # Tokenize source
        source_tokens = source_tokenizer(
            source_sentence, return_tensors='pt', padding=True, truncation=True
        )

        # Initialize with start token
        target_tokens = torch.tensor([[target_tokenizer.bos_token_id]]).to(self.device)

        with torch.no_grad():
            for _ in range(max_length):
                # Get attention mask
                source_attention_mask = source_tokens['attention_mask'].to(self.device)
                target_attention_mask = torch.ones_like(target_tokens).to(self.device)

                # Forward pass
                outputs = self.model(
                    source_tokens['input_ids'].to(self.device),
                    target_tokens,
                    source_attention_mask,
                    target_attention_mask
                )

                # Get next token
                next_token = outputs[:, -1, :].argmax(dim=-1)

                # Append to target tokens
                target_tokens = torch.cat([target_tokens, next_token.unsqueeze(1)], dim=1)

                # Check for end token
                if next_token.item() == target_tokenizer.eos_token_id:
                    break

        # Convert to text
        translated_text = target_tokenizer.decode(target_tokens[0], skip_special_tokens=True)
        return translated_text
```

### 3.2 Text Summarization

```python
from rouge_score import rouge_scorer

class ExtractiveSummarizer:
    """
    Extractive text summarization using various algorithms
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def sentence_importance(self, document: str) -> List[Tuple[str, float]]:
        """
        Calculate sentence importance scores
        """
        sentences = sent_tokenize(document)

        # Extract features for each sentence
        sentence_features = []
        for sentence in sentences:
            features = self.feature_extractor.extract_syntactic_features(sentence)
            sentence_features.append(features)

        # Calculate importance scores
        importance_scores = []

        for i, (sentence, features) in enumerate(zip(sentences, sentence_features)):
            score = 0

            # Position importance (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 0.3

            # Length importance (medium-length sentences are best)
            length_score = min(len(sentence.split()) / 20, 1.0)
            score += 0.2 * length_score

            # Keyword frequency
            tfidf_score = features.get('lexical_diversity', 0) * 0.3
            score += tfidf_score

            importance_scores.append((sentence, score))

        return importance_scores

    def tfidf_summarize(self, document: str, num_sentences: int = 3) -> str:
        """
        TF-IDF based extractive summarization
        """
        sentences = sent_tokenize(document)
        if len(sentences) <= num_sentences:
            return document

        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Select top sentences
        top_indices = np.argsort(sentence_scores)[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Maintain original order

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary

    def text_rank_summarize(self, document: str, num_sentences: int = 3) -> str:
        """
        TextRank algorithm for extractive summarization
        """
        sentences = sent_tokenize(document)
        if len(sentences) <= num_sentences:
            return document

        # Build similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    # Calculate cosine similarity between sentences
                    similarity = self._sentence_similarity(sentences[i], sentences[j])
                    similarity_matrix[i][j] = similarity

        # Apply PageRank algorithm
        sentence_scores = self._pagerank(similarity_matrix)

        # Select top sentences
        top_indices = np.argsort(sentence_scores)[-num_sentences:][::-1]
        top_indices = sorted(top_indices)

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate cosine similarity between two sentences
        """
        # Preprocess sentences
        preprocessor = TextPreprocessor()
        tokens1 = set(preprocessor.preprocess_pipeline(sent1))
        tokens2 = set(preprocessor.preprocess_pipeline(sent2))

        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0

        # Calculate Jaccard similarity (simplified)
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def _pagerank(self, similarity_matrix: np.ndarray, damping: float = 0.85,
                  max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        PageRank algorithm implementation
        """
        n = similarity_matrix.shape[0]

        # Normalize similarity matrix
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = similarity_matrix / row_sums

        # Initialize PageRank scores
        scores = np.ones(n) / n

        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * np.dot(transition_matrix.T, scores)

            if np.linalg.norm(new_scores - scores) < tol:
                break

            scores = new_scores

        return scores

class AbstractiveSummarizer:
    """
    Abstractive text summarization using sequence-to-sequence models
    """

    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40,
                 num_beams: int = 4, do_sample: bool = False) -> str:
        """
        Generate abstractive summary
        """
        # Tokenize input
        inputs = self.tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                do_sample=do_sample,
                early_stopping=True
            )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def evaluate_summary(self, generated_summary: str, reference_summaries: List[str]) -> Dict[str, float]:
        """
        Evaluate summary quality using ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for reference in reference_summaries:
            score = scorer.score(reference, generated_summary)
            for metric in scores:
                scores[metric].append(score[metric].fmeasure)

        # Calculate average scores
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}

        return avg_scores

class HybridSummarizer:
    """
    Hybrid extractive-abstractive summarization
    """

    def __init__(self):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()

    def hybrid_summarize(self, document: str, extractive_ratio: float = 0.3,
                        max_length: int = 150) -> str:
        """
        Generate hybrid summary
        """
        # Extractive step
        num_extractive = max(1, int(len(sent_tokenize(document)) * extractive_ratio))
        extractive_summary = self.extractive.tfidf_summarize(document, num_extractive)

        # Abstractive step
        final_summary = self.abstractive.summarize(
            extractive_summary, max_length=max_length
        )

        return final_summary

    def multi_document_summarize(self, documents: List[str], max_length: int = 200) -> str:
        """
        Summarize multiple documents
        """
        # Combine documents
        combined_document = ' '.join(documents)

        # Generate summary
        summary = self.abstractive.summarize(combined_document, max_length=max_length)

        return summary
```

## 4. Real-world Applications

### 4.1 Customer Service Chatbot

```python
class CustomerServiceBot:
    """
    Customer service chatbot with intent recognition and response generation
    """

    def __init__(self):
        self.intent_classifier = None
        self.response_templates = self._load_response_templates()
        self.knowledge_base = self._load_knowledge_base()

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """
        Load response templates for different intents
        """
        return {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! Welcome to our customer service. What can I assist you with?",
                "Good day! I'm here to help with any questions or concerns."
            ],
            'product_inquiry': [
                "I'd be happy to help you with product information. What specific product are you interested in?",
                "Certainly! Let me get you the details about that product.",
                "I can help you find information about our products. What would you like to know?"
            ],
            'order_status': [
                "I can check your order status for you. Could you please provide your order number?",
                "To check your order status, I'll need your order number. Do you have that available?",
                "I'd be happy to look up your order status. What's your order number?"
            ],
            'complaint': [
                "I'm sorry to hear that you're having an issue. I'd like to help resolve this for you. Could you tell me more about the problem?",
                "I understand you're experiencing an issue, and I want to help fix it. Can you provide more details?",
                "I apologize for the inconvenience. Let me help you resolve this issue. What seems to be the problem?"
            ],
            'refund_request': [
                "I can help you with a refund request. Could you tell me more about why you're requesting a refund?",
                "I understand you'd like to request a refund. Let me help you with that process. What's the reason for the refund?",
                "I can assist you with your refund request. Could you provide some details about your purchase and the reason for the refund?"
            ],
            'technical_support': [
                "I'm here to help with technical issues. Could you describe the problem you're experiencing?",
                "Technical support is available. What specific issue are you encountering?",
                "I can help you with technical problems. What seems to be the issue?"
            ],
            'farewell': [
                "Thank you for contacting us! Is there anything else I can help you with?",
                "I'm glad I could help today. Feel free to reach out if you need anything else!",
                "Thank you for choosing our service. Have a great day!"
            ]
        }

    def _load_knowledge_base(self) -> Dict[str, str]:
        """
        Load knowledge base with product information and policies
        """
        return {
            'shipping_policy': "We offer free standard shipping on orders over $50. Express shipping is available for an additional fee.",
            'return_policy': "You can return items within 30 days of purchase for a full refund. Items must be in original condition.",
            'warranty_info': "All products come with a 1-year manufacturer warranty covering defects and malfunctions.",
            'payment_methods': "We accept all major credit cards, PayPal, and Apple Pay.",
            'business_hours': "Our customer service is available Monday-Friday 9AM-6PM EST."
        }

    def train_intent_classifier(self, training_data: List[Tuple[str, str]]):
        """
        Train intent classifier
        """
        texts = [text for text, _ in training_data]
        intents = [intent for _, intent in training_data]

        self.intent_classifier = TextClassifier()
        X_train, X_test, y_train, y_test = self.intent_classifier.prepare_data(texts, intents)

        results = self.intent_classifier.train_models(X_train, y_train)
        evaluation = self.intent_classifier.evaluate_models(X_test, y_test)

        return results, evaluation

    def detect_intent(self, user_message: str) -> str:
        """
        Detect user intent from message
        """
        if self.intent_classifier is None:
            # Fallback to keyword-based intent detection
            return self._keyword_intent_detection(user_message)

        prediction = self.intent_classifier.predict(user_message)
        return prediction

    def _keyword_intent_detection(self, message: str) -> str:
        """
        Keyword-based intent detection as fallback
        """
        message_lower = message.lower()

        intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'product_inquiry': ['product', 'item', 'information', 'details', 'specifications'],
            'order_status': ['order', 'status', 'tracking', 'where is my order', 'delivery'],
            'complaint': ['problem', 'issue', 'wrong', 'broken', 'defective', 'not working'],
            'refund_request': ['refund', 'return', 'money back', 'cancel order'],
            'technical_support': ['technical', 'help', 'support', 'issue', 'error', 'not working'],
            'farewell': ['bye', 'goodbye', 'thank you', 'thanks', 'see you']
        }

        # Count keyword matches
        intent_scores = {intent: 0 for intent in intent_keywords}

        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    intent_scores[intent] += 1

        # Return intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        return best_intent if intent_scores[best_intent] > 0 else 'general_inquiry'

    def generate_response(self, user_message: str, intent: str, context: Dict = None) -> str:
        """
        Generate appropriate response based on intent
        """
        if context is None:
            context = {}

        # Get base response
        if intent in self.response_templates:
            response = np.random.choice(self.response_templates[intent])
        else:
            response = "I'm here to help. Could you please provide more details about what you need?"

        # Add context-specific information
        if intent == 'order_status' and 'order_number' in context:
            response += f" Your order {context['order_number']} is currently being processed."

        elif intent == 'technical_support' and 'product_name' in context:
            response += f" I can help you with {context['product_name']}. What specific issue are you experiencing?"

        elif intent == 'refund_request' and 'order_number' in context:
            response += f" I can process a refund for order {context['order_number']}."

        return response

    def extract_entities(self, message: str) -> Dict[str, str]:
        """
        Extract entities from user message
        """
        entities = {}

        # Extract order number (simplified pattern)
        order_pattern = r'\b[A-Z]{2}\d{9}\b|\b\d{4,}\b'
        order_matches = re.findall(order_pattern, message)
        if order_matches:
            entities['order_number'] = order_matches[0]

        # Extract product names (simplified)
        product_keywords = ['laptop', 'phone', 'tablet', 'headphones', 'monitor']
        for keyword in product_keywords:
            if keyword in message.lower():
                entities['product_name'] = keyword
                break

        return entities

    def handle_conversation(self, user_message: str, conversation_history: List = None) -> str:
        """
        Handle complete conversation turn
        """
        if conversation_history is None:
            conversation_history = []

        # Detect intent
        intent = self.detect_intent(user_message)

        # Extract entities
        entities = self.extract_entities(user_message)

        # Update context
        context = entities.copy()
        if conversation_history:
            # Add relevant context from previous turns
            last_intent = self.detect_intent(conversation_history[-1])
            if last_intent == 'order_status' and 'order_number' in context:
                context['previous_order_inquiry'] = True

        # Generate response
        response = self.generate_response(user_message, intent, context)

        return response, intent, entities

    def evaluate_performance(self, test_conversations: List[Dict]) -> Dict[str, float]:
        """
        Evaluate chatbot performance
        """
        total_turns = 0
        correct_intents = 0
        user_satisfaction_scores = []

        for conv in test_conversations:
            for turn in conv['turns']:
                total_turns += 1

                # Evaluate intent detection
                predicted_intent = self.detect_intent(turn['user_message'])
                if predicted_intent == turn['true_intent']:
                    correct_intents += 1

                # Evaluate response quality (simplified)
                response = self.generate_response(turn['user_message'], predicted_intent)
                satisfaction = self._evaluate_response_quality(response, turn['expected_response_type'])
                user_satisfaction_scores.append(satisfaction)

        intent_accuracy = correct_intents / total_turns if total_turns > 0 else 0
        avg_satisfaction = np.mean(user_satisfaction_scores) if user_satisfaction_scores else 0

        return {
            'intent_accuracy': intent_accuracy,
            'average_satisfaction': avg_satisfaction,
            'total_conversations': len(test_conversations),
            'total_turns': total_turns
        }

    def _evaluate_response_quality(self, response: str, expected_type: str) -> float:
        """
        Evaluate response quality (simplified)
        """
        # Simple heuristics for response quality
        quality_indicators = {
            'helpful': ['help', 'assist', 'solution', 'resolve', 'fix'],
            'polite': ['please', 'thank you', 'sorry', 'welcome', 'appreciate'],
            'clear': ['clear', 'understand', 'explain', 'detail', 'specific']
        }

        score = 0.5  # Base score

        for category, indicators in quality_indicators.items():
            for indicator in indicators:
                if indicator in response.lower():
                    score += 0.1

        return min(score, 1.0)
```

### 4.2 Document Analysis System

```python
class DocumentAnalyzer:
    """
    Comprehensive document analysis system
    """

    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.summarizer = HybridSummarizer()

    def analyze_document(self, document: str, analysis_type: str = 'comprehensive') -> Dict:
        """
        Analyze document based on specified analysis type
        """
        if analysis_type == 'comprehensive':
            return self._comprehensive_analysis(document)
        elif analysis_type == 'sentiment':
            return self._sentiment_analysis(document)
        elif analysis_type == 'topic':
            return self._topic_analysis(document)
        elif analysis_type == 'readability':
            return self._readability_analysis(document)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    def _comprehensive_analysis(self, document: str) -> Dict:
        """
        Perform comprehensive document analysis
        """
        analysis = {}

        # Basic statistics
        analysis['basic_stats'] = self._basic_document_stats(document)

        # Sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        analysis['sentiment'] = sentiment_analyzer.comprehensive_sentiment_analysis(document)

        # Readability analysis
        analysis['readability'] = self.feature_extractor.readability_features(document)

        # Topic analysis
        analysis['topics'] = self._topic_analysis(document)

        # Entity extraction
        analysis['entities'] = self.feature_extractor.extract_named_entities(document)

        # Summary
        analysis['summary'] = self.summarizer.hybrid_summarize(document)

        return analysis

    def _basic_document_stats(self, document: str) -> Dict:
        """
        Calculate basic document statistics
        """
        sentences = sent_tokenize(document)
        words = word_tokenize(document)
        characters = len(document)

        # Remove punctuation and stopwords for word count
        preprocessor = TextPreprocessor()
        clean_words = preprocessor.preprocess_pipeline(document)

        return {
            'total_characters': characters,
            'total_words': len(words),
            'total_sentences': len(sentences),
            'unique_words': len(set(clean_words)),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_characters_per_word': characters / max(len(words), 1)
        }

    def _sentiment_analysis(self, document: str) -> Dict:
        """
        Perform sentiment analysis on document
        """
        sentiment_analyzer = SentimentAnalyzer()
        return sentiment_analyzer.comprehensive_sentiment_analysis(document)

    def _topic_analysis(self, document: str) -> Dict:
        """
        Perform topic analysis on document
        """
        sentences = sent_tokenize(document)

        # Extract TF-IDF features
        tfidf_features, vectorizer = self.feature_extractor.tfidf_features(sentences, max_features=100)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(document)

        # Calculate topic coherence
        topic_coherence = self._calculate_topic_coherence(key_phrases)

        return {
            'key_phrases': key_phrases[:10],  # Top 10 key phrases
            'topic_coherence': topic_coherence,
            'document_topics': self._assign_document_topics(key_phrases)
        }

    def _extract_key_phrases(self, document: str, max_phrases: int = 20) -> List[Tuple[str, float]]:
        """
        Extract key phrases from document
        """
        # Simple key phrase extraction based on TF-IDF
        sentences = sent_tokenize(document)
        words = word_tokenize(document.lower())

        # Calculate word frequencies
        word_freq = Counter(words)
        total_words = len(words)

        # Calculate TF-IDF-like scores
        phrase_scores = {}
        window_size = 3

        for i in range(len(words) - window_size + 1):
            phrase = ' '.join(words[i:i + window_size])
            if phrase not in phrase_scores:
                # Calculate score based on word frequencies
                score = np.mean([word_freq[word] / total_words for word in words[i:i + window_size]])
                phrase_scores[phrase] = score

        # Return top phrases
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:max_phrases]

    def _calculate_topic_coherence(self, key_phrases: List[Tuple[str, float]]) -> float:
        """
        Calculate topic coherence score
        """
        if len(key_phrases) < 2:
            return 0.0

        # Simplified coherence calculation
        coherence_scores = []

        for i, (phrase1, score1) in enumerate(key_phrases[:10]):
            for phrase2, score2 in key_phrases[i+1:10]:
                # Calculate semantic similarity (simplified)
                similarity = self._phrase_similarity(phrase1, phrase2)
                coherence_scores.append(similarity * min(score1, score2))

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _phrase_similarity(self, phrase1: str, phrase2: str) -> float:
        """
        Calculate similarity between two phrases
        """
        words1 = set(phrase1.split())
        words2 = set(phrase2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _assign_document_topics(self, key_phrases: List[Tuple[str, float]]) -> List[str]:
        """
        Assign topic labels based on key phrases
        """
        # Simplified topic assignment
        topic_keywords = {
            'Technology': ['computer', 'software', 'data', 'digital', 'online'],
            'Business': ['company', 'market', 'sales', 'revenue', 'customer'],
            'Health': ['medical', 'patient', 'health', 'treatment', 'doctor'],
            'Education': ['school', 'student', 'learning', 'education', 'teacher'],
            'Environment': ['climate', 'environment', 'sustainability', 'green', 'energy']
        }

        topics = []
        for phrase, score in key_phrases[:5]:
            for topic, keywords in topic_keywords.items():
                if any(keyword in phrase for keyword in keywords):
                    if topic not in topics:
                        topics.append(topic)

        return topics if topics else ['General']

    def _readability_analysis(self, document: str) -> Dict:
        """
        Perform readability analysis
        """
        return self.feature_extractor.readability_features(document)

    def compare_documents(self, documents: List[Dict[str, str]]) -> Dict:
        """
        Compare multiple documents
        """
        if len(documents) < 2:
            raise ValueError("Need at least 2 documents to compare")

        # Analyze each document
        analyses = [self._comprehensive_analysis(doc['content']) for doc in documents]

        comparison = {
            'documents': [doc['name'] for doc in documents],
            'sentiment_comparison': self._compare_sentiments(analyses),
            'topic_comparison': self._compare_topics(analyses),
            'readability_comparison': self._compare_readability(analyses),
            'similarity_matrix': self._calculate_similarity_matrix(analyses)
        }

        return comparison

    def _compare_sentiments(self, analyses: List[Dict]) -> Dict:
        """
        Compare sentiment across documents
        """
        sentiments = [analysis['sentiment']['combined_sentiment']['score'] for analysis in analyses]

        return {
            'sentiment_scores': sentiments,
            'average_sentiment': np.mean(sentiments),
            'sentiment_variance': np.var(sentiments),
            'most_positive': np.argmax(sentiments),
            'most_negative': np.argmin(sentiments)
        }

    def _compare_topics(self, analyses: List[Dict]) -> Dict:
        """
        Compare topics across documents
        """
        all_topics = []
        for analysis in analyses:
            all_topics.extend(analysis['topics']['document_topics'])

        topic_counts = Counter(all_topics)

        return {
            'topic_distribution': dict(topic_counts),
            'common_topics': [topic for topic, count in topic_counts.items() if count > 1],
            'unique_topics': [topic for topic, count in topic_counts.items() if count == 1]
        }

    def _compare_readability(self, analyses: List[Dict]) -> Dict:
        """
        Compare readability across documents
        """
        readability_metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index']

        comparison = {}
        for metric in readability_metrics:
            scores = [analysis['readability'][metric] for analysis in analyses]
            comparison[metric] = {
                'scores': scores,
                'average': np.mean(scores),
                'range': (np.min(scores), np.max(scores))
            }

        return comparison

    def _calculate_similarity_matrix(self, analyses: List[Dict]) -> np.ndarray:
        """
        Calculate similarity matrix between documents
        """
        n_docs = len(analyses)
        similarity_matrix = np.zeros((n_docs, n_docs))

        for i in range(n_docs):
            for j in range(n_docs):
                if i != j:
                    similarity = self._document_similarity(analyses[i], analyses[j])
                    similarity_matrix[i][j] = similarity

        return similarity_matrix

    def _document_similarity(self, analysis1: Dict, analysis2: Dict) -> float:
        """
        Calculate similarity between two document analyses
        """
        # Compare topics
        topics1 = set(analysis1['topics']['document_topics'])
        topics2 = set(analysis2['topics']['document_topics'])
        topic_similarity = len(topics1.intersection(topics2)) / len(topics1.union(topics2))

        # Compare sentiment
        sentiment1 = analysis1['sentiment']['combined_sentiment']['score']
        sentiment2 = analysis2['sentiment']['combined_sentiment']['score']
        sentiment_similarity = 1 - abs(sentiment1 - sentiment2) / 6.0  # Normalize to [0,1]

        # Weighted combination
        similarity = 0.6 * topic_similarity + 0.4 * sentiment_similarity

        return similarity
```

## Conclusion

This comprehensive guide to core NLP techniques provides practical implementations for:

1. **Text Preprocessing**: Complete pipeline for cleaning, tokenization, and feature extraction
2. **Traditional ML**: Classification, NER, and sentiment analysis with classical algorithms
3. **Advanced NLP**: Machine translation and text summarization with deep learning
4. **Real Applications**: Customer service chatbots and document analysis systems

Each implementation includes detailed documentation, mathematical foundations, and practical considerations for production deployment. The code examples serve as building blocks for more sophisticated NLP applications and demonstrate the evolution from traditional to modern approaches in natural language processing.