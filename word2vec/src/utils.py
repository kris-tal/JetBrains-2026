import os
import numpy as np
import pandas as pd
import re
from collections import Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    return text


def load_data(file_path):
    if not os.path.exists(file_path):
        raise Exception("dataset not found")
    
    df = pd.read_csv(file_path, usecols=['text'])
    df = df.dropna(subset=['text'])

    processed_df = df['text'].apply(preprocess_text)
    tokenized_df = processed_df.apply(lambda x : x.split())

    return tokenized_df


def split_data(df, training_ratio=0.80):
    training_set = df.sample(frac=training_ratio, random_state=42)
    test_set = df.drop(training_set.index)

    return training_set, test_set


class Vocabulary:
    def __init__(self, dataset, min_count=2):
        self.word2idx = {}
        self.idx2word = {}
        self.probs = []
        self.size = 0
        self._build_vocab(dataset, min_count)


    def _build_vocab(self, dataset, min_count):
        words_counted = Counter()
        for text in dataset:
            words_counted.update(text)

        total_count = words_counted.total()
        
        words_ordered = sorted([word for word, count in words_counted.items() if count >= min_count])

        counts = [words_counted[word] for word in words_ordered]
        count_rare = np.sum([count for word, count in words_counted.items() if count < min_count])

        all_words = ['<UNK>'] + words_ordered
        all_counts = counts + [count_rare]
        
        self.word2idx = {word: i for i, word in enumerate(all_words)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.size = len(all_words)

        self.probs = [count / total_count for count in all_counts]

        
