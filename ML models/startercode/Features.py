""" 
    Basic feature extractor
"""
from math import log
from operator import methodcaller
import string 
from nltk.stem import PorterStemmer

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

class TfIDF:

    def __init__(self):
        self.idf = {}
        self.vocab = []
        self.word2idx = {}
        self.N = 0
        self.Ngram = 3
        self.punct_table = str.maketrans("", "", string.punctuation)
    
    def get_n_gram(self, tokens, gram_count):
        n_grams = []
        for i in range(len(tokens)-gram_count+1):
            n_gram_tokens = tokens[i:i+gram_count]
            n_gram_string = " ".join(n_gram_tokens)
            n_gram_string = n_gram_string.strip() 
            n_grams.append(n_gram_string)
        return n_grams

    def normalize(self, tokens, gram_count):
        cleaned = []
        for t in tokens:
            # lowercase
            t = t.casefold()
            # remove punctuation
            t = t.translate(self.punct_table)
            if t:
                cleaned.append(t)
        return self.get_n_gram(cleaned, gram_count)
    
    def fit(self, corpus_text):
        N = len(corpus_text)
        doc_counts = {}
        for gram_count in range(1, self.Ngram+1):
            for text in corpus_text:
                ngram_tokens = self.normalize(text, gram_count) 
                seen = set(ngram_tokens) 
                for word in seen: 
                    doc_counts[word] = doc_counts.get(word, 0) + 1

        self.vocab = sorted([w for w, df in doc_counts.items() if df < 0.95*N and df > 7])
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        self.idf = {word: log((N+1)/(doc_counts[word] + 1)) + 1 for word in self.vocab}
        
    def get_feature_vector_per_document(self, text):
        frequency = {}
        for gram_count in range(1, self.Ngram+1):
            ngram_tokens = self.normalize(text, gram_count) 
            for token in ngram_tokens:
                frequency[token] = frequency.get(token, 0) + 1
        features = {}
        for word, tf in frequency.items():
            idx = self.word2idx.get(word)
            if word not in self.idf:
                continue 
            features[idx] = tf * self.idf[word]
        return features
    
    def get_feature_vector(self, corpus):
        return [self.get_feature_vector_per_document(text) for text in corpus]


class Features:

    def __init__(self, data):
        tokenized_text = [tokenize(text) for text in data]
        self.tfIDF = TfIDF()
        self.tfIDF.fit(tokenized_text)

    def get_features(self, data):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features 
        tokenized_text = [tokenize(text) for text in data]
        return self.tfIDF.get_feature_vector(tokenized_text)
