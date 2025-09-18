"""
 Refer to Linear Models notes or Chapter 5 of J&M for more details on how to implement a Logistic Regression Model
"""

import numpy as np
from Model import *
from math import exp, log
from collections import Counter
from Features import Features
from operator import methodcaller


class LogisticRegression():
    
    def __init__(self):
        self.W = None
        self.b = 0.0 
        self.lr = 1e-3
        self.lmbda = 1e-4
        self.iterations = 50
     
    def sigmoid(self, x):
        if x >= 0:
            e = np.exp(-x)        # small when x is large+
            return 1.0 / (1.0 + e)
        else:
            e = np.exp(x)         # small when x is large-
            return e / (1.0 + e)

    def forward_pass(self, feature_vector):
        z = self.b
        for k, value in feature_vector.items():
            z += self.W[k] * value
        return self.sigmoid(z)

    def backward_pass(self, feature_vector, out, y):
        g = out - y
        for k, value in feature_vector.items():
            self.W[k] -= self.lr*(g*value + 2 * self.lmbda * self.W[k])
        self.b -= self.lr * g

    def train(self, feature_vector, labels):
        for _ in range(self.iterations):
            loss_list = []
            for i, vector in enumerate(feature_vector):
                out = self.forward_pass(vector) 
                eps = 1e-12
                loss = -1*(labels[i] * log(out + eps) + (1-labels[i]) * log(1-out+eps)) + self.lmbda * np.sum(self.W**2)
                loss_list.append(loss)
                self.backward_pass(vector, out, labels[i])
            average_loss = sum(loss_list) / len(loss_list)
            print(average_loss)

    def fit(self, data, labels):
        """
        This method is used to train your model and will generate a trained model file for some given input_file
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        self.features = Features(data)
        feature_vector = self.features.get_features(data)
        vocab_size = len(self.features.tfIDF.vocab)
        self.W = (np.random.randn(vocab_size) * 0.01).astype(np.float32)
        self.train(feature_vector, labels)


        # model = None
        # ## Save the model
        # self.save_model(model)
        # return model


    def classify(self, data, labels):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits that you make from the training sets given to you
        :param input_file: path to input file with a text per line **without labels** (note that this is different from the training data format!)
        :param model: the pretrained model
        :return: predictions list
        """
        feature_vector = self.features.get_features(data)
        correct = 0
        wrong = 0
        for label, feature in zip(labels, feature_vector):
            out = self.forward_pass(feature) 
            # if wrong <10:
            #     print(f'Out value : {out}')
            pred = 0 if out < 0.5 else 1
            if pred == label:
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong)
    

def train_test_split(text, labels):
    N = len(labels)
    ratio = 0.9
    rng = np.random.default_rng(42) 
    idx = rng.permutation(N)
    cut = int(ratio * N)
    train_idx, test_idx = idx[:cut], idx[cut:]
    texts = np.array(text, dtype=object)
    labels = np.array(labels, dtype=object)
    return texts[train_idx], labels[train_idx], texts[test_idx], labels[test_idx]

data_file = '/Users/anuragmudgil/Desktop/Study/AdvancedNLP/Hw1/startercode/datastores/products.train.txt'
with open(data_file) as file:
    data = file.read().splitlines()

data_split = map(methodcaller("rsplit", "\t", 1), data)
texts, text_labels = map(list, zip(*data_split))
labels = [0 if label == "neg" else 1 for label in text_labels]

# print(f'The split of labels are: {Counter(labels)}')
train_data, train_labels, test_data, test_lables = train_test_split(texts, labels)
lr = LogisticRegression()
lr.fit(train_data, train_labels)
accuracy = lr.classify(test_data, test_lables)
print(f'The accuracy is {accuracy}')