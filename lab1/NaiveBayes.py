import numpy as np
import pandas as pd
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, alpha=0.1):
        self.priors = {}  # Prior probabilities of classes
        self.likelihoods = {}  # Conditional probabilities P(X|Y)
        self.likelihoods_unseen = {}
        self.classes = []
        self.alpha = alpha  # Laplace smoothing parameter
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {c: np.mean(y == c) for c in self.classes}
        self.likelihoods = {c: defaultdict(lambda: defaultdict(float)) for c in self.classes}
        self.likelihoods_unseen = {c: defaultdict(float) for c in self.classes}
        
        for c in self.classes:
            X_c = X[y == c]
            for feature in X.columns:
                feature_counts = X_c[feature].value_counts().to_dict()
                total_count = len(X_c)
                unique_values = X[feature].nunique()
                for value in feature_counts.keys():
                    if len(feature_counts) != unique_values:
                        self.likelihoods[c][feature][value] = (feature_counts[value] + self.alpha) / (total_count + self.alpha * unique_values)
                    else:
                        self.likelihoods[c][feature][value] = feature_counts[value] / total_count
                    # print(c, feature, value, self.likelihoods[c][feature][value])
                self.likelihoods_unseen[c][feature] = self.alpha / (total_count + self.alpha * unique_values)
    
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_probs = {}
            for c in self.classes:
                prob = self.priors[c]
                for feature, value in row.items():
                    prob *= self.likelihoods[c][feature].get(value, self.likelihoods_unseen[c][feature])
                class_probs[c] = prob
            # print(class_probs)
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)
    
    def name(self):
        return "NaiveBayes"


# data = {
#     'Q1': [0, 1, 2, 0, 0, 0, 1, 2, 2, 0],
#     'Q2': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
#     'Q3': [1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
#     'Q4': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
#     'S':  [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
# }
# df = pd.DataFrame(data)
# X, y = df[['Q1', 'Q2', 'Q3', 'Q4']], df['S']

# model = NaiveBayesClassifier()
# model.fit(X, y)

# data = {
#     'Q1': [2],
#     'Q2': [1],
#     'Q3': [1],
#     'Q4': [1]
# }
# df = pd.DataFrame(data)
# X = df[['Q1', 'Q2', 'Q3', 'Q4']]

# res = model.predict(X)
# print('result = ' + str(res[0]))