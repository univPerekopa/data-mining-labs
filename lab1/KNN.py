import numpy as np
import pandas as pd
from collections import defaultdict

class KNNClassifier:
    def __init__(self, k=3, weights=False):
        self.k = k
        self.weights = weights

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def weight(self, point1, point2):
        return 1.0 / np.sum((point1 - point2) ** 2)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        predictions = [self._predict(x) for _, x in X_test.iterrows()]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.euclidean_distance(np.array(x), np.array(x_train)) for _, x_train in self.X_train.iterrows()]
        
        k_indices = np.argsort(distances)[:self.k]
        # print(k_indices)
        # print(self.y_train)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        if self.weights:
            k_nearest_weights = [self.weight(np.array(x), np.array(self.X_train.iloc[[i]])) for i in k_indices]
        else:
            k_nearest_weights = [1 for _ in k_indices]
        # print(k_indices)
        # print(k_nearest_labels)
        # print(k_nearest_weights)
        dict = defaultdict(float)
        best_w = -1
        best_c = 0
        for i in range(self.k):
            dict[k_nearest_labels[i]] += k_nearest_weights[i]
            if dict[k_nearest_labels[i]] > best_w:
                best_w = dict[k_nearest_labels[i]]
                best_c = k_nearest_labels[i]

        return best_c
    
    def name(self):
        return "KNN"

# data = {
#     'Q1': [0, 1, 2, 0, 0, 0, 1, 2, 2, 0],
#     'Q2': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
#     'Q3': [1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
#     'Q4': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
#     'S':  [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
# }
# df = pd.DataFrame(data)
# X, y = df[['Q1', 'Q2', 'Q3', 'Q4']], df['S']

# model = KNNClassifier(weights=True)
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
