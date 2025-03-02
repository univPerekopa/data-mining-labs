import numpy as np
import pandas as pd
from collections import defaultdict

class DecisionTreeNode:
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature
        self.children = children
        self.value = value

class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y):
        if len(set(y)) == 1:
            return DecisionTreeNode(value=y.iloc[0], children={})
    
        arr = np.array(y)
        values, counts = np.unique(arr, return_counts=True)
        most_frequent = values[np.argmax(counts)]
        if len(X.columns) == 0:
            return DecisionTreeNode(value=most_frequent, children={})
        
        best_feature = self._find_best_split(X, y)

        children_ = defaultdict(DecisionTreeNode)
        for x_value in X[best_feature].unique():
            mask = X[best_feature] == x_value
            new_X = X[mask]
            new_X = new_X.drop(columns=[best_feature])
            new_y = y[mask]

            if len(new_X) > 0:
                children_[x_value] = self._build_tree(new_X, new_y)
        
        return DecisionTreeNode(feature=best_feature, children=children_, value=most_frequent)
    
    def _find_best_split(self, X, y):
        best_feature = None
        best_info_gain = -float('inf')
        
        for feature in X.columns:
            info_gain = self._information_gain(X[feature], y)
            # print(feature, info_gain)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
        
        return best_feature
    
    def _information_gain(self, feature_column, y):
        gain = self._entropy(y)

        for x_value in feature_column.unique():
            mask = feature_column == x_value
            entropy = self._entropy(y[mask])
            weight = len(y[mask]) / len(y)
            gain -= weight * entropy
        
        return gain
    
    def _entropy(self, y):
        proportions = y.value_counts(normalize=True)
        return -sum(proportions * np.log2(proportions))
    
    def predict(self, X):
        return X.apply(lambda row: self._predict_single(row, self.root), axis=1)
    
    def _predict_single(self, row, node: DecisionTreeNode):
        if node.feature != None and row[node.feature] in node.children:
            return self._predict_single(row, node.children[row[node.feature]])
        else:
            return node.value
    
    def debug_tree(self, tab_c = 0, node=None):
        if node == None:
            node = self.root
        tabs = ""
        for _ in range(tab_c):
            tabs += '\t'
        print(tabs + "f=" + str(node.feature))
        print(tabs + "value=" + str(node.value))
        print(tabs + "child count = " + str(len(node.children)))
        for k in node.children:
            print(tabs + "key = " + str(k))
            self.debug_tree(tab_c + 1, node.children[k])

    def name(self):
        return "DecisionTree"

# data = {
#     'Q1': [0, 1, 2, 0, 0, 0, 1, 2, 2, 0],
#     'Q2': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
#     'Q3': [1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
#     'Q4': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
#     'S':  [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
# }
# df = pd.DataFrame(data)
# X, y = df[['Q1', 'Q2', 'Q3', 'Q4']], df['S']

# model = DecisionTreeClassifier()
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

# # model.debug_tree()
