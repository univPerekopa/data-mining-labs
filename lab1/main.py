import pandas as pd

file_path = 'datasets/data_train.csv'
df = pd.read_csv(file_path)

X = df.iloc[:, :-1]  # Features (all columns except the last)
y = df.iloc[:, -1]   # Target variable (last column)

from OneRule import OneRuleClassifier
from NaiveBayes import NaiveBayesClassifier
from DecisionTree import DecisionTreeClassifier
from KNN import KNNClassifier

clfs = [OneRuleClassifier(), NaiveBayesClassifier(), DecisionTreeClassifier(), KNNClassifier()]
for clf in clfs:
    clf.fit(X, y)

file_path = 'datasets/data_test.csv'
df = pd.read_csv(file_path)
X_test = df.iloc[:, :-1]  # Features (all columns except the last)

for clf in clfs:
    res = clf.predict(X_test)
    name = clf.name()
    print("Model " + name + " predicted " + str(res[0]))

