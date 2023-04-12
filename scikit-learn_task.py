# task 3

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('data/covtype.data', delimiter=',', header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

clf_logistic_regression = LogisticRegression(random_state=123).fit(X_train, y_train)
clf_logistic_regression_pred = clf_logistic_regression.predict(X_test)
clf_logistic_regression_score = clf_logistic_regression.score(X_test, y_test)
clf_logistic_regression_f1_score = f1_score(y_test, clf_logistic_regression_pred, average='weighted')

print(f"logistic regression score: {clf_logistic_regression_score:.3f}")
print(f"logistic regression F1-score: {clf_logistic_regression_f1_score:.3f}")

clf_decision_tree = DecisionTreeClassifier(random_state=123).fit(X_train, y_train)
clf_decision_tree_pred = clf_decision_tree.predict(X_test)
clf_decision_tree_score = clf_decision_tree.score(X_test, y_test)
clf_decision_tree_f1_score = f1_score(y_test, clf_decision_tree_pred, average='weighted')

print(f"decision tree score:{clf_decision_tree_score:.3f}")
print(f"decision tree F1-score:{clf_decision_tree_f1_score:.3f}")
