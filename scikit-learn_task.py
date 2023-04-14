# task 3, task 5

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./InternshipTaskRestAPI/data/covtype.data', delimiter=',', header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

clf_logistic_regression = LogisticRegression(multi_class='multinomial', random_state=123).fit(X_train,
                                                                                                              y_train)
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


# plots

models = ['Logistic Regression', 'Decision Tree']
scores = [clf_logistic_regression_score, clf_decision_tree_score]
f1_scores = [clf_logistic_regression_f1_score, clf_decision_tree_f1_score]

x = range(len(models))
width = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(x, scores, width, label='Accuracy')
rects2 = ax.bar([i + width for i in x], f1_scores, width, label='F1 Score')

ax.set_ylabel('Score')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(models)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()