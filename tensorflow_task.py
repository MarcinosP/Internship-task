from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('data/covtype.data', delimiter=',', header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

classes = len(y.unique())


def create_model(units1=100, units2=20, lr=0.001):
    model = Sequential([
        Dense(units1, activation='relu'),
        Dense(units2, activation='relu'),
        Dense(classes, activation='softmax')
    ], name="my_model")

    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


param_grid = {
    'units1': [50, 100, 150],
    'units2': [10, 20, 30],
    'lr': [0.001, 0.01, 0.1]
}

model = KerasClassifier(build_fn=create_model, epochs=4, verbose=0)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)
