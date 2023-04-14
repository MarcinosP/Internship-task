# task 6

import tensorflow as tf
import pickle
from django.core.management.base import BaseCommand
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class Command(BaseCommand):
    help = 'Train the machine learning model'

    def handle(self, *args, **options):
        df = pd.read_csv('data/covtype.data', delimiter=',', header=None)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1] - 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

        classes = len(y.unique())

        tensorflow_model = Sequential([
            Dense(160, activation='relu'),
            Dense(40, activation='relu'),
            Dense(classes, activation='softmax')
        ], name="my_model")

        optimizer = tf.keras.optimizers.Adam(0.001)
        tensorflow_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tensorflow_model.fit(
            X_train, y_train,
            epochs=20
        )

        logistic_regression_model = LogisticRegression(multi_class='multinomial', random_state=123).fit(X_train,
                                                                                                        y_train)
        decision_tree_model = DecisionTreeClassifier(random_state=123).fit(X_train, y_train)

        models = {'tensorflow_model': tensorflow_model, 'logistic_regression_model': logistic_regression_model,
                  'decision_tree_model': decision_tree_model}

        with open('models.pickle', 'wb') as f:
            pickle.dump(models, f)
