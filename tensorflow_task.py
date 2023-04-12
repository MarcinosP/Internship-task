# from tensorflow.keras.models import Sequential
# from keras.layers import Dense
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# df = pd.read_csv('data/covtype.data', delimiter=',', header=None)
#
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)
#
# model = Sequential(
#     [
#         Dense(25, activation='relu'),
#         Dense(15, activation='relu'),
#         Dense(1, activation='relu')
#     ], name="my_model"
# )

# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
