# task 2
import pandas as pd

df = pd.read_csv('./InternshipTaskRestAPI/data/covtype.data', delimiter=',', header=None)

def most_frequent_class_heuristic(df):
    return df.iloc[:, -1].value_counts().idxmax()

print(most_frequent_class_heuristic(df))