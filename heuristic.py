# task 2
def most_frequent_class_heuristic(df):
    return df.iloc[:, -1].value_counts().idxmax()
