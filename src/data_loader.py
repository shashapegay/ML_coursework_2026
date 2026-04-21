import pandas as pd

def load_data(data_path):
    X = pd.read_csv(data_path + "secom.data", sep=r"\s+", header=None)
    y = pd.read_csv(data_path + "secom_labels.data", sep=r"\s+", header=None)

    y = y[0].replace(-1, 0)
    return X, y