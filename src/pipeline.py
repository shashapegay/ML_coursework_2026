from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def create_pipeline(model):
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50)),
        ("model", model)
    ])