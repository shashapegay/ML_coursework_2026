from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def remove_constant_features(X):
    selector = VarianceThreshold(0.0)
    return selector.fit_transform(X)

def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)