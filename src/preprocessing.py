from sklearn.impute import SimpleImputer

def remove_nan_columns(X, threshold=0.5):
    missing_ratio = X.isnull().mean()
    return X.loc[:, missing_ratio < threshold]

def impute_data(X):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X)