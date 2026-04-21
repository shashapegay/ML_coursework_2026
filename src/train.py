import joblib

def train_and_save(model, X_train, y_train, path="models/model.joblib"):
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model