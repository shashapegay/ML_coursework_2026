from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=10),
        "gb": GradientBoostingClassifier(),
        "svm": SVC(probability=True, class_weight='balanced')
    }