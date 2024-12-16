import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Train KNN models
knn_brute = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
knn_kdtree = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')

knn_brute.fit(X_train, y_train)
knn_kdtree.fit(X_train, y_train)

# Evaluate models
y_pred_brute = knn_brute.predict(X_test)
y_pred_kdtree = knn_kdtree.predict(X_test)

metrics = {
    "brute": {
        "accuracy": accuracy_score(y_test, y_pred_brute),
        "confusion_matrix": confusion_matrix(y_test, y_pred_brute).tolist(),
        "classification_report": classification_report(y_test, y_pred_brute, output_dict=True)
    },
    "kd_tree": {
        "accuracy": accuracy_score(y_test, y_pred_kdtree),
        "confusion_matrix": confusion_matrix(y_test, y_pred_kdtree).tolist(),
        "classification_report": classification_report(y_test, y_pred_kdtree, output_dict=True)
    }
}

# Save metrics and models
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)
