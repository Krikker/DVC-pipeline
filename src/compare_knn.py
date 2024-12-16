import json
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Compare execution times and accuracies
results = {}

for algo in ['brute', 'kd_tree']:
    knn = KNeighborsClassifier(n_neighbors=3, algorithm=algo)
    start_time = time.time()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    execution_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    results[algo] = {
        "execution_time": execution_time,
        "accuracy": accuracy
    }

# Save results
with open('reports/comparison.txt', 'w') as f:
    for algo, metrics in results.items():
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Execution Time: {metrics['execution_time']:.4f} seconds\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write("\n")
