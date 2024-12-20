import joblib
import json
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from dvclive import Live
import os

EVAL_PATH = "eval"

# Load model and data
def load_model(path):
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise ValueError(f"The loaded object from {path} is not a valid model.")
    return model

def load_data(path):
    import pandas as pd
    return pd.read_csv(path)

# Main evaluation
if __name__ == "__main__":
    import sys
    model_paths = sys.argv[1:3]  # Two model paths
    features_path = sys.argv[3]

    X_test = load_data(f"{features_path}/X_test.csv")
    y_test = load_data(f"{features_path}/y_test.csv").values.ravel()

    os.makedirs(EVAL_PATH, exist_ok=True)

    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace(".pkl", "")
        try:
            model = load_model(model_path)
        except ValueError as e:
            print(e)
            continue

        y_pred = model.predict(X_test)

        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None

        # Metrics calculation
        if y_proba is not None:
            if y_proba.shape[1] > 1:  # Multiclass
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                avg_precision = average_precision_score(y_test, y_proba, average="macro")
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)
                precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=1)
            else:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 0])
                avg_precision = average_precision_score(y_test, y_proba[:, 0])
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
        else:
            roc_auc = None
            avg_precision = None
            fpr, tpr, precision, recall = None, None, None, None

        cm = confusion_matrix(y_test, y_pred)

        # Save metrics and plots
        model_eval_path = os.path.join(EVAL_PATH, model_name)
        os.makedirs(model_eval_path, exist_ok=True)

        # Save ROC curve if applicable
        if fpr is not None and tpr is not None:
            roc_path = os.path.join(model_eval_path, "roc.json")
            with open(roc_path, "w") as f:
                json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, f)

        # Save Precision-Recall curve if applicable
        if precision is not None and recall is not None:
            prc_path = os.path.join(model_eval_path, "prc.json")
            with open(prc_path, "w") as f:
                json.dump({"precision": precision.tolist(), "recall": recall.tolist()}, f)

        # Save confusion matrix
        cm_path = os.path.join(model_eval_path, "cm.json")
        with open(cm_path, "w") as f:
            json.dump({"confusion_matrix": cm.tolist()}, f)

        # Save metrics
        metrics_path = os.path.join(model_eval_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "roc_auc": roc_auc,
                    "avg_precision": avg_precision,
                    "confusion_matrix": cm.tolist(),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                },
                f,
            )

        # Log metrics with DVCLive
        with Live(model_eval_path, dvcyaml=False) as live:
            if roc_auc is not None:
                live.log_metric("roc_auc", roc_auc)
            if avg_precision is not None:
                live.log_metric("avg_precision", avg_precision)
