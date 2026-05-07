import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

EVAL_THRESHOLD = 0.70


def _build_model(params: dict):
    model_type = params.get("model_type", "random_forest")
    rf_gb_keys = {"n_estimators", "max_depth", "min_samples_split"}
    boost_keys  = {"n_estimators", "max_depth"}

    if model_type == "gradient_boosting":
        kw = {k: v for k, v in params.items() if k in rf_gb_keys and v is not None}
        return GradientBoostingClassifier(**kw, random_state=42)
    elif model_type == "catboost":
        kw = {k: v for k, v in params.items() if k in boost_keys and v is not None}
        return CatBoostClassifier(**kw, random_state=42, verbose=0)
    elif model_type == "xgboost":
        kw = {k: v for k, v in params.items() if k in boost_keys and v is not None}
        return XGBClassifier(**kw, random_state=42, eval_metric="mlogloss", verbosity=0)
    elif model_type == "lightgbm":
        kw = {k: v for k, v in params.items() if k in boost_keys and v is not None}
        return LGBMClassifier(**kw, random_state=42, verbose=-1)
    else:
        kw = {k: v for k, v in params.items() if k in rf_gb_keys and v is not None}
        return RandomForestClassifier(**kw, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: kiem tra phan phoi nhan truoc khi huan luyen
    label_dist = y_train.value_counts(normalize=True).to_dict()
    for label, ratio in label_dist.items():
        if ratio < 0.10:
            print(f"WARNING: class {label} chi chiem {ratio:.1%} tap huan luyen — co the bi mat can bang.")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = _build_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # Bonus 3: precision va recall tung lop
        precision_per_class = precision_score(y_eval, preds, average=None, zero_division=0)
        recall_per_class    = recall_score(y_eval, preds, average=None, zero_division=0)
        cm                  = confusion_matrix(y_eval, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for i, (p, r) in enumerate(zip(precision_per_class, recall_per_class)):
            mlflow.log_metric(f"precision_class_{i}", p)
            mlflow.log_metric(f"recall_class_{i}", r)

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)

        # Bonus 5: ghi phan phoi nhan vao metrics
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "label_distribution": {str(k): round(v, 4) for k, v in label_dist.items()},
        }
        for i, (p, r) in enumerate(zip(precision_per_class, recall_per_class)):
            metrics[f"precision_class_{i}"] = round(float(p), 4)
            metrics[f"recall_class_{i}"]    = round(float(r), 4)

        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Bonus 3: ghi report.txt
        report_lines = [
            f"Model type : {params.get('model_type', 'random_forest')}",
            f"Accuracy   : {acc:.4f}",
            f"F1 (weighted): {f1:.4f}",
            "",
            "Per-class metrics:",
        ]
        for i, (p, r) in enumerate(zip(precision_per_class, recall_per_class)):
            report_lines.append(f"  Class {i}: precision={p:.4f}  recall={r:.4f}")
        report_lines += ["", "Confusion matrix (rows=actual, cols=predicted):"]
        for row in cm:
            report_lines.append("  " + "  ".join(f"{v:4d}" for v in row))
        report_lines += ["", "Label distribution (train):"]
        for k, v in sorted(label_dist.items()):
            flag = " <-- WARNING: < 10%" if v < 0.10 else ""
            report_lines.append(f"  Class {k}: {v:.1%}{flag}")

        with open("outputs/report.txt", "w") as f:
            f.write("\n".join(report_lines) + "\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
