"""Train a customer churn model and save it for the Flask app.

Uses a synthetic dataset so the project runs out of the box.
To use real data, replace data.csv with the Kaggle "Telco Customer Churn"
dataset and adjust FEATURES below.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_months"]


def make_synthetic(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.normal(65, 25, n).clip(15, 120).round(2),
        "support_calls": rng.poisson(2, n),
        "contract_months": rng.choice([1, 12, 24], n, p=[0.55, 0.25, 0.20]),
    })
    score = (
        -0.04 * df.tenure_months
        + 0.02 * df.monthly_charges
        + 0.35 * df.support_calls
        - 0.06 * df.contract_months
        + rng.normal(0, 0.8, n)
    )
    df["churn"] = (score > np.percentile(score, 72)).astype(int)
    return df


def main():
    df = make_synthetic()
    df.to_csv("data.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["churn"], test_size=0.2, random_state=42, stratify=df["churn"]
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, model.predict(X_test)))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
