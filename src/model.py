"""
Train and evaluate an ML classifier for ASD physical-stability prediction.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "mol_weight", "logp", "hbd", "hba", "tpsa",
    "rotatable_bonds", "ring_count", "fraction_csp3",
    "delta_drug", "delta_polymer", "delta_diff", "chi",
    "hbond_complementarity", "polymer_tg", "polymer_enc",
]


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["polymer_enc"] = le.fit_transform(df["polymer"])
    X = df[FEATURE_COLS]
    y = df["stable"]
    return X, y, le


def train_and_evaluate(df: pd.DataFrame, seed: int = 42, n_estimators: int = 300,
                        max_depth: int = 6, test_size: float = 0.25):
    X, y, le = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return clf, le, metrics, (X_test, y_test, y_pred)


if __name__ == "__main__":
    from .dataset import build_dataset
    df = build_dataset()
    clf, le, metrics, _ = train_and_evaluate(df)
    for k, v in metrics.items():
        print(f"{k}: {v}")
