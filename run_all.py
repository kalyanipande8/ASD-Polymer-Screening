"""
End-to-end pipeline runner for ASD Polymer Screening.

1. Builds the synthetic 600+ drug-polymer pair dataset.
2. Trains a RandomForest classifier to predict ASD physical stability.
3. Reports held-out accuracy/precision/recall/F1.
4. Ranks candidate polymers for a few example drugs.
5. Computes the real screening-reduction metric on the held-out test set.
6. Writes dataset + results to the data/ and results/ folders.
"""
from __future__ import annotations

import json
import os

from src.dataset import build_dataset
from src.model import train_and_evaluate, FEATURE_COLS
from src.ranking import rank_polymers_for_drug, screening_reduction

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Building drug-polymer pair dataset ===")
    df = build_dataset()
    df.to_csv(os.path.join(DATA_DIR, "asd_pairs.csv"), index=False)
    print(f"Dataset: {df.shape[0]} pairs, {df['drug_id'].nunique()} drugs x "
          f"{df['polymer'].nunique()} polymers")
    print("Label balance:", df["stable"].value_counts(normalize=True).round(3).to_dict())

    print("\n=== Training RandomForest classifier ===")
    clf, le, metrics, (X_test, y_test, y_pred) = train_and_evaluate(df)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Polymer ranking example ===")
    example_drug = df["drug_id"].iloc[0]
    drug_rows = df[df["drug_id"] == example_drug]
    ranked = rank_polymers_for_drug(drug_rows, clf, FEATURE_COLS, le)
    print(f"Top polymer candidates for {example_drug}:")
    print(ranked[["polymer", "predicted_prob_stable", "chi", "combined_score", "stable"]].head(5).to_string(index=False))

    print("\n=== Screening-reduction metric (held-out test set) ===")
    test_df = df.loc[X_test.index]
    reduction, per_drug_df, extra = screening_reduction(test_df, clf, FEATURE_COLS, le, recall_target=0.90)
    print(f"Recall-constrained screening reduction (90% recall target): {reduction*100:.1f}%")
    print(f"Predicted-positive-only screening reduction: {extra['predicted_positive_reduction']*100:.1f}%")
    print(f"Precision of the tested (predicted-stable) set: {extra['precision_of_tested_set']*100:.1f}%")
    per_drug_df.to_csv(os.path.join(RESULTS_DIR, "screening_reduction_per_drug.csv"), index=False)

    results_summary = {
        "n_pairs": int(df.shape[0]),
        "n_drugs": int(df["drug_id"].nunique()),
        "n_polymers": int(df["polymer"].nunique()),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "recall_constrained_reduction_pct": round(reduction * 100, 2),
        "predicted_positive_reduction_pct": round(extra["predicted_positive_reduction"] * 100, 2),
        "precision_of_tested_set_pct": round(extra["precision_of_tested_set"] * 100, 2),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(results_summary, indent=2))


if __name__ == "__main__":
    main()
