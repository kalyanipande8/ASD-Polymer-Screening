"""
Polymer ranking and screening-reduction calculation.

For a given drug, candidate polymers are ranked by a combined score of
(a) the trained classifier's predicted probability of ASD stability and
(b) the Flory-Huggins miscibility proxy (lower chi = better), then compared
against ground-truth "true_prob_stable" labels used to generate the data.

Screening-reduction metric
---------------------------
Instead of testing all candidate polymers physically for every drug, a lab
would only run physical stability testing on the model's top-K predicted
polymers. The screening reduction is the real fraction of candidates that
can be skipped while still capturing a target share of the truly stable
candidates -- i.e. this is a recall-constrained reduction:

    reduction = 1 - (candidates_tested / total_candidates)

where candidates_tested is the smallest top-K (per drug) such that at
least `recall_target` (default 90%) of that drug's actually-stable
polymers are captured within the top-K. This is computed out-of-sample
(on the held-out test set) and averaged across drugs, so it is a measured
quantity, not an assumed one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_polymers_for_drug(df_drug: pd.DataFrame, clf, feature_cols, le) -> pd.DataFrame:
    df_drug = df_drug.copy()
    df_drug["polymer_enc"] = le.transform(df_drug["polymer"])
    X = df_drug[feature_cols]
    proba = clf.predict_proba(X)[:, 1]
    df_drug["predicted_prob_stable"] = proba
    # combined score: predicted probability minus a normalized chi penalty
    chi_norm = (df_drug["chi"] - df_drug["chi"].min()) / (df_drug["chi"].max() - df_drug["chi"].min() + 1e-9)
    df_drug["combined_score"] = df_drug["predicted_prob_stable"] - 0.15 * chi_norm
    return df_drug.sort_values("combined_score", ascending=False).reset_index(drop=True)


def screening_reduction(test_df: pd.DataFrame, clf, feature_cols, le, recall_target: float = 0.90):
    """Two real, measured quantities on the held-out test set:

    1. `recall_constrained_reduction`: per drug, the minimal top-K (ranked by
       predicted probability) needed to capture `recall_target` of that
       drug's TRUE stable polymers, averaged into an overall fraction of
       candidates skipped. This is the conservative, recall-safe number.

    2. `predicted_positive_reduction`: the practical screening workflow --
       a lab only physically tests polymers the classifier predicts stable
       (predict==1) instead of testing every candidate. The reduction is
       1 - (predicted-positive count / total candidates), and its
       trustworthiness is anchored by the classifier's measured precision
       on those same predicted-positive calls (i.e. we only skip testing
       candidates predicted negative, and we report what fraction of the
       predicted-positive set is actually correct, from the confusion
       matrix -- no assumed numbers).
    """
    total_candidates = 0
    total_tested = 0
    per_drug_rows = []

    for drug_id, group in test_df.groupby("drug_id"):
        n = len(group)
        n_true_stable = group["stable"].sum()
        if n_true_stable == 0:
            total_candidates += n
            total_tested += min(1, n)
            continue

        g = group.copy()
        g["polymer_enc"] = le.transform(g["polymer"])
        X = g[feature_cols]
        g["predicted_prob_stable"] = clf.predict_proba(X)[:, 1]
        g = g.sort_values("predicted_prob_stable", ascending=False).reset_index(drop=True)

        cum_true_stable = g["stable"].cumsum()
        needed_fraction = recall_target * n_true_stable
        k_idx = np.searchsorted(cum_true_stable.values, needed_fraction) + 1
        k = min(int(k_idx), n)

        total_candidates += n
        total_tested += k
        per_drug_rows.append({
            "drug_id": drug_id,
            "n_candidates": n,
            "n_true_stable": int(n_true_stable),
            "k_tested": k,
            "reduction": 1 - (k / n),
        })

    recall_constrained_reduction = 1 - (total_tested / total_candidates)
    per_drug_df = pd.DataFrame(per_drug_rows)

    # Practical "only test predicted-stable" workflow metric
    g_all = test_df.copy()
    g_all["polymer_enc"] = le.transform(g_all["polymer"])
    X_all = g_all[feature_cols]
    y_pred_all = clf.predict(X_all)
    n_total = len(g_all)
    n_tested = int(y_pred_all.sum())
    predicted_positive_reduction = 1 - (n_tested / n_total)

    tp = int(((y_pred_all == 1) & (g_all["stable"].values == 1)).sum())
    precision_of_tested = tp / n_tested if n_tested else 0.0

    extra_metrics = {
        "predicted_positive_reduction": predicted_positive_reduction,
        "precision_of_tested_set": precision_of_tested,
        "n_total": n_total,
        "n_tested": n_tested,
    }
    return recall_constrained_reduction, per_drug_df, extra_metrics
