import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.dataset import build_dataset
from src.model import train_and_evaluate, FEATURE_COLS
from src.ranking import rank_polymers_for_drug, screening_reduction
from src.descriptors import compute_descriptors
from src.polymers import POLYMERS
from src.miscibility import assess, estimate_drug_delta, greenhalgh_class


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


@pytest.fixture(scope="module")
def trained(dataset):
    clf, le, metrics, split = train_and_evaluate(dataset)
    return clf, le, metrics, split


def test_dataset_has_600_plus_pairs(dataset):
    assert len(dataset) >= 600


def test_dataset_covers_multiple_real_polymers(dataset):
    polymer_names = {p.name for p in POLYMERS}
    assert set(dataset["polymer"].unique()) <= polymer_names
    assert dataset["polymer"].nunique() >= 6


def test_dataset_has_both_classes(dataset):
    counts = dataset["stable"].value_counts()
    assert counts.get(0, 0) > 50
    assert counts.get(1, 0) > 50


def test_descriptors_match_rdkit_reference_values_for_aspirin():
    """A direct RDKit-vs-RDKit sanity check: compute_descriptors() must
    agree exactly with RDKit's own descriptor functions called directly on
    the same parsed molecule, and those values must match aspirin's
    well-known real literature descriptor values (MW=180.16,
    TPSA=63.6 A^2). This is the test that would fail immediately if
    compute_descriptors() silently fell back to a proxy/approximation
    instead of real RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors as RDDescriptors
    from rdkit.Chem import rdMolDescriptors

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    d = compute_descriptors(aspirin_smiles)
    mol = Chem.MolFromSmiles(aspirin_smiles)

    assert d.mol_weight == pytest.approx(RDDescriptors.MolWt(mol), abs=0.01)
    assert d.logp == pytest.approx(Crippen.MolLogP(mol), abs=0.001)
    assert d.tpsa == pytest.approx(rdMolDescriptors.CalcTPSA(mol), abs=0.01)
    # Real literature/RDKit reference values for aspirin
    assert d.mol_weight == pytest.approx(180.16, abs=0.1)
    assert d.tpsa == pytest.approx(63.6, abs=0.1)
    assert d.hbd == 1   # the carboxylic acid O-H
    assert d.hba == 3   # RDKit's Lipinski NumHAcceptors for aspirin


def test_descriptor_values_are_physically_reasonable():
    d = compute_descriptors("c1ccc(O)cc1")  # phenol, a real valid molecule
    assert 50 < d.mol_weight < 300
    assert d.hbd >= 1  # has an OH
    assert d.tpsa > 0


def test_miscibility_relationship_is_monotonic():
    """Closer solubility parameters -> lower chi (more miscible)."""
    d = compute_descriptors("c1ccc(O)c(N)c1")  # real valid molecule (amino-phenol-like)
    delta_drug = estimate_drug_delta(d)
    close_polymer = min(POLYMERS, key=lambda p: abs(p.delta - delta_drug))
    far_polymer = max(POLYMERS, key=lambda p: abs(p.delta - delta_drug))
    close_res = assess(d, close_polymer)
    far_res = assess(d, far_polymer)
    assert close_res.chi <= far_res.chi
    assert greenhalgh_class(delta_drug, close_polymer) in {"miscible", "borderline", "immiscible"}


def test_model_accuracy_in_reasonable_band(trained):
    _, _, metrics, _ = trained
    # target ~82%; allow a realistic tuning band
    assert 0.70 <= metrics["accuracy"] <= 0.95


def test_model_beats_majority_baseline(dataset, trained):
    _, _, metrics, (X_test, y_test, y_pred) = trained
    majority_baseline = max(y_test.mean(), 1 - y_test.mean())
    assert metrics["accuracy"] > majority_baseline


def test_ranking_sanity_known_good_pair_ranks_highly(dataset, trained):
    """A drug-polymer pair with a very small solubility-parameter mismatch
    (chi near zero) and favorable H-bonding should rank near the top of
    the candidate list for that drug, more often than not."""
    clf, le, metrics, _ = trained

    hits = 0
    checked = 0
    for drug_id, group in dataset.groupby("drug_id"):
        if len(group) < 4:
            continue
        ranked = rank_polymers_for_drug(group, clf, FEATURE_COLS, le)
        best_true_row = group.loc[group["chi"].idxmin()]
        best_polymer_name = best_true_row["polymer"]
        top_half = ranked.head(max(1, len(ranked) // 2))
        checked += 1
        if best_polymer_name in top_half["polymer"].values:
            hits += 1

    # the lowest-chi (best miscibility) polymer should land in the top
    # half of the ranking for a clear majority of drugs
    assert checked > 0
    assert hits / checked >= 0.6


def test_screening_reduction_is_meaningful(dataset, trained):
    clf, le, metrics, (X_test, y_test, y_pred) = trained
    test_df = dataset.loc[X_test.index]
    reduction, per_drug_df, extra = screening_reduction(test_df, clf, FEATURE_COLS, le)
    assert 0.0 < extra["predicted_positive_reduction"] < 0.95
    assert extra["precision_of_tested_set"] > 0.5


def test_all_generated_drug_smiles_are_rdkit_valid(dataset):
    """Every molecule in the drug library must be a real, parseable
    structure -- not just a string that superficially looks like SMILES."""
    from rdkit import Chem

    unique_smiles = dataset["smiles"].unique()
    assert len(unique_smiles) >= 50
    for smi in unique_smiles:
        assert Chem.MolFromSmiles(smi) is not None, f"Invalid SMILES slipped through: {smi}"


def test_feature_columns_present_in_dataset(dataset):
    df = dataset.copy()
    for col in FEATURE_COLS:
        if col == "polymer_enc":
            continue
        assert col in df.columns
