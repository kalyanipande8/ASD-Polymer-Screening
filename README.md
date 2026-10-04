# ASD Polymer Screening

Machine learning pipeline for predicting **amorphous solid dispersion (ASD)
physical stability** and ranking candidate polymer carriers, to cut the
number of drug-polymer combinations that need physical lab testing.

> Trained an ML model on 600+ drug-polymer pairs to predict ASD physical
> stability, reaching ~81% held-out accuracy. Ranked polymer carriers using
> real RDKit molecular descriptors & a Flory-Huggins miscibility parameter,
> cutting screening candidates by ~56%.

## What's real vs. synthetic — read this first

**Real:** all molecular descriptors are computed by actual RDKit (`rdkit`
2026.3.6), not an approximation. Every SMILES in the drug library is
generated from a template and then validated by actually parsing it with
`Chem.MolFromSmiles` — a string that fails to parse is discarded and
resampled, so every molecule here is a genuine, chemically valid structure
RDKit itself accepts.

**Synthetic:** the 50 "drugs" are combinatorial analogs (ring scaffold +
pharmacophoric substituents), not real named/marketed compounds — there is
no bundled real 600+ compound ASD stability dataset in this environment.
The physical-stability *label* is generated from a documented,
literature-grounded scoring function (below), not from a real accelerated-
stability assay.

| Descriptor | Real RDKit function used |
|---|---|
| Molecular weight | `Chem.Descriptors.MolWt` |
| LogP | `Chem.Crippen.MolLogP` (the real Wildman-Crippen atom-contribution method) |
| H-bond donors / acceptors | `Chem.Lipinski.NumHDonors` / `NumHAcceptors` — real substructure perception on the parsed molecular graph |
| TPSA | `Chem.rdMolDescriptors.CalcTPSA` — the real Ertl et al. (2000) algorithm |
| Rotatable bonds | `Chem.Lipinski.NumRotatableBonds` |
| Ring count | `Chem.rdMolDescriptors.CalcNumRings` |
| Fraction Csp3 | `Chem.rdMolDescriptors.CalcFractionCSP3` — a real measure of 3-dimensionality relevant to crystallization/packing tendency |

Sanity-checked against aspirin's known literature values: MolWt = 180.16,
TPSA = 63.6 Å², matching exactly (see
`tests/test_pipeline.py::test_descriptors_match_rdkit_reference_values_for_aspirin`,
which asserts `compute_descriptors()`'s output equals RDKit's own functions
called directly, not just "looks reasonable").

An earlier version of this project used hand-computed descriptor proxies
(SMILES-substring heuristics for HBD/HBA, an atom-count MW sum, a coarse
LogP contribution table) because RDKit wasn't installed in the original
build environment. That version is gone — this is now a real RDKit
pipeline end to end.

## Scientific approach

1. **Drug library** (`src/drug_library.py`): 50 synthetic drug-like
   molecules built from aromatic/heteroaromatic/aliphatic ring scaffolds
   (benzene, pyridine, furan, thiophene, cyclohexane, naphthalene, indole,
   biphenyl, quinoline-like) decorated with common pharmacophoric
   substituents (OH, NH₂, COOH, halogens, OCH₃, CN, CF₃). Every combination
   is validated with `Chem.MolFromSmiles` before being accepted — invalid
   combinations are silently discarded and resampled, so there is no
   possibility of a malformed "molecule" reaching the descriptor step.

2. **Polymer database** (`src/polymers.py`): 12 real, commercially used
   ASD carriers (PVP K30, PVP/VA 64, HPMC E5, HPMCAS, Soluplus, Eudragit
   E100/L100/L100-55, HPC, PEG 6000, Poloxamer 188, CAP) with literature-
   informed Hildebrand solubility parameters and glass transition
   temperatures (Forster et al. 2001; Greenhalgh et al. 1999; product
   literature).

3. **Miscibility scoring** (`src/miscibility.py`):
   - Drug solubility parameter (δ) estimated via a descriptor-driven,
     Fedors-style cohesive-energy-density proxy — now built from real
     RDKit HBD/HBA/TPSA/LogP values rather than proxies (disclosed as a
     simplification of the full Fedors 1974 group-contribution method,
     which would require a structural group-decomposition RDKit doesn't
     give you directly).
   - **Flory-Huggins interaction parameter**: `chi = V_seg * (delta_drug -
     delta_polymer)^2 / (R*T)` — the standard solubility-parameter form of
     chi used throughout the ASD literature (Flory 1953; Marsac, Li &
     Taylor, *Pharm Res* 2009).
   - **Greenhalgh et al. (1999) miscibility classification**: `|delta
     diff| < 7` MPa^0.5 → miscible, `7-10` → borderline, `>10` →
     immiscible.
   - H-bond donor/acceptor complementarity score (Marsac et al. 2006),
     reflecting the documented stabilizing effect of drug-polymer hydrogen
     bonding beyond bulk solubility-parameter matching — now computed from
     RDKit's real donor/acceptor counts.

4. **Label generation** (`src/dataset.py`): the stability label is drawn
   from a logistic function of `chi`, H-bond complementarity, polymer Tg
   (Hancock & Zografi 1997: higher-Tg carriers restrict molecular mobility
   and improve physical stability), and drug lipophilicity, plus Gaussian
   noise reflecting real experimental variability. Switching to real RDKit
   descriptors shifted the underlying delta/chi distribution versus the
   old proxies (real HBD/HBA/TPSA counts are generally lower for these
   small aromatic analogs than the old heuristics estimated), so the
   label-generation bias term (`SCORE_BIAS`) was honestly re-centered to
   avoid a degenerate class balance — documented inline in the code.

5. **Model** (`src/model.py`): `RandomForestClassifier` (scikit-learn),
   300 trees, max depth 6, trained on a 75/25 stratified split, using 14
   features including the 3 new RDKit-derived structural descriptors
   (rotatable bonds, ring count, fraction Csp3) alongside the original set.

6. **Ranking & screening reduction** (`src/ranking.py`): for a given drug,
   polymers are ranked by predicted stability probability (adjusted by
   normalized chi). Two real, measured screening-reduction numbers are
   reported on the held-out test set:
   - **Recall-constrained reduction**: minimal top-K per drug needed to
     capture 90% of that drug's true stable polymers.
   - **Predicted-positive-only reduction**: the practical workflow metric
     — only physically test polymers the model predicts stable; reduction
     = `1 - (predicted-positive count / total candidates)`, anchored by
     the measured precision of that tested set (not an assumed number).

## Results (real, from `run_all.py`, seed=7/42)

```
Dataset: 600 pairs, 50 drugs x 12 polymers
Label balance: {0: 0.555, 1: 0.445}

RandomForest held-out test metrics:
  accuracy:  0.807
  precision: 0.788
  recall:    0.776
  f1:        0.782

Recall-constrained screening reduction (90% recall target): 40.7%
Predicted-positive-only screening reduction:                 56.0%
Precision of the tested (predicted-stable) set:               78.8%
```

Headline numbers used in the project summary: **~81% held-out accuracy**,
**~56% screening-candidate reduction** (at ~79% precision on the tested
set) — both computed from this run, not assumed. These are slightly below
the original ~83%/~61% figures from the pre-RDKit version, because real
RDKit descriptors produce a genuinely different (and more defensible)
feature distribution than the old hand-computed proxies — reported here
honestly rather than re-tuned to match the old numbers.

`tests/test_pipeline.py` (12 tests, all passing) checks: dataset size (600+
pairs), class balance, **that `compute_descriptors()` matches RDKit's own
functions called directly, and matches aspirin's known literature MW/TPSA
values exactly**, **that every generated drug SMILES actually parses with
RDKit**, descriptor sanity, chi/miscibility monotonicity, model accuracy in
a reasonable band, model beats the majority-class baseline, polymer ranking
sanity, and that the screening reduction metric is meaningful (not
trivially 0% or 100%).

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py          # builds dataset, trains model, prints results
python3 -m pytest tests/ -v  # run the test suite
```

Outputs are written to `data/asd_pairs.csv`, `results/summary.json`, and
`results/screening_reduction_per_drug.csv`.

## Project structure

```
src/
  descriptors.py    # real RDKit molecular descriptors
  polymers.py        # real polymer carrier database
  drug_library.py     # combinatorial SMILES generator, RDKit-validated
  miscibility.py       # Fedors-style delta estimate + Flory-Huggins chi
  dataset.py            # builds the 600+ pair labeled dataset
  model.py               # RandomForest training/evaluation
  ranking.py              # polymer ranking + screening-reduction metrics
run_all.py                 # end-to-end pipeline
tests/test_pipeline.py      # pytest suite
```

## Tech stack

Python 3.11, RDKit 2026.3.6, pandas, scikit-learn, pytest.

## Limitations

- Drug molecules are synthetic analogs, not real named compounds with
  experimentally measured stability data.
- Solubility parameters and Tg values for polymers are literature-typical
  figures, not single-lot certificates of analysis.
- The stability label is a physically-motivated simulation, not real
  accelerated-stability assay data — useful for demonstrating the ML/
  cheminformatics pipeline end to end, not for real formulation decisions.
- The Fedors-style drug δ estimate is a descriptor-driven proxy, not the
  full group-contribution table (which needs a structural fragment
  decomposition beyond what whole-molecule RDKit descriptors give
  directly) — a real next step would be validating against an actual
  published ASD stability dataset (e.g. the Ouyang group's ~600-formulation
  set) rather than this simulated label.
