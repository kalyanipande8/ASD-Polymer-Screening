"""
Synthetic drug-molecule library for ASD screening.

Real poorly-water-soluble drug cores are mimicked with combinatorial
SMILES strings built from drug-like scaffolds (aromatic rings, fused rings)
decorated with common pharmacophoric substituents (OH, NH2, COOH, halogens,
ethers, additional rings). This produces a diverse set of >=50 distinct
"molecules" spanning a realistic range of MW / LogP / HBD / HBA / TPSA,
without claiming these are specific named drugs (they are synthetic
analogs, used because no real 600+ compound ASD stability dataset is
publicly bundled for this environment).
"""
from __future__ import annotations

import random

SCAFFOLDS = [
    "c1ccccc1",              # benzene
    "c1ccc2ccccc2c1",        # naphthalene
    "c1ccc2[nH]ccc2c1",      # indole-like
    "c1ccc(cc1)c1ccccc1",    # biphenyl
    "C1CCCCC1",              # cyclohexyl
    "c1ccoc1",               # furan
    "c1ccsc1",               # thiophene
    "c1ccncc1",              # pyridine
]

SUBSTITUENTS = [
    "O",        # ether/ketone oxygen
    "OH",       # hydroxyl -> HBD
    "NH2",      # amine -> HBD
    "NH",       # secondary amine -> HBD
    "COOH",     # carboxylic acid -> HBD + HBA
    "Cl",
    "Br",
    "F",
    "C(=O)O",   # ester-like
    "C",        # methyl
    "CC",       # ethyl
    "N",        # ring nitrogen substitution
]


def generate_drug_smiles(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    smiles_set = []
    seen = set()
    attempts = 0
    while len(smiles_set) < n and attempts < n * 50:
        attempts += 1
        scaffold = rng.choice(SCAFFOLDS)
        n_subs = rng.randint(1, 4)
        subs = "".join(rng.choice(SUBSTITUENTS) for _ in range(n_subs))
        maybe_second_ring = rng.choice(["", rng.choice(SCAFFOLDS)])
        smi = f"{scaffold}{subs}{maybe_second_ring}"
        if smi not in seen:
            seen.add(smi)
            smiles_set.append(smi)
    return smiles_set


def drug_names(n: int) -> list[str]:
    return [f"CMPD-{i+1:03d}" for i in range(n)]
