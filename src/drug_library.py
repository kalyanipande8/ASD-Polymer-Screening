"""
Synthetic drug-molecule library for ASD screening.

Real poorly-water-soluble drug cores are mimicked with combinatorial SMILES
built from drug-like ring scaffolds (aromatic/heteroaromatic rings, fused
bicyclics) decorated with common pharmacophoric substituents (OH, NH2,
COOH, halogens, ethers, nitrile, CF3). Every generated SMILES is validated
by actually parsing it with RDKit (`Chem.MolFromSmiles`) before it's
accepted into the library -- invalid combinations are discarded and
resampled, so every molecule in this library is a real, RDKit-parseable
structure, not a string that merely looks like SMILES.

These are synthetic analogs, not specific named/marketed drugs -- there is
no publicly bundled 600+ compound real ASD stability dataset available in
this environment, so a combinatorial library stands in for it (disclosed
in README.md).
"""
from __future__ import annotations

import random

from rdkit import Chem

# All substituents are written in "branch form" (valid as the first atom(s)
# immediately after a ring atom's open parenthesis), which keeps every
# combination chemically valid regardless of how many substituents are
# combined on one core -- unlike prefix-position attachment, branch
# attachment always starts from an atom with a free valence bonding back to
# the ring, so there is no directionality pitfall to get wrong.
SUBSTITUENTS = [
    "O",             # -OH (hydroxyl)
    "N",             # -NH2 (primary amine)
    "C(=O)O",        # -COOH (carboxylic acid)
    "Cl",
    "Br",
    "F",
    "C",             # -CH3 (methyl)
    "CC",            # -CH2CH3 (ethyl)
    "OC",            # -OCH3 (methoxy)
    "C#N",           # -CN (nitrile)
    "C(F)(F)F",      # -CF3 (trifluoromethyl)
]

# Monocyclic core templates with 1-3 open branch slots. `n_slots` tells the
# generator how many {s#} placeholders the template expects.
MONOCYCLIC_TEMPLATES = [
    ("c1ccc({s0})cc1", 1),                 # substituted benzene
    ("c1ccc({s0})nc1", 1),                 # substituted pyridine
    ("c1cc({s0})oc1", 1),                  # substituted furan
    ("c1cc({s0})sc1", 1),                  # substituted thiophene
    ("C1CCC({s0})CC1", 1),                 # substituted cyclohexane
    ("c1ccc({s0})c({s1})c1", 2),           # di-substituted benzene
    ("c1cc({s0})c({s1})c({s2})c1", 3),     # tri-substituted benzene
]

# Fixed bicyclic/biaryl scaffolds (unsubstituted, or with a single branch
# substituent on the pendant ring) -- these add real structural diversity
# (fused rings, biaryls, N-heterocycles) beyond simple monocyclic analogs.
BICYCLIC_TEMPLATES = [
    ("c1ccc2ccccc2c1", 0),                     # naphthalene
    ("c1ccc2[nH]ccc2c1", 0),                   # indole-like
    ("c1ccc(-c2ccccc2)cc1", 0),                # biphenyl
    ("c1ccc2ncccc2c1", 0),                     # quinoline-like
    ("c1ccc2cc({s0})ccc2c1", 1),               # substituted naphthalene
    ("c1ccc(-c2ccc({s0})cc2)cc1", 1),          # substituted biphenyl
]

ALL_TEMPLATES = MONOCYCLIC_TEMPLATES + BICYCLIC_TEMPLATES


def _try_build(rng: random.Random) -> str | None:
    template, n_slots = rng.choice(ALL_TEMPLATES)
    if n_slots == 0:
        smi = template
    else:
        slots = {f"s{i}": rng.choice(SUBSTITUENTS) for i in range(n_slots)}
        smi = template.format(**slots)
    # Validate by actually parsing with RDKit -- this is the real safety
    # net that makes every accepted molecule a genuine, valid structure.
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Canonicalize so syntactically-different-but-identical molecules
    # (e.g. different substituent orderings yielding the same structure)
    # are deduplicated correctly.
    return Chem.MolToSmiles(mol)


def generate_drug_smiles(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    smiles_set: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n * 200
    while len(smiles_set) < n and attempts < max_attempts:
        attempts += 1
        smi = _try_build(rng)
        if smi is None or smi in seen:
            continue
        seen.add(smi)
        smiles_set.append(smi)
    if len(smiles_set) < n:
        raise RuntimeError(
            f"Only generated {len(smiles_set)}/{n} unique valid molecules "
            f"after {attempts} attempts -- widen the template/substituent "
            f"pool."
        )
    return smiles_set


def drug_names(n: int) -> list[str]:
    return [f"CMPD-{i + 1:03d}" for i in range(n)]


if __name__ == "__main__":
    smis = generate_drug_smiles(50)
    print(f"Generated {len(smis)} unique, RDKit-valid molecules")
    for s in smis[:5]:
        print(" ", s)
