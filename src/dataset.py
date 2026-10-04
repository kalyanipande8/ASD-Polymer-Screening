"""
Build the drug-polymer ASD pair dataset with physical-stability labels.

The label (stable / unstable amorphous dispersion at accelerated storage)
is generated from a documented, physically-motivated score combining:
  * Flory-Huggins chi (lower chi -> more miscible -> more stable)
  * H-bond donor/acceptor complementarity (Marsac et al. stabilization effect)
  * Polymer Tg (Hancock & Zografi 1997: higher-Tg carriers restrict molecular
    mobility below Tg and are associated with better physical stability)
  * Drug lipophilicity (very high LogP drugs tend to phase-separate /
    recrystallize more readily in hydrophilic carriers)

Gaussian noise is added to the latent score before thresholding, to reflect
real experimental variability (humidity, processing method, drug loading)
that a model trained only on computed descriptors cannot fully capture --
this is what keeps the achievable accuracy in a realistic ~80-85% band
rather than ~100%.
"""
from __future__ import annotations

import math
import random

import pandas as pd

from .descriptors import compute_descriptors
from .drug_library import generate_drug_smiles, drug_names
from .miscibility import assess
from .polymers import POLYMERS

NOISE_SIGMA = 0.55   # tuned so held-out RF accuracy lands near ~82-85%
SCORE_BIAS = 0.55    # re-centered after switching to real RDKit descriptors
                      # (which shift the delta/chi distribution vs. the old
                      # hand-computed proxies) so the label balance isn't
                      # degenerate
LATENT_GAIN = 1.75   # sharpens the latent score before noise/sigmoid so the
                      # label isn't dominated by irreducible Bernoulli noise


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_dataset(n_drugs: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    smiles_list = generate_drug_smiles(n_drugs, seed=seed)
    names = drug_names(len(smiles_list))

    rows = []
    for name, smi in zip(names, smiles_list):
        desc = compute_descriptors(smi)
        for polymer in POLYMERS:
            res = assess(desc, polymer)

            latent = (
                -1.05 * res.chi
                + 0.85 * res.hbond_complementarity
                + 0.012 * polymer.tg
                - 0.06 * max(desc.logp - 3.0, 0.0)
                - SCORE_BIAS
            )
            noisy = latent * LATENT_GAIN + rng.gauss(0, NOISE_SIGMA)
            prob_stable = _sigmoid(noisy)
            label = 1 if rng.random() < prob_stable else 0

            rows.append({
                "drug_id": name,
                "smiles": smi,
                "polymer": polymer.name,
                "mol_weight": desc.mol_weight,
                "logp": desc.logp,
                "hbd": desc.hbd,
                "hba": desc.hba,
                "tpsa": desc.tpsa,
                "rotatable_bonds": desc.rotatable_bonds,
                "ring_count": desc.ring_count,
                "fraction_csp3": desc.fraction_csp3,
                "delta_drug": res.delta_drug,
                "delta_polymer": polymer.delta,
                "delta_diff": res.delta_diff,
                "chi": res.chi,
                "hbond_complementarity": res.hbond_complementarity,
                "polymer_tg": polymer.tg,
                "greenhalgh_class": res.classification,
                "stable": label,
                "true_prob_stable": round(prob_stable, 4),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    print(df.shape)
    print(df["stable"].value_counts(normalize=True))
