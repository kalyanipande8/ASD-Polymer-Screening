"""
Drug solubility-parameter estimation and Flory-Huggins miscibility scoring.

Method
------
1. Estimate the drug's Hildebrand solubility parameter (delta) with a
   simplified Fedors-style group-contribution proxy:

       delta = sqrt( E_coh / V_molar )

   where E_coh (cohesive energy, J/mol) is approximated from polar
   descriptor counts (HBD, HBA, TPSA) and V_molar (molar volume, cm^3/mol)
   is approximated from molecular weight and an assumed typical small-
   molecule drug density (1.2 g/cm^3). This is NOT the full Fedors (1974)
   group-contribution table (which requires per-functional-group Ecoh/V
   lookups from a structural parse) — it is a descriptor-driven proxy that
   follows the same "cohesive energy density" logic and is disclosed as
   such. It produces deltas in the realistic 15-30 MPa^0.5 range reported
   for small-molecule drugs in the literature (e.g. ibuprofen ~19,
   indomethacin ~24).

2. Compute a Flory-Huggins-style interaction parameter proxy:

       chi = V_seg * (delta_drug - delta_polymer)^2 / (R * T)

   This is the standard solubility-parameter form of chi used throughout
   the ASD literature (Flory 1953; Hildebrand & Scott 1950; applied to
   drug-polymer systems by Marsac, Li & Taylor, Pharm Res 2009).

3. Classify miscibility using the documented Greenhalgh et al. (1999)
   rule of thumb for solubility-parameter differences:
       |delta_drug - delta_polymer| <  7  MPa^0.5  -> miscible
       7 <= |delta_drug - delta_polymer| <= 10      -> borderline
       |delta_drug - delta_polymer| > 10            -> immiscible
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .descriptors import Descriptors
from .polymers import Polymer

R_GAS = 8.314       # J / (mol K)
T_KELVIN = 298.15    # standard screening temperature, K
DRUG_DENSITY = 1.2   # g/cm^3, typical small-molecule organic solid


def estimate_drug_delta(d: Descriptors) -> float:
    v_molar = d.mol_weight / DRUG_DENSITY  # cm^3/mol
    e_coh = (
        2200.0
        + 420.0 * d.hbd
        + 260.0 * d.hba
        + 14.0 * d.tpsa
        - 90.0 * max(d.logp, 0.0)
    )
    e_coh = max(e_coh, 400.0)
    delta = math.sqrt(e_coh / v_molar) * 5.0  # scale factor -> MPa^0.5 range
    return round(delta, 2)


def flory_huggins_chi(delta_drug: float, polymer: Polymer) -> float:
    delta_diff_sq = (delta_drug - polymer.delta) ** 2
    chi = polymer.molar_volume * delta_diff_sq / (R_GAS * T_KELVIN)
    return round(chi, 4)


def greenhalgh_class(delta_drug: float, polymer: Polymer) -> str:
    diff = abs(delta_drug - polymer.delta)
    if diff < 7.0:
        return "miscible"
    if diff <= 10.0:
        return "borderline"
    return "immiscible"


@dataclass
class MiscibilityResult:
    delta_drug: float
    delta_diff: float
    chi: float
    classification: str
    hbond_complementarity: float


def hbond_complementarity(d: Descriptors, polymer: Polymer) -> float:
    """Documented concept (Marsac et al., Pharm Res 2006/2009): drug-polymer
    H-bonding (drug donor <-> polymer acceptor, or vice versa) can stabilize
    an ASD beyond what the bulk solubility-parameter match alone predicts.
    This computes a simple complementarity score: normalized overlap of
    donor/acceptor capacity between the two components."""
    drug_donor_poly_acceptor = min(d.hbd, polymer.hba_capacity)
    drug_acceptor_poly_donor = min(d.hba, polymer.hbd_capacity)
    score = drug_donor_poly_acceptor + 0.5 * drug_acceptor_poly_donor
    return round(score, 3)


def assess(d: Descriptors, polymer: Polymer) -> MiscibilityResult:
    delta_drug = estimate_drug_delta(d)
    diff = abs(delta_drug - polymer.delta)
    chi = flory_huggins_chi(delta_drug, polymer)
    cls = greenhalgh_class(delta_drug, polymer)
    hb = hbond_complementarity(d, polymer)
    return MiscibilityResult(delta_drug, diff, chi, cls, hb)
