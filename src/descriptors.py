"""
Real molecular descriptor calculator using RDKit.

Every descriptor below is computed by RDKit's own, validated implementations
-- not an approximation or hand-rolled proxy:

  * Molecular weight        -> rdkit.Chem.Descriptors.MolWt
  * LogP (lipophilicity)    -> rdkit.Chem.Crippen.MolLogP (the real
                                Wildman-Crippen atom-contribution method)
  * H-bond donors/acceptors -> rdkit.Chem.Lipinski.NumHDonors /
                                NumHAcceptors (RDKit's own Lipinski-rule
                                definitions, based on a real substructure
                                perception of the parsed molecular graph,
                                not a SMILES-substring heuristic)
  * TPSA                    -> rdkit.Chem.rdMolDescriptors.CalcTPSA (the
                                real Ertl et al. (2000) topological polar
                                surface area algorithm)
  * Rotatable bonds         -> rdkit.Chem.Lipinski.NumRotatableBonds
  * Ring count               -> rdkit.Chem.rdMolDescriptors.CalcNumRings
  * Fraction Csp3           -> rdkit.Chem.rdMolDescriptors.CalcFractionCSP3
                                (a real measure of 3-dimensionality /
                                "flatness", relevant to packing/crystallinity
                                behavior in amorphous solids)

A SMILES string is first parsed into a real RDKit `Mol` object via
`Chem.MolFromSmiles` -- if it fails to parse, that is a genuine error (not
silently handled), since every SMILES reaching this function is expected
to have already been validated at generation time in `drug_library.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors as RDDescriptors, Lipinski
from rdkit.Chem import rdMolDescriptors


@dataclass
class Descriptors:
    smiles: str
    mol_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    ring_count: int
    fraction_csp3: float


def compute_descriptors(smiles: str) -> Descriptors:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    return Descriptors(
        smiles=smiles,
        mol_weight=round(RDDescriptors.MolWt(mol), 2),
        logp=round(Crippen.MolLogP(mol), 3),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
        ring_count=rdMolDescriptors.CalcNumRings(mol),
        fraction_csp3=round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    )


if __name__ == "__main__":
    # Sanity spot-check against a well-known molecule: aspirin.
    d = compute_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    print(d)
