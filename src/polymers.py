"""
Polymer carrier database for amorphous solid dispersion (ASD) screening.

Each entry is a real, commercially used pharmaceutical polymer carrier.
Hildebrand solubility parameter (delta, MPa^0.5) and glass transition
temperature (Tg, degC) values below are approximate figures commonly cited
in the ASD / hot-melt-extrusion literature (e.g. Forster et al. 2001;
Greenhalgh et al. 1999; Six et al. 2004; BASF/Evonik/DuPont product
literature for Soluplus, Eudragit, Klucel). They are representative
"typical" values used for screening-stage decision making, not
single-batch certificate-of-analysis numbers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Polymer:
    name: str
    delta: float      # Hildebrand solubility parameter, MPa^0.5
    tg: float          # glass transition temperature, degC
    hbd_capacity: int  # relative H-bond donor sites per repeat unit (0-3 scale)
    hba_capacity: int  # relative H-bond acceptor sites per repeat unit (0-3 scale)
    molar_volume: float  # approx repeat-unit molar volume, cm^3/mol


POLYMERS = [
    Polymer("PVP K30", delta=23.0, tg=164, hbd_capacity=0, hba_capacity=3, molar_volume=70.0),
    Polymer("PVP/VA 64", delta=21.5, tg=101, hbd_capacity=0, hba_capacity=3, molar_volume=75.0),
    Polymer("HPMC E5", delta=22.0, tg=170, hbd_capacity=2, hba_capacity=3, molar_volume=145.0),
    Polymer("HPMCAS", delta=20.5, tg=120, hbd_capacity=2, hba_capacity=3, molar_volume=160.0),
    Polymer("Soluplus", delta=19.4, tg=70, hbd_capacity=0, hba_capacity=3, molar_volume=210.0),
    Polymer("Eudragit E100", delta=18.5, tg=50, hbd_capacity=0, hba_capacity=2, molar_volume=130.0),
    Polymer("Eudragit L100", delta=20.0, tg=190, hbd_capacity=1, hba_capacity=2, molar_volume=100.0),
    Polymer("Eudragit L100-55", delta=19.8, tg=110, hbd_capacity=1, hba_capacity=2, molar_volume=105.0),
    Polymer("Klucel EF (HPC)", delta=21.0, tg=130, hbd_capacity=2, hba_capacity=2, molar_volume=150.0),
    Polymer("PEG 6000", delta=20.3, tg=-40, hbd_capacity=1, hba_capacity=1, molar_volume=40.0),
    Polymer("Poloxamer 188", delta=18.0, tg=-50, hbd_capacity=1, hba_capacity=1, molar_volume=50.0),
    Polymer("CAP (cellulose acetate phthalate)", delta=22.5, tg=170, hbd_capacity=1, hba_capacity=3, molar_volume=170.0),
]


def get_polymer(name: str) -> Polymer:
    for p in POLYMERS:
        if p.name == name:
            return p
    raise KeyError(name)
