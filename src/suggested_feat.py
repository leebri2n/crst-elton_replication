
"""
    def sum_over_bonds(m):
    m = Chem.AddHs(m)                                 # make hydrogens real atoms so C-H etc. get counted
    counts = Counter()
    for b in m.GetBonds():
        a1, a2 = sorted([b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()])
        counts[a1 + BOND_SYMBOL[b.GetBondType()] + a2] += 1   # e.g. "C-H", "N=O", "C:N"
    return counts

def oxygen_balance_100(m):
    m = Chem.AddHs(m)
    n = Counter(a.GetSymbol() for a in m.GetAtoms())
    return 100 * (n["O"] - 2*n["C"] - n["H"]/2) / m.GetNumAtoms()

def estate_counts(m):
    counts, sums = FingerprintMol(m)                  # 79 slots each
    return counts[6:38]    


    from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from collections import Counter
import pandas as pd
"""
"""
def load(smi):
    Parse SMILES and convert N(=O)=O nitro groups to [N+](=O)[O-].
    m = Chem.MolFromSmiles(smi, sanitize=False)   # sanitize=False so 5-bond N doesn't error out
    m = Chem.RWMol(m)
    for a in m.GetAtoms():
        if a.GetSymbol() == "N" and a.GetFormalCharge() == 0:
            dbl_O = [b for b in a.GetBonds()
                     if b.GetBondType() == Chem.BondType.DOUBLE
                     and b.GetOtherAtom(a).GetSymbol() == "O"]
            if len(dbl_O) == 2:                       # it's a nitro written the wrong way
                a.SetFormalCharge(1)
                dbl_O[1].SetBondType(Chem.BondType.SINGLE)
                dbl_O[1].GetOtherAtom(a).SetFormalCharge(-1)
    Chem.SanitizeMol(m)                               # now run RDKit's checks
    return m.GetMol()

BOND_SYMBOL = {Chem.BondType.SINGLE: "-", Chem.BondType.DOUBLE: "=",
               Chem.BondType.TRIPLE: "#", Chem.BondType.AROMATIC: ":"}

def sum_over_bonds(m):
    m = Chem.AddHs(m)                                 # make hydrogens real atoms so C-H etc. get counted
    counts = Counter()
    for b in m.GetBonds():
        a1, a2 = sorted([b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()])
        counts[a1 + BOND_SYMBOL[b.GetBondType()] + a2] += 1   # e.g. "C-H", "N=O", "C:N"
    return counts

def oxygen_balance_100(m):
    m = Chem.AddHs(m)
    n = Counter(a.GetSymbol() for a in m.GetAtoms())
    return 100 * (n["O"] - 2*n["C"] - n["H"]/2) / m.GetNumAtoms()

def estate_counts(m):
    counts, sums = FingerprintMol(m)                  # 79 slots each
    return counts[6:38]                               # keep the 32 C/N/O/F slots (slots 7–38)

# Build the feature table
df = pd.read_excel("Huang_Massa_data_with_all_SMILES.xlsx")
mols = [load(s) for s in df["SMILES"]]
X_sob = pd.DataFrame([sum_over_bonds(m) for m in mols]).fillna(0)   # columns = bond types
print(X_sob.shape)          # expect (110, ~20)
print(X_sob.columns.tolist()) 
"""