import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from collections import Counter
import pandas as pd
import collections

import os

BOND_SYMBOL = {Chem.BondType.SINGLE: "-", Chem.BondType.DOUBLE: "=",
               Chem.BondType.TRIPLE: "#", Chem.BondType.AROMATIC: ":"}
CNOF = ("C", "N", "O", "F", "CH", "CH2", "CH3", "NH", "NH2", "NH3", "OH")
SIGNATURES = {
    # Nitrogen-centered groups
    "CNO2":    ("N", {"C": 1, "O": 2}, "carbon nitro group"),
    "NNO2":    ("N", {"N": 1, "O": 2}, "nitrogen nitro group"),
    "ONO":     ("N", {"O": 2},         "nitrite group"),
    "ONO2":    ("N", {"O": 3},         "oxygen nitro group (nitrate)"),
    "CNN":     ("N", {"C": 1, "N": 1}, "azo group"),
    "NNN":     ("N", {"N": 2},         "azide group"),
    "CNO":     ("N", {"C": 1, "O": 1}, "fulminate group"),
    "CNH2":    ("N", {"C": 1, "H": 2}, "amine group"),
    "CN(O)C":  ("N", {"C": 2, "O": 1}, "N-oxide nitrogen"),
    "CNF":     ("N", {"C": 1, "N": 2}, "nitrogen-fluorine group (as coded; contains no F)"),

    # Fluorine-centered
    "CF":      ("F", {"C": 1},         "carbon-fluorine group"),

    # Oxygen-centered (n_O1 to n_O4 in their code)
    "NO":      ("O", {"N": 1},         "nitrate/nitrite oxygen"),
    "NOC":     ("O", {"N": 1, "C": 1}, "N-oxide oxygen"),
    "CO":      ("O", {"C": 1},         "ketone/carboxyl oxygen"),
    "COH":     ("O", {"C": 1, "H": 1}, "hydroxyl oxygen")
}

GROUPS_LAYERED = {
    "N": {
        (("C", 1), ("O", 2),):  "CNO2",
        (("N", 1), ("O", 2),):  "NNO2",
        (("O", 2),):           "ONO",
        (("O", 3),):           "ONO2",
        (("C", 1), ("N", 1),):  "CNN",
        (("N", 2),):           "NNN",
        (("C", 1), ("O", 1),):  "CNO",
        (("C", 1), ("H", 2),):  "CNH2",
        (("C", 2), ("O", 1),):  "CN(O)C",
        (("C", 1), ("N", 2),):  "CNF"
    },
    "O": {
        (("N", 1),):           "NO",
        (("C", 1), ("N", 1),):  "NOC",
        (("C", 1),):           "CO",
        (("C", 1), ("H", 1),):  "COH"
    },
    "F": {
        (("C", 1),):           "CF"
    }
}

def load(smi):
    """Parse SMILES and convert N(=O)=O nitro groups to [N+](=O)[O-]."""
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
    try:
        Chem.SanitizeMol(m)                               # now run RDKit's checks
    except Exception as e:
        print(f"Molecule SMILE {smi} not correct.")

    m = Chem.AddHs(m.GetMol())
    return m


def atom_counts(m):
    counts = collections.defaultdict(int)
    for atom in m.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] += 1
    return counts

def sum_over_bonds(m):
    bonds = m.GetBonds()

    bond_counts = {bond_type: 0 for bond_type in BOND_SYMBOL}

    for bond in bonds:
        bond_type = bond.GetBondType()
        bond_counts[bond_type] += 1

    counts_vec = list(bond_counts.values())
    return counts_vec

def oxygen_balance_100(m):
    n_atoms = m.GetNumAtoms(onlyExplicit=False)
    n_heavy = m.GetNumAtoms()
    n_H = n_atoms-n_heavy

    atomic_counts = atom_counts(m)

    OB = (100 / n_atoms) * (atomic_counts['O']  - 2*atomic_counts['C'] - atomic_counts['H']/2)
    return [OB]

def group_count(m, debug=False):
    group_counts = {group_name: 0 for group_name in SIGNATURES}

    for i, atom in enumerate(m.GetAtoms()):
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in GROUPS_LAYERED:
            continue

        neighbor_counts = {}
        for atom_neighbor in atom.GetNeighbors():
            neighbor_symbol = atom_neighbor.GetSymbol()
            neighbor_counts[neighbor_symbol] = neighbor_counts.get(neighbor_symbol, 0) + 1
            if debug:
                print(f"Central atom: {atom_symbol}, Neighbor: {neighbor_symbol}")
        
        neighbor_signature = tuple(sorted(neighbor_counts.items()))  # fix        
        
        if neighbor_signature in GROUPS_LAYERED[atom_symbol].keys():
            group_counts[GROUPS_LAYERED[atom_symbol][neighbor_signature]] += 1
            if debug:
                print("Found group:", GROUPS_LAYERED[atom_symbol][neighbor_signature])
    
    return group_counts

def elton_descriptor_set(m, debug=True):
    ob_100 = oxygen_balance_100(m)[0]

    atomic_counts = atom_counts(m)
    group_counts = group_count(m)

    n_C = atomic_counts['C']
    n_N = atomic_counts['N']
    n_O = atomic_counts['O']
    n_F = atomic_counts['F']
    n_H = atomic_counts['H']

    if n_C == 0:
        nNnC_ratio = n_N/0.00001
    else:
        nNnC_ratio = n_N/n_C

    n_NNO2 = group_counts['NNO2']
    n_CNO2 = group_counts['CNO2']
    n_ONO2 = group_counts['ONO2']
    n_ONO = group_counts['ONO']
    n_CNO = group_counts['CNO']
    n_CNN = group_counts['CNN']
    n_NNN = group_counts['NNN']
    n_CNH2 = group_counts['CNH2']
    n_CNOC = group_counts['CN(O)C']
    n_CNF = group_counts['CNF']
    n_CF = group_counts['CF']

    n_NO = group_counts['NO']
    n_COH = group_counts['COH']
    n_NOC = group_counts['NOC']
    n_CO = group_counts['CO']

    if debug:
        print(group_counts)

    x_CDS = [ob_100, nNnC_ratio, n_NNO2, n_CNO2, n_ONO2, n_ONO, n_CNO, n_CNN, n_NNN, n_CNH2, n_CNOC, n_CNF, n_CF, n_C, n_N, n_NO, n_COH, n_NOC, n_CO, n_H, n_F]
    
    return x_CDS

def estate_counts(m):
    # For sanity:
    #estate_names = [atom_name for atom_name in rdkit.Chem.EState.Fingerprinter.GetAtomEStateNames()]
    #rdkit.Chem.EState.AtomTypes.BuildPatts()
    
    m = Chem.RemoveHs(m) 
    counts, sums = FingerprintMol(m)

    estate_names = [atom_name for atom_name, atom_pattern in rdkit.Chem.EState.AtomTypes.esPatterns]
    keep_indices = [i for i, name in enumerate(estate_names) if name.endswith((CNOF))]

    cnof_counts = counts[keep_indices]
    
    return cnof_counts

def load_data_as_mols(data_path):
    df = pd.read_excel(os.path.join(data_path, "Huang_Massa_data_with_all_SMILES.xlsx")).drop(index = [109])
    df = df.sort_values(by='num').reset_index(drop=True)
    mols = [load(s) for s in df["SMILES"]]

    return tuple(mols), df

def construct_dataset(data_path, featurization_scheme={}):
    featurized_data = pd.DataFrame()
    mols, df = load_data_as_mols(data_path)

    print(Chem.MolToSmiles(mols[108]))

    #featurized_data = pd.DataFrame([elton_descriptor_set(m) for m in mols if m is not None])
    #featurized_data.columns = ["OB_100", "nNnC_ratio", "n_NNO2", "n_CNO2", "n_ONO2", "n_ONO", "n_CNO", "n_CNN", "n_NNN", "n_CNH2", "n_CNOC", "n_CNF", "n_CF", "n_C", "n_N", "n_NO", "n_COH", "n_NOC", "n_CO", "n_H", "n_F"]

    if featurization_scheme.get("CDS", True):
        cds_df = elton_descriptor_set(m)

    return featurized_data, df

def __main__():
    data_path = os.path.abspath("/scratch/leebrian/elton/Machine-Learning-Energetic-Molecules-Notebooks/datasets/")
    construct_dataset(data_path)

    #mols: list of rdkit.Chem.rdchem.Mol objects

    for m in mols:
        print("MOLECULE: ", Chem.MolToSmiles(m))
        ob = oxygen_balance_100(m)
        bs = sum_over_bonds(m)
        estate_fp = estate_counts(m)
        cds = elton_descriptor_set(m, debug=True)
        
        print("O balance: ", ob)
        print("Sum of bonds: ", bs)
        print("Estate counts: ", estate_fp, len(estate_fp))
        print("Custom descriptor set: ", cds, len(cds))

if __name__ == "__main__":
    #X_sob = pd.DataFrame([sum_over_bonds(m) for m in mols if m is not None])
    #X_sob = X_sob.fillna(0).astype(int)   # columns = bond types
    __main__()