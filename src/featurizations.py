import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.EState import AtomTypes
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

def load_smi(smi):
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

def __group_count__(m, debug=False):
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

def feat_sum_over_bonds(df):
    print(type(df))
    mols = df['mol']
    rows = []

    for m in mols:
        bonds = m.GetBonds()
        bond_counts = {bond_type: 0 for bond_type in BOND_SYMBOL}

        for bond in bonds:
            bond_type = bond.GetBondType()
            bond_counts[bond_type] += 1

        counts_vec = list(bond_counts.values())
        rows.append(counts_vec)

    X_SOB = pd.DataFrame(rows, columns=[BOND_SYMBOL.values()], index=mols.index)
    return X_SOB

def feat_oxygen_balance_100(df):
    mols = df['mol']
    rows = []

    for m in mols:
        n_atoms = m.GetNumAtoms(onlyExplicit=False)
        n_heavy = m.GetNumAtoms()
        n_H = n_atoms-n_heavy
        atomic_counts = atom_counts(m)

        OB = (100 / n_atoms) * (atomic_counts['O']  - 2*atomic_counts['C'] - atomic_counts['H']/2)
        rows.append([OB])
    
    X_OB = pd.DataFrame(rows, columns=['ob_100'], index=mols.index)
    return X_OB

def group_count_reader(m):
    group_counts = __group_count__(m)
    
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

    return n_NNO2, n_CNO2, n_ONO2, n_ONO, n_CNO, n_CNN, n_NNN, n_CNH2, n_CNOC, n_CNF, n_CF, n_NO, n_COH, n_NOC, n_CO

def feat_elton_descriptor_set(df):
    mols = df['mol']
    rows = []

    df_ob = feat_oxygen_balance_100(df)

    for m in mols:
        # Oxygen balance
        
        # Atom counts
        atomic_counts = atom_counts(m)
        n_C = atomic_counts['C']
        n_N = atomic_counts['N']
        n_O = atomic_counts['O']
        n_F = atomic_counts['F']
        n_H = atomic_counts['H']

        # Ratio
        if n_C == 0:
            nNnC_ratio = n_N/0.00001
        else:
            nNnC_ratio = n_N/n_C

        # Functional group counts
        n_NNO2, n_CNO2, n_ONO2, n_ONO, n_CNO, n_CNN, n_NNN, n_CNH2, n_CNOC, n_CNF, n_CF, n_NO, n_COH, n_NOC, n_CO = group_count_reader(m)
    
        x_CDS = [nNnC_ratio, n_NNO2, n_CNO2, n_ONO2, n_ONO, n_CNO, n_CNN, n_NNN, n_CNH2, n_CNOC, n_CNF, n_CF, n_C, n_N, n_NO, n_COH, n_NOC, n_CO, n_H, n_F]
        rows.append(x_CDS)

    X_CDS = pd.DataFrame(rows, columns=['nNnC_ratio', 'n_NNO2', 'n_CNO2', 'n_ONO2', 'n_ONO', 'n_CNO', 'n_CNN', 'n_NNN', 'n_CNH2', 'n_CNOC', 'n_CNF', 'n_CF', 'n_C', 'n_N', 'n_NO', 'n_COH', 'n_NOC', 'n_CO', 'n_H', 'n_F'], index=mols.index)
    X_CDS = pd.concat([df_ob, X_CDS], axis=1)
    return X_CDS

def feat_estate_counts(df):
    mols = df['mol']
    rows = []

    AtomTypes.TypeAtoms(Chem.MolFromSmiles("C")) 
    estate_names = [atom_name for atom_name, atom_pattern in rdkit.Chem.EState.AtomTypes.esPatterns]
    keep_indices = [i for i, name in enumerate(estate_names) if name.endswith((CNOF))]

    for m in mols:
        m = Chem.RemoveHs(m) 
        counts, sums = FingerprintMol(m)

        cnof_counts = counts[keep_indices]
        rows.append(cnof_counts)

    X_ES = pd.DataFrame(rows, columns=[estate_names[i] for i in keep_indices], index=mols.index)
    return X_ES

def load_data_with_mols(datafile_path):
    df = pd.read_excel(datafile_path).drop(index = [109])
    df = df.sort_values(by='num').reset_index(drop=True)

    df['mol'] = df['SMILES'].apply(load_smi)
    #mols = [load_smi(s) for s in df["SMILES"]]

    return df

def construct_dataset(data_path, filename, featurization_scheme={}, save=True):
    featurization_string = ""
    datafile_path = os.path.join(data_path, filename)
    df = load_data_with_mols(data_path)

    featurized_data = pd.DataFrame()

    print(feat_oxygen_balance_100(df))
    print(feat_sum_over_bonds(df))
    print(feat_elton_descriptor_set(df))
    print(feat_estate_counts(df))

    if featurization_scheme.get("CDS", True):
        featurization_string=featurization_string+"CDS"
        cds_data = feat_elton_descriptor_set(df)
        featurized_data = pd.concat([featurized_data, cds_data], axis=1)
    if featurization_scheme.get("ES", True):
        es_data = feat_estate_counts(df)
        featurized_data = pd.concat([featurized_data, es_data], axis=1)
    if featurization_scheme.get("SoB", True):
        sob_data = feat_sum_over_bonds(df)
        featurized_data = pd.concat([featurized_data, sob_data], axis=1)

    if save:
        featurized_data.to_csv(os.path.join(data_path, f"repfeat-{filename}_{featurization_string}.csv"))

    return featurized_data

def __main__():
    data_path = os.path.abspath("/scratch/leebrian/elton/Machine-Learning-Energetic-Molecules-Notebooks/datasets/")
    feat = construct_dataset(data_path, filename="Huang_Massa_data_with_all_SMILES.xlsx", save=True)
    print(feat)


if __name__ == "__main__":
    __main__()