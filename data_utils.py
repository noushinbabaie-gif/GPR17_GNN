import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ------------------------------
# 1. Data Featurization Functions
# ------------------------------
def featurize_atom(atom):
    """Extract feature vector for one atom."""
    features = []
    # (1) Atomic number (one-hot or integer)
    features.append(atom.GetAtomicNum())
    # (2) Degree
    features.append(atom.GetDegree())
    # (3) Formal charge
    features.append(atom.GetFormalCharge())
    
    features.append(atom.GetTotalValence())
    
    features.append(atom.GetNumRadicalElectrons())
    # (4) Hybridization
    hybridization = atom.GetHybridization()
    # Create a simple one-hot encoding for some common hybridizations:
    # [SP, SP2, SP3, OTHER]
    hybrid_map = {
        Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0],
        Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0],
        Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0]
    }
    if hybridization in hybrid_map:
        features.extend(hybrid_map[hybridization])
    else:
        features.extend([0, 0, 0, 1])
    # (5) Is aromatic
    features.append(atom.GetIsAromatic())

    return features

def featurize_bond(bond):
    """
    Returns a list of bond features, including bond type, conjugation, ring membership.
    """
    bt = bond.GetBondType()
    bond_type = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0
    ]
    return bond_type + [
        1 if bond.GetIsConjugated() else 0,
        1 if bond.IsInRing() else 0
    ]

def get_global_features(mol):
    """Extract global features for the molecule such as MW, LogP, etc."""
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    logp = Descriptors.MolLogP(mol)
    num_val = Descriptors.NumValenceElectrons(mol)
    lip_HBA = Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)
    lip_HBD = Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)
    rot_bond = Descriptors.NumRotatableBonds(mol)
    volume = AllChem.ComputeMolVolume(mol)
    return [mw, tpsa, num_rings, logp, num_val, lip_HBA, lip_HBD, rot_bond, volume]

# ------------------------------
# 2. Custom PyTorch Geometric Dataset
# ------------------------------
class SMILESDataset(Dataset):
    def __init__(self, dataframe, transform=None, pre_transform=None):
        """
        dataframe: pandas DataFrame containing 'SMILES' and 'pAffinity' columns.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.dataframe)

    def get(self, idx):
        smiles = self.dataframe.loc[idx, 'SMILES']
        y_value = self.dataframe.loc[idx, 'pAffinity']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # In case of invalid SMILES, return an empty Data object
            # or handle accordingly
            return Data()

        # Add Hs for a better atom environment
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        # Atom features
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(featurize_atom(atom))
        x = torch.tensor(atom_features_list, dtype=torch.float)

        # Edge (bond) features
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            bond_feature = featurize_bond(bond)

            edge_indices.append([start, end])
            edge_indices.append([end, start])
            edge_features.append(bond_feature)
            edge_features.append(bond_feature)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Molecules with no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)

        # Global features
        global_feat = get_global_features(mol)
        u = torch.tensor(global_feat, dtype=torch.float).view(1, -1)  # shape [1, num_global_features]

        y = torch.tensor([y_value], dtype=torch.float)

        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y)

        # Attach global features
        data.u = u

        return data

