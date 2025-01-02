import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler 

# ignore warnings regarding "not removing hydrogen atom without neighbors" 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


def preprocessing(df, train_triplet = None):

    # Step1: Check if there are Molecules which have no meassurements for train tasks
    exclusionNeeded = False
    if train_triplet is not None:
        exclude = train_triplet.groupby('mol_id')['label'].apply(lambda x: x.isna().all()).loc[lambda x: x].index.tolist()

        if len(exclude)!= 0:
           exclusionNeeded = True
    
    # Step 2: Extract SMILES and initialize storage
    smiles_list = df["smiles"].tolist()
    ecfps = []  # Container for ECFP fingerprints
    mols = []   # Container for molecular objects

    # Step 3: Generate molecular objects and fingerprints
    for smiles in smiles_list:
        # Convert SMILES to RDKit molecular object
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

        # Compute ECFP fingerprints
        fp_sparseVec = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fp = np.zeros((0,), np.int8)  # Create target pointer to fill
        DataStructs.ConvertToNumpyArray(fp_sparseVec, fp)
        ecfps.append(fp)
    
    # Step 4: Compute RDKit descriptors for each molecule
    # Filter out certain descriptors by index 
    filter = list(range(17, 25))
    real_descr = [i for i in range(208) if i not in filter]
    
    rdkit_descriptors = []  # Container for descriptor vectors
    for i, mol in enumerate(mols):
        descrs = [calc_fn(mol) for _, calc_fn in Descriptors._descList]  # All descriptors
        descrs = np.array(descrs)[real_descr]  # Filter unwanted descriptors
        rdkit_descriptors.append(descrs)

    # Convert descriptor list to numpy array
    rdkit_descriptors = np.array(rdkit_descriptors) # convert to numpy

    # Step 5: Quantile normalization for descriptors
    if exclusionNeeded:
        trdkit_descriptors = np.delete(rdkit_descriptors,exclude,axis=0) # Filter descriptors for training molecules only (not neccessary for sider)
    else:
        trdkit_descriptors = rdkit_descriptors
        
    rdkit_descriptors_quantils = np.zeros_like(rdkit_descriptors)
    for column in range(rdkit_descriptors.shape[1]): 
        raw_values_ecdf = trdkit_descriptors[:, column].reshape(-1)  #  Flatten for ECDF 
        raw_values = rdkit_descriptors[:, column]  # Raw column values

        ecdf = ECDF(raw_values_ecdf)
        quantils = ecdf(raw_values)
        rdkit_descriptors_quantils[:, column] = quantils

    # Step 6: Stack & Scale
    data = np.hstack([ecfps, rdkit_descriptors_quantils])
    scaler = StandardScaler()
    data = scaler.fit_transform(data) # -> need to adapt for tox

    print("Preprocessing done...\n")
    return data     