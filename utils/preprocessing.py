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

def data_split(df, seed):
    np.random.seed(seed)

    # Unique tasks
    unique_targets = df['target_id'].unique()

    # Shuffle the unique tasks
    np.random.shuffle(unique_targets)

    # Define the proportions for each set
    train_prop, val_prop, test_prop = 0.6, 0.2, 0.2

    # Calculate the number of tasks for each set
    n_tasks = len(unique_targets)
    n_train = int(train_prop * n_tasks)
    n_val = int(val_prop * n_tasks)

    # Split the tasks into train, validation, and test sets
    train_targets = unique_targets[:n_train]
    val_targets = unique_targets[n_train:n_train + n_val]
    test_targets = unique_targets[n_train + n_val:]

    # Filter the DataFrame based on the selected tasks for each set
    train_triplet = df[(df['target_id'].isin(train_targets))]
    val_triplet = df[(df['target_id'].isin(val_targets))]
    test_triplet = df[(df['target_id'].isin(test_targets))]

    return train_triplet, val_triplet, test_triplet

def preprocessing(df, triplet_df,seed):

    """
    To do:
    - move datasplit in here -> done
    - delete rows of fps and descrs where mols were invalid -> done
    - delete rows of triplets were mol-ids were invalid
    - initialize fp-array for allocation during loop -> done
    """

    # Step 1: Datasplit
    train_triplet, val_triplet, test_triplet = data_split(triplet_df, seed)

    # Step 2: Check if there are Molecules which have no measurements for train tasks
    exclude = train_triplet.groupby('mol_id')['label'].apply(lambda x: x.isna().all()).loc[lambda x: x].index.tolist()

    # Step 3: Extract SMILES and initialize storage
    smiles_list = df["smiles"].tolist()
    nMols = len(smiles_list)

    filter_indices = list(range(17, 25))  # Indices of unwanted descriptors
    real_descr_indices = [i for i in range(208) if i not in filter_indices]

    rdkit_descriptors = np.zeros(( nMols, len(real_descr_indices) ), dtype=np.float64) # Container for descriptors
    ecfps = np.zeros(( nMols, 2048 ), dtype=np.int8) # Container for ECFP fingerprints, 2048 only possible for MorganFpGen

    # Step 4: Generate molecular objects, fingerprints, and descriptors on-the-fly
    invalidCount = 0
    invalidIdxs = []
    for i,smiles in enumerate(smiles_list):
        # Convert SMILES to RDKit molecular object
        mol = Chem.MolFromSmiles(smiles)

        if mol:
            # Compute ECFP fingerprints
            fp_sparseVec = rdFingerprintGenerator.GetCountFPs(
                [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            #fp = np.zeros((0,), np.int8)  # Create target pointer to fill
            DataStructs.ConvertToNumpyArray(fp_sparseVec, ecfps[i])

            # Compute RDKit descriptors
            descrs = np.array([calc_fn(mol) for _, calc_fn in Descriptors._descList])[real_descr_indices]  # Filter unwanted descriptors
            rdkit_descriptors[i] = descrs

        else:
            invalidCount += 1
            invalidIdxs.append(i)
            continue

    # delete rows if invalid mols occurred
    rdkit_descriptors = np.delete(rdkit_descriptors, invalidIdxs, axis=0)
    ecfps = np.delete(ecfps, invalidIdxs, axis=0)

    # Filter for training molecules, trdkit_descriptors ==  rdkit_descriptors if exclude is empty
    trdkit_descriptors = np.delete(rdkit_descriptors, exclude, axis=0)

    # Step 5: Quantile normalization for descriptors
    rdkit_descriptors_quantils = np.zeros_like(rdkit_descriptors)
    for column in range(rdkit_descriptors.shape[1]):
        raw_values_ecdf = trdkit_descriptors[:, column].reshape(-1)  # Flatten for ECDF
        raw_values = rdkit_descriptors[:, column]  # Raw column values

        ecdf = ECDF(raw_values_ecdf)
        quantils = ecdf(raw_values)
        rdkit_descriptors_quantils[:, column] = quantils

    # Step 7: Stack & Scale
    data = np.hstack([ecfps, rdkit_descriptors_quantils])
    tData = np.delete(data, exclude, axis=0)

    scaler = StandardScaler()
    scaler.fit(tData)
    data = scaler.transform(data)

    # Step 8: Update Df's if invalid mols occurred
    if invalidCount > 0:
        # drop invalid mols
        df = df.drop(index=invalidIdxs).reset_index(drop=True)
        train_triplet = train_triplet[~train_triplet["mol_id"].isin(invalidIdxs)].reset_index(drop=True)
        val_triplet = val_triplet[~val_triplet["mol_id"].isin(invalidIdxs)].reset_index(drop=True)
        test_triplet = test_triplet[~test_triplet["mol_id"].isin(invalidIdxs)].reset_index(drop=True)

        # reset "mol index"
        unique_mol_ids = train_triplet["mol_id"].unique()
        mol_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_mol_ids)}

        train_triplet["mol_id"] = train_triplet["mol_id"].map(mol_id_mapping)
        val_triplet["mol_id"] = val_triplet["mol_id"].map(mol_id_mapping)
        test_triplet["mol_id"] = test_triplet["mol_id"].map(mol_id_mapping)
        
        print(f"{invalidCount} mols have been removed.\n")

    print("Preprocessing done!\n")
    return data, df, train_triplet, val_triplet, test_triplet