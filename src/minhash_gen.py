#of course things can never be easy, I like it this way
from map4 import MAP4Calculator, MHFPEncoder
from yaml import load, Loader
from data_preprocessing import *
import pandas as pd
from rdkit import Chem
import numpy as np
import json
# spome duplication to avoid circular.
def parse_identifiertype(idtype:str):
    try:
        return IdentifierType(idtype)
    except ValueError:
        return None

def load_dataset(config):
    """use config to prep dataset objects"""
    data_path = config.get("data_source", None) # could move to get if not local.
    
    target_col = config.get("target_col", None)
    input_col = config.get("input_col", None)
    input_format = config.get("input_format", None)
    
    if config.get("data_islocal", False) == True:
        data_table = pd.read_csv(data_path, sep = '\t', header = 0)
    else: # here fetch.
        data_table = pd.DataFrame([0])
    
    # sanity checks
    if target_col in data_table:
        target_list = data_table[target_col].values.tolist()
    else:
        raise KeyError(f"provided target col {target_col} not found in data table")
    if input_col in data_table:
        entry_list = data_table[input_col].values.tolist()
    else:
        raise KeyError(f"provided compound col {input_col} not found in data table")
    identifier_type = parse_identifiertype(input_format)
    entries = zip(entry_list,target_list)
    dataset = CompoundDataset([CompoundEntry(id, identifier_type, target) for (id,target) in entries])
    return dataset

def main():
    # very similar to the stream of data in the prediction case. but needed seperate env due to conflicts.
    # Ill keep my rant to myself, bandaid fix GO.
    config_path:str = "config/config.yaml"
    with open(config_path,'r') as stream:
        config:dict = load(stream, Loader)
    print(config)
    dataset = load_dataset(config)
    # simple, we create a injectable JSON dict of the unfolded hashes:
    #since we only get shingles we do not need to set vars.
    MAP4_folded = MAP4Calculator()
    Encoder = MHFPEncoder()
    # touching the encoder class. we can enforce larger ints to not overflow...
    Encoder.permutations_a = Encoder.permutations_a.astype(np.uint64)
    Encoder.permutations_b = Encoder.permutations_b.astype(np.uint64)
    Encoder.max_hash = np.uint64(Encoder.max_hash)
    # roundabout way but fast to modify.
    
    print("Generating Map4 unfolded - Minhashed fingerprints")
    results = {}
    for entry in dataset:
        mol = Chem.MolFromSmiles(entry.id)
        envs = MAP4_folded._get_atom_envs(mol)
        pairs = MAP4_folded._all_pairs(mol, envs) #set of pairs
        # return hash_values.reshape((1, self.n_permutations))[0]
        hashies = Encoder.from_molecular_shingling(pairs)
        results[entry.id] = hashies.tolist()
    with open("map4_unfolded.json", 'w') as file:
        json.dump(results, file)

    print("Generating SECFP6 unfolded - Minhashed fingerprints")
    results = {}
    # recreate secfp from mol but unfolded
    for entry in dataset:
        mol = Chem.MolFromSmiles(entry.id)
        shinglings = Encoder.shingling_from_mol(mol)
        hashies = Encoder.from_molecular_shingling(shinglings)
        results[entry.id] = hashies.tolist()
    for k,v in results.items():
        print(k, v)
    with open("secfp6_unfolded.json", 'w') as file:
        json.dump(results,file)

if __name__ == "__main__":
    main()

