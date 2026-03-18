# Map4 fingerprint folding to fix suspicous collision count.
from yaml import load, Loader
from data_preprocessing import *
import pandas as pd
from rdkit import Chem
import numpy as np
import json

from map4 import MAP4Calculator

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
        data_table = pd.read_csv(data_path, sep = ',', header = 0)
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
    config_path:str = "config/lipo.yaml"
    with open(config_path,'r') as stream:
        config:dict = load(stream, Loader)
    print(config)
    dataset = load_dataset(config)
    # lc for fast parrl.
    smiles = [x.id for x in dataset]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    # JERRY RIG: 
    sizes = [128,256,1024,2048,4096]
    # no need for defaultdic
    map_folded = {}
    for bitlen in sizes:
        print(f"Generating folded MAP4 fingerpints for bitlen {bitlen}")
        fpgen_map = MAP4Calculator(dimensions = bitlen, is_folded = True) #chiral disabled by default
        # somehow this feels disgusting? aside from that, try to generate this genius piece chatgpt, flesh wins again
        map_folded[f"{bitlen}"] = dict(zip(smiles, [x.tolist() for x in fpgen_map.calculate_many(mols)]))
    # no indent, the human readable part.... 
    with open("map4_folded_lipo_native.json", 'w') as file:
        json.dump(map_folded, file,)

if __name__ == "__main__":
    main()
