from yaml import ValueToken, load, Loader

import data_preprocessing
import pandas as pd


from data_preprocessing import *

# could be moved to data part.
def parse_identifiertype(idtype:str) -> IdentifierType | None:
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


    bit_lengths = config.get("bit_lengths",[1024])
    encoding_types = config.get("encoding_types",["ecfp"])
    n_splits = config.get("runs", 10)

def main():
    config_path:str = "config/config.yaml"
    with open(config_path,'r') as stream:
        config:dict = load(stream, Loader)
    print(config)
    dataset = load_dataset(config)
    # no compound cleaning?
    dataset.build_scaffolds()
    dataset.print_scaffoldstats()

    dataset.build_scaffolds()
if __name__ == "__main__":
    main()
