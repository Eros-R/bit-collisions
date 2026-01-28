from yaml import load, Loader
import pandas as pd
# WE LOVE STAR IMPORTS. WE ARE LAAAZYYY
from data_preprocessing import *
from models import *
import json

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

def main():
    config_path:str = "config/config.yaml"
    with open(config_path,'r') as stream:
        config:dict = load(stream, Loader)
    print(config)
    test_fraction = config.get("test_fraction",0.2)
    n_samples = config.get("runs", 10) #bit confusing var...
    model_list = config.get("models", [])
    dataset = load_dataset(config)
    # no compound cleaning?
    dataset.build_scaffolds()
    #dataset.print_scaffoldstats()
    #print(full_dict)

    print(f"Generating {n_samples} Data splits: ")
    split_list = dataset.scaffold_exclusive_sampling(test_fraction=test_fraction,n_samples= n_samples, seed = 1508)
    encoding_types = config.get("encoding_types",["ecfp"])
    bit_lengths = config.get("bit_lengths",[1024])
    resdic = {} # lsp crying
    for encoding_type in encoding_types:
        try:
            reptype = RepresentationType(encoding_type)
        except ValueError:
            print(f"Unrecognized representation type: {encoding_type}")
            continue
        match reptype:
            case RepresentationType.ecfp4:
                print(f"Generating ECFP4 fingerprints")
                dataset.set_representations(RepresentationType.ecfp4)
                for bit_len in bit_lengths:
                    dataset.get_folded_dataset(reptype,bit_len)
                    dataset.get_dataset_self_collisions(reptype,bit_len)
            case RepresentationType.rdkit:
                print("Generating rdkit fingerprints")
                dataset.set_representations(RepresentationType.rdkit)
                for bit_len in bit_lengths:
                    dataset.get_folded_dataset(reptype, bit_len)
                    dataset.get_dataset_self_collisions(reptype,bit_len)
            case RepresentationType.map4:
                print("retrieving pre-calculated map4 fingerprints")
                map4_path = config.get("map4_unfolded_path", None)
                if map4_path is None:
                    raise RuntimeError("Invalid path")
                with open(map4_path, 'r') as file:
                    map4_data = json.load(file)
                dataset._precomputed = map4_data
                dataset.set_representations(RepresentationType.map4)
                for bit_len in bit_lengths:
                    dataset.get_folded_dataset(reptype,bit_len)
                    dataset.get_dataset_self_collisions(reptype,bit_len)
            case RepresentationType.secfp6:
                print("retrieving pre-calculated secfp6 fingerprints")
                secfp6_unfolded_path = config.get("secfp6_unfolded_path", None)
                if secfp6_unfolded_path is None:
                    raise RuntimeError("Invalid path")
                with open(secfp6_unfolded_path, 'r') as file:
                    secfp6_data = json.load(file)
                dataset._precomputed = secfp6_data
                dataset.set_representations(RepresentationType.secfp6)
                for bit_len in bit_lengths:
                    dataset.get_folded_dataset(reptype,bit_len)
                    dataset.get_dataset_self_collisions(reptype,bit_len)
            case _:
                raise KeyError("unknown representation type provided") #type analysis says this will never happen but alas
        for bit_len in bit_lengths:
            print(f"Generating {encoding_type} fingerprints on {bit_len} bits")
            dataset.set_representations(reptype)
            for model_type in model_list:
                try:
                    mtype = ModelType.from_string(model_type)
                except ValueError as e:
                    print(e)
                    continue

                splitstack = {}
                # WARNING: for speed sake, first 2 splits.
                for i,data_split in enumerate(split_list[0:2]):
                    train_ds, test_ds = data_split[1], data_split[0]  # train/test, note the inverted index ;)
                    results = mtype.loader(
                        train=train_ds,
                        test=test_ds,
                        representation=reptype,
                        bits=bit_len
                    )
                    splitstack[i] = results
                reskey = f"{model_type}_{encoding_type}_{str(bit_len)}"
                resdic[reskey] = splitstack

    for split_idx, split_results in resdic.items():
        for model_type, metrics in split_results.items():
            print(f"Split {split_idx}, Split: {model_type}: MAE = {metrics['mae']:.4f}")
    # TODO: gather relevant collision data for individual at folding level...

    return # return call here so I can shelf some analytic junk :) 

if __name__ == "__main__":
    main()
