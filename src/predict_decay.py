from yaml import load, Loader
import pandas as pd
# WE LOVE STAR IMPORTS. WE ARE LAAAZYYY
from data_preprocessing import *
from models import *
import json
import copy

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
    config_path = "config/config.yaml"
    with open(config_path,'r') as stream:
        config = load(stream, Loader)
    test_fraction = config.get("test_fraction",0.2)
    n_samples = config.get("runs", 10)
    model_list = config.get("models", [])
    dataset = load_dataset(config)
    dataset.build_scaffolds()

    split_list = dataset.scaffold_exclusive_sampling(test_fraction=test_fraction, n_samples=n_samples, seed=1508)
    encoding_types = config.get("encoding_types", ["ecfp"])
    bit_lengths = config.get("bit_lengths", [1024])

    resdic = defaultdict(dict)
    all_self_collisions = defaultdict(dict)
    all_dataset_collisions = defaultdict(dict)

    for encoding_type in encoding_types:
        try:
            reptype = RepresentationType(encoding_type)
        except ValueError:
            continue

        dataset._dataset_collisions = {}
        dataset._dataset_self_collisions = {}

        if reptype in (RepresentationType.map4, RepresentationType.secfp6):
            path_key = "map4_unfolded_path" if reptype is RepresentationType.map4 else "secfp6_unfolded_path"
            file_path = config.get(path_key)
            if file_path is None:
                raise RuntimeError("Invalid path")
            with open(file_path, "r") as f:
                dataset._precomputed = json.load(f)
        else:
            dataset._precomputed = None

        for entry in dataset._entries:
            rep = entry._representation
            rep._self_collisions = {}
            rep._folded = {}
            rep._folded_counts = {}
            rep._bitfinfo = None
            if reptype in (RepresentationType.map4, RepresentationType.secfp6):
                rep._unfolded = copy.deepcopy(dataset._precomputed.get(entry.id, []))
            else:
                rep._unfolded = None

        if reptype in (RepresentationType.ecfp4, RepresentationType.rdkit):
            dataset.set_representations(reptype)

        for bit_len in bit_lengths:
            reskey = f"{encoding_type}_{bit_len}"
            dataset.get_folded_dataset(reptype, bit_len)
            self_coll = dataset.get_dataset_self_collisions(reptype, bit_len)
            dataset_coll = dataset.get_dataset_collisions(reptype, bit_len)
            all_self_collisions[reskey] = {cid: bool(collisions) for cid, collisions in self_coll.items()}
            all_dataset_collisions[reskey] = sum(len(v) for v in dataset_coll.values())

        for bit_len in bit_lengths:
            for model_type in model_list:
                try:
                    mtype = ModelType.from_string(model_type)
                except ValueError:
                    continue
                splitstack = {}
                for i, data_split in enumerate(split_list):
                    train_ds, test_ds = data_split[1], data_split[0]
                    results = mtype.loader(train=train_ds, test=test_ds, representation=reptype, bits=bit_len)
                    splitstack[i] = results
                reskey = f"{model_type}_{encoding_type}_{bit_len}"
                resdic[reskey] = splitstack
    mae_rows = [
        {"split": split_idx, "model": model_type, "mae": metrics["mae"]}
        for split_idx, split_results in resdic.items()
        for model_type, metrics in split_results.items()
    ]
    mae_df = pd.DataFrame(mae_rows).pivot(index="split", columns="model", values="mae")
    print(mae_df)

    dataset_coll_rows = [
        {"encoding_bit": reskey, "num_dataset_collisions": val}
        for reskey, val in all_dataset_collisions.items()
    ]
    dataset_coll_df = pd.DataFrame(dataset_coll_rows)
    print("\nDataset-level collisions per encoding/bit:")
    print(
        dataset_coll_df.sort_values("encoding_bit")
        .set_index("encoding_bit")
    )

    rows = [
        {
            "representation": reskey.rsplit("_", 1)[0],
            "bits": int(reskey.rsplit("_", 1)[1]),
            "has_self_collision": has_coll
        }
        for reskey, comp_dict in all_self_collisions.items()
        for has_coll in comp_dict.values()
    ]

    df = pd.DataFrame(rows)

    print(
        df.groupby(["representation", "bits", "has_self_collision"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "no_collision", True: "with_collision"})
        .sort_index()
    )
    with open("model_results.json", "w") as f:
        json.dump(resdic, f, indent=2)

    with open("self_collisions.json", "w") as f:
        json.dump(all_self_collisions, f, indent=2)

    with open("dataset_collisions.json", "w") as f:
        json.dump(all_dataset_collisions, f, indent=2)
if __name__ == "__main__":
    main()
