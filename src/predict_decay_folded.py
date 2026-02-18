from yaml import load, Loader
import pandas as pd
from data_preprocessing import *
from models import *
import json
import copy
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

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

def run_split(i, split, split_identities, dataset, reptype, bit_len, mtype, model_type):
    test_ids = split_identities[split]["test"]
    train_ids = split_identities[split]["train"]

    target_dict = dataset.get_targets_dict()
    folded_ds = dataset.get_folded_dataset(reptype, bit_len)

    test_targets = [target_dict.get(id) for id in test_ids]
    test_fp = [folded_ds.get(id) for id in test_ids]
    train_targets = [target_dict.get(id) for id in train_ids]
    train_fp = [folded_ds.get(id) for id in train_ids]

    results = mtype.loader(
        x_train=train_fp,
        y_train=train_targets,
        x_test=test_fp,
        y_test=test_targets,
        representation=reptype,
        bits=bit_len,
        extract=False
    )

    if "model" in results:
        results["model"] = model_type

    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            results[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            results[k] = int(v)

    return i, results

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
    # save split identities to dict:
    split_identities = {}
    for i, split in enumerate(split_list):
        i += 1
        train, test =  split #holy fuck why did I invert this?
        test_ids = [entry.id for entry in test._entries]
        train_ids = [entry.id for entry in train._entries]
        split_identities[f"split_{i}"] = {"test":test_ids,"train":train_ids}
    resdic = defaultdict(dict)

    for encoding_type in encoding_types:
        try:
            reptype = RepresentationType(encoding_type)
        except ValueError:
            continue

        if reptype in (RepresentationType.map4, RepresentationType.secfp6):
            path_key = "map4_folded_path" if reptype is RepresentationType.map4 else "secfp6_folded_path"
            file_path = config.get(path_key)
            if file_path is None:
                raise RuntimeError("Invalid path")
            with open(file_path, "r") as f:
                precomputed_folded = json.load(f)
            dataset._precomputed_folded = precomputed_folded
            dataset.add_folded_representation(reptype, precomputed_folded)
        else:
            dataset._precomputed_folded = None

        for entry in dataset._entries:
            rep = entry._representation
            rep._folded = {}
            if reptype in (RepresentationType.map4, RepresentationType.secfp6) and dataset._precomputed_folded:
                rep._folded = copy.deepcopy(dataset._precomputed_folded.get(entry.id, []))
            else:
                rep._folded = None

        if reptype in (RepresentationType.ecfp4, RepresentationType.rdkit):
            dataset.set_representations(reptype)

        for bit_len in bit_lengths:
            reskey = f"{encoding_type}_{bit_len}"
            dataset.get_folded_dataset(reptype, str(bit_len))

        for bit_len in bit_lengths:
            for model_type in model_list:
                try:
                    mtype = ModelType.from_string(model_type)
                except ValueError:
                    continue

                splitstack = {}
                with ProcessPoolExecutor() as executor:
                    futures = []
                    for split in split_identities:
                        i = int(split.split('_')[1]) - 1
                        print(f"split {i}")
                        futures.append(
                            executor.submit(
                                run_split,
                                i,
                                split,
                                split_identities,
                                dataset,
                                reptype,
                                bit_len,
                                mtype,
                                model_type
                            )
                        )

                    for future in as_completed(futures):
                        i, results = future.result()
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

    # warp dict for lazinesss.
    with open("split_identities.json", "w") as f:
        json.dump(dict(split_identities), f)

    with open("model_results.json", "w") as f:
        json.dump(dict(resdic), f, indent=2)

if __name__ == "__main__":
    main()
