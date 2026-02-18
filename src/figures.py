# generating figures on predict_decay.py output.
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
# loading helper
def load_output(json_path):
    with open(json_path, 'r') as f:
        output = json.load(f)
    return output

# similartiy to compare hashes.
def jaccard_similarity(fp1, fp2):
    s1, s2 = set(fp1), set(fp2)
    return len(s1 & s2) / len(s1 | s2)
# reused fold check for sanity check...
def fold_and_count(fp_list, n_bits=128):
    folded = [h % n_bits for h in fp_list]
    counts = Counter(folded)
    num_collisions = sum(v-1 for v in counts.values() if v > 1)
    total_bits = len(folded)
    proportion = num_collisions / total_bits if total_bits else 0
    return set(folded), proportion
def main():
    model_output_path = "model_results.json"
    dataset_collisions_path = "dataset_collisions.json"
    self_collisions_path = "self_collisions.json"
    split_identities_path = "split_identities.json"
    # Average runs, flag "has collisions", plot
    model_output = load_output(model_output_path)
    self_collisons = load_output(self_collisions_path)
    split_identities = load_output(split_identities_path)
    # Folding collisions proportional to the set bits:
    

    # order is perseved to ID / pred 892 * bit len = total bits
    # proportion of (collided bits / total bits.)
    dataset_collisions = load_output(dataset_collisions_path)
    # dirty stringsplit, do not do this :)
    n_entries = 982
    df_dict = {}
    for key, value in dataset_collisions.items():
        hashtype, bit_len = key.split('_')
        proportion = value / (n_entries * int(bit_len)) * 100
        df_dict[(hashtype, int(bit_len))] = proportion

    proportion_df = pd.Series(df_dict).unstack(level=1)
    proportion_df.index = [i[0] if isinstance(i, tuple) else i for i in proportion_df.index]
    proportion_df = proportion_df.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8,5))
    # styles look shit
    styles = {
        "ecfp4": {"linestyle": "-", "marker": "o", "color": "blue"},
        "map4": {"linestyle": "-", "marker": "s", "color": "red"},
        "rdkit": {"linestyle": "-", "marker": "^", "color": "green"},
        "secfp6": {"linestyle": "-", "marker": "v", "color": "orange"},
    }

    for idx, row in proportion_df.iterrows():
        x = proportion_df.columns.tolist()
        y = row.values.astype(float)
        ax.plot(x, y, label=idx, **styles[idx])

    ax.set_xlabel("Fingerprint size")
    ax.set_ylabel("Collision proportion")
    ax.set_xticks(proportion_df.columns)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig("assets/collision_proportion_vs_fingerprint_size.png", dpi=300)
    plt.close(fig)

    # sanity check of similarity of SECFP6 / MAP4
    map4_data = load_output("map4_unfolded.json")
    secfp6_data = load_output("secfp6_unfolded.json")
    # wanted maybe check per split, but can wrangle entire set.
    common_smiles = set(map4_data.keys()) & set(secfp6_data.keys())
    print(f"identical molecules: {len(common_smiles)}")


    similarities = []
    for smi in common_smiles:
        fp1 = map4_data[smi]
        fp2 = secfp6_data[smi]
        similarities.append(jaccard_similarity(fp1, fp2))

    average_similarity = np.mean(similarities)
    print("Average Jaccard similarity:", average_similarity)
    # @Daniel, checked pred. no way it was not overwritten. so ?
    collision_stats = defaultdict(dict)
    for smi in common_smiles:
        fp_map4 = map4_data[smi]
        fp_secfp6 = secfp6_data[smi]

        folded_map4, prop_map4 = fold_and_count(fp_map4, 128)
        folded_secfp6, prop_secfp6 = fold_and_count(fp_secfp6, 128)

        collision_stats[smi]["map4_folded_set"] = folded_map4
        collision_stats[smi]["map4_collision_prop"] = prop_map4
        collision_stats[smi]["secfp6_folded_set"] = folded_secfp6
        collision_stats[smi]["secfp6_collision_prop"] = prop_secfp6

    avg_map4 = sum(d["map4_collision_prop"] for d in collision_stats.values()) / len(collision_stats)
    avg_secfp6 = sum(d["secfp6_collision_prop"] for d in collision_stats.values()) / len(collision_stats)

    print("Average collision proportion at 128 bits:")
    print(f"MAP4:    {avg_map4:.4f}")
    print(f"SECFP6:  {avg_secfp6:.4f}")
    print(f"near identical fold clashes. So it is valid but very strange?")
    
    # s2 Compound level collisions:

    collision_data = self_collisons
    collision_stats = []

    for type_bitlen, smiles_dict in collision_data.items():
        encoding, bitlen = type_bitlen.rsplit("_", 1)
        proportion = sum(1 for collided in smiles_dict.values() if collided) / len(smiles_dict)
        collision_stats.append({
            "encoding": encoding,
            "bits": int(bitlen),
            "proportion_with_collision": proportion
        })

    df = pd.DataFrame(collision_stats)
    collision_df = df.pivot(index="encoding", columns="bits", values="proportion_with_collision").sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8,5))

    styles = {
        "ecfp4": {"linestyle": "-", "marker": "o", "color": "blue"},
        "map4": {"linestyle": "--", "marker": "s", "color": "red"},
        "rdkit": {"linestyle": "-.", "marker": "^", "color": "green"},
        "secfp6": {"linestyle": ":", "marker": "v", "color": "orange"},
    }

    for encoding, row in collision_df.iterrows():
        x = collision_df.columns.tolist()
        y = row.values.astype(float)
        ax.plot(x, y, label=encoding, **styles.get(encoding, {"marker":"o"}))

    ax.set_xlabel("Fingerprint size (bits)")
    ax.set_ylabel("Proportion of molecules with collisions")
    ax.set_xticks(collision_df.columns)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig("assets/compound_collisions_vs_fingerprint_size.png", dpi=300)
    plt.close(fig)
    print("Consider that the amount of collisions/ molecules with a collision are a bit misleading due to the folding process.")
    print("'Sane' folding lengths produce clashes even on the set of hashes")
    bit_length = 2048 
    print(f"bit len {bit_length}")
    collision_stats = {}
    for name, data in [("map4", map4_data), ("secfp6", secfp6_data)]:
        n_total = len(data)
        n_collided = 0
        for fp in data.values():
            folded = [h % bit_length for h in set(fp)]
            counts = Counter(folded)
            if any(v > 1 for v in counts.values()):
                n_collided += 1
        collision_stats[name] = n_collided / n_total

    for enc, prop in collision_stats.items():
        print(f"{enc}: {prop:.4f}")
    print("Unreasonably large folding lengths produce non clashing compounds")
    bit_length = int(1e6)
    print(f"bit len {bit_length}")
    collision_stats = {}
    for name, data in [("map4", map4_data), ("secfp6", secfp6_data)]:
        n_total = len(data)
        n_collided = 0
        for fp in data.values():
            folded = [h % bit_length for h in set(fp)]
            counts = Counter(folded)
            if any(v > 1 for v in counts.values()):
                n_collided += 1
        collision_stats[name] = n_collided / n_total

    for enc, prop in collision_stats.items():
        print(f"{enc}: {prop:.4f}")
    rows = []
    for key, split_dict in model_output.items():
        model_type, encoding, bitlen = key.rsplit("_", 2)
        bitlen = int(bitlen)
        for split_idx, metrics in split_dict.items():
            mae = metrics.get("mae", None)
            if mae is not None:
                rows.append({
                    "model_type": model_type,
                    "encoding": encoding,
                    "bits": bitlen,
                    "split": int(split_idx),
                    "mae": mae
                })

    df = pd.DataFrame(rows)

    mean_mae_df = df.groupby(["model_type", "encoding", "bits"])["mae"].mean().reset_index()

    for model_type, model_df in mean_mae_df.groupby("model_type"):
        pivot_df = model_df.pivot(index="encoding", columns="bits", values="mae").sort_index(axis=1)
        
        fig, ax = plt.subplots(figsize=(8,5))
        
        styles = {
            "ecfp4": {"linestyle": "-", "marker": "o", "color": "blue"},
            "map4": {"linestyle": "--", "marker": "s", "color": "red"},
            "rdkit": {"linestyle": "-.", "marker": "^", "color": "green"},
            "secfp6": {"linestyle": ":", "marker": "v", "color": "orange"},
        }
        
        for encoding, row in pivot_df.iterrows():
            x = pivot_df.columns.tolist()
            y = row.values.astype(float)
            ax.plot(x, y, label=encoding, **styles.get(encoding, {"marker":"o"}))
        
        ax.set_xlabel("Fingerprint size (bits)")
        ax.set_ylabel(f"Mean MAE across splits ({model_type})")
        ax.set_xticks(pivot_df.columns)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(f"assets/mean_mae_{model_type}.png", dpi=300)
        plt.close(fig)

    # BEST model ecfp "do collisions matter?"
    model_key = "xgb_ecfp4_256"
    # obscure previous split dict:
    split_dict = split_identities 
    splits = model_output[model_key]
    best_split_idx = min(splits, key=lambda s: splits[s]["mae"])
    best_split = splits[best_split_idx]
    print(f"Using split {best_split_idx}")
    y_p = best_split["y_pred"]
    y_truth = best_split["y_test"]
    split_name = f"split_{int(best_split_idx) + 1}"
    print(split_name)# split idx0based
    test_smiles = split_dict[split_name]["test"]
    print(len(test_smiles))
    print(len(y_p))
    assert len(test_smiles) == len(y_p) == len(y_truth), "Mismatch in test size"
    
    collision_key = "ecfp4_2048"
    collision_flags = collision_data[collision_key]
    colors = ["red" if collision_flags.get(smi, False) else "blue" for smi in test_smiles]

    plt.figure(figsize=(6,6))
    plt.scatter(y_truth, y_p, c=colors, alpha=0.7, edgecolors="k")
    plt.xlabel("Ground truth")
    plt.ylabel("Predictions")
    plt.title(f"Pred vs Truth ({model_key}, best split {best_split_idx})")
    plt.plot([min(y_truth), max(y_truth)], [min(y_truth), max(y_truth)], "k--")  # identity line
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/pred_vs_truth_xgboost_ecfp4_2048.png", dpi=300)
    plt.close()

    df = pd.read_csv("data/cpd_data_soil_all_data.tsv", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df['DT50_gmean'], bins=50, color='dodgerblue', edgecolor='black')
    axes[0].set_title("DT50_gmean")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Count")

    axes[1].hist(df['DT50_log_gmean'], bins=50, color='orange', edgecolor='black')
    axes[1].set_title("DT50_log_gmean")
    axes[1].set_xlabel("Log(Days)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("assets/dt50_log_gmean_hist.png", dpi=300)
    plt.close()
    print("DT50_gmean:", np.mean(df['DT50_gmean']), np.std(df['DT50_gmean']))
    print("DT50_log_gmean:", np.mean(df['DT50_log_gmean']), np.std(df['DT50_log_gmean']))

if __name__ == "__main__":
    main()
