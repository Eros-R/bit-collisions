# Script to find 3 candidate molecules for illustration of collisions
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator


def get_first_self_collision(dataset, radius = 2, bitlen = 1024):
    fp = rdFingerprintGenerator.GetMorganGenerator(
        radius = radius,
        fpSize = bitlen,
    )

    for smiles, mol in dataset:
        info = rdFingerprintGenerator.AdditionalOutput()
        info.CollectBitInfoMap()
        fp_bitvec = fp.GetFingerprint(mol, additionalOutput = info)
        bit_info = info.GetBitInfoMap()

        for bit_id, radius_list in bit_info.items():
            if len(radius_list) < 2:
                continue
            env_smiles_set = set()
            env_map = {}

            for atom_index, radius in radius_list:
                env = Chem.FindAtomEnvironmentOfRadiusN(
                    mol, radius, atom_index
                )
                submol = Chem.PathToSubmol(mol, env)
                env_smiles = Chem.MolToSmiles(submol, canonical=True)
                env_map.setdefault(env_smiles, []).append((atom_index, radius))
                env_smiles_set.add(env_smiles)
            # stop if hit
            if len(env_map) > 1:
                return {
                    "smiles" : smiles,
                    "bit_id" : bit_id,
                    "environments": env_map,
                    "fingerprint": list(fp_bitvec),
                }
    return None

def get_first_cross_collision(dataset, radius=2, bitlen=1024):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=bitlen
    )
    bit_registry = {}

    for smiles, mol in dataset:
        info = rdFingerprintGenerator.AdditionalOutput()
        info.CollectBitInfoMap()
        fp_bitvec = fpgen.GetFingerprint(mol, additionalOutput=info)
        bit_info = info.GetBitInfoMap()

        env_map_per_bit = {}
        for bit_id, atom_rad_list in bit_info.items():
            env_map = {}
            for atom_idx, rad in atom_rad_list:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
                submol = Chem.PathToSubmol(mol, env)
                env_smi = Chem.MolToSmiles(submol, canonical=True)
                env_map.setdefault(env_smi, []).append((atom_idx, rad))
            env_map_per_bit[bit_id] = env_map

        for bit_id, curr_env_map in env_map_per_bit.items():
            if bit_id in bit_registry:
                for prev_entry in bit_registry[bit_id]:
                    prev_env_set = set(prev_entry["environments"].keys())
                    curr_env_set = set(curr_env_map.keys())
                    # real collision if environments differ
                    if prev_env_set != curr_env_set:
                        return {
                            "SMILES_1": prev_entry["smiles"],
                            "SMILES_2": smiles,
                            "clashing_bit": bit_id,
                            "environments_1": prev_entry["environments"],
                            "environments_2": curr_env_map,
                            "fingerprint_1": prev_entry["fingerprint"],
                            "fingerprint_2": list(fp_bitvec)
                        }

            bit_registry.setdefault(bit_id, []).append({
                "smiles": smiles,
                "environments": curr_env_map,
                "fingerprint": list(fp_bitvec)
            })

    return None

def main():
    data = pd.read_csv("data/cpd_data_soil_all_data.tsv", sep = '\t')
    smiles = data["SMILES"].to_list()
    # tie smiles to mols, for ez retrieval
    dataset = list(zip(smiles, [Chem.MolFromSmiles(smile) for smile in smiles]))
    x = get_first_self_collision(dataset)
    print(x)
    y = get_first_cross_collision(dataset)
    print(y)

if __name__ == "__main__":
    main()
