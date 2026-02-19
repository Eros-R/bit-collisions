# Script to find 3 candidate molecules for illustration of collisions
import pandas as pd
import json

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D


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
                if radius == 0:
                    continue # no center atoms
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
        for bit_id, atom_radius_list in bit_info.items():
            env_map = {}
            for atom_index, radius in atom_radius_list:
                if radius == 0:
                    continue # do not want atoms
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_index)
                submol = Chem.PathToSubmol(mol, env)
                env_smi = Chem.MolToSmiles(submol, canonical=True)
                env_map.setdefault(env_smi, []).append((atom_index, radius))
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


def draw_collision_molecules(collision_entries, file_name):
    base_name = file_name
    for i, entry in enumerate(collision_entries, start=1):
        mol = Chem.MolFromSmiles(entry['smiles'])
        if mol is None:
            raise ValueError(f"Invalid SMILES: {entry['smiles']}")

        highlight_atoms = {}
        envs = list(entry['env_map'].keys())
        colors = entry['color_map']

        for env_smi, color in zip(envs, colors):
            tuples = entry['env_map'][env_smi]
            atoms_in_env = set()
            for atom_idx, radius in tuples:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                atoms_in_env.add(atom_idx)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms_in_env.add(bond.GetBeginAtomIdx())
                    atoms_in_env.add(bond.GetEndAtomIdx())
            for aidx in atoms_in_env:
                highlight_atoms[aidx] = color

        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(highlight_atoms.keys()),
            highlightAtomColors=highlight_atoms
        )
        drawer.FinishDrawing()
        file_name = f"{base_name}_{i}.png"
        # could also use drawer.WriteDrawingTExt, but this feels nicer?
        with open(file_name, "wb") as file:
            file.write(drawer.GetDrawingText())

def main():
    data = pd.read_csv("data/cpd_data_soil_all_data.tsv", sep = '\t')
    smiles = data["SMILES"].to_list()
    # tie smiles to mols, for ez retrieval
    full_dataset = list(zip(smiles, [Chem.MolFromSmiles(smile) for smile in smiles]))

    # fuck it, taking some other molecules, first ones suck 365
    dataset = full_dataset[234:]

    cross_collision = get_first_cross_collision(dataset)
    entries = [
        {'smiles': cross_collision['SMILES_1'], 'env_map': cross_collision['environments_1'], 'color_map': [(1, 0, 0)]},
        {'smiles': cross_collision['SMILES_2'], 'env_map': cross_collision['environments_2'], 'color_map': [(1, 0, 0)]}
    ]
    draw_collision_molecules(entries, "assets/cross_collision")

    dataset = full_dataset[690:]

    self_collision = get_first_self_collision(dataset)
    entries = [{
        'smiles': self_collision['smiles'],
        'env_map': self_collision['environments'],
        'color_map': [(1, 0, 0), (1, 0.5, 0)]  # red and orange for different environments on the same
    }]
    draw_collision_molecules(entries, "assets/self_collision")

    # save prints in dict, unpacked , for lookup
    collisions = []
    collisions.append({"type":"self_collision", **self_collision})
    collisions.append({"type":"cross_collision", **cross_collision})
    with open("collisions.json", "w") as file:
        # not readable but do not intend to
        json.dump(collisions,file)

if __name__ == "__main__":
    main()
