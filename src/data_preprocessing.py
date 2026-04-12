# Dataset & Entry classes for downstream ML tasks

from enum import Enum
from typing import Any, Awaitable, List
from numpy import isin
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
import random
import re
from rdkit.Chem import rdFingerprintGenerator
import json

from collections import Counter
from rdkit import RDLogger

# We could infer type based on string, but doubt it's relevant.


class IdentifierType(Enum):
    smiles = "smiles"
    inchi = "inchi"


# EFCP [2,4,6,8,10]
# - set of hashes.
# RDKIT [5,6,7,8,9]

_DEFAULTS_MAP = {
    "ecfp2": {"radius": 1},
    "ecfp4": {"radius": 2},
    "ecfp6": {"radius": 3},
    "ecfp8": {"radius": 4},
    "ecfp10": {"radius": 5},
    "rdkit": {"minPath": 1, "maxPath": 7},
    "rdkit5": {"minPath": 1, "maxPath": 5},
    "rdkit6": {"minPath": 1, "maxPath": 6},
    "rdkit8": {"minPath": 1, "maxPath": 8},
    "rdkit9": {"minPath": 1, "maxPath": 9},
    "map4": None,
    "secfp6": None,
}
# even with decorator feels a bit wonky..


class RepresentationType(Enum):
    ecfp2 = "ecfp2"
    ecfp4 = "ecfp4"
    ecfp6 = "ecfp6"
    ecfp8 = "ecfp8"
    ecfp10 = "ecfp10"
    rdkit = "rdkit"
    rdkit5 = "rdkit5"
    rdkit6 = "rdkit6"
    rdkit8 = "rdkit8"
    rdkit9 = "rdkit9"
    map4 = "map4"
    secfp6 = "secfp6"

    @property
    def defaults(self):
        return _DEFAULTS_MAP[self.value]


class CompoundEntry:
    # init w/ one "type"
    def __init__(self, id, id_type, target=None):
        if not isinstance(id_type, IdentifierType):
            raise TypeError("unrecognized ID type")
        self.id = id
        self.id_type = id_type
        self._mol = None
        self._scaffold = None
        # agnostic "target" does not allow multiple targets.
        self._target = target
        self._representation = CompoundRepresentation(self)
        self._bifinfo = None  # we can propegate this later if we want to track

    def get_mol(self):
        if self.id_type is IdentifierType.inchi:
            self._mol = Chem.inchi.MolFromInchi(
                inchi=self.id, sanitize=True, removeHs=True, logLevel=None, treatWarningAsError=False)
        if self.id_type is IdentifierType.smiles:
            self._mol = Chem.MolFromSmiles(self.id)
    # using scaffold for stratification, note may be better?
    # https://greglandrum.github.io/rdkit-blog/posts/2024-05-31-scaffold-splits-and-murcko-scaffolds1.html

    def get_mol_scaffold(self):
        if self._mol is None:
            self.get_mol()
        u_scaffold = Chem.MurckoDecompose(self._mol)
        Chem.SanitizeMol(u_scaffold)
        # WARNING: This is not a valid smiles to use downstream.
        scaffold_smiles = Chem.MolToSmiles(u_scaffold, canonical=True)
        # decompose returns ring systems. so no cyclic = NONE. need to tag.
        if len(scaffold_smiles) == 0:
            scaffold_smiles = "acyclic"
        self._scaffold = scaffold_smiles
        return self._scaffold

    def set_target(self, target_value):
        self._target = target_value
    # need params to be equal? or just take expand and pray?


class CompoundRepresentation:
    # holder class for representation (unfolded/folded etc.)
    def __init__(self, entry):
        self._entry = entry  # a compound entry
        self._unfolded = None
        self._folded_counts = {}
        self._folded = {}  # can dump multiple "folds" to single dict
        self._self_collisions = {}
        self._bitfinfo = None
        self._precomputed_unfolded = None
    # perhaps magic default for radius? IDK if all use radii

    def set_representation(self, representation):
        if self._entry._mol is None:
            raise RuntimeError(
                "Mol must be set before constructing representation")
        if not isinstance(representation, RepresentationType):
            raise TypeError("Unrecognized representation type")
        # kill set rep.
        self._unfolded = None
        self._folded = {}
        self._folded_counts = {}
        self._self_collisions = {}
        self._bitfinfo = None
        # fuuhhh
        params = representation.defaults
        # we care very little about code duplication. Sue me.
        if representation is RepresentationType.ecfp4:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=params.get("radius", 2))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.ecfp2:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=params.get("radius", 1))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.ecfp6:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=params.get("radius", 3))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.ecfp8:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=params.get("radius", 4))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.ecfp10:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=params.get("radius", 5))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.rdkit:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            # Using defaults here?
            gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                minPath=params.get("minPath", 1), maxPath=params.get("maxPath", 7))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.rdkit5:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            # Using defaults here?
            gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                minPath=params.get("minPath", 1), maxPath=params.get("maxPath", 5))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.rdkit6:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            # Using defaults here?
            gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                minPath=params.get("minPath", 1), maxPath=params.get("maxPath", 6))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.rdkit8:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            # Using defaults here?
            gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                minPath=params.get("minPath", 1), maxPath=params.get("maxPath", 8))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        if representation is RepresentationType.rdkit9:
            bitinfo = rdFingerprintGenerator.AdditionalOutput()
            bitinfo.AllocateBitInfoMap()
            # Using defaults here?
            gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                minPath=params.get("minPath", 1), maxPath=params.get("maxPath", 9))
            self._unfolded = gen.GetSparseCountFingerprint(
                self._entry._mol, additionalOutput=bitinfo)
            self._bitinfo = bitinfo.GetBitInfoMap()
        # nx opening the file is stupid af.
        if representation is RepresentationType.map4:
            if self._precomputed_unfolded is None:
                raise RuntimeError(
                    "no precomputed map4 representation provided")
            self._unfolded = self._precomputed_unfolded[self._entry.id]

        if representation is RepresentationType.secfp6:
            if self._precomputed_unfolded is None:
                raise RuntimeError(
                    "no precomputed secfp6 representation provided")
            self._unfolded = self._precomputed_unfolded[self._entry.id]

    def _get_nonzero_elements(self):
        if hasattr(self._unfolded, "GetNonzeroElements"):
            return self._unfolded.GetNonzeroElements()
        elif isinstance(self._unfolded, list):
            return {i: 1 for i in self._unfolded}
        elif isinstance(self._unfolded, dict):
            return self._unfolded
        else:
            raise TypeError("Unknown unfolded type")

    def get_folded(self, bits=1024):
        folded = self._folded.get(bits, None)
        if folded:
            return self._folded[bits]
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        if bits not in self._folded:
            folded = [0] * bits
            for hash_idx in self._get_nonzero_elements():
                folded[hash_idx % bits] = 1
            self._folded[bits] = folded
        return self._folded[bits]

    def get_folded_counts(self, bits=1024):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        if bits not in self._folded_counts:
            folded = [0] * bits
            for hash_idx, count in self._get_nonzero_elements().items():
                folded[hash_idx % bits] += count
            self._folded_counts[bits] = folded
        return self._folded_counts[bits]

    def get_compound_collisions(self, bits):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        if bits not in self._self_collisions:
            collisions = defaultdict(set)
            for hash_val in self._get_nonzero_elements().keys():
                collisions[hash_val % bits].add(hash_val)
            self._self_collisions[bits] = {
                k: v for k, v in collisions.items() if len(v) > 1}
        return self._self_collisions[bits]

    def get_unfolded(self):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        return self._unfolded

# we can derive the amount of atoms directly from the Inchi string :)


def n_atoms_from_inchi(inchi: str):
    form = inchi.split('/')[1]
    # though there is a regex. should be faster/easier than making a mol and counting.
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', form)  # grep the atoms
    return sum(int(count) if count else 1 for _, count in matches)

# NOTE: unimplemented


def n_atoms_from_smiles(smiles: str):
    matches = re.findall(r'[A-Z][a-z]?', smiles)
    return len(matches)

# Might give some issues if ds is REALLY huge, but doubt it.


class CompoundDataset:
    def __init__(self, entries=None, precomputed_unfolded=None, precomputed_folded=None):
        self._entries = []
        if entries is not None:
            for entry in entries:
                self.add(entry)
        self._dataset_collisions = {}
        self._dataset_self_collisions = {}
        self._folded_dataset = {}
        self._precomputed_unfolded = precomputed_unfolded
        self._precomputed_folded = precomputed_folded

    def add(self, entry):
        if not isinstance(entry, CompoundEntry):
            raise TypeError("Dataset should consist of CompoundEntry objects")
        self._entries.append(entry)

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)
    # make slicable/indexable

    def __getitem__(self, key):
        if isinstance(key, slice):
            return CompoundDataset(self._entries[key])
        elif isinstance(key, int):
            return self._entries[key]

    # we need to sanitize, general "wrapper" of cleaning with extendable submods.
    def clean_compounds(self, drop_salts=True, drop_monodi=True):
        # NOTE: Though not optimal, multiple passes alow for a more modular system,
        # NOTE: since this is still a relatively small set, I could not care less about optimizing it.
        # WARNING: At this level only on inchi keys. needs refactoring to accept SMILES cleaning aswell
        if drop_salts:
            unparsed = len(self._entries)
            self._entries = [
                entry for entry in self._entries
                if not '.' in entry.id
            ]
            print(f"Dropped {unparsed - len(self._entries)} salts")
            print(f"{len(self._entries)} entries remain")
        if drop_monodi:
            unparsed = len(self._entries)
            self._entries = [
                entry for entry in self._entries
                if n_atoms_from_inchi(entry.id) > 2
            ]
            print(f"Dropped {unparsed - len(self._entries)} mono/di-atoms")
            print(f"{len(self._entries)} entries remain")

        # NOTE: should always check for failed mols and Kekulize.
        n_entries = len(self._entries)
        valid_entires = []
        for entry in self._entries:
            entry.get_mol()  # again lazily constructs if empty
            # 2 birds in 1 stone, we can check Kekulize here
            if entry._mol is not None:
                try:
                    Chem.Kekulize(entry._mol)
                    valid_entires.append(entry)
                except Chem.KekulizeException:
                    continue
            self._entries = valid_entires
        print(f"Dropped {n_entries - len(self._entries)
                         } unsolved mol structures.")
        print(f"{len(self._entries)} entries remain")

    def build_scaffolds(self):
        for entry in self._entries:
            entry.get_mol_scaffold()
    # there are some packages that do this, but its simple to implement, so no external matching nonsense
    # this also means we can perhaps bin later, or other strat target.
    # WRONG -> THEY SHOULD NOT SHARE SPLITS?

    def scaffold_exclusive_kfold(self, k: int = 5, seed: int = 1508):
        groups: defaultdict = defaultdict(list)
        acyclic_scaffold = "acyclic"
        acyclic_entries = []
        for entry in self._entries:
            scaffold = entry.get_mol_scaffold()
            if scaffold == acyclic_scaffold:
                acyclic_entries.append(entry)
            else:
                groups[scaffold].append(entry)
        seeded_random = random.Random(seed)
        scaffold_list = list(groups.values())
        seeded_random.shuffle(scaffold_list)
        folds = [[] for _ in range(k)]
        for i, group in enumerate(scaffold_list):
            fold_index = i % k
            folds[fold_index].extend(group)
        seeded_random.shuffle(acyclic_entries)
        for i, entry in enumerate(acyclic_entries):
            fold_index = i % k
            folds[fold_index].append(entry)
        split_list = []
        for i in range(k):
            test_entries = folds[i]
            train_entries = [e for j, fold in enumerate(
                folds) if j != i for e in fold]
            split_list.append((CompoundDataset(train_entries),
                              CompoundDataset(test_entries)))
        return split_list

    def scaffold_exclusive_sampling(
        self,
        test_fraction: float = 0.2,
        n_samples: int = 10,
        seed: int = 1508
    ):
        groups = defaultdict(list)
        acyclic_scaffold = "acyclic"
        acyclic_entries = []

        for entry in self._entries:
            scaffold = entry.get_mol_scaffold()
            if scaffold == acyclic_scaffold:
                acyclic_entries.append(entry)
            else:
                groups[scaffold].append(entry)

        scaffold_items_master = list(groups.values())
        total_scaffold_size = sum(len(g) for g in scaffold_items_master)
        target_test_size = int(test_fraction * total_scaffold_size)

        split_list = []

        for i in range(n_samples):
            rng = random.Random(seed + i)

            scaffold_items = scaffold_items_master.copy()
            rng.shuffle(scaffold_items)

            test_entries = []
            train_entries = []

            current_test_size = 0

            for group in scaffold_items:
                if current_test_size < target_test_size:
                    test_entries.extend(group)
                    current_test_size += len(group)
                else:
                    train_entries.extend(group)

            rng.shuffle(acyclic_entries)
            acyclic_test_size = int(test_fraction * len(acyclic_entries))
            test_entries.extend(acyclic_entries[:acyclic_test_size])
            train_entries.extend(acyclic_entries[acyclic_test_size:])

            print(i, len(test_entries), len(train_entries))

            split_list.append(
                (
                    CompoundDataset(train_entries),
                    CompoundDataset(test_entries),
                )
            )

        return split_list

    def to_dict(self):
        # NOTE: should be expanded with relevant slop for tables.
        return [{
            "identifier": entry.id,
            "id_type": entry.id_type.name,
            "scaffold": entry._scaffold,
            "target": entry._target,
            # etc, etc,
        }
            for entry in self._entries
        ]

    def set_targets(self, target_values):
        assert len(self) == len(
            target_values), "Unequal number of target values and compounds"

    def get_dataset_self_collisions(self, representation, bits):
        key = (representation, bits)
        if not hasattr(self, "_dataset_self_collisions"):
            self._dataset_self_collisions = {}
        if key in self._dataset_self_collisions:
            return self._dataset_self_collisions[key]

        collisions = {}
        for entry in self._entries:
            rep = entry._representation
            if rep._unfolded is None:
                rep.set_representation(representation)
            folded_counts = rep.get_folded_counts(bits)
            collisions_per_compound = defaultdict(set)
            for idx, count in enumerate(folded_counts):
                if count > 1:
                    collisions_per_compound[idx].add(idx)
            collisions[entry.id] = {
                k: v for k, v in collisions_per_compound.items() if len(v) > 0}

        self._dataset_self_collisions[key] = collisions
        return collisions

    def get_dataset_collisions(self, representation, bits):
        key = (representation, bits)
        if not hasattr(self, "_dataset_collisions"):
            self._dataset_collisions = {}
        if key in self._dataset_collisions:
            return self._dataset_collisions[key]

        collisions = defaultdict(set)
        for entry in self._entries:
            rep = entry._representation
            if rep._unfolded is None:
                rep.set_representation(representation)
            folded_counts = rep.get_folded_counts(bits)
            for idx, count in enumerate(folded_counts):
                if count > 0:
                    collisions[idx].add(entry.id)

        collisions = {k: v for k, v in collisions.items() if len(v) > 1}
        self._dataset_collisions[key] = collisions
        return collisions

    def set_representations(self, representation):
        if not isinstance(representation, RepresentationType):
            raise TypeError(f"unrecognized representation type provided: {
                            representation}")
        for entry in self._entries:
            if entry._mol is None:
                entry.get_mol()
            if self._precomputed_unfolded is not None:
                entry._representation._unfolded = self._precomputed_unfolded.get(
                    entry.id, None)
            else:
                entry._representation.set_representation(representation)
        # flush cached collision dicts to avoid cross-contamination
        self._dataset_collisions = {}
        self._dataset_self_collisions = {}
    # we actually do not need? to set the individual entries as all are aggregated

    def add_folded_representation(self, representation, folded_representation):
        if not isinstance(representation, RepresentationType):
            raise TypeError(f"unrecognized representation type provided: {
                            representation}")
        # note that this does not care what goes in...
        self._folded_dataset[representation] = folded_representation

    def print_scaffoldstats(self):
        scaffolds = [x._scaffold for x in self._entries]
        print(Counter(scaffolds))
    # needed to wrap some fringe cases

    def get_folded_dataset(self, representation, bits):
        bits = int(bits)
        bitstr = bits
        if isinstance(bitstr, int):
            bitstr = str(bits)
        # ensure top-level dict exists (already set in __init__)
        if representation not in self._folded_dataset:
            self._folded_dataset[representation] = {}
        if bitstr not in self._folded_dataset[representation]:
            self._folded_dataset[representation][bitstr] = {}
            for entry in self._entries:
                rep = entry._representation
                if rep._folded is None:
                    rep.set_representation(representation)
                self._folded_dataset[representation][bitstr][entry.id] = rep.get_folded(
                    bits)
        return self._folded_dataset[representation][bitstr]

    # helpers since the "keys" are tuples of type/bits
    def get_collisions_for(self, representation, bits):
        key = (representation, bits)
        if not hasattr(self, "_dataset_self_collisions"):
            self._dataset_self_collisions = {}
        if key not in self._dataset_self_collisions:
            self.get_dataset_self_collisions(representation, bits)
        return self._dataset_self_collisions.get(key, {})

    def get_folded_for(self, representation, bits):
        if not hasattr(self, "_folded_dataset"):
            self._folded_dataset = {}
        if representation not in self._folded_dataset:
            self._folded_dataset[representation] = {}
        if bits not in self._folded_dataset[representation]:
            self.get_folded_dataset(representation, bits)
        return self._folded_dataset[representation].get(bits, {})

    def get_targets_dict(self):
        if any(entry._target is None for entry in self._entries):
            raise RuntimeError("Some entries have no target set")
        return {entry.id: entry._target for entry in self._entries}


def sybau_rdkit():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


def main():
    sybau_rdkit()
    # pandas only used in demo so...
    import pandas as pd
    input_df = pd.read_csv("~/bit-collisions/substances.csv")
    inchi_list = input_df["inchi"].values.tolist()
    dummy_targets = input_df["inchi_id"]
    entries = zip(inchi_list, dummy_targets)
    test_ds = CompoundDataset(
        [CompoundEntry(id, IdentifierType.inchi, target) for (id, target) in entries])

    print(len(test_ds))
    test_ds.clean_compounds()
    test_ds.build_scaffolds()
    test_ds.print_scaffoldstats()
    # little print to show  some scaffolds
    # boohoo, warning because DS can be initialized empty
    for x in test_ds[30:50]:
        print(x._scaffold)

    split_list = test_ds.scaffold_exclusive_kfold(k=5, seed=1508)
    collisions = test_ds.get_dataset_collisions(RepresentationType.ecfp4, 1024)
    print(collisions)
    # unique hashes per bit
    count_collisions = {k: len(v) for k, v in collisions.items()}
    print(count_collisions)


if __name__ == "__main__":
    main()
