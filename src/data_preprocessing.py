# Dataset & Entry classes for downstream ML tasks

from enum import Enum
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
import random
import re

from collections import Counter
from rdkit import RDLogger

# We could infer type based on string, but doubt it's relevant.
class IdentifierType(Enum):
    smiles = "smiles"
    inchi = "inchi"

class RepresentationType(Enum):
    ecfp4 = "ecfp4"
    map4 = "map4"
    ... #?

class CompoundEntry:
    # init w/ one "type"
    def __init__(self, id, id_type, target = None):
        if not isinstance(id_type, IdentifierType):
            raise TypeError("unrecognized ID type")
        self.id = id
        self.id_type = id_type
        self._mol = None
        self._scaffold = None
        self._target = target #agnostic "target" does not allow multiple targets.
        self._representation = CompoundRepresentation(self)
        self._bifinfo = None # we can propegate this later if we want to track
    def get_mol(self):
        if self.id_type is IdentifierType.inchi:
            self._mol = Chem.inchi.MolFromInchi(inchi = self.id, sanitize=True, removeHs=True, logLevel=None, treatWarningAsError=False)
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
        if len(scaffold_smiles) == 0: # decompose returns ring systems. so no cyclic = NONE. need to tag.
            scaffold_smiles = "acyclic"
        self._scaffold = scaffold_smiles
    def set_target(self, target_value):
        self._target = target_value
    # need params to be equal? or just take expand and pray?



class CompoundRepresentation:
    # holder class for representation (unfolded/folded etc.)
    def __init__(self, entry):
        self._entry = entry #a compound entry
        self._unfolded = None
        self._folded_counts = {}
        self._folded = {} # can dump multiple "folds" to single dict
        self._self_collisions = None
        self._bifinfo = None
    def set_representation(self, representation):
        if not isinstance(representation, RepresentationType):
            raise TypeError("Unrecognized representation type")
        # kill set rep.
        self._unfolded = None
        self._folded = None
        self._folded_counts = None
        self._self_collisions = None
        self._bifinfo = None

        if representation is RepresentationType.ecfp4:
            bitinfo = {} # radius is hard to acces!
            self._unfolded = rdMolDescriptors.GetMorganFingerprint(self._entry._mol, radius = 2, bitInfo=bitinfo)
            self._bifinfo = bitinfo
    def get_folded(self,bits = 1024):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        if bits not in self._folded:
            folded = [0] * bits
            for hash in self._unfolded.GetNonzeroElements():
                folded[hash % bits] = 1
            self._folded[bits] = folded
        # oop, still return the folded vec dict?\
    #NOTE: maybe duplication > check every it?
    def get_folded_counts(self,bits = 1024):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        if bits not in self._folded_counts:
            folded = [0] * bits
            for hash in self._unfolded.GetNonzeroElements():
                folded[hash % bits] += 1
            self._folded_counts[bits] = folded
    def get_compount_collisions(self, bits):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        if bits not in self._self_collisions:
            collisions = defaultdict(set)
            for hash in self._unfolded.GetNonzeroElements():
                collisions[hash % bits].add(hash)
            self._self_collisions[bits] = {
                k:v for k,v in collisions.items() if len(v) > 1
            }
    # should make getters return
    def get_unfolded(self):
        if self._unfolded is None:
            raise RuntimeError("representation not set")
        return self._unfolded

# we can derive the amount of atoms directly from the Inchi string :)
def n_atoms_from_inchi(inchi:str):
    form = inchi.split('/')[1]
    #though there is a regex. should be faster/easier than making a mol and counting.
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', form) # grep the atoms
    return sum(int(count) if count else 1 for _, count in matches)

# NOTE: unimplemented
def n_atoms_from_smiles(smiles:str):
    matches = re.findall(r'[A-Z][a-z]?', smiles)
    return len(matches)

# Might give some issues if ds is REALLY huge, but doubt it.
class CompoundDataset:
    def __init__(self,entries=None):
        self._entries = []
        if entries is not None:
            for entry in entries:
                self.add(entry)
        self._dataset_collisions = {}
    def add(self,entry):
        if not isinstance(entry, CompoundEntry):
            raise TypeError("Dataset should consist of CompoundEntry objects")
        self._entries.append(entry)
    def __len__(self):
        return len(self._entries)
    def __iter__(self):
        return iter(self._entries)
    # make slicable/indexable
    def __getitem__(self,key):
        if isinstance(key,slice):
            return CompoundDataset(self._entries[key])
        elif isinstance(key,int):
            return self._entries[key]

    # we need to sanitize, general "wrapper" of cleaning with extendable submods.
    def clean_compounds(self, drop_salts = True, drop_monodi = True):
        #NOTE: Though not optimal, multiple passes alow for a more modular system,
        #NOTE: since this is still a relatively small set, I could not care less about optimizing it.
        #WARNING: At this level only on inchi keys. needs refactoring to accept SMILES cleaning aswell
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
            self._entries =[
                entry for entry in self._entries
                if n_atoms_from_inchi(entry.id) > 2 
            ]
            print(f"Dropped {unparsed - len(self._entries)} mono/di-atoms")
            print(f"{len(self._entries)} entries remain")

        #NOTE: should always check for failed mols and Kekulize.
        n_entries = len(self._entries)
        valid_entires = []
        for entry in self._entries:
            entry.get_mol() # again lazily constructs if empty
            # 2 birds in 1 stone, we can check Kekulize here
            if entry._mol is not None:
                try:
                    Chem.Kekulize(entry._mol)
                    valid_entires.append(entry)
                except Chem.KekulizeException:
                    continue
            self._entries = valid_entires
        print(f"Dropped {n_entries - len(self._entries)} unsolved mol structures.")
        print(f"{len(self._entries)} entries remain")

    def build_scaffolds(self):
        for entry in self._entries:
            entry.get_mol_scaffold()
    # there are some packages that do this, but its simple to implement, so no external matching nonsense
    # this also means we can perhaps bin later, or other strat target.
    # WRONG -> THEY SHOULD NOT SHARE SPLITS?
    def scaffold_exclusive_kfold(self, k:int=5,seed:int=1508):
        groups:defaultdict = defaultdict(list)
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
            train_entries = [e for j, fold in enumerate(folds) if j != i for e in fold]
            split_list.append((CompoundDataset(train_entries), CompoundDataset(test_entries)))
        return split_list

    def scaffold_exclusive_sampling(
        self,
        test_fraction: float = 0.2,
        n_samples: int = 10, #
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

        scaffold_items = list(groups.values())
        split_list = []

        for i in range(n_samples):
            rng = random.Random(seed + i) # note the seed +1 if wanted to reprod. later.
            rng.shuffle(scaffold_items)

            test_entries = []
            train_entries = []

            target_test_size = int(test_fraction * sum(len(g) for g in scaffold_items))
            current_size = 0

            for group in scaffold_items:
                if current_size < target_test_size:
                    test_entries.extend(group)
                    current_size += len(group)
                else:
                    train_entries.extend(group)

            rng.shuffle(acyclic_entries)
            acyclic_test_size = int(test_fraction * len(acyclic_entries))
            acyclic_test = acyclic_entries[:acyclic_test_size]
            acyclic_train = acyclic_entries[acyclic_test_size:]

            test_entries.extend(acyclic_test)
            train_entries.extend(acyclic_train)

            split_list.append(
                (
                    CompoundDataset(train_entries),
                    CompoundDataset(test_entries),
                )
            )

        return split_list
    def to_dict(self):
    #NOTE: should be expanded with relevant slop for tables.
        return[{
            "identifier": entry.id,
            "id_type": entry.id_type.name,
            "scaffold": entry._scaffold,
            "target":entry._target,
            # etc, etc,
        }
        for entry in self._entries
    ]
    def set_targets(self,target_values):
        assert len(self) == len(target_values), "Unequal number of target values and compounds"

    def get_dataset_collisions(self, representation, bits):
        key = representation,bits
        # if it is already there return.
        if key in self._dataset_collisions:
            return self._dataset_collisions[key]
        collisions = defaultdict(set)
        for entry in self._entries:
            rep = entry._representation
            if rep._unfolded is None:
                rep.set_representation(representation) #here radius is hardcoded,
            unfolded = rep.get_unfolded()
            for hash in unfolded.GetNonzeroElements():
                collisions[hash % bits].add(hash)
        collisions = {k:v for k,v in collisions.items() if len(v) > 1}
        self._dataset_collisions[key] = collisions
        return collisions

    def print_scaffoldstats(self):
        scaffolds = [x._scaffold for x in self._entries]
        print(Counter(scaffolds))
    


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
    test_ds = CompoundDataset([CompoundEntry(id,IdentifierType.inchi,target) for (id,target) in entries])

    print(len(test_ds))
    test_ds.clean_compounds()
    test_ds.build_scaffolds()
    test_ds.print_scaffoldstats()
    # little print to show  some scaffolds
    # boohoo, warning because DS can be initialized empty
    for x in test_ds[30:50]:
        print(x._scaffold)
        
    split_list = test_ds.scaffold_exclusive_kfold(k = 5, seed = 1508)
    collisions = test_ds.get_dataset_collisions(RepresentationType.ecfp4,1024)
    print(collisions)
    # unique hashes per bit
    count_collisions = {k:len(v) for k,v in collisions.items()}
    print(count_collisions)

if __name__ == "__main__":
    main()
