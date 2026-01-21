# Dataset & Entry classes for downstream ML tasks

from enum import Enum
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
import random
import re

from collections import Counter
import rdkit
from rdkit import RDLogger
from sqlalchemy.sql.schema import HasSchemaAttr

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
        self._representation = None
        self._bifinfo = None # we can propegate this later if we want to track
    def get_mol(self):
        if self.id_type is IdentifierType.inchi:
            self._mol = Chem.inchi.MolFromInchi(inchi = self.id, sanitize=True, removeHs=True, logLevel=None, treatWarningAsError=False)
        if self.id_type is IdentifierType.smiles:
            self._mol = Chem.MolFromSmiles(id) 
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
    def set_representation(self,representation, fold = False):
        # NOTE: maybe later fold self?
        if not isinstance(representation, RepresentationType):
            raise TypeError("unrecognized representation type requested")
        if representation is RepresentationType.ecfp4:
            bitinfo = {}
            unfolded = rdMolDescriptors.GetMorganFingerprint(self._mol,radius = 2, bitInfo=bitinfo)
            self._bifinfo = bitinfo
            self._representation = unfolded


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
        self.bit_collisions = None
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
    # NOTE:some wonk, think it would be smarter to do this bottom-up. (entry construction)
    def set_targets(self,target_values):
        assert len(self) == len(target_values), "Unequal number of target values and compounds"
    def fold_ecfp(self, bits = None):
        bits = bits or 1024
        collisions = defaultdict(set) #not list :)
        for entry in self._entries:
            if entry._representation is None:
                entry.set_representation(RepresentationType.ecfp4)
            folded = [0] * bits
            for env_hash in entry._representation.GetNonzeroElements():
                bit = env_hash % bits
                folded[bit] = 1
                collisions[bit].add(env_hash)
            entry._representation = folded # maybe not update, but what use is unfolded atp.
        self.bit_collisions = {bit:hashes for bit, hashes in collisions.items() if len(hashes) > 1}

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
    test_ds.fold_ecfp()
    print(test_ds.bit_collisions)
    collision_counts = {bit: len(hashes) for bit, hashes in test_ds.bit_collisions.items()}
    print(collision_counts)
    print(len(collision_counts))
    print((len(collision_counts)/1024)*100)
if __name__ == "__main__":
    main()
