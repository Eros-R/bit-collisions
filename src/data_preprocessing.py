# Ignore the placeholder name.
# want to achieve a few things here

from enum import Enum
from rdkit import Chem
from collections import defaultdict
import random
import re

import rdkit
from rdkit import RDLogger

# We could infer type based on string, but doubt it's relevant.
class IdentifierType(Enum):
    smiles = "smiles"
    inchi = "inchi"

class CompoundEntry:
    # init w/ one "type"
    def __init__(self, id, id_type):
        if not isinstance(id_type, IdentifierType):
            raise TypeError("unrecognized ID type")
        self.id = id
        self.id_type = id_type
        self._mol = None
        self._scaffold = None
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
        # Doubt we need the mol so we can save the SMILES for strat.
        # WARNING: This is not a valid smiles to use downstream. 
        scaffold_smiles = Chem.MolToSmiles(u_scaffold)
        if len(scaffold_smiles) == 0: # decompose returns ring systems. so no cyclic = NONE. need to tag.
            scaffold_smiles = "acyclic"
        self._scaffold = scaffold_smiles

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
    def scaffold_stratified_kfold(self, k:int=5,seed:int=1508):
        groups:defaultdict = defaultdict(list)
        for entry in self._entries:
            groups[entry.get_mol_scaffold()].append(entry)
        # randomize scaffold groups
        seeded_random = random.Random(seed)
        for group in groups.values():
            seeded_random.shuffle(group)
        folds = [[] for _ in range(k)]

        scaffold_list = list(groups.values())
        scaffold_list.sort(key=lambda g: len(g), reverse=True)
        for i, group in enumerate(scaffold_list):
            fold_index = i % k #haha fold, like in fingerprint (modulo)
            folds[fold_index].extend(group)
        split_list = []
        for i in range(k):
            test_entries = folds[i]
            train_entries = [e for j, fold in enumerate(folds) if j !=i for e in fold]
            split_list.append((CompoundDataset(train_entries), CompoundDataset(test_entries)))
        return split_list
    def to_dict(self):
    #NOTE: should be expanded with relevant slop for tables.
        return[{
            "identifier": entry.id,
            "id_type": entry.id_type.name,
            "scaffold": entry._scaffold,
            # etc, etc,
        }
        for entry in self._entries
    ]

def sybau_rdkit():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


def main():
    sybau_rdkit()
    # demo. might be a bit more than needed
    import pandas as pd
    input_df = pd.read_csv("~/bit-collisions/substances.csv")
    inchi_list = input_df["inchi"].values.tolist()
    test_ds = CompoundDataset([CompoundEntry(key,IdentifierType.inchi) for key in inchi_list])

    print(len(test_ds))
    test_ds.clean_compounds()
    test_ds.build_scaffolds()
    # little print to show  some scaffolds
    for x in test_ds[30:50]:
        print(x._scaffold)
    split_list = test_ds.scaffold_stratified_kfold()
    print(split_list)
    print(split_list[0][1].to_dict()) # test of fold 1



if __name__ == "__main__":
    main()
