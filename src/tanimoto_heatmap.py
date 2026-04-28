"""
Heatmap tanimoto difference sparse/folded RDkitFP 1024, show difference in ds increase
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger

import numpy as np
from matplotlib.colors import LogNorm
from itertools import combinations
import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing import Pool,cpu_count

SIZES = [256, 1024, 2048,4096, 8192]

def parse_soil(cpd_path:str) -> list[tuple[str,Chem.rdchem.Mol]]:
    data = pd.read_csv(cpd_path, sep = "\t")
    #data = data.sample(n=100000, random_state = 158)
    return [(name, Chem.MolFromSmiles(smiles)) for name,smiles in zip(data.compound_name.tolist(), data.SMILES.tolist())]

def parse_substances(path:str) -> list[tuple[str,Chem.rdchem.Mol]]:
    data = pd.read_csv(path) #csv
    data = data[~data.inchi.str.contains(r"\.")]
    #data = data.sample(n=100000, random_state = 158)
    return [(name, Chem.inchi.MolFromInchi(key)) for name,key in zip(data.inchikey.tolist(), data.inchi.tolist()) if Chem.inchi.MolFromInchi(key) is not None and Chem.inchi.MolFromInchi(key).GetNumAtoms() > 2]

def gen_fingerprints(mol_tuple:list[tuple[str,Chem.rdchem.Mol]], fpsize:int ,sparse:bool=False) -> list[DataStructs.cDataStructs.ExplicitBitVect | DataStructs.cDataStructs.SparseBitVect]:
    fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpsize) #default minmax(1,7)
    fplist = []
    if sparse:
        for _,mol in tqdm(mol_tuple, desc = f"Generating sparse RDkit FPs"):
            fplist.append(fpgen.GetSparseFingerprint(mol))
        return fplist
    else:
        for _,mol in tqdm(mol_tuple, desc = f"Generating RDkit FPs of size {fpsize}"):
            fplist.append(fpgen.GetFingerprint(mol))
        return fplist

_fp = None
_sparse_fp = None

def _init_worker(fp, sparse_fp):
    global _fp, _sparse_fp
    _fp = fp
    _sparse_fp = sparse_fp

def _worker(i):
    sims_fold = DataStructs.BulkTanimotoSimilarity(_fp[i], _fp[i+1:])
    sims_sparse = DataStructs.BulkTanimotoSimilarity(_sparse_fp[i], _sparse_fp[i+1:])
    hist, _, _ = np.histogram2d(
        sims_fold, sims_sparse,
        bins=100, range=[[0,1],[0,1]]
    )
    return hist

def gen_figure(mol_tuple:list[tuple[str, Chem.rdchem.Mol]], sizes:list[int]=SIZES) -> None:
    sparse_fp = gen_fingerprints(mol_tuple,1024,True)
    fingerprints = {size:gen_fingerprints(mol_tuple,size) for size in sizes}
    n = len(sparse_fp)

    n_proc = max(1, cpu_count() // 2)

    fig,axes = plt.subplots(1,len(sizes), figsize=(6*len(sizes),4), sharex=True,sharey=True)

    for ax,size in zip(axes,sizes):
        fp = fingerprints[size]
        counts = np.zeros((100,100))

        with Pool(processes=n_proc, initializer=_init_worker, initargs=(fp, sparse_fp)) as pool:
            results = pool.imap_unordered(_worker, range(n))

            for hist in tqdm(results, total=n, desc=f"Pairwise similarities", unit="mol"):
                counts += hist

        counts_masked = np.ma.masked_where(counts == 0, counts)

        mesh = ax.pcolormesh(
            np.linspace(0,1,101),
            np.linspace(0,1,101),
            counts_masked.T,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=counts.max())
        )

        ax.set_xlabel("Tanimoto folded")
        ax.set_ylabel("Tanimoto sparse")
        ax.set_title(f"RDKit; {size} bits")

    fig.colorbar(mesh, ax=axes, label="count", pad=0.02)
    plt.savefig("assets/gen_rdkit1k_similarity_full.svg", bbox_inches="tight")

def main() -> None:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    substances_path:str = "data/substances.csv"
    data_tuples = parse_substances(substances_path)
    gen_figure(data_tuples)

if __name__ == "__main__":
    main()
