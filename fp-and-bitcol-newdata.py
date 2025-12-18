# -*- coding: utf-8 -*-

# %% Import

#the usual 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#rdkit
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs, rdFingerprintGenerator, PandasTools
PandasTools.RenderImagesInAllDataFrames(images=True) # when true, it displays the structures in the DF output

#others
from matplotlib.colors import LogNorm
from tqdm import tqdm
from itertools import combinations

# %% load data

url = "https://media.githubusercontent.com/media/FennerLabs/pepper/refs/heads/main/data/soil/cpd_data_soil_all_data.tsv"
dataset = pd.read_csv(url, sep='\t')

df = dataset[['compound_name', 'SMILES']].copy()
df['Structure'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))

# %% define bit col 

# generators 
sizes = [1024, 2048, 4096, 8192, 16384, 32768]

fp_generators = {
    "TopologicalTorsion": {
        "sparse": rdFingerprintGenerator.GetTopologicalTorsionGenerator(),
        "folded": {size: rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=size) for size in sizes}
    },
    "ECFP4": {
        "sparse": rdFingerprintGenerator.GetMorganGenerator(radius=2),
        "folded": {size: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size) for size in sizes}
    },
    "ECFP6": {
        "sparse": rdFingerprintGenerator.GetMorganGenerator(radius=3),
        "folded": {size: rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=size) for size in sizes}
    },
    "AtomPair": {
        "sparse": rdFingerprintGenerator.GetAtomPairGenerator(),
        "folded": {size: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=size) for size in sizes}
    },
    "RDKit": {
        "sparse": rdFingerprintGenerator.GetRDKitFPGenerator(),
        "folded": {size: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=size) for size in sizes}
    }
}

all_results = []

# loop over each generator
for fp_name, gens in fp_generators.items():
    records = []

    # calculate the number of on bits for sparse and folded for each molecule
    for mol in tqdm(df["Structure"], desc=f"Processing {fp_name}"):
        rec = {"fingerprint": fp_name}

        # sparse bits
        rec["sparse_bits"] = len(gens["sparse"].GetSparseFingerprint(mol).GetOnBits())

        # folded bits
        for size, gen in gens["folded"].items():
            rec[f"folded_bits_{size}"] = gen.GetFingerprint(mol).GetNumOnBits()

        records.append(rec)

    fp_df = pd.DataFrame(records)

    # percentage of molecules where folding changed bit count
    percent_col = {
        size:
        (fp_df["sparse_bits"] != fp_df[f"folded_bits_{size}"]).sum()
        * 100 / len(fp_df)
        for size in sizes
    }
    
    s = pd.Series(percent_col, name=fp_name)
    all_results.append(s)

summary_df = pd.DataFrame(all_results)
summary_df = summary_df.T
summary_df
# %% plot the results 

labels = ["RDKit FP", "Atom Pair FP", "ECFP6" , "ECFP4" , "Torsion FP"  ]
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(labels)+1))

order = ["RDKit", "AtomPair", "ECFP6" , "ECFP4" , "TopologicalTorsion"]  

i=0
plt.figure()
for fp in order:
    plt.plot(summary_df[fp], marker='.', label=labels[i], color=colors[i])
    i=i+1

plt.xlabel("Bit length", size=10, labelpad=15)
plt.ylabel('Molecules with atleast one bit collision (%)',size=10)
plt.xticks([1024, 2048, 4096, 8192, 16384, 32768], rotation=90, size=10)
plt.yticks(size=10)
plt.legend()

#plt.savefig('svg_pesticide_bit_col_plot.svg', bbox_inches='tight')
plt.show()

# %% example of bit collision
from IPython.display import SVG, display

def draw_top_morgan_bits(mols, bit_indices, radius=2, nBits=2048, max_mols=3, molsPerRow=3):
    """
    Draws substructures for ecfp bits across a list of RDKit molecules.
    
    Input:
    mols: list rdkit structures (from molforsmiles)
    bit_indices: list of integers of the bit to visualize 
    radius: Morgan fingerprint radius (default=2)
    nBits: fingerprint size (default=2048)
    max_mols: how many molecules to show per bit
    molsPerRow: number of molecules per row in the output grid
    """
    
    # generate morgan fp with a specific radius and bit length
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)

    for i, bit_idx in enumerate(bit_indices):
        tuples = []
        shown = 0

        # itteratively per molecule in the list of mols provided, untill max_mols is reached
        for mol in mols:
            if shown >= max_mols:
                break
            
            # store information about the bit, needed for visalisation     
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = fpg.GetFingerprint(mol, additionalOutput=ao)
            bitInfo = ao.GetBitInfoMap()
            
            if bit_idx in bitInfo:
                tuples.append((mol, bit_idx, bitInfo))
                shown += 1
        
        if tuples:
            # image of the bit with its ID and the position on bit_indices
            print(f"Bit {bit_idx} (Top {i+1})")
            img = Draw.DrawMorganBits(tuples, molsPerRow=molsPerRow)
            display(SVG(img))
        else:
            print(f"Bit {bit_idx} (Top {i+1}) not found in any of the molecules.")

draw_top_morgan_bits(df['Structure'], [917,1], nBits=1024  , radius=2, max_mols=40, molsPerRow=5)

# %% similarity effects just for ecfp 4

mols = df.Structure
n = len(mols)

# fingerprints
ecfp4 = [rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize=1024).GetFingerprint(m) for m in mols]
sparse_ecfp4 = [rdFingerprintGenerator.GetMorganGenerator(radius=2).GetSparseFingerprint(m) for m in mols]

# initialize histogram
counts = np.zeros((100, 100))

# loop over molecules, compare against all later ones
for i in range(n):
    sims_fold = DataStructs.BulkTanimotoSimilarity(ecfp4[i], ecfp4[i+1:])
    sims_sparse = DataStructs.BulkTanimotoSimilarity(sparse_ecfp4[i], sparse_ecfp4[i+1:])
    
    # update histogram 
    hist, _, _ = np.histogram2d(
        sims_fold, sims_sparse,
        bins=100, range=[[0,1],[0,1]]
    )
    counts += hist

# mask zero-count bins
counts_masked = np.ma.masked_where(counts == 0, counts)

# plot
plt.figure(figsize=(8, 6))

plt.pcolormesh(
    np.linspace(0,1,101),
    np.linspace(0,1,101),
    counts_masked.T,
    cmap="viridis",
    norm=LogNorm(vmin=1, vmax=counts.max())) # log scale is needed

plt.colorbar(label="count")
plt.xlabel("Tanimoto ECFP4 folded bit (1024)")
plt.ylabel("Tanimoto ECFP4 sparse bit")
plt.title(f"Similarity scores for unfolded vs folded\nfor {n} compounds")
#plt.savefig('svg_pesticide_sim_plot.svg', bbox_inches='tight')
plt.show()

# %% now for RDkit 

mols = df.Structure
n = len(mols)

# fingerprints
rdkitfp = [rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024).GetFingerprint(m) for m in mols]
sparse_rdkitfp= [rdFingerprintGenerator.GetRDKitFPGenerator().GetSparseFingerprint(m) for m in mols]

# initialize histogram
counts = np.zeros((100, 100))

# loop over molecules, compare against all later ones
for i in range(n):
    sims_fold = DataStructs.BulkTanimotoSimilarity(rdkitfp[i], rdkitfp[i+1:])
    sims_sparse = DataStructs.BulkTanimotoSimilarity(sparse_rdkitfp[i], sparse_rdkitfp[i+1:])
    
    # update histogram 
    hist, _, _ = np.histogram2d(
        sims_fold, sims_sparse,
        bins=100, range=[[0,1],[0,1]]
    )
    counts += hist

# mask zero-count bins
counts_masked = np.ma.masked_where(counts == 0, counts)

# plot
plt.figure(figsize=(8, 6))

plt.pcolormesh(
    np.linspace(0,1,101),
    np.linspace(0,1,101),
    counts_masked.T,
    cmap="viridis",
    norm=LogNorm(vmin=1, vmax=counts.max())) # log scale is needed

plt.colorbar(label="count")
plt.xlabel("Tanimoto RDKit folded bit (1024)")
plt.ylabel("Tanimoto RDKit sparse bit")
plt.title(f"Similarity scores for unfolded vs folded\nfor {n} compounds")
#plt.savefig('svg_pesticide_rdkit_sim_plot.svg', bbox_inches='tight')
plt.show()

# %% find the ones with high folded sim and low sparse sim 

# %%% normal fp
df["rdkit_fp"] = df['Structure'].apply(lambda x: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024).GetFingerprint(x))

jaccard_similarities = {}

for row1, row2 in combinations(df.index, 2):
    sim = DataStructs.cDataStructs.TanimotoSimilarity(df["rdkit_fp"][row1], df["rdkit_fp"][row2])
    jaccard_similarities[(row1, row2)] = sim

# to Series and filter
high_jaccard = pd.Series(jaccard_similarities)
high_jaccard = high_jaccard[high_jaccard > 0.85]
high_jaccard = high_jaccard.sort_values(ascending=False)

high_jaccard

# %%% sprase fp 
df["rdkit_sparse_fp"] = df['Structure'].apply(lambda x: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024).GetSparseFingerprint(x))

sparse_jaccard_similarities = {}

for row1, row2 in combinations(df.index, 2):
    sim = DataStructs.cDataStructs.TanimotoSimilarity(df["rdkit_sparse_fp"][row1], df["rdkit_sparse_fp"][row2])
    sparse_jaccard_similarities[(row1, row2)] = sim

# to Series and filter
low_jaccard = pd.Series(sparse_jaccard_similarities)
low_jaccard = low_jaccard[low_jaccard < 0.1]
low_jaccard = low_jaccard.sort_values(ascending=False)

low_jaccard

# %%% combine

pairs = high_jaccard.index.intersection(low_jaccard.index)
pairs

for pair in pairs:
    mols = [df.iloc[pair[0]]['Structure'] , df.iloc[pair[1]]['Structure']] 
    img = Draw.MolsToGridImage( mols, molsPerRow=2,subImgSize=(200,200))
    display(img)
    
# ok this does not seem to be a big impact for this dataset tbh 

# %%% 

# somthing about the size