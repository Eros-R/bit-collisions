"""
Script for fingerprinting sparse, and 2048 folded fingerprints ECFP2.
Compare tanimoto similarity of sparse vs folded, find biggest discrepancies.
Save in DataFrame and save
"""

import pandas as pd
from rdkit import Chem, DataStructs
from tqdm import tqdm
from rdkit import RDLogger
from rdkit.Chem import AllChem
from itertools import combinations
# max 50 heap
import heapq
from multiprocessing import Pool
import os

CHUNK_SIZE = 200
TOP_K = 50

# no passing C++ objs in chunker
def parse_inchi_dataset(inchi_csv_path: str):
    df = pd.read_csv(inchi_csv_path)
    df = df[~df.inchi.str.contains(r"\.")] #why is it regex by default?
    data = list(df.inchi.tolist())
    return data
# (key, folded, sparse)
def gen_fps_from_inchi(inchi: str, length=2048):
    mol = Chem.inchi.MolFromInchi(inchi)
    if mol is None or mol.GetNumAtoms() >= 5: # also drop small ones
        return None

    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=length)

    return (
        inchi,
        fpgen.GetFingerprint(mol),
        fpgen.GetSparseFingerprint(mol),
    )

# (keyA, keyB, tanimoto_folded, tanimoto_sparse)
def calculate_similarity(
    generated_fpA: tuple[
        str,
        DataStructs.cDataStructs.ExplicitBitVect,
        DataStructs.cDataStructs.SparseBitVect,
    ],
    generated_fpB: tuple[
        str,
        DataStructs.cDataStructs.ExplicitBitVect,
        DataStructs.cDataStructs.SparseBitVect,
    ],
) -> tuple[str, str, float, float]:
    folded_sim = DataStructs.TanimotoSimilarity(generated_fpA[1], generated_fpB[1])
    unfolded_sim = DataStructs.TanimotoSimilarity(generated_fpA[2], generated_fpB[2])
    return (generated_fpA[0], generated_fpB[0], folded_sim, unfolded_sim)


# pairing molecules
def calculate_pairs(
    generated_fp_list: list[
        tuple[
            str,
            DataStructs.cDataStructs.ExplicitBitVect,
            DataStructs.cDataStructs.SparseBitVect,
        ]
    ],
) -> list[str, str, float, float]:
    pairs = list(combinations(generated_fp_list, r=2))
    similarities: list[str, str, float, float] = [
        calculate_similarity(pair[0], pair[1]) for pair in tqdm(pairs)
    ]
    return similarities


# chunker
def chunkify(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


# block processing
def process_block(args):
    inchis, start, end = args

    local_fps = []
    for i in range(start, end):
        fp = gen_fps_from_inchi(inchis[i])
        if fp is not None:
            local_fps.append(fp)

    global_fps = []
    for inchi in inchis:
        fp = gen_fps_from_inchi(inchi)
        if fp is not None:
            global_fps.append(fp)

    local_heap = []

    for fp_a in local_fps:
        for fp_b in global_fps:
            if fp_a[0] >= fp_b[0]:
                continue

            a, b, folded, sparse = calculate_similarity(fp_a, fp_b)
            diff = abs(folded - sparse)

            item = (diff, a, b, folded, sparse)

            if len(local_heap) < TOP_K:
                heapq.heappush(local_heap, item)
            else:
                heapq.heappushpop(local_heap, item)

    return local_heap


# formatting, ordering, and saving
def process_dataset(inchi_path: str, out_file: str = "similarities.csv"):
    print("loading dataset...")
    # still taking a subset due to O2 complexity of pairing.
    inchis = parse_inchi_dataset(inchi_path)
    n = len(inchis)
    # abuse cores
    step = max(1, n // (int(int(os.cpu_count())/4) or 4))
    tasks = [(inchis, i, min(i + step, n)) for i in range(0, n, step)]
    print("processing...")
    heap = []
    # dropped tqdm here. w/e
    with Pool() as pool:
        for local in pool.imap_unordered(process_block, tasks):
            for item in local:
                if len(heap) < TOP_K:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)
    top = sorted(heap, reverse=True)
    df = pd.DataFrame(
        top,
        columns=["diff", "inchi_A", "inchi_B", "folded_sim", "sparse_sim"]
    )
    df.to_csv(out_file, index=False)
    print(f"saved -> {out_file}")

def main() -> None:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    inchi_csv_path: str = "data/substances.csv"
    process_dataset(inchi_csv_path)


if __name__ == "__main__":
    main()
