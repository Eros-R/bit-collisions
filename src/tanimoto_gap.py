"""
Script for fingerprinting sparse, and 2048 folded fingerprints ECFP2.
Compare tanimoto similarity of sparse vs folded, find biggest discrepancies.
Take 10 and visualize.
"""
import pandas as pd
from rdkit import Chem

def parse_inchi_dataset(inchi_csv_path:str)->pd.DataFrame: 
    inchi_df:pd.DataFrame = pd.read_csv(inchi_csv_path)
    keys = inchi_df.inchi.tolist()
    molset = [Chem.inchi.MolFromInchi(key) for key in keys]
    print(molset)

def main() -> None:
    print("hi")
    inchi_csv_path:str = "data/substances.csv"
    parse_inchi_dataset(inchi_csv_path)

if __name__ == "__main__":
    main()
