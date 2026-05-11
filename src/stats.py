"""
Significance tests model outputs
"""
import json
from typing import Any
import pandas as pd

import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


def load_output(json_path) -> Any:
    with open(json_path, 'r') as f:
        output = json.load(f)
    return output

def parse_results(datadict:dict) -> pd.DataFrame:
    parsed_results = {combination_key:{
        split:result["mae"]
        for split, result in combination.items()
    }
    for combination_key,combination in datadict.items()
    }
    return pd.DataFrame(parsed_results)

def reshape_results(parsed_results:pd.DataFrame) -> pd.DataFrame:
    parsed_results["split"] = parsed_results.index
    reshaped_df = (
        parsed_results.melt(
            id_vars = "split",
            var_name = "config",
            value_name = "mae",
        )
    )
    #regex colnames for splitting 
    sections = reshaped_df["config"].str.extract(
        r"(?P<model>[^_]+)_(?P<fingerprint>[^_]+)_(?P<bits>\d+)"
    )
    reshaped_df = pd.concat([reshaped_df,sections], axis = 1)
    reshaped_df["bits"] = reshaped_df["bits"].astype(int) # cast to int
    return reshaped_df


def main() -> None:
    hafner_path:str = "results/model_results.json"
    lipo_path:str = "results/model_results_lipo.json"
    hafner_data:dict = load_output(hafner_path)
    lipo_data:dict = load_output(lipo_path)
    # parse to df
    hafner_df = parse_results(hafner_data)
    lipo_df = parse_results(lipo_data)
    lipo_df = reshape_results(lipo_df)
    hafner_df = reshape_results(hafner_df)
    # save intermediates for tab.
    hafner_df.to_csv("results/mae_hafner.csv")
    lipo_df.to_csv("results/mae_lipo.csv")
    #stats. mixed lm over rm ANOVA (given splits are not completely balanced due to scaffold split differences)
    model_hafner = smf.mixedlm(
        "mae ~ C(model) + C(fingerprint) * C(bits)",
        data = hafner_df,
        groups = hafner_df["split"],
    )
    hafner_result = model_hafner.fit()
    print(hafner_result.summary())
    print(hafner_result.scale)
    print(hafner_result.cov_re)
    model_lipo = smf.mixedlm(
        "mae ~ C(model) + C(fingerprint) * C(bits)",
        data = lipo_df,
        groups = lipo_df["split"],
    )
    lipo_result = model_lipo.fit()
    print(lipo_result.summary())
    print(lipo_result.scale)
    print(lipo_result.cov_re)
    # since no big split interactions:
    #
    model_hafner = smf.ols(
        "mae ~ C(model) + C(fingerprint) * C(bits) + C(split)",
        data=hafner_df
    ).fit()

    print(model_hafner.summary())

    anova_table_hafner = sm.stats.anova_lm(model_hafner, typ=2)
    print(anova_table_hafner)

    model_lipo = smf.ols(
        "mae ~ C(model) + C(fingerprint) * C(bits) + C(split)",
        data=lipo_df
    ).fit()

    print(model_lipo.summary())

    anova_table_lipo = sm.stats.anova_lm(model_lipo, typ=2)
    print(anova_table_lipo)

if __name__ == "__main__":
    main()
