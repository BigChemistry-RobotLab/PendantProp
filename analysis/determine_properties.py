# Imports

## Packages
import pandas as pd
import numpy as np

## Custom code
from analysis.models import szyszkowski_model
from analysis.utils import fit_model
from analysis.utils import calculate_st_at_cmc, calculate_gamma_max

def extract_properties_from_isotherm(results_solution: pd.DataFrame) -> pd.DataFrame:
    
    # extract metadata
    properties = {}
    properties["solution"] = results_solution["solution"].iloc[0]

    # fit model to the isotherm data
    c = results_solution["concentration"] / 1000
    st = results_solution["surface tension eq. (mN/m)"] / 1000
    obs = (c, st)
    parameters = ["cmc", "gamma_max", "Kad"]
    post_pred, x_new = fit_model(
        obs, model=szyszkowski_model, parameters=parameters, outlier_check=False
    )

    # extract properties from the fitted model
    for parameter in parameters:
        if parameter == "cmc":
            properties[parameter] = float(post_pred[parameter].mean(axis=0) * 1000)
        elif parameter == "gamma_max":
            if properties["solution"] == "C12E3" or properties["solution"] == "C12E4": 
                properties[parameter] = calculate_gamma_max(x_new=x_new, post_pred=post_pred, n = 1) # non-ionic
            else:
                properties[parameter] = calculate_gamma_max(x_new=x_new, post_pred=post_pred, n = 2) # ionic
        else:
            properties[parameter] = float(post_pred[parameter].mean(axis=0))
    
    properties["st_at_cmc"] = float(calculate_st_at_cmc(x_new, post_pred))
    properties = pd.DataFrame([properties])
    return properties

def extract_total_properties() -> pd.DataFrame:
    overview = pd.read_csv("data/single_surfactant_characterization/overview.csv")
    total_properties = pd.DataFrame()
    for idx, row in overview.iterrows():
        experiment_tag = row["experiment name"]
        sample = row["sample"]
        print(f"Processing {sample}")
        for i in range(1, 4): # change to range(1, 4) to process all solutions
            solution_name = f"{sample}_{i}"
            results = pd.read_csv(f"data/single_surfactant_characterization/{experiment_tag}/results.csv")
            results_solution = results[results["solution"] == solution_name]
            print("extracting properties for solution:", solution_name)
            properties = extract_properties_from_isotherm(results_solution=results_solution)
            total_properties = pd.concat([total_properties, pd.DataFrame(properties)])
    return total_properties

def aggregate_properties(total_properties: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate properties for each sample to a mean and std.
    """
    total_properties["sample"] = (
        total_properties["solution"].str.rsplit("_", n=1).str[0]
    )
    total_properties["log(Kad)"] = np.log10(total_properties["Kad"])
    total_properties = total_properties.drop(columns=["Kad"])
    numeric_cols = total_properties.select_dtypes(include="number").columns
    agg_df = total_properties.groupby("sample")[numeric_cols].agg(["mean", "std"])
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()

    return agg_df

def format_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the aggregated properties table for better readability.
    """

    # Rename columns
    agg_df["sample"] = agg_df["sample"].str.replace(r"_[124]$", "", regex=True)
    agg_df = agg_df.rename(columns={"sample": "surfactant"})

    # Remove relative err columns
    agg_df = agg_df.loc[:, ~agg_df.columns.str.contains("err")]

    # Convert gamma_max to mol/cm^2 * 1e10
    agg_df["gamma_max_mean"] = agg_df["gamma_max_mean"] / 1e4 * 1e10
    agg_df["gamma_max_std"] = agg_df["gamma_max_std"] / 1e4 * 1e10

    # Round columns
    agg_df = agg_df.round({
        "cmc_mean": 2,
        "gamma_max_mean": 1,
        "st_at_cmc_mean": 1,
        "log(Kad)_mean": 2
    })

    # Add the  error behind the mean value in brackets
    for col in ["cmc", "gamma_max", "log(Kad)", "st_at_cmc"]:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        agg_df[mean_col] = agg_df.apply(
            lambda x: (
                f"{x[mean_col]} ({x[std_col]:.2f})"
                if pd.notna(x[std_col])
                else x[mean_col]
            ),
            axis=1,
        )

    # Drop the std columns
    agg_df = agg_df.drop(columns=[f"{col}_std" for col in ["cmc", "gamma_max", "log(Kad)", "st_at_cmc"]])

    return agg_df