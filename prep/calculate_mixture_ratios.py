import numpy as np
import pandas as pd

# Input
stock_surf1 = "SOS_2"
stock_surf2 = "CTAB_1"
alphas = [0.2, 0.4, 0.6, 0.8]
total_surfactant_concentration = 1.14  # mM
total_volume = 3
  # mL

# Load stock concentrations
df = pd.read_csv("prep/stock_solutions_concentrations.csv")
concentration_surf1 = df[df["stock name"] == stock_surf1]["concentration (mM)"].values[
    0
]
concentration_surf2 = df[df["stock name"] == stock_surf2]["concentration (mM)"].values[
    0
]


# Prepare output data
data = []

for alpha in alphas:
    n_surf1 = alpha * total_surfactant_concentration * total_volume  # µmol
    n_surf2 = (1 - alpha) * total_surfactant_concentration * total_volume  # µmol

    vol_surf1 = n_surf1 / concentration_surf1  # mL
    vol_surf2 = n_surf2 / concentration_surf2  # mL
    vol_water = total_volume - vol_surf1 - vol_surf2

    data.append(
        {
            "alpha": alpha,
            f"{stock_surf1} (mL)": round(vol_surf1, 2),
            f"{stock_surf2} (mL)": round(vol_surf2, 2),
            "Water (mL)": round(vol_water, 2),
        }
    )

# Create DataFrame
pipetting_df = pd.DataFrame(data)
# pipetting_df.to_csv("CTAB_SDS_pipetting_scheme.csv")
print(pipetting_df)
