import pandas as pd
import numpy as np

surfactant_properties = pd.read_csv('prep/surfactant_properties.csv')

# input parameters
surfactant = "12-BAC" # acronym of the surfactant
volume = 50 # in mL
factor_above_CMC = 4 # factor above CMC

CMC_literature = surfactant_properties.loc[surfactant_properties['surfactant'] == surfactant, 'CMC (mM)'].values[0] # in mM
molar_mass = surfactant_properties.loc[surfactant_properties['surfactant'] == surfactant, 'molar mass (g/mol)'].values[0] # in g/mol

required_amount_umol = factor_above_CMC * CMC_literature * volume
required_amount_mg = required_amount_umol * molar_mass / 1e3 # convert to miligrams

print(f"To prepare {volume} mL of a solution with a concentration of {factor_above_CMC} times the CMC of {surfactant}, you need to weigh {required_amount_mg:.2f} mg of {surfactant}.")