import pandas as pd
import numpy as np

surfactant_properties = pd.read_csv('prep/surfactant_properties.csv')
stock_solutions_info = pd.read_csv('prep/stock_solutions.csv')
stock_solutions_concentrations = pd.DataFrame(columns=["stock name", "concentration (mM)"])
stock_solutions = stock_solutions_info["stock name"]

for stock_solution in stock_solutions:
    stock_solution_info = stock_solutions_info.loc[stock_solutions_info['stock name'] == stock_solution]
    volume = stock_solution_info['volume (mL)'].values[0] # in mL
    mass = stock_solution_info['mass (mg)'].values[0] # in mg
    surfactant = stock_solution_info['surfactant'].values[0] # acronym of the surfactant
    molar_mass = surfactant_properties.loc[surfactant_properties['surfactant'] == surfactant, 'molar mass (g/mol)'].values[0] # in g/mol
    concentration_mM = mass / molar_mass * 1e3 / volume # convert to mM
    stock_solutions_concentrations = pd.concat([stock_solutions_concentrations, pd.DataFrame({"stock name": [stock_solution], "concentration (mM)": [concentration_mM]})], ignore_index=True)

# Save the concentrations to a CSV file
stock_solutions_concentrations.to_csv('prep/stock_solutions_concentrations.csv', index=False)

print(stock_solutions_concentrations)