from analysis.utils import suggest_volume
import pandas as pd

results = pd.read_csv("results.csv")
vol = suggest_volume(results=results, next_concentration=3.5, solution_name="SDS_1")
print(vol)
