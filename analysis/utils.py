import pandas as pd
import numpy as np

def predict_surface_tension(results: pd.DataFrame, next_concentration: float):
    if results.empty:
        predicted_surface_tension = 35
    else:
        concentrations = results["concentration"].to_numpy()
        surface_tensions = results["surface tension eq. (mN/m)"].to_numpy()
        if len(surface_tensions) == 0:
            predicted_surface_tension = 35
        elif len(surface_tensions) == 1:
            predicted_surface_tension = surface_tensions[0]
        elif len(surface_tensions) > 1:
            dif_st = surface_tensions[-1]-surface_tensions[-2]
            dif_conc = concentrations[-1]-concentrations[-2]
            gradient = dif_st / dif_conc
            dif_conc_sugg = next_concentration-concentrations[-1]
            predicted_surface_tension = gradient*dif_conc_sugg+surface_tensions[-1]
        else:
            print("error in predicted surface tension.")
    
    # no surfactant concentration with larger st than pure water

    if predicted_surface_tension > 72:
        predicted_surface_tension = 72
    return predicted_surface_tension

def volume_for_st(st: float):
    # could be more fancy but suffice for now
    max_drop_72 = 12
    max_drop_35 = 6
    volume = max_drop_35 + (max_drop_72 - max_drop_35) / (72 - 33) * (st - 33)

    if volume < 6:
        volume = 6
    elif volume > 12:
        volume = 12
    return volume

def suggest_volume(results: pd.DataFrame, next_concentration: float, solution_name: str):
    results_solution = results.loc[results["solution"] == solution_name]
    st = predict_surface_tension(results=results_solution, next_concentration=next_concentration)
    suggested_volume = volume_for_st(st=st)
    return suggested_volume
