import pandas as pd
import numpy as np

def predict_surface_tension(results: pd.DataFrame, next_concentration: float):
    if results.empty:
        predicted_surface_tension = 72
    else:
        concentrations = results["concentration"].to_numpy()
        surface_tensions = results["surface tension eq. (mN/m)"].to_numpy()
        if len(surface_tensions) == 0:
            predicted_surface_tension = 72
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
    max_drop_72 = 11
    max_drop_37 = 6.5
    volume = max_drop_37 + (max_drop_72 - max_drop_37) / (72 - 37) * (st - 37)

    if volume < max_drop_37:
        volume = max_drop_37
    elif volume > max_drop_72:
        volume = max_drop_72

    return volume

def suggest_volume(results: pd.DataFrame, next_concentration: float, solution_name: str):
    results_solution = results.loc[results["solution"] == solution_name]
    predicted_st = predict_surface_tension(results=results_solution, next_concentration=next_concentration)
    suggested_volume = volume_for_st(st=predicted_st)
    return suggested_volume

def gauge2mm(gauge_no: int) -> float:
    df = pd.read_csv("analysis/gauge_table.csv")
    df_gauge = df[df["Gauge No"] == gauge_no]
    od = df_gauge["Needle Nominal O.D. (mm)"].values[0]
    return od

if __name__ == "__main__":
    # results = pd.DataFrame()
    # results["concentration"] = [1, 2, 4, 8]
    # results["surface tension eq. (mN/m)"] = [70, 69.5, 68, 55]
    # predicted_st = predict_surface_tension(results=results, next_concentration=16)
    # print(predicted_st)
    od = gauge2mm(gauge_no=23)
    print(od)