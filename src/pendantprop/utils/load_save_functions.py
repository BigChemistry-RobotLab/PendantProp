# Imports

## Packages
import os
import json
import csv
import subprocess
import pandas as pd

from opentrons_api.containers import Container
from pendantprop.hardware.sensor_api import SensorAPI
from pendantprop.utils.utils import calculate_equillibrium_value

def save_csv_file(exp_name: str, subdir_name: str, csv_file, app):
    #TODO fix
    """
    Save csv file in experiment directory

    :param exp_name: Experiment name
    :param subdir_name: Subdirectory name (meta_data, experiment_data, etc)
    :param csv_file: File to save
    :param app: Flask app

    :return: None
    """

    exp_dir = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{exp_name}/{subdir_name}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    file_path = os.path.join(exp_dir, csv_file.filename)
    csv_file.save(file_path)



# def save_calibration_data(scale_t: list):
#     settings = load_settings()
#     df = pd.DataFrame(scale_t, columns=["time (s)", "scale"])
#     df.to_csv(f"experiments/{settings['EXPERIMENT_NAME']}/calibration.csv")


def initialize_results(type: str) -> pd.DataFrame: 
    # Initialize an empty DataFrame with the required columns
    if type == "wells":
        return pd.DataFrame(
            columns=[
                "sample id",
                "surface tension eq. (mN/m)",
                "drop count",
                "drop volume (uL)",
                "measure time (s)",
                "temperature (C)",
                "humidity (%)",
                "pressure (Pa)",
            ]
        )
    elif type == "serial_dilution":
        return pd.DataFrame(
            columns=[
                "sample id",
                "solution",
                "concentration",
                "surface tension eq. (mN/m)",
                "drop count",
                "drop volume (uL)",
                "measure time (s)",
                "temperature (C)",
                "humidity (%)",
                "pressure (Pa)",
            ]
        )

def append_results(
    results: pd.DataFrame,
    settings: dict,
    dynamic_surface_tension: list,
    container: Container,
    drop_parameters: dict,
    n_eq_points: int,
    sensor_api: SensorAPI,
    type: str,
):
    if dynamic_surface_tension:
        save_dynamic_surface_tension(settings, dynamic_surface_tension, container=container)
        st_eq = calculate_equillibrium_value(
            x=dynamic_surface_tension,
            n_eq_points=n_eq_points,
            column_index=1,
        )
        results = add_data_to_results(
            type=type,
            results=results,
            container=container,
            surface_tension_eq=st_eq,
            drop_parameters=drop_parameters,
            sensor_api=sensor_api,
        )
    else:
        print("Was not able to measure pendant drop!")
    return results


def save_dynamic_surface_tension(settings: dict, dynamic_surface_tension: list, container: Container):
    sample_id = container.sample_id
    if sample_id is None:
        sample_id = container.WELL_ID
    file_settings = settings["file_settings"]
    df = pd.DataFrame(
        dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
    )
    folder = f"{file_settings['output_folder']}/{file_settings['exp_tag']}/{file_settings['data_folder']}/{sample_id}"
    os.makedirs(folder, exist_ok=True)
    df.to_csv(
        f"{folder}/dynamic_surface_tension.csv"
    )


def add_data_to_results(
    type: str,
    results: pd.DataFrame,
    container: Container,
    surface_tension_eq: float,
    drop_parameters: dict,
    sensor_api=SensorAPI,
):
    sensor_data = sensor_api.capture_sensor_data()
    if type == "wells":
        new_row = pd.DataFrame(
            {
                "sample id": [container.sample_id],
                "surface tension eq. (mN/m)": [surface_tension_eq],
                "drop count": [drop_parameters["drop_count"]],
                "drop volume (uL)": [drop_parameters["drop_volume"]],
                "measure time (s)": [drop_parameters["measure_time"]],
                "temperature (C)": [float(sensor_data["Temperature (C)"])],
                "humidity (%)": [float(sensor_data["Humidity (%)"])],
                "pressure (Pa)": [float(sensor_data["Pressure (Pa)"])],
            }
        )
        results = pd.concat([results, new_row], ignore_index=True)
        return results
    elif type == "serial_dilution":
        new_row = pd.DataFrame(
            {
                "well id": [container.WELL_ID],
                "solution": [container.solution_name],
                "concentration": [container.concentration],
                "surface tension eq. (mN/m)": [surface_tension_eq],
                "drop count": [drop_parameters["drop_count"]],
                "drop volume (uL)": [drop_parameters["drop_volume"]],
                "measure time (s)": [drop_parameters["measure_time"]],
                "flow rate (uL/s)": [drop_parameters["flow_rate"]],
                "temperature (C)": [float(sensor_data["Temperature (C)"])],
                "humidity (%)": [float(sensor_data["Humidity (%)"])],
                "pressure (Pa)": [float(sensor_data["Pressure (Pa)"])],
            }
        )
        results = pd.concat([results, new_row], ignore_index=True)
        return results
    else:
        print(f"Unknown results type: {type}")

def save_results(results: pd.DataFrame, settings: dict):
    file_settings = settings["file_settings"]
    folder = f"{file_settings['output_folder']}/{file_settings['exp_tag']}/{file_settings['data_folder']}"
    os.makedirs(folder, exist_ok=True)
    file_name_results = (
        f"{folder}/results.csv"
    )
    results.to_csv(file_name_results, index=False)

def load_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError:
        return None


if __name__ == "__main__":
    pass
