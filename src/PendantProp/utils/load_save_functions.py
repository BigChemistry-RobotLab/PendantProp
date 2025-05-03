import os
import json
import csv
import subprocess
import pandas as pd

from PendantProp.utils.utils import calculate_equillibrium_value
from PendantProp.hardware.sensor.sensor_api import SensorAPI


def save_csv_file(exp_name: str, subdir_name: str, csv_file, app):
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
    )  # TODO check
    os.makedirs(exp_dir, exist_ok=True)
    file_path = os.path.join(exp_dir, csv_file.filename)
    csv_file.save(file_path)


def load_settings(file_name="settings.json"):
    """
    Load settings from json file

    :param file_name: File name to load settings from
    """
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            return json.load(file)


def save_settings(settings, file_name="settings.json"):
    """
    Save settings to json file

    :param settings: Settings to save
    :param file_name: File name to save settings to
    """
    with open(file_name, "w") as file:
        json.dump(settings, file, indent=4)


def save_settings_meta_data(settings: dict, experiment_dir="experiments"):
    file_path = (
        f"{experiment_dir}/{settings['EXPERIMENT_NAME']}/meta_data/settings.json"
    )
    with open(file_path, "w") as file:
        json.dump(settings, file, indent=4)


def save_instances_to_csv(instances, filename):
    # Get the attribute names from the first instance
    fieldnames = [attr for attr in vars(instances[0])]

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data rows
        for instance in instances:
            writer.writerow(vars(instance))


def save_calibration_data(scale_t: list, experiment_dir="experiments"):
    settings = load_settings()
    df = pd.DataFrame(scale_t, columns=["time (s)", "scale"])
    df.to_csv(f"{experiment_dir}/{settings['EXPERIMENT_NAME']}/calibration.csv")


def initialize_results():
    # Initialize an empty DataFrame with the required columns
    return pd.DataFrame(
        columns=[
            "well id",
            "solution",
            "concentration",
            "point type",
            "surface tension eq. (mN/m)",
            "drop count",
            "drop volume (uL)",
            "max measure time (s)",
            "flow rate (uL/s)",
            "temperature (C)",
            "humidity (%)",
            "pressure (Pa)",
        ]
    )


def load_info(file_name: str, experiment_dir="experiments"):
    settings = load_settings()
    file_path = f"{experiment_dir}/{settings['EXPERIMENT_NAME']}/meta_data"
    return pd.read_csv(f"{file_path}/{file_name}")


def append_results(
    results: pd.DataFrame,
    point_type: str,
    dynamic_surface_tension: list,
    well_id: str,
    drop_parameters: dict,
    n_eq_points: int,
    containers: list,
    sensor_api: SensorAPI,
):
    if dynamic_surface_tension:
        save_dynamic_surface_tension(dynamic_surface_tension, well_id)
        st_eq = calculate_equillibrium_value(
            x=dynamic_surface_tension,
            n_eq_points=n_eq_points,
            column_index=1,
        )
        results = add_data_to_results(
            results=results,
            point_type=point_type,
            well_id=well_id,
            surface_tension_eq=st_eq,
            drop_parameters=drop_parameters,
            containers=containers,
            sensor_api=sensor_api,
        )
    else:
        print("Was not able to measure pendant drop!")
    return results


def save_dynamic_surface_tension(
    dynamic_surface_tension, well_id, experiment_dir="experiments"
):
    settings = load_settings()
    df = pd.DataFrame(
        dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
    )
    df.to_csv(
        f"{experiment_dir}/{settings['EXPERIMENT_NAME']}/data/{well_id}/dynamic_surface_tension.csv"
    )


def add_data_to_results(
    results: pd.DataFrame,
    point_type: str,
    well_id: str,
    surface_tension_eq: float,
    drop_parameters: dict,
    containers=list,
    sensor_api=SensorAPI,
):
    sensor_data = sensor_api.capture_sensor_data()
    container = containers[well_id]
    new_row = pd.DataFrame(
        {
            "well id": [well_id],
            "solution": [container.solution_name],
            "concentration": [container.concentration],
            "point type": [point_type],
            "surface tension eq. (mN/m)": [surface_tension_eq],
            "drop count": [drop_parameters["drop_count"]],
            "drop volume (uL)": [drop_parameters["drop_volume"]],
            "max measure time (s)": [drop_parameters["max_measure_time"]],
            "flow rate (uL/s)": [drop_parameters["flow_rate"]],
            "temperature (C)": [float(sensor_data["Temperature (C)"])],
            "humidity (%)": [float(sensor_data["Humidity (%)"])],
            "pressure (Pa)": [float(sensor_data["Pressure (Pa)"])],
        }
    )
    results = pd.concat([results, new_row], ignore_index=True)
    return results


def save_results(results: pd.DataFrame, experiments_dir="experiments"):
    settings = load_settings()
    file_name_results = f"{experiments_dir}/{settings['EXPERIMENT_NAME']}/results.csv"
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
    settings = load_settings()
    print(settings)
