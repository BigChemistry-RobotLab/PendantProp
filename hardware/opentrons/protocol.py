import warnings
import pandas as pd
import numpy as np

# Suppress the specific FutureWarning of Pandas
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)


from analysis.plots import Plotter
from analysis.active_learning import ActiveLearner
from analysis.models import szyszkowski_model
from analysis.utils import suggest_volume, volume_for_st
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.droplet_manager import DropletManager
from hardware.opentrons.formulater import Formulater
from hardware.opentrons.configuration import Configuration
from hardware.opentrons.containers import Container
from hardware.cameras import PendantDropCamera
from hardware.sensor.sensor_api import SensorAPI
from utils.load_save_functions import (
    load_settings,
    save_calibration_data,
    initialize_results,
    load_info,
    append_results,
    save_results,
)
from utils.logger import Logger
from utils.utils import play_sound, calculate_average_in_column


class Protocol:
    def __init__(
        self,
        opentrons_api: OpentronsAPI,
        sensor_api: SensorAPI,
        pendant_drop_camera: PendantDropCamera,
    ):
        self.settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )
        self.logger.info("Initialization starting...")
        self.opentrons_api = opentrons_api
        self.opentrons_api.initialise()
        self.sensor_api = sensor_api
        self.pendant_drop_camera = pendant_drop_camera
        self.config = Configuration(opentrons_api=opentrons_api)
        self.labware = self.config.load_labware()
        self.containers = self.config.load_containers()
        pipettes = self.config.load_pipettes()
        self.right_pipette = pipettes["right"]
        self.left_pipette = pipettes["left"]
        self.n_measurement_in_eq = 100  # number of data points which is averaged for equillibrium surface tension
        self.results = initialize_results()
        self.learner = ActiveLearner(
            model=szyszkowski_model,
            parameters=["cmc", "gamma_max", "Kad"],
        )
        self.plotter = Plotter()
        self.droplet_manager = DropletManager(
            left_pipette=self.left_pipette,
            containers=self.containers,
            pendant_drop_camera=self.pendant_drop_camera,
            opentrons_api=self.opentrons_api,
            plotter=self.plotter,
        )
        self.formulater = Formulater(
            left_pipette=self.left_pipette,
            right_pipette=self.right_pipette,
            containers=self.containers,
        )
        self.opentrons_api.home()
        self.logger.info("Initialization finished.")
        play_sound("Ready to measure.")

    def calibrate(self):
        self.logger.info("Starting calibration...")
        drop_parameters = {
            "drop_volume": 12,
            "max_measure_time": 60,
            "flow_rate": 1,
        }  # standard settings for calibration
        scale_t, drop_parameters = self.droplet_manager.measure_pendant_drop(
            source=self.containers["7A1"],
            drop_parameters=drop_parameters,
            calibrate=True,
        )
        save_calibration_data(scale_t)
        average_scale = calculate_average_in_column(x=scale_t, column_index=1)
        self.logger.info(f"Finished calibration, average scale is: {average_scale}")
        play_sound("Calibration done.")

    def measure_wells(self):
        self.logger.info("Starting measure wells protocol...")
        self.settings = load_settings()  # update settings
        well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
        wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)
        for i, well_id in enumerate(wells_ids):
            drop_parameters = {
                "drop_volume": float(well_info["drop volume (uL)"][i]),
                "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
                "flow_rate": float(self.settings["FLOW_RATE"]),
            }
            dynamic_surface_tension, drop_parameters = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id], drop_parameters=drop_parameters
                )
            )
            self.formulater.wash(repeat=1)
            self.results = append_results(
                results=self.results,
                point_type="None",
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id,
                drop_parameters=drop_parameters,
                n_eq_points=self.n_measurement_in_eq,
                containers=self.containers,
                sensor_api=self.sensor_api,
            )
            self.plotter.plot_results_well_id(df=self.results)
            save_results(results=self.results)
        self.left_pipette.return_needle()
        self.logger.info("Finished measure wells protocol.")
        play_sound("Moar measurement.")

    def characterize_surfactant(self):
        self.logger.info("Starting characterization protocol...")

        # general information
        self.settings = load_settings()  # update settings
        characterization_info = load_info(
            file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
        )
        explore_points = int(self.settings["EXPLORE_POINTS"])
        exploit_points = int(self.settings["EXPLOIT_POINTS"])

        for i, surfactant in enumerate(characterization_info["surfactant"]):
            row_id = characterization_info["row id"][i]
            measure_time = float(characterization_info["measure time"][i])
            self.formulater.serial_dilution(
                row_id=row_id,
                solution_name=surfactant,
                n_dilutions=explore_points,
                well_volume=float(self.settings["WELL_VOLUME"]),
            )
            for i in reversed(range(explore_points)): #reverse order to go from low to high concentration
                well_id_explore = f"{row_id}{i+1}"
                drop_volume_suggestion = suggest_volume(
                    results=self.results,
                    next_concentration=float(self.containers[well_id_explore].concentration),
                    solution_name=surfactant
                )
                drop_parameters = {
                    "drop_volume": drop_volume_suggestion,
                    "max_measure_time": measure_time,
                    "flow_rate": float(self.settings["FLOW_RATE"]),
                }
                dynamic_surface_tension, drop_parameters = (
                    self.droplet_manager.measure_pendant_drop(
                        source=self.containers[well_id_explore], drop_parameters=drop_parameters
                    )
                )
                self.results = append_results(
                    results=self.results,
                    point_type="explore",
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=well_id_explore,
                    drop_parameters=drop_parameters,
                    n_eq_points=self.n_measurement_in_eq,
                    containers=self.containers,
                    sensor_api=self.sensor_api,
                )
                save_results(self.results)
                self.plotter.plot_results_concentration(
                    df=self.results, solution_name=surfactant
                )

            self.formulater.wash(repeat=3, return_needle=True)

            for i in range(exploit_points):
                well_id_exploit = f"{row_id}{explore_points+i+1}"
                suggest_concentration, st_at_suggestion = self.learner.suggest(results=self.results, solution_name=surfactant)
                drop_volume_suggestion = volume_for_st(st_at_suggestion)
                self.formulater.formulate_exploit_point(
                    suggest_concentration=suggest_concentration,
                    solution_name=surfactant,
                    well_volume=float(self.settings["WELL_VOLUME"]),
                    well_id_exploit=well_id_exploit,
                )
                drop_parameters = {
                    "drop_volume": drop_volume_suggestion,
                    "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
                    "flow_rate": float(self.settings["FLOW_RATE"]),
                }
                dynamic_surface_tension, drop_parameters = (
                    self.droplet_manager.measure_pendant_drop(
                        source=self.containers[well_id_exploit], drop_parameters=drop_parameters
                    )
                )
                self.results = append_results(
                    results=self.results,
                    point_type="exploit",
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=well_id_exploit,
                    drop_parameters=drop_parameters,
                    n_eq_points=self.n_measurement_in_eq,
                    containers=self.containers,
                    sensor_api=self.sensor_api,
                )
                save_results(self.results)
                self.plotter.plot_results_concentration(
                    df=self.results, solution_name=surfactant
                )
                self.formulater.wash(repeat=3, return_needle=True)

        self.logger.info("Finished characterization protocol.")
        play_sound("DATA DATA.")

    def measure_same_well(self, well_id: str, repeat: int = 3):
        drop_parameters = {"drop_volume": 6, "max_measure_time": 60, "flow_rate": 1}
        for i in range(repeat):
            dynamic_surface_tension, drop_parameters = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id], drop_parameters=drop_parameters
                )
            )
            # self.left_pipette.wash()
            df = pd.DataFrame(
                dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
            )
            df.to_csv(
                f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/dynamic_surface_tension_{i}.csv"
            )
        if self.left_pipette.has_needle:
            self.left_pipette.return_needle()

    def measure_same_well_cali(self, well_id: str, repeat: int = 3):
        drop_parameters = {"drop_volume": 11, "max_measure_time": 30, "flow_rate": 1}
        for i in range(repeat):
            scale = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id], drop_parameters=drop_parameters, calibrate=True
                )
            )
            # self.left_pipette.wash()
            df = pd.DataFrame(
                scale, columns=["time (s)", "scale"]
            )
            df.to_csv(
                f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/scale{i}.csv"
            )

        if self.left_pipette.has_needle:
            self.left_pipette.return_needle()

    def measure_plate(self, well_volume: float, solution_name: str, plate_location: int):
        # TODO save results correctly!
        self.logger.info("Starting measure whole plate protocol...")
        self.droplet_manager.set_max_retries = 1
        print(self.droplet_manager.MAX_RETRIES)
        self.settings = load_settings()  # update settings
        well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
        wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)
        # self.formulater.fill_plate(well_volume=well_volume, solution_name=solution_name, plate_location=plate_location)

        for i, well_id in enumerate(wells_ids):
            drop_parameters = {
                "drop_volume": float(well_info["drop volume (uL)"][i]),
                "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
                "flow_rate": float(well_info["flow rate (uL/s)"][i]),
            }
            dynamic_surface_tension, drop_parameters = self.droplet_manager.measure_pendant_drop(
                source=self.containers[well_id], drop_parameters=drop_parameters
            )
            self.results = append_results(
                results=self.results,
                point_type="None",
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id,
                drop_parameters=drop_parameters,
                n_eq_points=self.n_measurement_in_eq,
                containers=self.containers,
                sensor_api=self.sensor_api,
            )
            save_results(self.results)
            self.plotter.plot_results_well_id(df=self.results)

        self.logger.info("Done measuring plate.")
        play_sound("DATA DATA.")
