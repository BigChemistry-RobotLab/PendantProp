import warnings
import pandas as pd

# Suppress the specific FutureWarning of Pandas
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)


from analysis.plots import Plotter
from analysis.utils import suggest_volume
from hardware.opentrons.http_communications import OpentronsAPI
from hardware.opentrons.droplet_manager import DropletManager
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
        self.config = Configuration(http_api=opentrons_api)
        self.labware = self.config.load_labware()
        self.containers = self.config.load_containers()
        pipettes = self.config.load_pipettes()
        self.right_pipette = pipettes["right"]
        self.left_pipette = pipettes["left"]
        self.n_measurement_in_eq = 100  # number of data points which is averaged for equillibrium surface tension
        self.results = initialize_results()
        self.plotter = Plotter()
        self.droplet_manager = DropletManager(
            left_pipette=self.left_pipette,
            containers=self.containers,
            pendant_drop_camera=self.pendant_drop_camera,
            opentrons_api=self.opentrons_api,
            plotter=self.plotter,
            logger=self.logger,
        )
        self.opentrons_api.home()
        self.logger.info("Initialization finished.")
        play_sound("Irem & Wojtek.")

    def calibrate(self):
        self.logger.info("Starting calibration...")
        drop_parameters = {
            "drop_volume": 12,
            "max_measure_time": 60,
            "flow_rate": 1,
        }  # standard settings for calibration
        scale_t = self.droplet_manager.measure_pendant_drop(
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
                "flow_rate": float(well_info["flow rate (uL/s)"][i]),
            }
            dynamic_surface_tension, drop_parameters = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id], drop_parameters=drop_parameters
                )
            )
            self.results = append_results(
                results=self.results,
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id,
                drop_parameters=drop_parameters,
                n_eq_points=self.n_measurement_in_eq,
                containers=self.containers,
                sensor_api=self.sensor_api,
            )
            self.plotter.plot_results_well_id(df=self.results)
            save_results(results=self.results)

        self.logger.info("Finished measure wells protocol.")
        play_sound("DATA DATA.")

    def characterize_surfactant(self):
        self.logger.info("Starting characterization protocol...")
        self.settings = load_settings()  # update settings
        characterization_info = load_info(
            file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
        )
        explore_points = int(self.settings["EXPLORE_POINTS"])

        for i, surfactant in enumerate(characterization_info["surfactant"]):
            row_id = characterization_info["row id"][i]
            self.right_pipette.serial_dilution(
                row_id=row_id,
                solution_name=surfactant,
                n_dilutions=explore_points,
                well_volume=float(self.settings["WELL_VOLUME"]),
            )
            for i in range(explore_points):
                well_id = f"{row_id}{i+1}"
                drop_volume_suggestion = suggest_volume(
                    results=self.results,
                    next_concentration=float(self.containers[well_id].concentration),
                )
                drop_parameters = {
                    "drop_volume": drop_volume_suggestion,
                    "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
                    "flow_rate": float(self.settings["FLOW_RATE"]),
                }
                dynamic_surface_tension, drop_parameters = (
                    self.droplet_manager.measure_pendant_drop(
                        source=self.containers[well_id], drop_parameters=drop_parameters
                    )
                )
                self.results = append_results(
                    results=self.results,
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=well_id,
                    drop_parameters=drop_parameters,
                    n_eq_points=self.n_measurement_in_eq,
                    containers=self.containers,
                    sensor_api=self.sensor_api,
                )
                save_results(self.results)
                self.plotter.plot_results_concentration(
                    df=self.results, solution_name=surfactant
                )

        self.logger.info("Finished characterization protocol.")
        play_sound("DATA DATA.")

    def measure_same_well(self, well_id: str, repeat: int = 3):
        drop_parameters = {"drop_volume": 12, "max_measure_time": 30, "flow_rate": 1}
        for i in range(repeat):
            dynamic_surface_tension, drop_parameters = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id], drop_parameters=drop_parameters
                )
            )
            df = pd.DataFrame(
                dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
            )
            df.to_csv(
                f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/dynamic_surface_tension_{i}.csv"
            )

    def measure_plate(self, well_volume: float, solution_name: str, plate_location: int):
        self.logger.info("Starting measure whole plate protocol...")
        self.settings = load_settings()  # update settings
        well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
        wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)
        self.logger.info("Filling plate with solution.")
        self.right_pipette.fill_plate(well_volume=well_volume, solution_name=solution_name, plate_location=plate_location)
        for i, well_id in enumerate(wells_ids):
            self.left_pipette.measure_pendant_drop(
                source=self.containers[well_id],
                drop_volume=float(well_info["drop volume (uL)"][i]),
                delay=60,
                flow_rate=float(well_info["flow rate (uL/s)"][i]),
                pendant_drop_camera=self.pendant_drop_camera
            )
        
