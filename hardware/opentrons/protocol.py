import warnings
import pandas as pd
import numpy as np
import time

# Suppress the specific FutureWarning of Pandas
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)


from analysis.plots import Plotter
from analysis.active_learning import ActiveLearner
from analysis.models import szyszkowski_model
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.droplet_manager import DropletManager
from hardware.opentrons.formulater import Formulater
from hardware.opentrons.configuration import Configuration
from hardware.opentrons.containers import Container
from hardware.cameras import PendantDropCamera
from hardware.sensor.sensor_api import SensorAPI
from hardware.opentrons.gridsearch import GridSearch
# from hardware.opentrons.gridsearch import PointSelection
from utils.search_containers import find_container, find_stock_tubes
from utils.load_save_functions import (
    load_settings,
    save_calibration_data,
    initialize_results,
    load_info,
    append_results,
    append_results_binary,
    save_results,
)
from utils.logger import Logger
from utils.utils import play_sound, calculate_average_in_column
import math


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
        self.gridsearch = GridSearch(opentrons_api=opentrons_api)
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
            labware=self.labware,
            opentrons_api=self.opentrons_api,
        )
        # self.ps = PointSelection(opentrons_api=self.opentrons_api)
        self.opentrons_api.home()
        self.batch_size = self.settings["BATCH_SIZE"]
        self.repeat = self.settings["WASH_REPEATS"]

        self.logger.info("Initialization finished.\n\n\n")
        # play_sound("Lets go.")

    def measure_wells(self) -> None:
        """
        Perform pendant drop measurements for all wells specified in the settings.
        """
        self.logger.info("Starting measure wells protocol...\n\n\n")
        self.settings = load_settings()  # Update settings
        well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
        wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)

        for well_id in wells_ids:
            self._measure_single_well(well_id)

        self.left_pipette.return_needle()
        self.logger.info("Finished measure wells protocol.\n\n\n")

    def _measure_single_well(self, well_id: str) -> None:
        """
        Measure pendant drop for a single well and process the results.
        """
        self.logger.info(f"Start pendant drop measurement of {well_id}.\n")
        dynamic_surface_tension, drop_volume, drop_count = (
            self.droplet_manager.measure_pendant_drop(
                source=self.containers[well_id],
                # max_measure_time=float(self.settings["EQUILIBRATION_TIME"]),
            )
        )

        drop_parameters = self._create_drop_parameters(
            drop_volume=drop_volume,
            measure_time=float(self.settings["EQUILIBRATION_TIME"]),
            drop_count=drop_count,
        )

        self.formulater.wash(repeat=self.settings["WASH_REPEATS"])

        self._append_and_save_results(
            point_type="None",
            dynamic_surface_tension=dynamic_surface_tension,
            well_id=well_id,
            drop_parameters=drop_parameters,
            solution_name=self.containers[well_id].solution_name,
            plot_type="wells",
        )

        self.logger.info(f"End of pendant drop measurement of {well_id}.\n")

    def prepare_grid_scan(
        self,
        solution1,
        solution2,
        sol1_max_conc,
        sol2_max_conc,
        x_dil_range,
        y_dil_range,
        samples,
    ):
        self.logger.info(
            "Preparing grid scan...\n"
        )  # Introduce functionality to set bounds instead of range? so 10^-5 -> 10^1? give choice
        success = False
        while not success:
            try:
                df = self.gridsearch.generate_grid(
                    solution1=solution1,
                    solution2=solution2,
                    x_max_conc=sol1_max_conc,
                    y_max_conc=sol2_max_conc,
                    x_dilution=x_dil_range,
                    y_dilution=y_dil_range,
                    n_samples=samples,
                    plot=False,
                )
                grid, columns = self.gridsearch.process_df(df=df)

                if (solution1 or solution2) not in columns:
                    self.logger.error("Surfactants not found! Quitting...\n\n\n")
                    self.logger.info(
                        "Needs: ", solution1, " ", solution2, "\n Found: ", columns
                    )
                    return

                dilution_factor, bdil1, bdil2, _ = (
                    self.gridsearch.find_optimal_dilution_setup(
                        stock1=sol1_max_conc, stock2=sol2_max_conc, grid=grid
                    )
                )
                success = True
            except Exception as e:
                self.logger.info(f"Retrying, error: {e}.")
        tubes_req = self.gridsearch.count_total_dilutions(
            bdil1, bdil2
        )  # Add check for empty tubes after

        amount_empty = len(find_container(containers=self.containers, type="tube 15"))
        if tubes_req > amount_empty:
            self.logger.error("Not enough tubes to create dilutions! Quitting...\n\n\n")
            return

        if "form_scheme" not in locals():
            form_scheme = pd.DataFrame(
                columns=[
                    "well_id",
                    "dil1",
                    "vol1",
                    "conc1",
                    "dil2",
                    "vol2",
                    "conc2",
                    "vol_water",
                ]
            )
        for c1, c2 in grid:
            form_scheme = self.gridsearch.generate_feasible_combinations_for_sample(
                form_scheme, c1=c1, c2=c2, d1_list=bdil1, d2_list=bdil2
            )
        empty_wells = find_container(
            containers=self.containers, type="Plate well", amount=len(form_scheme)
        )
        form_scheme["well_id"] = empty_wells
        form_scheme.to_csv(
            f"experiments/{self.settings['EXPERIMENT_NAME']}/formulation_scheme.csv"
        )
        return form_scheme, tubes_req, dilution_factor

    def formulate_gridscan(
        self,
        tubes_req,
        form_scheme: pd.DataFrame,
        solution1,
        solution2,
        dilution_factor,
    ):
        useable_tubes = find_container(
            containers=self.containers, type="tube 15", amount=tubes_req
        )
        sum_vol1_per_conc1 = form_scheme.groupby("dil1")["vol1"].sum().reset_index()
        sum_vol2_per_conc2 = form_scheme.groupby("dil2")["vol2"].sum().reset_index()
        self.formulater.formulate_dilution_tube(
            dilution_df=sum_vol1_per_conc1,
            solution=solution1,
            dilution_factor=dilution_factor,
        )
        self.formulater.formulate_dilution_tube(
            dilution_df=sum_vol2_per_conc2,
            solution=solution2,
            dilution_factor=dilution_factor,
        )
  
    def _perform_grid_measurement(
        self, solutions, form_scheme, concentrations, max_measure_time, well_volume
        ):
        batches = [
            form_scheme.iloc[i : i + self.batch_size]
            for i in range(0, len(form_scheme), self.batch_size)
        ]

        for idx, batch_df in enumerate(batches, start=1):
            print(f"Processing batch {idx} with {len(batch_df)} wells")
            print(batch_df)
            print("-" * 40)

            self.formulater.formulate_batches(
                batch_df=batch_df,
                well_volume=well_volume
            )

            for _, row in batch_df.iterrows():
                well_id = row["well_id"]
                self.logger.info(
                    f"Start pendant drop measurement of {well_id}, "
                    f"containing {concentrations[0]} mM {solutions[0]} and "
                    f"{concentrations[1]} mM {solutions[1]}."
                )

                (   dynamic_surface_tension,
                    drop_volume,
                    drop_count,
                    measure_time,
                    wt_number,
                ) = self.droplet_manager.measure_pendant_drop(
                    source=well_id,
                    max_measure_time=max_measure_time
                )

                drop_parameters = self._create_drop_parameters(
                    drop_volume=drop_volume,
                    measure_time=measure_time,
                    drop_count=drop_count,
                    wt_number=wt_number,
                )

                self._append_and_save_results_binary(
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=well_id,
                    drop_parameters=drop_parameters,
                    solutions=solutions,
                    concentrations=concentrations,
                )
    
    # def active_learning_loop()

    def characterize_surfactant(self) -> None:
        """
        Characterize surfactants by performing explore and exploit phases.
        """
        self.logger.info("Starting characterization protocol...\n\n\n")
        self.settings = load_settings()  # Update settings
        characterization_info = load_info(
            file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
        )
        explore_points = int(self.settings["EXPLORE_POINTS"])
        exploit_points = int(self.settings["EXPLOIT_POINTS"])

        self._check_needle_position()
        self.formulater.wash()
        for i, surfactant in enumerate(characterization_info["surfactant"]):
            self.logger.info(f"Start characterization of {surfactant}.\n\n")
            row_id = characterization_info["row id"][i]
            max_measure_time = float(characterization_info["measure time"][i])

            # Perform serial dilution
            self.formulater.serial_dilution(
                row_id=row_id,
                solution_name=surfactant,
                n_dilutions=explore_points,
                well_volume=float(self.settings["WELL_VOLUME"]),
                dilution_factor=float(self.settings["DILUTION_FACTOR"]),
            )

            # Explore phase
            self._perform_explore_phase(
                surfactant=surfactant,
                row_id=row_id,
                explore_points=explore_points,
                max_measure_time=max_measure_time,
            )

            # Exploit phase
            self._perform_exploit_phase(
                surfactant=surfactant,
                row_id=row_id,
                explore_points=explore_points,
                exploit_points=exploit_points,
                max_measure_time=max_measure_time,
            )
        if self.left_pipette.has_needle:
            self.left_pipette.return_needle()
        self.logger.info("Finished characterization protocol.\n\n\n")
        play_sound("DATA DATA.")

    def _perform_explore_phase(
        self, surfactant: str, row_id: str, explore_points: int, max_measure_time: float
    ) -> None:
        """
        Perform the explore phase for a given surfactant.
        """
        self.logger.info(f"Start explore phase for surfactant {surfactant}.\n")
        for i in reversed(
            range(explore_points)
        ):  # Reverse order for low to high concentration
            well_id_explore = f"{row_id}{i+1}"
            source_well = self.containers[well_id_explore]
            self.logger.info(
                f"Start pendant drop measurement of {source_well.WELL_ID}, containing {source_well.concentration} mM {source_well.solution_name}.\n"
            )
            (
                dynamic_surface_tension,
                drop_volume,
                drop_count,
                measure_time,
                wt_number,
            ) = self.droplet_manager.measure_pendant_drop(
                source=source_well, max_measure_time=max_measure_time
            )
            drop_parameters = self._create_drop_parameters(
                drop_volume=drop_volume,
                measure_time=measure_time,
                drop_count=drop_count,
                wt_number=wt_number,
            )
            self._append_and_save_results(
                point_type="explore",
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id_explore,
                drop_parameters=drop_parameters,
                solution_name=surfactant,
                plot_type="concentrations",
            )

    def _perform_exploit_phase(
        self,
        surfactant: str,
        row_id: str,
        explore_points: int,
        exploit_points: int,
        max_measure_time: float,
    ) -> None:
        """
        Perform the exploit phase for a given surfactant.
        """
        self.logger.info(f"Start exploit phase for surfactant {surfactant}.\n")
        for i in range(exploit_points):
            well_id_exploit = f"{row_id}{explore_points+i+1}"

            suggest_concentration, st_at_suggestion = self.learner.suggest(
                results=self.results, solution_name=surfactant
            )
            self.formulater.wash(return_needle=False, repeat=self.repeat)
            self.formulater.formulate_exploit_point(
                suggest_concentration=suggest_concentration,
                solution_name=surfactant,
                well_volume=float(self.settings["WELL_VOLUME"]),
                well_id_exploit=well_id_exploit,
            )
            self.logger.info(
                f"Start pendant drop measurement of {well_id_exploit}, containing {self.containers[well_id_exploit].concentration} mM {surfactant}.\n"
            )
            self.left_pipette.mixing(
                container=self.containers[well_id_exploit], mix=("before", 20, 5)
            )
            dynamic_surface_tension, drop_volume, drop_count = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id_exploit],
                    max_measure_time=max_measure_time,
                )
            )
            drop_parameters = self._create_drop_parameters(
                drop_volume=drop_volume,
                measure_time=max_measure_time,
                drop_count=drop_count,
            )
            self._append_and_save_results(
                point_type="exploit",
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id_exploit,
                drop_parameters=drop_parameters,
                solution_name=surfactant,
                plot_type="concentrations",
            )
        self.formulater.wash(return_needle=False, repeat=self.repeat)

    def _create_drop_parameters(
        self,
        drop_volume: float,
        measure_time: float,
        drop_count: int,
        wt_number: float,
    ) -> dict:
        """
        Create a dictionary of drop parameters.
        """
        return {
            "drop_volume": drop_volume,
            "measure_time": measure_time,
            "drop_count": drop_count,
            "wt_number": wt_number,
        }

    def _append_and_save_results(
        self,
        point_type: str,
        dynamic_surface_tension: list,
        well_id: str,
        drop_parameters: dict,
        solution_name: str,
        plot_type: str,
    ) -> None:
        """
        Append results, save them, and plot the results. Plots based on the specified plot type (either well or concentration)
        """
        self.results = append_results(
            results=self.results,
            point_type=point_type,
            dynamic_surface_tension=dynamic_surface_tension,
            well_id=well_id,
            drop_parameters=drop_parameters,
            n_eq_points=self.n_measurement_in_eq,
            containers=self.containers,
            sensor_api=self.sensor_api,
        )
        save_results(self.results)
        if plot_type == "wells":
            self.plotter.plot_results_well_id(df=self.results)
        elif plot_type == "concentrations":
            self.plotter.plot_results_concentration(
                df=self.results, solution_name=solution_name
            )

    def _append_and_save_results_binary(
        self,
        dynamic_surface_tension: list,
        well_id: str,
        drop_parameters: dict,
        solutions: list,
        concentrations: list,
    ) -> None:
        # Append base results
        self.results = append_results_binary(
            results=self.results,
            dynamic_surface_tension=dynamic_surface_tension,
            well_id=well_id,
            drop_parameters=drop_parameters,
            n_eq_points=self.n_measurement_in_eq,
            solutions=solutions,
            concentrations=concentrations,
            sensor_api=self.sensor_api,
        )
        # Save updated results
        save_results(self.results)

        # Plotting logic
        # self.plotter.plot_results_well_id(df=self.results) TODO Probably broken, automatic heatmap?, only after multiple points?

    def _check_needle_position(self):
        self.left_pipette.pick_up_needle()
        self.left_pipette.move_to_well(container=self.containers["drop_stage"])

        time.sleep(30)
        # play_sound("Remove hands from the deck within 5 seconds.")
        play_sound("Beep")
        time.sleep(5)

        self.left_pipette.move_to_well(
            container=self.containers["drop_stage"],
            depth_offset=-10,
        )


### legacy ###

# def random_measurement(self) -> None:
# """
# Perform pendant drop measurements for all wells specified in the settings.
# """
# self.logger.info("Starting measure wells protocol...\n\n\n")
# self.settings = load_settings()
# well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
# wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)

# try:
#     df = pd.read_csv("experiments/csv_file.csv")
# except Exception as e:
#     self.logger.error(f"Error loading formulation CSV: {e}")
#     return

# if len(df) != len(wells_ids):
#     self.logger.error(f"Mismatch: {len(wells_ids)} wells but {len(df)} formulation rows.")
#     return

# for i, well_id in enumerate(wells_ids):
#     row = df.iloc[i]
#     conc_dict = row.dropna().to_dict()
#     print(f"Row {i}: {conc_dict}")
# for i, well_id in enumerate(wells_ids):
#     row = df.iloc[i]
#     conc_dict = row.dropna().to_dict()
#     # self.formulater.formulate_random_single_well(well_id=well_id, concentrations=conc_dict)   single well

#     self._measure_single_well(well_id)

# def calibrate(self):
# self.logger.info("Starting calibration...")
# drop_parameters = {
# "drop_volume": 12,
# "max_measure_time": 60,
# "flow_rate": 1,
# }  # standard settings for calibration
# scale_t, drop_parameters = self.droplet_manager.measure_pendant_drop(
# source=self.containers["7A1"],
# drop_parameters=drop_parameters,
# calibrate=True,
# )
# save_calibration_data(scale_t)
# average_scale = calculate_average_in_column(x=scale_t, column_index=1)
# self.logger.info(f"Finished calibration, average scale is: {average_scale}")
# play_sound("Calibration done.")

# def characterize_surfactant_old(self):
#     self.logger.info("Starting characterization protocol...")

#     # general information
#     self.settings = load_settings()  # update settings
#     characterization_info = load_info(
#         file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
#     )
#     explore_points = int(self.settings["EXPLORE_POINTS"])
#     exploit_points = int(self.settings["EXPLOIT_POINTS"])

#     for i, surfactant in enumerate(characterization_info["surfactant"]):
#         row_id = characterization_info["row id"][i]
#         measure_time = float(characterization_info["measure time"][i])
#         self.formulater.serial_dilution(
#             row_id=row_id,
#             solution_name=surfactant,
#             n_dilutions=explore_points,
#             well_volume=float(self.settings["WELL_VOLUME"]),
#         )
#         for i in reversed(
#             range(explore_points)
#         ):  # reverse order to go from low to high concentration
#             well_id_explore = f"{row_id}{i+1}"
#             drop_volume_suggestion = suggest_volume(
#                 results=self.results,
#                 next_concentration=float(
#                     self.containers[well_id_explore].concentration
#                 ),
#                 solution_name=surfactant,
#             )
#             drop_parameters = {
#                 "drop_volume": drop_volume_suggestion,
#                 "max_measure_time": measure_time,
#                 "flow_rate": float(self.settings["FLOW_RATE"]),
#             }
#             dynamic_surface_tension, drop_parameters = (
#                 self.droplet_manager.measure_pendant_drop(
#                     source=self.containers[well_id_explore],
#                     drop_parameters=drop_parameters,
#                 )
#             )
#             self.results = append_results(
#                 results=self.results,
#                 point_type="explore",
#                 dynamic_surface_tension=dynamic_surface_tension,
#                 well_id=well_id_explore,
#                 drop_parameters=drop_parameters,
#                 n_eq_points=self.n_measurement_in_eq,
#                 containers=self.containers,
#                 sensor_api=self.sensor_api,
#             )
#             save_results(self.results)
#             self.plotter.plot_results_concentration(
#                 df=self.results, solution_name=surfactant
#             )

#         self.formulater.wash(repeat=3, return_needle=True)

#         for i in range(exploit_points):
#             well_id_exploit = f"{row_id}{explore_points+i+1}"
#             suggest_concentration, st_at_suggestion = self.learner.suggest(
#                 results=self.results, solution_name=surfactant
#             )
#             drop_volume_suggestion = volume_for_st(st_at_suggestion)
#             self.formulater.formulate_exploit_point(
#                 suggest_concentration=suggest_concentration,
#                 solution_name=surfactant,
#                 well_volume=float(self.settings["WELL_VOLUME"]),
#                 well_id_exploit=well_id_exploit,
#             )
#             drop_parameters = {
#                 "drop_volume": drop_volume_suggestion,
#                 "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
#                 "flow_rate": float(self.settings["FLOW_RATE"]),
#             }
#             dynamic_surface_tension, drop_parameters = (
#                 self.droplet_manager.measure_pendant_drop(
#                     source=self.containers[well_id_exploit],
#                     drop_parameters=drop_parameters,
#                 )
#             )
#             self.results = append_results(
#                 results=self.results,
#                 point_type="exploit",
#                 dynamic_surface_tension=dynamic_surface_tension,
#                 well_id=well_id_exploit,
#                 drop_parameters=drop_parameters,
#                 n_eq_points=self.n_measurement_in_eq,
#                 containers=self.containers,
#                 sensor_api=self.sensor_api,
#             )
#             save_results(self.results)
#             self.plotter.plot_results_concentration(
#                 df=self.results, solution_name=surfactant
#             )
#             self.formulater.wash(repeat=3, return_needle=True)

#     self.logger.info("Finished characterization protocol.")
#     play_sound("DATA DATA.")

# def measure_same_well(self, well_id: str, repeat: int = 3):
#     drop_parameters = {"drop_volume": 6, "max_measure_time": 60, "flow_rate": 1}
#     for i in range(repeat):
#         dynamic_surface_tension, drop_parameters = (
#             self.droplet_manager.measure_pendant_drop(
#                 source=self.containers[well_id], drop_parameters=drop_parameters
#             )
#         )
#         # self.left_pipette.wash()
#         df = pd.DataFrame(
#             dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
#         )
#         df.to_csv(
#             f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/dynamic_surface_tension_{i}.csv"
#         )
#     if self.left_pipette.has_needle:
#         self.left_pipette.return_needle()

# def measure_same_well_cali(self, well_id: str, repeat: int = 3):
#     drop_parameters = {"drop_volume": 11, "max_measure_time": 30, "flow_rate": 1}
#     for i in range(repeat):
#         scale = (
#             self.droplet_manager.measure_pendant_drop(
#                 source=self.containers[well_id], drop_parameters=drop_parameters, calibrate=True
#             )
#         )
#         # self.left_pipette.wash()
#         df = pd.DataFrame(
#             scale, columns=["time (s)", "scale"]
#         )
#         df.to_csv(
#             f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/scale{i}.csv"
#         )

#     if self.left_pipette.has_needle:
#         self.left_pipette.return_needle()

# def measure_plate(self, well_volume: float, solution_name: str, plate_location: int):
# # TODO save results correctly!
# self.logger.info("Starting measure whole plate protocol...")
# self.droplet_manager.set_max_retries = 1
# print(self.droplet_manager.MAX_RETRIES)
# self.settings = load_settings()  # update settings
# well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
# wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)
# # self.formulater.fill_plate(well_volume=well_volume, solution_name=solution_name, plate_location=plate_location)

# for i, well_id in enumerate(wells_ids):
#     drop_parameters = {
#         "drop_volume": float(well_info["drop volume (uL)"][i]),
#         "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
#         "flow_rate": float(well_info["flow rate (uL/s)"][i]),
#     }
#     dynamic_surface_tension, drop_parameters = self.droplet_manager.measure_pendant_drop(
#         source=self.containers[well_id], drop_parameters=drop_parameters
#     )
#     self.results = append_results(
#         results=self.results,
#         point_type="None",
#         dynamic_surface_tension=dynamic_surface_tension,
#         well_id=well_id,
#         drop_parameters=drop_parameters,
#         n_eq_points=self.n_measurement_in_eq,
#         containers=self.containers,
#         sensor_api=self.sensor_api,
#     )
#     save_results(self.results)
#     self.plotter.plot_results_well_id(df=self.results)

# self.logger.info("Done measuring plate.")
# play_sound("DATA DATA.")
