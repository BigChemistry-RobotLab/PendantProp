# Imports

## Packages
import warnings
import pandas as pd
import numpy as np

## Custom code
from analysis.plots import Plotter
from analysis.active_learning import ActiveLearner
from analysis.models import szyszkowski_model
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.droplet_manager import DropletManager
from hardware.opentrons.formulater import Formulater
from hardware.opentrons.configuration import Configuration
from hardware.opentrons.containers import Container
from hardware.cameras.pendant_drop_camera import PendantDropCamera
from hardware.sensor.sensor_api import SensorAPI
from utils.load_save_functions import (
    load_settings,
    initialize_results,
    load_info,
    append_results,
    save_results,
)
from utils.logger import Logger

## Surpress warning pandas
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)

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
            labware=self.labware,
        )
        self.opentrons_api.home()
        self.logger.info("Initialization finished.\n\n\n")

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
        container = self.containers[well_id]
        dynamic_surface_tension, drop_volume, drop_count = (
            self.droplet_manager.measure_pendant_drop(
                source=container,
                max_measure_time=float(self.settings["EQUILIBRATION_TIME"]),
            )
        )

        drop_parameters = self._create_drop_parameters(
            drop_volume=drop_volume,
            measure_time=float(self.settings["EQUILIBRATION_TIME"]),
            drop_count=drop_count,
        )

        # self.formulater.wash(repeat=self.settings["WASH_REPEATS"]) #! for now

        self._append_save_plot_results(
            point_type="None",
            dynamic_surface_tension=dynamic_surface_tension,
            container=container,
            drop_parameters=drop_parameters,
            solution_name=container.solution_name,
            plot_type="wells",
        )

        self.logger.info(f"End of pendant drop measurement of {well_id}.\n")

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

        for i, surfactant in enumerate(characterization_info["surfactant"]):
            self.logger.info(f"Start characterization of {surfactant}.\n\n")
            row_id = characterization_info["row id"][i]
            measure_time = float(characterization_info["measure time"][i])

            # Perform serial dilution
            self.formulater.serial_dilution(
                row_id=row_id,
                solution_name=surfactant,
                n_dilutions=explore_points,
                well_volume=float(self.settings["WELL_VOLUME"]),
            )

            # Explore phase
            self._perform_explore_phase(
                surfactant=surfactant,
                row_id=row_id,
                explore_points=explore_points,
                measure_time=measure_time,
            )

            # Exploit phase
            self._perform_exploit_phase(
                surfactant=surfactant,
                row_id=row_id,
                explore_points=explore_points,
                exploit_points=exploit_points,
                measure_time=measure_time,
            )

        self.logger.info("Finished characterization protocol.\n\n\n")

    def _perform_explore_phase(
        self, surfactant: str, row_id: str, explore_points: int, measure_time: float
    ) -> None:
        """
        Perform the explore phase for a given surfactant.
        """
        self.logger.info(f"Start explore phase for surfactant {surfactant}.\n")
        for i in reversed(
            range(explore_points)
        ):  # Reverse order for low to high concentration
            well_id_explore = f"{row_id}{i+1}"
            container = self.containers[well_id_explore]
            self.logger.info(
                f"Start pendant drop measurement of {container.WELL_ID}, containing {container.concentration} mM {container.solution_name}.\n"
            )
            dynamic_surface_tension, drop_volume, drop_count = (
                self.droplet_manager.measure_pendant_drop(
                    source=container, max_measure_time=measure_time
                )
            )
            drop_parameters = self._create_drop_parameters(
                drop_volume=drop_volume,
                measure_time=measure_time,
                drop_count=drop_count,
            )
            self._append_save_plot_results(
                point_type="explore",
                dynamic_surface_tension=dynamic_surface_tension,
                container=container,
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
        measure_time: float,
    ) -> None:
        """
        Perform the exploit phase for a given surfactant.
        """
        self.logger.info(f"Start exploit phase for surfactant {surfactant}.\n")
        for i in range(exploit_points):
            well_id_exploit = f"{row_id}{explore_points+i+1}"
            container = self.containers[well_id_exploit]
            suggest_concentration, st_at_suggestion = self.learner.suggest(
                results=self.results, solution_name=surfactant
            )
            self.formulater.wash(return_needle=True)
            if suggest_concentration is None:
                self.logger.warning(
                    f"Suggestion for {surfactant} is None. Skipping this point.\n"
                )
            else:
                self.formulater.formulate_exploit_point(
                    suggest_concentration=suggest_concentration,
                    solution_name=surfactant,
                    well_volume=float(self.settings["WELL_VOLUME"]),
                    well_id_exploit=well_id_exploit,
                )
                self.logger.info(
                    f"Start pendant drop measurement of {well_id_exploit}, containing {self.containers[well_id_exploit].concentration} mM {surfactant}.\n"
                )
                # self.left_pipette.mixing(container=self.containers[well_id_exploit], mix=("before", 20, 5))
                dynamic_surface_tension, drop_volume, drop_count = (
                    self.droplet_manager.measure_pendant_drop(
                        source=container,
                        max_measure_time=measure_time,
                    )
                )
                drop_parameters = self._create_drop_parameters(
                    drop_volume=drop_volume,
                    measure_time=measure_time,
                    drop_count=drop_count,
                )
                self._append_save_plot_results(
                    point_type="exploit",
                    dynamic_surface_tension=dynamic_surface_tension,
                    container=container,
                    drop_parameters=drop_parameters,
                    solution_name=surfactant,
                    plot_type="concentrations",
                )
        self.formulater.wash(return_needle=True)

    def _create_drop_parameters(
        self, drop_volume: float, measure_time: float, drop_count: int
    ) -> dict:
        """
        Create a dictionary of drop parameters.
        """
        return {
            "drop_volume": drop_volume,
            "max_measure_time": measure_time,
            "flow_rate": float(self.settings["FLOW_RATE"]),
            "drop_count": drop_count,
        }

    def _append_save_plot_results(
        self,
        point_type: str,
        dynamic_surface_tension: list,
        container: Container,
        drop_parameters: dict,
        solution_name: str,
        plot_type: str
    ) -> None:
        """
        Append results, save them, and plot the results. Plots based on the specified plot type (either well or concentration)
        """
        self.results = append_results(
            results=self.results,
            point_type=point_type,
            dynamic_surface_tension=dynamic_surface_tension,
            container=container,
            drop_parameters=drop_parameters,
            n_eq_points=self.n_measurement_in_eq,
            sensor_api=self.sensor_api,
        )
        save_results(self.results)
        if plot_type == "wells":
            self.plotter.plot_results_well_id(
                df=self.results
            )
        elif plot_type == "concentrations":
            self.plotter.plot_results_concentration(
                df=self.results, solution_name=solution_name
            )
