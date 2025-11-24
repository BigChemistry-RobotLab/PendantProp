import pandas as pd
import warnings

from opentrons_api.load_save_functions import load_settings
from opentrons_api.logger import Logger
from opentrons_api.containers import Container
from pendantprop.hardware.opentrons.config import Config
from pendantprop.hardware.droplet_management import DropletManager
from pendantprop.analysis.plots import Plotter
from pendantprop.hardware.sensor_api import SensorAPI
from pendantprop.utils.load_save_functions import (
    initialize_results,
    append_results,
    save_results,
)

## Surpress warning pandas
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)

class Protocol:
    def __init__(self, settings: dict = None):
        if settings is None:
            self.settings = load_settings(file_path="config/settings.json")
        else:
            self.settings = settings
        self.file_settings = self.settings["file_settings"]
        self.config = Config(settings=self.settings)
        self.left_pipette, self.right_pipette, self.containers = self.config.load_all()
        self.plotter = Plotter(settings=self.settings)
        self.sensor_api = SensorAPI(settings=self.settings)
        self.droplet_manager = DropletManager(
            settings=self.settings,
            left_pipette=self.left_pipette,
            containers=self.containers,
        )
        self.logger = Logger(
            name="protocol",
            file_path=f'{self.file_settings["output_folder"]}/{self.file_settings["exp_tag"]}/{self.file_settings["meta_data_folder"]}',
        )

        self.results = None
        self.config.home()
        self.left_pipette.pick_up_tip()
        self.logger.info("Protocol initialized successfully.")
    
    def measure_well(self, well_id: str, type: str = "wells"):
        if self.results is None:
            self.results = initialize_results(type=type)
        well_to_measure = self.containers[well_id]
        st_t, drop_parameters = self.droplet_manager.measure(source=well_to_measure)
        self._append_save_plot_results(
            dynamic_surface_tension=st_t,
            container=well_to_measure,
            drop_parameters=drop_parameters,
            type=type
        )


    def measure_wells(self):
        self.logger.info("Starting measure wells protocol...\n\n\n")
        type = "wells"
        sample_info = pd.read_csv(f"{self.file_settings['sample_info_filepath']}")
        sample_info.to_csv(f"{self.file_settings['output_folder']}/{self.file_settings['exp_tag']}/{self.file_settings["meta_data_folder"]}/sample_info.csv", index=False)
        self._set_sampleID_in_container(sample_info=sample_info)
        if self.results is None:
            self.results = initialize_results(type=type)
        well_ids = sample_info["well ID"].tolist()
        for well_id in well_ids:
            self.measure_well(well_id=well_id, type=type)

        self.logger.info("Finished measure wells protocol.\n\n\n")
        self.config.home()

    def characterise_solution(self):
        self.logger.info("Starting measure wells protocol...\n\n\n")
        type = "characterization"
        sample_info = pd.read_csv(f"{self.file_settings['sample_info_filepath']}")
        sample_info.to_csv(f"{self.file_settings['output_folder']}/{self.file_settings['exp_tag']}/{self.file_settings["meta_data_folder"]}/sample_info.csv", index=False)
        pass

    def _append_save_plot_results(
        self,
        dynamic_surface_tension: list,
        container: Container,
        drop_parameters: dict,
        type: str
    ) -> None:
        """
        Append results, save them, and plot the results. Plots based on the specified plot type (either well or concentration)
        """
        self.results = append_results(
            results=self.results,
            settings=self.settings,
            dynamic_surface_tension=dynamic_surface_tension,
            container=container,
            drop_parameters=drop_parameters,
            n_eq_points=self.settings["pendant_drop_settings"]["n_equilibration_points"],
            sensor_api=self.sensor_api,
            type = type
        )
        save_results(self.results, settings=self.settings)
        if type == "wells":
            self.plotter.plot_results_sample_id(
                df=self.results
            )
        elif type == "characterization":
            pass
            # self.plotter.plot_results_concentration(
            #     df=self.results, container=container
            # )

    def _set_sampleID_in_container(self, sample_info: pd.DataFrame):
        well_ids = sample_info["well ID"].tolist()
        sample_ids = sample_info["sample ID"].tolist()
        for well_id, sample_id in zip(well_ids, sample_ids):
            if well_id in self.containers:
                self.containers[well_id].sample_id = sample_id
        