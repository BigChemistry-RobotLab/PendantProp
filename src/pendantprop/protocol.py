import pandas as pd
import warnings

from opentrons_api.load_save_functions import load_settings
from opentrons_api.utils import find_empty_rows, get_well_ids_compounds
from opentrons_api.logger import Logger
from opentrons_api.containers import Container
from opentrons_api.formulater import Formulater
from pendantprop.hardware.opentrons.config import Config
from pendantprop.hardware.droplet_management import DropletManager
from pendantprop.hardware.opentrons.washing import Washer
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
        self.formulater = Formulater(
            settings=self.settings,
            left_pipette=self.left_pipette,
            right_pipette=self.right_pipette,
            containers=self.containers,
        )
        self.plotter = Plotter(settings=self.settings)
        self.sensor_api = SensorAPI(settings=self.settings)
        self.droplet_manager = DropletManager(
            settings=self.settings,
            left_pipette=self.left_pipette,
            containers=self.containers,
        )
        self.washer = Washer(
            settings=self.settings,
            left_pipette=self.left_pipette,
            right_pipette=self.right_pipette,
            containers=self.containers,
            labware=self.config.labware,
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
            self.results = initialize_results()
        well_to_measure = self.containers[well_id]
        st_t, drop_parameters = self.droplet_manager.measure(source=well_to_measure)
        self._append_save_plot_results(
            dynamic_surface_tension=st_t,
            container=well_to_measure,
            drop_parameters=drop_parameters,
            type=type,
        )

    def measure_wells(self):
        self.logger.info("Starting measure wells protocol...\n\n\n")
        type = "wells"
        sample_info = pd.read_csv(f"{self.file_settings['sample_info_filepath']}")
        sample_info.to_csv(
            f"{self.file_settings['output_folder']}/{self.file_settings['exp_tag']}/{self.file_settings["meta_data_folder"]}/sample_info.csv",
            index=False,
        )
        self._set_sampleID_in_container(sample_info=sample_info)
        well_ids = sample_info["well ID"].tolist()
        for well_id in well_ids:
            self.measure_well(well_id=well_id, type=type)

        self.logger.info("Finished measure wells protocol.\n\n\n")
        self.config.home()
        self.config.save_layout_final()
        self.config.log_protocol_summary()

    def characterise_solutions(self):
        self.logger.info("Starting characterisation protocol...\n\n\n")
        type = "characterization"
        sample_info = pd.read_csv(f"{self.file_settings['sample_info_filepath']}")
        sample_info.to_csv(
            f"{self.file_settings['output_folder']}/{self.file_settings['exp_tag']}/{self.file_settings["meta_data_folder"]}/sample_info.csv",
            index=False,
        )
        self._set_sampleID_in_container(sample_info=sample_info)
        empty_rows =find_empty_rows(containers=self.containers)
        empty_rows_filtered = self._filter_empty_rows(
            row_ids=empty_rows,
            keywords = ["sample"] #? in settings? for now hardcoded 
            )
        
        well_id_samples = sample_info["well ID"].tolist()
        if len(well_id_samples)>len(empty_rows_filtered):
            self.logger.error("Not enough empty rows to perform characterisation protocol.")
            return
        
        solvents = sample_info["solvent"].tolist()

        solvent_ids = []
        for solvent in solvents:
            solvent_id = get_well_ids_compounds(
                containers=self.containers,
                compound=solvent,
            )[0]
            solvent_ids.append(solvent_id)

        for row_id, well_id_sample, solvent_id in zip(empty_rows_filtered, well_id_samples, solvent_ids):
            sample_id = self.containers[well_id_sample].sample_id
            self.formulater.serial_dilution(
                row_id=row_id,
                stock_id=well_id_sample,
                solvent_id=solvent_id,
                n_dil=self.settings["pendant_drop_settings"]["n_dilutions"],
                well_volume=self.settings["pendant_drop_settings"]["well_volume"],
                mix_repeat=self.settings["pendant_drop_settings"]["mix_repeat"],
                drop_right_tip_only=True,
            )
            for i in range(self.settings["pendant_drop_settings"]["n_dilutions"], 0, -1):
                well_id_to_measure = f"{row_id}{i}"
                sample_id_dilution = f"{sample_id}_{i}"
                self.containers[well_id_to_measure].sample_id = sample_id_dilution
                self.measure_well(well_id=well_id_to_measure, type=type)
            self.washer.wash()

        self.logger.info("Finished characterisation protocol.\n\n\n")
        self.config.home()
        self.config.save_layout_final()
        self.config.log_protocol_summary()

    def calibrate(self, well_id_water: str):
        self.logger.info(f"Starting calibration protocol with water well ID: {well_id_water}...\n\n\n")
        pass

    def _append_save_plot_results(
        self,
        dynamic_surface_tension: list,
        container: Container,
        drop_parameters: dict,
        type: str,
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
            n_eq_points=self.settings["pendant_drop_settings"][
                "n_equilibration_points"
            ],
            sensor_api=self.sensor_api,
        )
        save_results(self.results, settings=self.settings)
        if type == "wells":
            self.plotter.plot_results_sample_id(df=self.results)
        elif type == "characterization":
            self.plotter.plot_results_concentration(
                df=self.results
            )

    def _set_sampleID_in_container(self, sample_info: pd.DataFrame):
        well_ids = sample_info["well ID"].tolist()
        sample_ids = sample_info["sample ID"].tolist()
        for well_id, sample_id in zip(well_ids, sample_ids):
            if well_id in self.containers:
                self.containers[well_id].sample_id = sample_id
    
    def _filter_empty_rows(self, row_ids: list, keywords: list) -> list:
        """Filter out empty rows from a list of row IDs."""
        
        
        filtered_rows = []
        for row_id in row_ids:
            well_id_first_well = f"{row_id}1"
            container = self.containers.get(well_id_first_well)
            labware_name = container.LABWARE_NAME.lower()
            if any(keyword in labware_name for keyword in keywords):
                filtered_rows.append(row_id)
        return filtered_rows
