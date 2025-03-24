import pandas as pd
import os
from hardware.opentrons.containers import *
from hardware.opentrons.pipette import Pipette
from utils.load_save_functions import load_settings
from utils.logger import Logger


class Configuration:

    def __init__(self, http_api=None):
        settings = load_settings()
        self.api = http_api
        self.LABWARE_DEFINITIONS_FOLDER = self.api.LABWARE_DEFINITIONS_FOLDER
        self.ROBOT_TYPE = settings["ROBOT_TYPE"]
        self.FILE_PATH_CONFIG = f'experiments/{settings["EXPERIMENT_NAME"]}/meta_data/{settings["CONFIG_FILENAME"]}'
        self.LAYOUT = self.load_config_file()
        self.RIGHT_PIPETTE_NAME = "p1000_single_gen2"
        self.LEFT_PIPETTE_NAME = "p20_single_gen2"
        self.RIGHT_PIPETTE_ID = None
        self.LEFT_PIPETTE_ID = None
        self.LABWARE = None
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{settings["EXPERIMENT_NAME"]}/meta_data',
        )

    def load_config_file(self):
        try:
            return pd.read_csv(self.FILE_PATH_CONFIG)
        except FileNotFoundError:
            self.logger.error(
                f"Configuration file not found at {self.FILE_PATH_CONFIG}"
            )
            return None

    def check_if_custom_labware(self, labware_file: str) -> bool:
        list_of_custom_labware = os.listdir(self.LABWARE_DEFINITIONS_FOLDER)
        for custom_labware in list_of_custom_labware:
            if labware_file == custom_labware[:-5]:
                return True
        return False

    def load_pipettes(self):
        try:
            self.RIGHT_PIPETTE_ID = self.api.load_pipette(
                name=self.RIGHT_PIPETTE_NAME, mount="right"
            )
            right_pipette = Pipette(
                opentrons_api=self.api,
                mount="right",
                pipette_name=self.RIGHT_PIPETTE_NAME,
                pipette_id=self.RIGHT_PIPETTE_ID,
                tips_info=self.LABWARE["tips P1000, 1"],
            )

            self.LEFT_PIPETTE_ID = self.api.load_pipette(
                name=self.LEFT_PIPETTE_NAME, mount="left"
            )
            left_pipette = Pipette(
                opentrons_api=self.api,
                mount="left",
                pipette_name=self.LEFT_PIPETTE_NAME,
                pipette_id=self.LEFT_PIPETTE_ID,
                tips_info=self.LABWARE["tips P20, 1"],
            )
            self.logger.info("Pipettes loaded successfully")
            return {"right": right_pipette, "left": left_pipette}
        except Exception as e:
            self.logger.error(f"Error loading pipettes: {e}")
            return None

    def load_labware(self):
        layout = self.LAYOUT
        labware = {}
        try:
            for position in layout["deck position"].unique():
                # robot type causes different position (int or str)
                if self.ROBOT_TYPE == "OT2":
                    position = int(position)
                elif self.ROBOT_TYPE == "Flex":
                    position = str(position)
                else:
                    print("Robot type not recognised")

                labware_df = layout[layout["deck position"] == position]
                labware_file = labware_df["labware file"].values[0]
                labware_name = labware_df["labware name"].values[0]
                # TODO: how to load trash in Flex?

                # check if labware is custom
                custom_labware = self.check_if_custom_labware(labware_file)

                labware_info = self.api.load_labware(
                    labware_name=labware_name,
                    labware_file=labware_file,
                    location=position,
                    custom_labware=custom_labware,
                )
                labware[labware_name] = labware_info

                self.LABWARE = labware
            self.logger.info("Labware loaded successfully")
            return self.LABWARE
        except Exception as e:
            self.logger.error(f"Error loading labware: {e}")

    def load_containers(self):
        try:
            containers = {}
            layout = self.LAYOUT

            # Define a mapping from labware name to container class
            labware_mapping = {
                "tube rack 15 mL": FalconTube15,
                "glass vial rack": GlassVial,
                "tube rack 50 mL": FalconTube50,
            }

            for i, function in enumerate(layout["function"]):
                if function == "source":  # check if function is source
                    name_solution = layout.loc[i, "solution"]
                    labware_name = layout.loc[i, "labware name"]
                    labware_info = self.LABWARE[labware_name]
                    well = layout.loc[i, "well"]
                    initial_volume = layout.loc[i, "initial volume (mL)"]
                    container_class = labware_mapping.get(labware_name)

                    containers[name_solution] = container_class(
                        labware_info=labware_info,
                        well=well,
                        initial_volume_mL=initial_volume,
                        solution_name=name_solution,
                    )
                elif function == "destination":  # check if function is destination
                    labware_name = layout.loc[i, "labware name"]
                    labware_info = self.LABWARE[labware_name]
                    location = labware_info["location"]
                    for well in labware_info["ordering"]:
                        well_id = f"{location}{well}"
                        containers[well_id] = PlateWell(
                            labware_info=labware_info, well=well
                        )

            self.logger.info("Containers loaded successfully")
            return containers
        except Exception as e:
            self.logger.error(f"Error loading containers: {e}")
            return None

    # def load_destinations(self):
    #     try:
    #         destinations = {}
    #         layout = self.LAYOUT

    #         for i, function in enumerate(layout["function"]):
    #             if function == "target":  # TODO -> target or destination?
    #                 labware_name = layout.loc[i, "labware name"]
    #                 labware_info = self.LABWARE[labware_name]
    #                 location = labware_info["location"]
    #                 for well in labware_info["ordering"]:
    #                     well_id = f"{location}{well}"
    #                     destinations[well_id] = Well(well, labware_info)
    #         self.logger.info("Destinations loaded successfully")
    #         return destinations
    #     except Exception as e:
    #         self.logger.error(f"Error loading destinations: {e}")
    #         return None
