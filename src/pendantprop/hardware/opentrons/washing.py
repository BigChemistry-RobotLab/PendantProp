from typing import Dict

from opentrons_api.containers import Container
from opentrons_api.pipette import Pipette
from opentrons_api.logger import Logger
from opentrons_api.utils import get_well_ids_compounds


class Washer:
    def __init__(
        self,
        settings: Dict,
        left_pipette: Pipette,
        right_pipette: Pipette,
        containers: Dict[str, Container],
        labware: Dict,
    ):
        self.settings = settings
        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.containers = containers
        self.labware = labware
        self.logger = Logger(
            name="protocol",
            file_path=f'{settings["OUTPUT_FOLDER"]}/{self.settings["EXP_TAG"]}/{self.settings["META_DATA_FOLDER"]}',
        )
        self.wash_index = 0

    def wash(self, wash_settings: Dict = None):
        if wash_settings is None:
            wash_settings = self.settings["WASH_SETTINGS"]

        self.logger.info(
            f"Starting needle wash procedure - "
            f"Repeats: {wash_settings['wash_repeats']}, "
            f"Wash volume: {wash_settings['wash_volume_ul']} µL, "
            f"Mixing: {wash_settings['mixing_volume_ul']} µL × {wash_settings['mix_repeats']}"
        )
        try:
            well_id_water_wash = get_well_ids_compounds(
                containers=self.containers,
                compound=wash_settings["name_wash_container"],
            )[0]
            well_id_trash = get_well_ids_compounds(
                containers=self.containers,
                compound=wash_settings["name_trash_container"],
            )[0]
            well_id_wash_well = self._get_well_id_from_index(
                well_index=self.wash_index,
                plate_location=self.labware[wash_settings["name_wash_plate"]][
                    "location"
                ],
            )
        except ValueError as e:
            self.logger.error(f"Error getting well IDs for washing: {e}")
            return

        if not self.left_pipette.has_tip:
            self.logger.error("Left pipette has no needle for washing!")
            return

        for i in range(wash_settings["wash_repeats"]):
            if not self.right_pipette.has_tip:
                self.right_pipette.pick_up_tip()

            # transfer water to cleaning well
            self.right_pipette.aspirate(
                volume=wash_settings["wash_volume_ul"],
                source=self.containers[well_id_water_wash],
                touch_tip=True,
            )
            self.right_pipette.dispense(
                volume=wash_settings["wash_volume_ul"],
                destination=self.containers[well_id_wash_well],
                touch_tip=True,
                update_info=False,
            )

            # flush needle with water via mixing
            self.left_pipette.mixing(
                container=self.containers[well_id_wash_well],
                volume_mix=wash_settings["mixing_volume_ul"],
                repeat=wash_settings["mix_repeats"],
                touch_tip=False,
            )

            # transfer water in cleaning well to trash falcon tube
            self.right_pipette.aspirate(
                volume=wash_settings["wash_volume_ul"],
                source=self.containers[well_id_wash_well],
                touch_tip=True,
                update_info=False,
            )
            # TODO: what if trash is full? Handle that case.
            self.right_pipette.dispense(
                volume=wash_settings["wash_volume_ul"],
                destination=self.containers[well_id_trash],
                update_info=False,
            )

            self.right_pipette.drop_tip()

        self.wash_index += 1

    def _get_well_id_from_index(self, well_index: int, plate_location: int):
        """
        Assumes 96 well plate

        TODO: move to opentrons_api.utils
        """
        list_of_wells = []
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            for i in range(1, 13):
                well_id = f"{plate_location}{letter}{i}"
                list_of_wells.append(well_id)
        return list_of_wells[well_index]
