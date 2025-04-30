from PendantProp.hardware.opentrons.pipette import Pipette
from PendantProp.hardware.opentrons.containers import Container
from PendantProp.utils.logger import Logger
from PendantProp.utils.search_containers import (
    get_list_of_well_ids_concentration,
    get_well_id_solution,
    get_plate_ids,
)
from PendantProp.utils.load_save_functions import load_settings
from PendantProp.utils.utils import get_well_id_from_index


class Formulater:
    def __init__(
        self,
        left_pipette: Pipette,
        right_pipette: Pipette,
        containers: dict,
        labware: dict,
        experiments_dir="experiments",
    ):
        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.containers = containers
        self.labware = labware
        settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f"{experiments_dir}/{settings['EXPERIMENT_NAME']}/meta_data",
        )
        self.wash_index = settings["WASH_INDEX"]

    def formulate_exploit_point(
        self,
        suggest_concentration: float,
        solution_name: str,
        well_volume: float,
        well_id_exploit: str,
    ):
        self.logger.info(
            f"Formulating exploit point with concentration {suggest_concentration} mM, at well ID {well_id_exploit}."
        )

        # Find relevant well IDs
        try:
            well_ids_source = get_list_of_well_ids_concentration(
                containers=self.containers,
                solution=solution_name,
                requested_concentration=suggest_concentration,
            )
        except ValueError as e:
            self.logger.error(f"Error finding source wells: {e}")
            return
        print(well_ids_source)
        # Initialize variables
        volume_from_source = None
        volume_water = None
        well_id_source = None

        # Validate source wells and calculate volumes
        for well_id in well_ids_source:
            volume_in_source = self.containers[well_id].volume_mL * 1000
            volume_from_source, volume_water = self._calculate_volumes_from_ratios(
                suggest_concentration=suggest_concentration,
                well_concentration=self.containers[well_id].concentration,
                well_volume=well_volume,
            )
            if volume_from_source / volume_in_source < 0.5:
                well_id_source = well_id
                break

        if not well_id_source:
            self.logger.error(
                "None of the sources have enough volume to formulate the exploit point."
            )
            return

        self.logger.info(
            f"Calculated volumes for exploit point: {volume_from_source} uL source from {well_id_source}, {volume_water} uL water."
        )

        # Transfer source to exploit point
        self._transfer(
            volume=volume_from_source,
            source=self.containers[well_id_source],
            destination=self.containers[well_id_exploit],
        )

        # Transfer water to exploit point
        well_id_water = get_well_id_solution(
            containers=self.containers, solution_name="water"
        )
        self._transfer(
            volume=volume_water,
            source=self.containers[well_id_water],
            destination=self.containers[well_id_exploit],
            mix=("after", well_volume / 1.2, 12),
        )

        self.logger.info("Finished formulating exploit point.")

    def _calculate_volumes_from_ratios(
        self,
        suggest_concentration: float,
        well_concentration: float,
        well_volume: float,
    ):
        ratio = suggest_concentration / float(well_concentration)
        volume_source = ratio * well_volume
        volume_water = well_volume - volume_source
        return volume_source, volume_water

    def _transfer(
        self,
        volume: float,
        source: Container,
        destination: Container,
        mix=None,
        drop_tip=True,
    ):
        if volume < 20:
            pipette = self.left_pipette
        elif volume < 1000:
            pipette = self.right_pipette
        else:
            self.logger.error(f"Volume {volume} uL is too big for both pipettes.")

        if not pipette.has_tip:
            pipette.pick_up_tip()

        pipette.transfer(
            volume=volume,
            source=source,
            destination=destination,
            blow_out=True,
        )
        if mix:
            pipette.mixing(container=destination, mix=mix)
        if drop_tip:
            pipette.drop_tip()

    def serial_dilution(
        self, row_id: str, solution_name: str, n_dilutions: int, well_volume: float
    ):
        pipette = self.right_pipette
        # find relevant well id's
        well_id_trash = get_well_id_solution(
            containers=self.containers, solution_name="trash"
        )  # well ID liquid waste
        well_id_water = get_well_id_solution(
            containers=self.containers, solution_name="water"
        )  # well ID water stock
        well_id_solution = get_well_id_solution(
            containers=self.containers, solution_name=solution_name
        )

        # log start of serial dilution
        self.logger.info(
            f"Start of serial dilution of {solution_name} in row {row_id}, with {n_dilutions} dilutions."
        )

        # pick up tip if pipette has no tip
        if not pipette.has_tip:
            pipette.pick_up_tip()

        # adding water to all wells except the first one
        for i in range(n_dilutions - 1):
            pipette.aspirate(
                volume=well_volume,
                source=self.containers[well_id_water],
                touch_tip=True,
            )
            pipette.dispense(
                volume=well_volume,
                destination=self.containers[f"{row_id}{i + 2}"],
                blow_out=True,
            )

        pipette.drop_tip()

        # adding surfactant to the first well
        pipette.pick_up_tip()
        pipette.aspirate(
            volume=well_volume * 2,
            source=self.containers[well_id_solution],
            touch_tip=True,
        )
        pipette.dispense(
            volume=well_volume * 2,
            destination=self.containers[f"{row_id}1"],
            blow_out=True,
        )

        # serial dilution of surfactant
        for i in range(1, n_dilutions):
            pipette.aspirate(
                volume=well_volume,
                source=self.containers[f"{row_id}{i}"],
                touch_tip=True,
            )
            pipette.dispense(
                volume=well_volume, destination=self.containers[f"{row_id}{i + 1}"]
            )
            pipette.mixing(
                container=self.containers[f"{row_id}{i + 1}"],
                mix=("after", well_volume / 1.2, 12),
            )
            pipette.blow_out(container=self.containers[f"{row_id}{i + 1}"])

        # transfering half of the volume of the last well to trash to ensure equal volume in all wells (handy for dye check, not per se for surfactants dilutions)
        pipette.aspirate(
            volume=well_volume,
            source=self.containers[f"{row_id}{n_dilutions}"],
            touch_tip=True,
        )
        pipette.dispense(
            volume=well_volume,
            destination=self.containers[well_id_trash],
            touch_tip=True,
            update_info=False,
        )
        pipette.drop_tip()

        # log end of serial dilution
        self.logger.info("End of serial dilution.")

    def fill_plate(self, well_volume: float, solution_name: str, plate_location: int):
        self.logger.info(f"Start filling plate with {solution_name}.")
        well_id_stock = get_well_id_solution(
            containers=self.containers, solution_name=solution_name
        )
        well_ids = get_plate_ids(location=plate_location)
        self.right_pipette.pick_up_tip()
        for well_id in well_ids:
            self._transfer(
                volume=well_volume,
                source=self.containers[well_id_stock],
                destination=self.containers[well_id],
                drop_tip=False,
            )
        self.right_pipette.drop_tip()
        self.logger.info("Done filling plate.")

    def wash(self, repeat=3, return_needle=False):
        self.logger.info("Start washing needle.")
        well_id_water = get_well_id_solution(
            containers=self.containers, solution_name="water_wash"
        )
        well_id_trash = get_well_id_solution(
            containers=self.containers, solution_name="trash"
        )
        well_id_wash_well = get_well_id_from_index(
            well_index=self.wash_index,
            plate_location=self.labware["plate wash"]["location"],
        )

        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        for i in range(repeat):
            if not self.right_pipette.has_tip:
                self.right_pipette.pick_up_tip()

            # transfer water to cleaning well
            self.right_pipette.aspirate(
                volume=300, source=self.containers[well_id_water], touch_tip=True
            )
            self.right_pipette.dispense(
                volume=300,
                destination=self.containers[well_id_wash_well],
                touch_tip=True,
                update_info=False,
            )

            # flush needle with water via mixing
            self.left_pipette.mixing(
                container=self.containers[well_id_wash_well], mix=("after", 20, 5)
            )

            # transfer water in cleaning well to trash falcon tube
            self.right_pipette.aspirate(
                volume=300,
                source=self.containers[well_id_wash_well],
                touch_tip=True,
                update_info=False,
            )
            self.right_pipette.dispense(
                volume=300,
                destination=self.containers[well_id_trash],
                update_info=False,
            )

            self.right_pipette.drop_tip()

        self.left_pipette.clean_on_sponge()
        if return_needle:
            self.left_pipette.return_needle()

        self.wash_index += 1
