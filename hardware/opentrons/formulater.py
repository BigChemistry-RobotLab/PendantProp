from hardware.opentrons.pipette import Pipette
from hardware.opentrons.containers import Container
from utils.logger import Logger
from utils.search_containers import (
    get_well_id_concentration,
    get_well_id_solution,
    get_plate_ids,
)


class Formulater:
    def __init__(
        self,
        left_pipette: Pipette,
        right_pipette: Pipette,
        containers: dict,
        logger: Logger,
    ):
        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.containers = containers
        self.logger = logger

    def formulate_exploit_point(
        self,
        suggest_c: float,
        solution_name: str,
        well_volume: float,
        well_id_exploit: str,
    ):
        self.logger.info(
            f"Formulating exploit point with concentration {suggest_c} mM, at well ID {well_id_exploit}."
        )
        well_id_source = get_well_id_concentration(
            containers=self.containers,
            solution=solution_name,
            requested_concentration=suggest_c,
        )
        well_id_water = get_well_id_solution(
            containers=self.containers, solution_name="water"
        )
        ratio = suggest_c / float(self.containers[well_id_source].concentration)
        volume_source = ratio * well_volume
        self._transfer(
            volume=volume_source,
            source=self.containers[well_id_source],
            destination=self.containers[well_id_exploit],
        )
        volume_water = well_volume - volume_source
        self._transfer(
            volume=volume_water,
            source=self.containers[well_id_water],
            destination=self.containers[well_id_exploit],
            mix=("after", well_volume / 2, 5),
        )
        self.logger.info("Finished formulating exploit point.")

    def _transfer(
        self, volume: float, source: Container, destination: Container, mix=None
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
            touch_tip=True,
            blow_out=True,
        )
        if mix:
            pipette.mixing(container=destination, mix=mix)

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
        if pipette.has_tip == False:
            pipette.pick_up_tip()

        # adding water to all wells
        for i in range(n_dilutions):
            pipette.transfer(
                volume=well_volume,
                source=self.containers[well_id_water],
                destination=self.containers[f"{row_id}{i+1}"],
                touch_tip=True,
                blow_out=True,
            )
        pipette.drop_tip()

        # adding surfactant to the first well
        pipette.pick_up_tip()
        pipette.aspirate(volume=well_volume, source=self.containers[well_id_solution])
        pipette.touch_tip(container=self.containers[well_id_solution])
        pipette.dispense(
            volume=well_volume,
            destination=self.containers[f"{row_id}1"],
            mix=("after", well_volume / 2, 5),
        )
        pipette.blow_out(container=self.containers[f"{row_id}1"])

        # serial dilution of surfactant
        for i in range(1, n_dilutions):
            pipette.aspirate(
                volume=well_volume,
                source=self.containers[f"{row_id}{i}"],
                touch_tip=True,
            )
            pipette.dispense(
                volume=well_volume, destination=self.containers[f"{row_id}{i+1}"]
            )
            pipette.mixing(
                container=self.containers[f"{row_id}{i+1}"],
                mix=("after", well_volume / 2, 5),
            )
            pipette.blow_out(container=self.containers[f"{row_id}{i+1}"])

        # transfering half of the volume of the last well to trash to ensure equal volume in all wells
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
        for well_id in well_ids:
            self._transfer(
                volume=well_volume,
                source=self.containers[well_id_stock],
                destination=self.containers[well_id],
                touch_tip=True,
            )
        self.logger.info("Done filling plate.")
