from opentrons_api.utils import get
class ComplexCommander:
    def __init__(self, left_pipette, right_pipette, containers, labware, logger):
        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.containers = containers
        self.labware = labware
        self.logger = logger
        self.wash_index = 0

    def wash(self, repeat = 3, return_needle = False):
        self.logger.info("Start washing needle.")
        well_id_water = get_well_id_solution(containers=self.containers, solution="water_wash")
        well_id_trash = get_well_id_solution(containers=self.containers, solution="trash")
        well_id_wash_well = get_well_id_from_index(
            well_index=self.wash_index, plate_location=self.labware["plate wash"]["location"]
        )

        if not self.left_pipette.has_tip:
            self.left_pipette.pick_up_needle()

        for i in range(repeat):
            if not self.right_pipette.has_tip:
                self.right_pipette.pick_up_tip()

            # transfer water to cleaning well
            self.right_pipette.aspirate(
                volume=300, source=self.containers[well_id_water], touch_tip=True
            )
            self.right_pipette.dispense(volume=300, destination=self.containers[well_id_wash_well], touch_tip=True, update_info=False)

            # flush needle with water via mixing
            self.left_pipette.mixing(container=self.containers[well_id_wash_well], mix=("after", 20, 5))

            # transfer water in cleaning well to trash falcon tube
            self.right_pipette.aspirate(
                volume=300,
                source=self.containers[well_id_wash_well],
                touch_tip=True,
                update_info=False
            )
            self.right_pipette.dispense(
                volume=300, destination=self.containers[well_id_trash], update_info=False
            )

            self.right_pipette.drop_tip()

        self.left_pipette.clean_on_sponge()
        if return_needle:
            self.left_pipette.return_needle()

        self.wash_index += 1