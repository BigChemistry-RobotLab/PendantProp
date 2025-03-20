from utils.logger import Logger
from utils.load_save_functions import load_settings
from utils.search_containers import get_well_id_solution, get_plate_ids
from hardware.opentrons.containers import *
from hardware.opentrons.http_communications import OpentronsAPI
from hardware.cameras import PendantDropCamera
import time


class Pipette:
    def __init__(
        self,
        http_api: OpentronsAPI,
        mount: str,
        pipette_name: str,
        pipette_id: str,
        tips_info: dict,
        containers: dict,
    ):
        settings = load_settings()
        self.api = http_api
        self.MOUNT = mount
        self.PIPETTE_NAME = pipette_name
        self.PIPETTE_ID = pipette_id
        self.TIPS_INFO = tips_info
        self.CONTAINERS = containers
        self.DIFFERENCE_NEEDLE_PICK_UP = 6
        self.DIFFERENCE_NEEDLE_Z = 14.65
        self.DIFFERENCE_NEEDLE_Y = 1 
        self.has_tip = False
        self.has_needle = False
        self.volume = 0
        self.current_solution = "empty"
        self.clean = True  # boolean to check if tip is clean
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{settings["EXPERIMENT_NAME"]}/meta_data',
        )

        # warning if no tips information is provided
        if self.TIPS_INFO == None:
            self.logger.warning("No tips information provided for pipette!")

        # set max volume
        if self.PIPETTE_NAME == "p20_single_gen2":
            self.MAX_VOLUME = 20
            self.OFFSET = {"x": -1, "y": 1.4, "z": 0}

        elif self.PIPETTE_NAME == "p1000_single_gen2":
            self.MAX_VOLUME = 1000
            self.OFFSET = dict(x=-0.7, y=0.3, z=0.8)

        else:
            self.logger.error("Pipette name not recognised!")

        self.well_index = 0
        self.last_source = None
        self.last_destination = None
        self.air_gap_volume = 0

    def pick_up_tip(self, well=None):
        if self.has_tip:
            self.logger.error(
                f"Could not pick up tip as {self.MOUNT} pipette already has one!"
            )
            return

        if self.has_needle:
            self.logger.error(
                f"Could not pick up tip as {self.MOUNT} pipette has a needle attached."
            )
            return

        if well == None:
            tips_id, well = self._find_well_and_tips_id()
            if tips_id == None:
                self.logger.error("Well index is out of bounds.")
        else:
            tips_id =  self.TIPS_INFO[next(iter(self.TIPS_INFO))]["labware_id"] # takes from the first tip rack

        self.api.pick_up_tip(
            pipette_id=self.PIPETTE_ID,
            labware_id=tips_id,
            well=well,
            offset=self.OFFSET,
        )
        self.has_tip = True
        self.well_index += 1
        self.logger.info("Picked up tip.")

    def drop_tip(self):
        if not self.has_tip:
            self.logger.error("Pipette does not have a tip to drop!")
            return

        if self.has_needle:
            self.logger.warning("Pipette has needle, which should be returned.")

        self.api.drop_tip(pipette_id=self.PIPETTE_ID)
        self.logger.info(
            f"{self.MOUNT.capitalize()} pipette dropped tip into trash."
        )
        self.has_tip = False
        self.volume = 0
        self.current_solution = "empty"

    def pick_up_needle(self):
        if self.has_tip:
            self.logger.error("tried to pick up needle, while pipette has tip.")
            return

        depth_offset = self.DIFFERENCE_NEEDLE_PICK_UP
        offset = self.OFFSET.copy()
        offset["z"] += depth_offset
        tips_id = self.TIPS_INFO[next(iter(self.TIPS_INFO))]["labware_id"]
        well = "H12"
        self.api.pick_up_tip(
            pipette_id=self.PIPETTE_ID,
            labware_id=tips_id,
            well=well,
            offset=offset,
        )
        self.has_needle = True
        self.OFFSET['z'] += self.DIFFERENCE_NEEDLE_Z
        self.OFFSET['y'] += self.DIFFERENCE_NEEDLE_Y
        self.logger.info("Picked up needle.")

    def return_needle(self):
        if not self.has_needle:
            self.logger.info("No needle to return!")
            return

        depth_offset = self.DIFFERENCE_NEEDLE_PICK_UP
        offset = self.OFFSET.copy()
        offset["z"] -= self.DIFFERENCE_NEEDLE_Z - self.DIFFERENCE_NEEDLE_PICK_UP
        tips_id = self.TIPS_INFO[next(iter(self.TIPS_INFO))]["labware_id"]
        well = "H12"
        self.api.drop_tip(
            pipette_id=self.PIPETTE_ID,
            labware_id=tips_id,
            well=well,
            offset=offset,
        )
        self.has_tip = False
        self.volume = 0
        self.OFFSET["z"] -= self.DIFFERENCE_NEEDLE_Z
        self.OFFSET["y"] -= self.DIFFERENCE_NEEDLE_Y
        self.logger.info("Returned needle.")

    def _find_well_and_tips_id(self):
        tips_ids = []
        tips_orderings = []
        for tip_name in self.TIPS_INFO.keys():
            tips_ids.append(self.TIPS_INFO[tip_name]["labware_id"])
            tips_orderings.append(self.TIPS_INFO[tip_name]["ordering"])
        total_wells = len(
            tips_orderings[0]
        ) 
        for i, tips_id in enumerate(tips_ids):
            if self.well_index < (i + 1) * total_wells:
                tips_id = tips_ids[i]
                well = tips_orderings[i][self.well_index - i * total_wells]
                return tips_id, well

    def aspirate(
        self,
        volume: float,
        source: Container,
        touch_tip=False,
        mix=None,
        depth_offset=0,
        flow_rate=100,
        log=True,
        update_info=True,
    ):

        # check if pipette has tip
        if not self.has_tip and not self.has_needle:
            self.logger.error(
                f"{self.MOUNT.capitalize()} pipette ({self.PIPETTE_NAME}) does not have a tip or needle! Cancelled aspirating step."
            )
            return

        # check if volume exceeds pipette capacity
        if self.volume + volume > self.MAX_VOLUME and update_info:
            self.logger.error(
                f"{self.MOUNT.capitalize()} pipette ({self.PIPETTE_NAME}) does not have enough free volume to aspirate {volume} uL! Cancelled aspirating step."
            )
            return

        if mix:
            mix_order = mix[0]
            if mix_order not in ["before", "after", "both"]:
                self.logger.warning(f"mix_order {mix_order} not recognized.")

        if mix and (mix_order == "before" or mix_order == "both"):
            self.mixing(container=source, mix=mix)

        self.api.aspirate(
            pipette_id=self.PIPETTE_ID,
            labware_id=source.LABWARE_ID,
            well=source.WELL,
            volume=volume,
            depth=source.height_mm - source.DEPTH + depth_offset,
            offset=self.OFFSET,
            flow_rate=flow_rate
        )
        if mix and (mix_order == "after" or mix_order == "both"):
            self.mixing(container=source, mix=mix)

        if touch_tip:
            self.touch_tip(container=source)

        # update information:
        if update_info:
            source.aspirate(volume, log=log)
            if (
                self.current_solution != "empty"
                and self.current_solution != source.solution_name
            ):
                self.clean = False
            self.last_source = source
            self.current_solution = source.solution_name
            self.volume += volume

        if log:
            self.logger.info(
                f"Aspirated {volume} uL from {source.WELL_ID} with {self.MOUNT} pipette."
            )

    def dispense(
        self,
        volume: float,
        destination: Container,
        touch_tip=False,
        mix=None,
        blow_out=False,
        depth_offset=0,
        flow_rate=100,
        log=True,
        update_info=True,
    ):
        if not self.has_tip and not self.has_needle:
            self.logger.error(
                f"{self.MOUNT} pipette ({self.PIPETTE_NAME}) does not have a tip or needle! Cancelled dispensing step."
            )
            return

        # TODO this gives a weird error with mixing
        # if self.volume - volume < 0:
        #     self.protocol_logger.error(
        #         f"{self.MOUNT} pipette ({self.PIPETTE_NAME}) does not have enough volume to dispense {volume} uL! Cancelled dispensing step."
        #     )
        #     return

        if mix:
            mix_order = mix[0]
            if mix_order not in ["before", "after", "both"]:
                self.logger.warning(f"mix_order {mix_order} not recognized.")

        if mix and (mix_order == "before" or mix_order == "both"):
            self.mixing(container=destination, mix=mix)

        self.api.dispense(
            pipette_id=self.PIPETTE_ID,
            labware_id=destination.LABWARE_ID,
            well=destination.WELL,
            volume=volume,
            depth=destination.height_mm - destination.DEPTH + depth_offset,
            offset=self.OFFSET,
            flow_rate=flow_rate,
        )
        if mix and (mix_order == "after" or mix_order == "both"):
            self.mixing(container=destination, mix=mix)
        if blow_out:
            self.blow_out(container=destination)
        if touch_tip:
            self.touch_tip(container=destination)

        if update_info:
            self.volume -= volume
            self.last_destination = destination
            destination.dispense(volume=volume, source=self.last_source, log=log)

        if log:
            self.logger.info(
                f"Dispensed {volume} uL into well {destination.WELL_ID} with {self.MOUNT} pipette."
            )

    def transfer(
        self,
        volume: float,
        source: Container,
        destination: Container,
        touch_tip=False,
        mix=None,
        blow_out=False,
    ):
        self.logger.info(
            f"Transferring {volume} uL from {source.WELL_ID} to well {destination.WELL_ID} with {self.MOUNT} pipette."
        )
        self.aspirate(volume=volume, source=source, touch_tip=touch_tip, mix=mix)
        self.dispense(
            volume=volume,
            destination=destination,
            touch_tip=touch_tip,
            mix=mix,
            blow_out=blow_out,
        )

    def move_to_well(self, container: Container, offset=None):
        if offset == None:
            offset_move = self.OFFSET.copy()
        else:
            offset_move = self.OFFSET.copy()
            for key in offset:
                offset_move[key] += offset[key]

        self.api.move_to_well(
            pipette_id=self.PIPETTE_ID,
            labware_id=container.LABWARE_ID,
            well=container.WELL,
            offset=offset_move,
        )

    def move_to_tip_calibrate(self, well: str, offset: dict = dict(x=0, y=0, z=0)):
        # This is used to check offset of pipettes!!
        tips_id = self.TIPS_INFO[next(iter(self.TIPS_INFO))]["labware_id"]
        self.api.move_to_well(
            pipette_id=self.PIPETTE_ID,
            labware_id=tips_id,
            well=well,
            offset=offset,
        )

    def move_to_well_calibrate(self, container: Container, well: str, offset: dict = dict(x=0, y=0, z=0)):
        # This is used to check offset of pipettes!!
        self.api.move_to_well(
            pipette_id=self.PIPETTE_ID,
            labware_id=container.LABWARE_ID,
            well=well,
            offset=offset,
        )

    def touch_tip(self, container: Container, repeat=1):
        if not self.has_tip and not self.has_needle:
            self.logger.error("No tip or needle attached to perform touch_tip!")
            return
        depth = (
            0.05 * container.DEPTH
        )  # little depth to ensure the tip touches the wall of the container
        initial_offset = self.OFFSET.copy()
        initial_offset["z"] -= 0.05 * container.DEPTH
        self.api.move_to_well(
            pipette_id=self.PIPETTE_ID,
            labware_id=container.LABWARE_ID,
            well=container.WELL,
            offset=initial_offset,
        )
        radius = container.WELL_DIAMETER / 2
        radius = radius * 0.9  # safety TODO fix
        for n in range(repeat):
            for i in range(4):
                offset = (
                    self.OFFSET.copy()
                )  # Create a copy of the offset to avoid modifying the original
                offset["z"] -= depth
                if i == 0:
                    offset["x"] -= radius
                elif i == 1:
                    offset["x"] += radius
                elif i == 2:
                    offset["y"] -= radius
                elif i == 3:
                    offset["y"] += radius
                self.api.move_to_well(
                    pipette_id=self.PIPETTE_ID,
                    labware_id=container.LABWARE_ID,
                    well=container.WELL,
                    offset=offset,
                    speed=30,
                )
                self.api.move_to_well(
                    pipette_id=self.PIPETTE_ID,
                    labware_id=container.LABWARE_ID,
                    well=container.WELL,
                    offset=initial_offset,
                    speed=30,
                )

        self.logger.info(f"Touched tip performed, repeated {repeat} times")

    def mixing(self, container: Container, mix: any):
        mix_order, volume_mix, repeat_mix = mix
        for n in range(repeat_mix):
            self.aspirate(volume=volume_mix, source=container, log=False, update_info=False)
            self.dispense(volume=volume_mix, destination=container, log=False, update_info=False)
        self.logger.info(
            f"Done with mixing in {container.WELL_ID} with order {mix_order}, with volume {volume_mix} uL, repeated {repeat_mix} times"
        )

    def blow_out(self, container: Container):
        self.api.blow_out(
            pipette_id=self.PIPETTE_ID,
            labware_id=container.LABWARE_ID,
            well=container.WELL,
            offset=self.OFFSET,
        )
        self.logger.info(f"blow out done in container {container.WELL_ID}")

    def air_gap(self, air_volume: float):
        if not self.has_tip and not self.has_needle:
            self.logger.error("No tip/needle attached to perform air_gap!")
            return

        if air_volume + self.volume > self.MAX_VOLUME:
            self.logger.warning("Air gap exceeds pipette capacity!")
            # return

        if self.last_source == None:
            self.logger.error(
                "No source location found, needed to perform air gap!"
            )
            return
        height_percentage = 0.05
        if self.last_source.CONTAINER_TYPE == "Plate well":
            height_percentage = 1 # needed as plate height is standard 2 mm at the moment #TODO change?
        depth_offset = height_percentage * self.last_source.DEPTH + self.last_source.height_mm
        flow_rate = air_volume / 3
        self.aspirate(
            volume=air_volume,
            source=self.last_source,
            depth_offset=depth_offset,
            flow_rate=flow_rate,
            log=False,
            update_info=False,
        )
        self.air_gap_volume = air_volume
        self.logger.info(
            f"Air gap of {air_volume} uL performed in {self.MOUNT} pipette."
        )

    def remove_air_gap(self, at_drop_stage: bool = False):
        if not self.has_tip and not self.has_needle:
            self.logger.error("No tip/needle attached to remove air_gap!")
            return

        if at_drop_stage:
            container = self.CONTAINERS["drop_stage"]
        else:
            if self.last_destination is not None:
                container = self.last_destination
            elif self.last_source is not None:
                container = self.last_source
            else:
                self.logger.error(
                    "No source or destination location found, needed to remove air gap!"
                )
                return
        height_percentage = 0.05
        if container.CONTAINER_TYPE == "Plate Well":
            height_percentage = 1
        depth_offset = height_percentage * container.DEPTH + container.height_mm
        flow_rate = self.air_gap_volume / 3
        self.dispense(
            volume=self.air_gap_volume,
            destination=container,
            depth_offset=depth_offset,
            flow_rate=flow_rate,
            log=False,
            update_info=False,
        )
        self.logger.info(f"Air gap of {self.air_gap_volume} uL removed in {self.MOUNT} pipette.")
        self.air_gap_volume = 0

    def clean_on_sponge(self):
        if not self.has_tip and not self.has_needle:
            self.logger.error("No tip or needle attached to clean on sponge!")
            return
        try:
            sponge = self.CONTAINERS["sponge"]
        except KeyError:
            self.logger.error("No sponge container found!")
            return

        self.api.move_to_well(
            pipette_id=self.PIPETTE_ID,
            labware_id=sponge.LABWARE_ID,
            well=sponge.well,
            offset=self.OFFSET,
        )
        for i in range(3):
            offset = self.OFFSET.copy()
            offset["z"] -= 3

            self.api.move_to_well(
                pipette_id=self.PIPETTE_ID,
                labware_id=sponge.LABWARE_ID,
                well=sponge.well,
                offset=offset,
                speed=30,
            )
            self.api.move_to_well(
                pipette_id=self.PIPETTE_ID,
                labware_id=sponge.LABWARE_ID,
                well=sponge.well,
                offset=self.OFFSET,
                speed=30,
            )
        self.logger.info("Tip/needle cleaned on sponge.")
        sponge.update_well()

    def wash(self):

        if not self.has_needle:
            self.logger.error("No needle attached. Cancelled washing step.")
            return

        if self.has_tip:
            self.logger.error("Tip attached, instead of needle. Cancelled washing step.")
            return

        try:
            ipa_well_id = get_well_id_solution(containers=self.CONTAINERS, solution_name="ipa")
        except:
            self.logger.error("Could not find well ID for IPA. Cancelled washing step.")
            return

        try:
            nitrogen_well_id = get_well_id_solution(containers=self.CONTAINERS, solution_name="nitrogen")
        except:
            self.logger.error("Could not find well ID for nitrogen flow. Cancelled washing step.")
        self.logger.info("Starting washing needle.")
        self.mixing(container=self.CONTAINERS[ipa_well_id], mix=("after", 20, 4))
        offset = {
            "x": 0,
            "y": 0,
            "z": 10,
        }
        self.move_to_well(container=self.CONTAINERS[ipa_well_id], offset=offset)
        self.logger.info("Drying needle.")
        for i in range(3):
            self.blow_out(container=self.CONTAINERS[nitrogen_well_id])
            offset = {
                "x": 0,
                "y": 1,
                "z": -30,
            }
            self.move_to_well(container=self.CONTAINERS[nitrogen_well_id], offset=offset)
            time.sleep(5)
        self.logger.info("Done washing needle.")

    def __str__(self):
        return f"""
        Pipette object

        Mount: {self.MOUNT}
        Pipette name: {self.PIPETTE_NAME}
        Pipette ID: {self.PIPETTE_ID}
        Tips ID: {self.TIPS_ID} 
        Has tip: {self.has_tip}
        Has needle: {self.has_needle}
        Current volume: {self.volume} uL
        Current solution: {self.current_solution}
        Clean: {self.clean}
        Tip well index: {self.ORDERING[self.well_index]}
        """
