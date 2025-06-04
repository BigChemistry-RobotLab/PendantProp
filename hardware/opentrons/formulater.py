from hardware.opentrons.pipette import Pipette
from hardware.opentrons.containers import Container
from utils.logger import Logger
from utils.search_containers import (
    get_list_of_well_ids_concentration,
    get_well_id_solution,
    get_plate_ids,
)
from utils.load_save_functions import load_settings, load_info
from utils.utils import get_well_id_from_index
import pandas as pd


class Formulater:
    def __init__(
        self,
        left_pipette: Pipette,
        right_pipette: Pipette,
        containers: dict,
        labware: dict
    ):
        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.containers = containers
        self.labware = labware
        settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{settings["EXPERIMENT_NAME"]}/meta_data',
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

    def _calculate_volumes_from_ratios(self, suggest_concentration: float, well_concentration: float, well_volume: float
    ):
        ratio = suggest_concentration / float(
            well_concentration
        )
        volume_source = ratio * well_volume
        volume_water = well_volume - volume_source
        return volume_source, volume_water

    def _transfer(
        self,
        volume: float,
        source: Container,
        destination: Container,
        mix=None,
        drop_tip_behavior="trash",  # or "return", or "keep"
        solution=None,
        tip_return_info=None,
        depth_offset=0
    ):
        # Select pipette
        if volume < 20:
            pipette = self.left_pipette
        elif volume < 1000:
            pipette = self.right_pipette
        else:
            self.logger.error(f"Volume {volume} uL is too big for both pipettes.")
            return

        # Pick up tip if needed
        if not pipette.has_tip:
            pipette.pick_up_tip()
            pipette.contains_solution = solution

        # Do the transfer
        pipette.transfer(
        volume=volume,
        source=source,
        destination=destination,
        blow_out=True,
        depth_offset=depth_offset
        )

        if mix:
            pipette.mixing(container=destination, mix=mix)

        # Drop, return, or keep tip
        if drop_tip_behavior == "trash":
            pipette.drop_tip()
        elif drop_tip_behavior == "return" and tip_return_info:
            tips_id, well = tip_return_info
            pipette.drop_tip(return_to_tiprack=True, tiprack_labware_id=tips_id, tip_well_name=well)
        elif drop_tip_behavior == "keep":
            pass  # Keep it on pipette for next use

    def serial_dilution(
        self, row_id: str, solution_name: str, n_dilutions: int, well_volume: float, dilution_factor: float
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
            f"Start of serial dilution of {solution_name} in row {row_id}, with {n_dilutions} dilutions and dilution factor {dilution_factor}."
        )
        dilution_volume = round(well_volume/dilution_factor)
        # pick up tip if pipette has no tip
        if pipette.has_tip == False:
            pipette.pick_up_tip()

        # print("Available container keys:", self.containers.keys())
        # adding water to all wells except the first one
        for i in range(n_dilutions-1):
            pipette.aspirate(
                volume=well_volume-dilution_volume,
                source=self.containers[well_id_water],
                touch_tip=True
            )
            pipette.dispense(
                volume=well_volume, destination=self.containers[f"{row_id}{i+2}"], blow_out=True
            )

        # pipette.drop_tip()
        pipette.dispense(
            volume=0,
            destination=self.containers[well_id_trash],
            touch_tip=True,
            update_info=False,
            blow_out=True
        )
        # pipette.dispense(
        #     volume=0,
        #     destination=self.containers[well_id_trash],
        #     touch_tip=True,
        #     update_info=False
        # )
        # adding surfactant to the first well
        # pipette.pick_up_tip()
        pipette.aspirate(volume=dilution_volume, source=self.containers[well_id_solution], touch_tip=True)
        pipette.dispense(
            volume=dilution_volume,
            destination=self.containers[f"{row_id}1"],
            blow_out=True
        )

        # serial dilution of surfactant
        for i in range(1, n_dilutions):
            pipette.aspirate(
                volume=dilution_volume,
                source=self.containers[f"{row_id}{i}"],
                touch_tip=True,
            )
            pipette.dispense(
                volume=dilution_volume, destination=self.containers[f"{row_id}{i+1}"]
            )
            pipette.mixing(
                container=self.containers[f"{row_id}{i+1}"],
                mix=("after", well_volume / 1.2, 12),
            )
            pipette.blow_out(container=self.containers[f"{row_id}{i+1}"])

        # transfering half of the volume of the last well to trash to ensure equal volume in all wells (handy for dye check, not per se for surfactants dilutions)
        pipette.aspirate(
            volume=dilution_volume,
            source=self.containers[f"{row_id}{n_dilutions}"],
            touch_tip=True,
        )
        pipette.dispense(
            volume=dilution_volume,
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
                drop_tip=False
            )
        self.right_pipette.drop_tip()
        self.logger.info("Done filling plate.")
    
    def wash(self, repeat = 2, return_needle = False):
        self.logger.info("Start washing needle.")
        well_id_water = get_well_id_solution(containers=self.containers, solution_name="water_wash")
        well_id_trash = get_well_id_solution(containers=self.containers, solution_name="trash")
        # if self.wash_index >= 96:
        #     well_id_wash_well = get_well_id_from_index(
        #         well_index=self.wash_index, plate_location=self.labware["plate wash 1"]["location"]
        #     )
        # else:    
        #     well_id_wash_well = get_well_id_from_index(
        #         well_index=self.wash_index, plate_location=self.labware["plate wash"]["location"]
        #     )
        well_id_wash_well = get_well_id_from_index(
            well_index=self.wash_index, plate_location=self.labware["plate wash"]["location"]
        )

        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        for i in range(repeat):
            if not self.right_pipette.has_tip:  # redundant check, you always drop if you have one before this step
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

    # def formulate_random_single_well(self, well_id: str, concentrations: dict, well_volume: float = 150.0):
    #     self.logger.info("Starting formulation protocol...\n\n\n")
    #     self.settings = load_settings()
    #     characterization_info = load_info(
    #         file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
    #     )
    #     safety_margin = 0.01

    #     try:
    #         well_id_water = get_well_id_solution(self.containers, "water")
    #     except ValueError as e:
    #         self.logger.error(f"Water source not found: {e}")
    #         return

    #     surfactants = characterization_info.get("surfactant", [])
    #     if isinstance(surfactants, pd.Series):
    #         surfactants = surfactants.tolist()
    #     elif not isinstance(surfactants, list) or not surfactants:
    #         self.logger.error("Invalid or empty surfactant list in characterization info.")
    #         return
        
    #     solution_plan = {}
    #     tip_map = {}
    #     solution_index = 0
    #     total_volume = 0.0

    #     for solution in solution_plan:
    #         if solution.lower() != "water":
    #             tip_map[solution] = solution_index
    #             solution_index += 1

    #     for solution, target_conc in concentrations.items():
    #         if pd.isna(target_conc) or target_conc <= 0:
    #             self.logger.warning(f"Skipping {solution} due to missing or invalid concentration.")
    #             continue

    #         try:
    #             suitable_wells = get_list_of_well_ids_concentration(
    #                 containers=self.containers,
    #                 solution=solution,
    #                 requested_concentration=target_conc,
    #             )
    #             stock_well_id = suitable_wells[0]
    #             stock_conc = float(self.containers[stock_well_id].concentration)
    #             volume_needed = (target_conc * well_volume) / stock_conc

    #             if not (0 <= volume_needed <= well_volume):
    #                 raise ValueError(f"Calculated volume {volume_needed:.2f} µL out of range.")

    #             solution_plan[solution] = (target_conc, stock_conc, volume_needed, stock_well_id)
    #             total_volume += volume_needed
    #         except ValueError as e:
    #             self.logger.error(f"{e}")
    #             continue

    #     volume_water = well_volume - total_volume
    #     if volume_water < -safety_margin:
    #         self.logger.error(f"Total volume {total_volume:.2f} µL exceeds well capacity.")
    #         return

    #     volume_water = max(volume_water, 0.0)

    #     # Log formulation plan
    #     log_msg = f"Formulating well {well_id}:"
    #     for sol, (_, _, vol, _) in solution_plan.items():
    #         log_msg += f" {vol:.2f} µL of {sol},"
    #     log_msg += f" {volume_water:.2f} µL of water."
    #     self.logger.info(log_msg)

    #     for solution, (_, _, volume, stock_well_id) in solution_plan.items():
            


    #         # TODO fix someday
    #         # is_water = solution.lower() == "water"
    #         # drop_tip_behavior = "trash" if is_water else "return"
    #         # depth_offset = 0 if is_water else 4
    #         # tip_return_info = None

    #         # if not is_water:
    #         #     tip_index = tip_map[solution]
    #         #     tips_id, tip_well = self.right_pipette._find_well_and_tips_id(tip_index)
    #         #     tip_return_info = (tips_id, tip_well)

    #         self._transfer(
    #             volume=volume,
    #             source=self.containers[stock_well_id],
    #             destination=self.containers[well_id],
    #             solution=solution,
    #             # drop_tip_behavior=drop_tip_behavior,
    #             # tip_return_info=tip_return_info,
    #             # depth_offset=depth_offset,
    #         )

    #         self.logger.debug(
    #             f"Transferred {volume:.2f} µL of {solution} from {stock_well_id} to {well_id}."
    #         )

    #     self.logger.info("Finished formulating random single wells.")

    def formulate_random_single_well(
        self,
        well_ids: list[str],
        concentrations_per_well: dict[str, dict[str, float]],
        well_volume: float = 150.0,
    ):
        self.logger.info("Starting formulation protocol...\n\n\n")
        self.settings = load_settings()
        characterization_info = load_info(
            file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
        )
        safety_margin = 0.01

        try:
            well_id_water = get_well_id_solution(self.containers, "water")
        except ValueError as e:
            self.logger.error(f"Water source not found: {e}")
            return

        surfactants = characterization_info.get("surfactant", [])
        if isinstance(surfactants, pd.Series):
            surfactants = surfactants.tolist()
        elif not isinstance(surfactants, list) or not surfactants:
            self.logger.error("Invalid or empty surfactant list in characterization info.")
            return

        # Find all components to dispense
        all_solutions = set()
        for well_id in well_ids:
            all_solutions.update(concentrations_per_well.get(well_id, {}).keys())

        # solution_plan: well_id → solution → (target_conc, stock_conc, volume, stock_well_id)
        solution_plan = {well_id: {} for well_id in well_ids}
        total_volumes = {well_id: 0.0 for well_id in well_ids}

        for well_id in well_ids:
            concentrations = concentrations_per_well.get(well_id, {})
            for solution, target_conc in concentrations.items():
                if pd.isna(target_conc) or target_conc <= 0:
                    self.logger.warning(f"Skipping {solution} in {well_id} due to missing or invalid concentration.")
                    continue

                try:
                    suitable_wells = get_list_of_well_ids_concentration(
                        containers=self.containers,
                        solution=solution,
                        requested_concentration=target_conc,
                    )
                    stock_well_id = suitable_wells[0]
                    stock_conc = float(self.containers[stock_well_id].concentration)
                    volume_needed = (target_conc * well_volume) / stock_conc

                    if not (0 <= volume_needed <= well_volume):
                        raise ValueError(f"Calculated volume {volume_needed:.2f} µL for {solution} in {well_id} is out of range.")

                    solution_plan[well_id][solution] = (target_conc, stock_conc, volume_needed, stock_well_id)
                    total_volumes[well_id] += volume_needed
                except ValueError as e:
                    self.logger.error(f"{e}")
                    continue

        # Transfer each solution across all wells that need it
        for solution in all_solutions:
            for well_id in well_ids:
                if solution not in solution_plan[well_id]:
                    continue
                _, _, volume, stock_well_id = solution_plan[well_id][solution]

                self._transfer(
                    volume=volume,
                    source=self.containers[stock_well_id],
                    destination=self.containers[well_id],
                    solution=solution,
                )

                self.logger.debug(
                    f"Transferred {volume:.2f} µL of {solution} from {stock_well_id} to {well_id}."
                )

        # Transfer water last
        for well_id in well_ids:
            remaining = well_volume - total_volumes[well_id]
            if remaining < -safety_margin:
                self.logger.error(f"Total volume {total_volumes[well_id]:.2f} µL in {well_id} exceeds well capacity.")
                continue

            volume_water = max(remaining, 0.0)

            self._transfer(
                volume=volume_water,
                source=self.containers[well_id_water],
                destination=self.containers[well_id],
                solution="water",
            )

            self.logger.debug(
                f"Transferred {volume_water:.2f} µL of water to {well_id}."
            )

        self.logger.info("Finished formulating wells: " + ", ".join(well_ids))
