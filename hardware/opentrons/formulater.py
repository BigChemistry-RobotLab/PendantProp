from hardware.opentrons.pipette import Pipette
from hardware.opentrons.containers import Container
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.configuration import Configuration
from utils.logger import Logger
from utils.search_containers import (
    get_list_of_well_ids_concentration,
    get_well_id_solution,
    get_plate_ids,
    get_solution_and_conc,
    find_container,
)
from utils.load_save_functions import load_settings, load_info
from utils.utils import get_well_id_from_index, get_tube_id_from_index
import pandas as pd
import numpy as np


class Formulater:
    def __init__(
        self,
        left_pipette: Pipette,
        right_pipette: Pipette,
        containers: dict,
        labware: dict,
        opentrons_api: OpentronsAPI,
    ):
        self.config = Configuration(opentrons_api=opentrons_api)
        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.containers = containers
        self.labware = labware
        self.settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )
        self.wash_index = self.settings["WASH_INDEX"]
        self.mixing_steps = self.settings["MIXING_STEPS_FORMULATION"]
        self.tube_index = 0
        self.well_tracker = []

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
            mix=("after", well_volume / 1.2, self.mixing_steps),
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
        drop_tip_behavior="trash",  # or "return", or "keep"
        solution=None,
        tip_return_info=None,
        depth_offset=0,
        side=None,
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
            depth_offset=depth_offset,
            side=side,
        )

        if mix:
            pipette.mixing(container=destination, mix=mix)

        # Drop, return, or keep tip
        if drop_tip_behavior == "trash":
            pipette.drop_tip()
        elif drop_tip_behavior == "return" and tip_return_info:
            tips_id, well = tip_return_info
            pipette.drop_tip(
                return_to_tiprack=True, tiprack_labware_id=tips_id, tip_well_name=well
            )
        elif drop_tip_behavior == "keep":
            pass  # Keep it on pipette for next use

    def serial_dilution(
        self,
        row_id: str,
        solution_name: str,
        n_dilutions: int,
        well_volume: float,
        dilution_factor: float,
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
        )  # Make it jump out of serial dilution and measurement if tube is not found

        # log start of serial dilution
        self.logger.info(
            f"Start of serial dilution of {solution_name} in row {row_id}, with {n_dilutions} dilutions and dilution factor {dilution_factor}."
        )
        dilution_volume = round(well_volume / dilution_factor)
        # pick up tip if pipette has no tip
        if pipette.has_tip == False:
            pipette.pick_up_tip()

        # print("Available container keys:", self.containers.keys())
        # adding water to all wells except the first one
        for i in range(n_dilutions - 1):
            pipette.aspirate(
                volume=well_volume - dilution_volume,
                source=self.containers[well_id_water],
                touch_tip=True,
            )
            pipette.dispense(
                volume=well_volume - dilution_volume,
                destination=self.containers[f"{row_id}{i+2}"],
                blow_out=True,
            )

        pipette.aspirate(
            volume=well_volume,
            source=self.containers[well_id_solution],
            touch_tip=True,
            mix=("before", well_volume / 1.2, int((self.mixing_steps / 2))),
        )
        pipette.dispense(
            volume=well_volume, destination=self.containers[f"{row_id}1"], blow_out=True
        )

        # serial dilution of surfactant
        for i in range(1, n_dilutions):
            pipette.aspirate(
                volume=dilution_volume,
                source=self.containers[f"{row_id}{i}"],
                touch_tip=True,
                depth_offset=0.5,
            )
            pipette.dispense(
                volume=dilution_volume,
                destination=self.containers[f"{row_id}{i+1}"],
                depth_offset=0.5,
            )
            pipette.mixing(
                container=self.containers[f"{row_id}{i+1}"],
                mix=("after", well_volume / 1.2, self.mixing_steps),
            )
            pipette.blow_out(container=self.containers[f"{row_id}{i+1}"])

        # transferring half of the volume of the last well to trash to ensure equal volume in all wells (handy for dye check, not per se for surfactants dilutions)
        pipette.aspirate(
            volume=dilution_volume,
            source=self.containers[f"{row_id}{n_dilutions}"],
            touch_tip=True,
            depth_offset=0.5,
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
                drop_tip=False,
            )
        self.right_pipette.drop_tip()
        self.logger.info("Done filling plate.")

    def find_wash_plate(self) -> int:
        return self.labware["plate wash"]["location"]

    def wash(self, repeat=2, return_needle=False) -> None:
        self.logger.info("Start washing needle.")
        well_id_water = get_well_id_solution(
            containers=self.containers, solution_name="water"
        )
        well_id_trash = get_well_id_solution(
            containers=self.containers, solution_name="trash"
        )
        # if self.wash_index >= 96:
        #     well_id_wash_well = get_well_id_from_index(
        #         well_index=self.wash_index, plate_location=self.labware["plate wash 1"]["location"]
        #     )
        # else:# TODO Can implement another wash plate for more extensive washing if required
        #     well_id_wash_well = get_well_id_from_index(
        #         well_index=self.wash_index, plate_location=self.labware["plate wash"]["location"]
        #     )
        if self.wash_index + repeat > 95:
            self.wash_index = 0
            self.logger.warning("Reset wash wells, not enough!")
        well_id_wash_well, amount_wells = get_well_id_from_index(
            well_index=self.wash_index,
            plate_location=self.labware["plate wash"]["location"],
            amount=repeat,
        )

        if self.left_pipette.has_tip:  # redundant, I dont use p20 tips.
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        for well_wash in well_id_wash_well:
            if not self.right_pipette.has_tip:
                self.right_pipette.pick_up_tip()
            self.right_pipette.aspirate(
                volume=300, source=self.containers[well_id_water], touch_tip=True
            )
            self.right_pipette.dispense(
                volume=300,
                destination=self.containers[well_wash],
                touch_tip=True,
                update_info=False,
            )
            self.left_pipette.mixing(
                container=self.containers[well_wash], mix=("after", 20, 10)
            )

        for well_wash in well_id_wash_well:
            self.right_pipette.aspirate(
                volume=300,
                source=self.containers[well_wash],
                touch_tip=True,
                update_info=False,
                depth_offset=0.5,
            )
            self.right_pipette.dispense(
                volume=300,
                destination=self.containers[well_id_trash],
                update_info=False,
            )
        self.right_pipette.drop_tip()

        # self.left_pipette.clean_on_sponge()
        if return_needle:
            self.left_pipette.return_needle()
        self.wash_index += amount_wells

    def _transfer_in_chunks(
        self, volume, source, destination, mix=None, max_pipette_vol=1000
    ) -> None:
        """
        A helper function to transfer liquid, automatically handling volumes
        larger than the pipette's maximum capacity.
        """
        remaining_vol = volume

        while remaining_vol > 0:
            transfer_vol = min(remaining_vol, max_pipette_vol)
            if mix == "before":
                mix_params = ("before", transfer_vol * 0.9, 5)
            else:
                mix_params = None

            self.right_pipette.transfer(
                volume=transfer_vol,
                source=source,
                destination=destination,
                touch_tip=True,
                mix=mix_params,
            )
            remaining_vol -= transfer_vol

    def formulate_serial_dilution(
        self, goals, vials, solution, stock_location, total_vol_dil=15000
    ) -> dict:
        """
        Performs a serial dilution to achieve a list of target concentrations.

        Args:
            goals (list): A list of target final concentrations in mM.
            vials (list): A list of available empty vial locations.
                          This list will be modified as vials are used.
            solution (str): The name of the solution being diluted.
            stock_location (str): The well location of the master stock solution.
        """
        goals.sort(reverse=True)

        # Find the water source
        water_location = find_container(
            containers=self.containers, content="water", type="tube"
        )
        if not water_location:
            raise ValueError("Water container could not be found.")

        # The source is the master stock provided
        current_source_loc = stock_location
        current_source_conc = self.containers[current_source_conc[0]].get_concentration(
            solute_name=solution
        )

        assert (
            goals[0] <= current_source_conc
        ), f"Highest target concentration ({goals[0]}) must be less than the stock concentration ({current_source_conc})."

        prepared_dilutions = {}

        self.right_pipette.pick_up_tip()

        for target_conc in goals:
            assert (
                current_source_conc <= target_conc
            ), f"Cannot create target {target_conc}mM from source {current_source_conc}mM."

            # Get an empty vial
            destination_vial = vials.pop(0)

            # Calc volume for concentration V1 = (C2 * V2) / C1
            stock_to_transfer_uL = (target_conc * total_vol_dil) / current_source_conc
            water_to_transfer_uL = total_vol_dil - stock_to_transfer_uL

            # Add stock first for better mixing
            self._transfer_in_chunks(
                volume=stock_to_transfer_uL,
                source=self.containers[current_source_loc],
                destination=self.containers[destination_vial],
                mix="before",
            )

            self.right_pipette.drop_tip()
            self.right_pipette.pick_up_tip()

            self._transfer_in_chunks(
                volume=water_to_transfer_uL,
                source=self.containers[water_location],
                destination=self.containers[destination_vial],
            )

            current_source_loc = destination_vial
            current_source_conc = target_conc

            prepared_dilutions[destination_vial] = target_conc

        self.right_pipette.drop_tip()
        self.logger.info(f"Serial dilution complete for {solution}.")
        return prepared_dilutions

    def formulate_new_master_stock(
        self, upper_bounds, master_stock, solution, total_volume=15000
    ) -> str:
        """
        Create a new master stock dilution from an existing master stock solution.

        Args:
            upper_bounds (float): Target upper concentration (in same units as stock).
            master_stock (str): Name or ID of the current master stock container.
            solution (str): Name of the solute to be diluted.
            total_volume (float, optional): Total volume of the new stock in uL. Default 15000 uL.

        Returns:
            str: The well ID of the new prepared vial.
        """

        # Find containers
        water_id = find_container(
            containers=self.containers, content="water", type="tube 50"
        )
        if not water_id:
            raise ValueError("Water container could not be found.")

        most_water = max(water_id, key=lambda entry: self.containers[entry].volume_mL)

        new_vial = find_container(
            containers=self.containers, content="empty", type="tube 50", amount=1
        )
        if not new_vial:
            raise ValueError("No empty 50 mL tube available for new master stock.")

        # Calculate dilution factor
        current_conc = self.containers[master_stock].get_concentration(
            solute_name=solution
        )
        new_factor = upper_bounds / current_conc
        if new_factor >= 1:
            raise ValueError(
                "Target concentration must be lower than current master stock concentration."
            )

        # Compute transfer volumes
        stock_to_transfer_uL = total_volume * new_factor
        water_to_transfer_uL = total_volume - stock_to_transfer_uL

        destination_vial = new_vial[0]
        current_source_loc = master_stock

        # Perform transfers
        # Add stock first for better mixing
        self.right_pipette.pick_up_tip()
        self._transfer_in_chunks(
            volume=stock_to_transfer_uL,
            source=self.containers[current_source_loc],
            destination=self.containers[destination_vial],
            mix="before",
        )

        self.right_pipette.drop_tip()
        self.right_pipette.pick_up_tip()

        self._transfer_in_chunks(
            volume=water_to_transfer_uL,
            source=self.containers[most_water],
            destination=self.containers[destination_vial],
        )

        self.right_pipette.drop_tip()
        self.logger.info(
            f"New master stock prepared for {solution} at {upper_bounds:.2f} units."
        )
        return self.containers[destination_vial].WELL_ID  # str

    def compute_total_required_volumes(
        self, df: pd.DataFrame, dilution_factor: float
    ) -> pd.DataFrame:
        df = df.sort_values("dil", ascending=False).reset_index(drop=True)
        df["total_volume"] = (
            df["volume_needed"] + 1000
        )  # Start with direct usage, +1000 should be done here

        for i in range(
            len(df) - 1, 0, -1
        ):  # Work from most dilute to most concentrated
            required_input = df.loc[i, "total_volume"] / dilution_factor
            df.loc[i - 1, "total_volume"] += required_input

        return df

    def formulate_single_point(
        self,
        surfactant_1: str,
        concentration_1: float,
        stock_conc_1: float,
        volume_1: float,
        surfactant_2: str,
        concentration_2: float,
        stock_conc_2: float,
        volume_2: float,
        total_well_volume: float,  
    ):  
        # Finds initial containers
        surf_1_stock = find_container(
            containers=self.containers,
            type="tube",
            content=surfactant_1,
            conc=stock_conc_1,
        )[0]
        surf_2_stock = find_container(
            containers=self.containers,
            type="tube",
            content=surfactant_2,
            conc=stock_conc_2,
        )[0]
        water_id = find_container(
            containers=self.containers, content="water", type="tube 50"
        )

        # Finds the tube with most water
        most_water = max(water_id, key=lambda entry: self.containers[entry].volume_mL)

        # Finds a list of all empty wells
        empty_well_id = find_container(
            containers=self.containers, content="empty", type="Plate well"
        )
        # Finds wash plate so it doesn't use those wells
        wash_loc = self.find_wash_plate()
        usable_wells = [
            item
            for item in empty_well_id
            if not self.containers[item].LOCATION == wash_loc
        ]
        dest_well = usable_wells.pop[0]

        # Define water volume
        volume_water = total_well_volume - volume_1 - volume_2

        self.logger.info(
            f"Formulating data point with concentration {concentration_1} mM of {surfactant_1} and concentration {concentration_2} mM of {surfactant_2}, at well ID {dest_well}."
        )

        self.logger.info(
            f"Calculated volumes for data point: {volume_1} uL of {surfactant_1} from {self.containers[surf_1_stock].get_concentration(solute_name=surfactant_1)}, {volume_2} uL of {surfactant_2} from {self.containers[surf_2_stock].get_concentration(solute_name=surfactant_2)} and {volume_water} uL water."
        )

        self._transfer(
            volume=volume_water,
            source=self.containers[most_water],
            destination=self.containers[dest_well],
            drop_tip_behavior="keep",
        )

        self._transfer(
            volume=volume_1,
            source=self.containers[surf_1_stock],
            destination=self.containers[dest_well],
        )

        self._transfer(
            volume=volume_2,
            source=self.containers[surf_2_stock],
            destination=self.containers[dest_well],
            mix=("after", total_well_volume / 1.2, self.mixing_steps),
        )

        self.logger.info("Finished formulating point.")
        return self.containers[dest_well]

    # Legacy version

    # def formulate_dilution_tube(self, dilution_df: pd.DataFrame, solution, dilution_factor):
    #     dilution_df = dilution_df.rename(columns={
    #         dilution_df.columns[0]: "dil",           # concentration values
    #         dilution_df.columns[1]: "volume_needed"  # raw usage volumes in uL
    #     })
    #     dilution_df = self.compute_total_required_volumes(dilution_df, dilution_factor)
    #     first_time = True
    #     water_location = find_container(    # Only needs to be found once
    #             containers=self.containers,
    #             content="water",
    #             type="tube",
    #         )
    #     for _, row in dilution_df.iterrows():
    #         conc = row["dil"]
    #         vol_uL = row["total_volume"]

    #         if vol_uL > 14950:
    #             raise ValueError(f"Too much volume required ({vol_uL} mL) for a 15 mL tube.")

    #         tube_location = find_container( # Needs to be found every run, same for updated stocks
    #             containers=self.containers, # since this script creates them.
    #             content="empty",
    #             type="tube 15",
    #             amount=1
    #         )
    #         stock_location = find_container(
    #             containers=self.containers,
    #             content=solution,
    #             type="tube",
    #             amount=10
    #         )

    #         if len(stock_location) > 1:
    #             # pick the tube whose concentration is closest to the target
    #             eligible_stocks = [loc for loc in stock_location
    #                if float(self.containers[loc].concentration) >= conc]

    #             if not eligible_stocks:
    #                 raise ValueError(f"No stock found with concentration >= target {conc} mM")

    #             # Pick the one closest to the target
    #             source_stock = min(
    #                 eligible_stocks,
    #                 key=lambda loc: float(self.containers[loc].concentration) - conc
    #             )
    #         else:
    #             source_stock = stock_location[0]

    #         if first_time:
    #             first_dil_factor = float(self.containers[source_stock].concentration)/conc
    #             total_vol = round(dilution_df["total_volume"][0], 0)
    #             first_vol_sol = round(total_vol / first_dil_factor, 0)   # stock volume
    #             first_vol_water = round(total_vol - first_vol_sol, 0)    # water volume
    #             print(first_vol_water, first_vol_sol)
    #             quit()
    #             # Transfer stock in chunks
    #             while first_vol_sol > 0:
    #                 transfer_vol_sol = min(1000, max(20, first_vol_sol))
    #                 self.left_pipette.transfer(
    #                     volume=transfer_vol_sol,
    #                     source=source_stock,
    #                     destination=tube_location,
    #                     touch_tip=True
    #                 )
    #                 first_vol_sol -= transfer_vol_sol

    #             # Transfer water in chunks
    #             while first_vol_water > 0:
    #                 transfer_vol_water = min(1000, max(20, first_vol_water))
    #                 self.left_pipette.transfer(
    #                     volume=transfer_vol_water,
    #                     source=water_location,
    #                     destination=tube_location,
    #                     touch_tip=True,
    #                     mix=("after", transfer_vol_water*0.9, 3)
    #                 )
    #                 first_vol_water -= transfer_vol_water
    #             first_time = False
    #             break   # Should be here?

    #         while vol_uL > 0:
    #             transfer_vol_sol = min(1000, max(20, vol_uL))
    #             self.left_pipette.transfer(
    #                 volume=vol_uL,  # mL
    #                 source=source_stock,
    #                 touch_tip=True,
    #                 destination=tube_location,
    #             )
    #             vol_uL -= transfer_vol_sol

    #         while vol_uL > 0:
    #             transfer_vol_sol = min(1000, max(20, vol_uL))
    #             self.left_pipette.transfer(
    #                 volume=vol_uL,  # mL
    #                 source=source_stock,
    #                 destination=tube_location,
    #                 touch_tip=True,
    #                 mix=("after", transfer_vol_water*0.9, 3)
    #             )
    #             vol_uL -= transfer_vol_sol

    # def formulate_dilution_tube(self, dilution_df: pd.DataFrame, label: str, containers):
    #         self.containers = containers  # Set initial in-memory container state

    #         for _, row in dilution_df.iterrows():
    #             conc = row.iloc[0]
    #             vol = row.iloc[1]
    #             updated_solution = f"{label}_{conc}"
    #             updated_conc = conc
    #             updated_mL = vol + 1  # Adjusted volume

    #             # Find an available empty 15 mL tube
    #             tube_location = find_container(containers=self.containers, content="empty", type="tube 15", amount=1)
    #             print("tl",tube_location)
    #             match = re.match(r"(\d+)([A-H]\d+)", tube_location[0])
    #             if not match:
    #                 raise ValueError(f"Invalid tube location format: {tube_location}")
    #             print("match",match)
    #             deck_pos = match.group(1)
    #             well = match.group(2)   # Splitting works properly
    #             print(deck_pos, well, "dp_w")
    #             for location, c in self.containers.items():
    #                 print("cloc",c, location)
    #                 match = re.match(r"(\d+)([A-H]\d+)", location)
    #                 print("match2", match)
    #                 if not match:
    #                     print("faulty")
    #                     continue  # skip malformed keys

    #                 c_deck = match.group(1)
    #                 c_well = match.group(2)

    #                 if c.LOCATION == int(c_deck) and c.WELL == c_well:
    #                     print(c_deck, c_well, c_deck+c_well)
    #                     c.solution_name = updated_solution
    #                     c.concentration = updated_conc
    #                     c.volume_mL = updated_mL  # or c.initial_volume_mL if that's the attribute
    #                     self.containers[location] = c
    #                     print(location)
    #                     # print(self.containers[location])
    #                     # print(self.containers[c_deck+c_well])
    #                     break
    #             else:
    #                 raise ValueError(f"Could not find matching container at {deck_pos}{well} in self.containers")

    #             print(f"Updated container at {deck_pos}{well} with {updated_solution}, {updated_conc}, {updated_mL} mL")
    #             print(self.containers["3A2"])
    #         print(self.containers["3A2"])
    #         return self.containers

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
    #                 raise ValueError(f"Calculated volume {volume_needed:.2f} uL out of range.")

    #             solution_plan[solution] = (target_conc, stock_conc, volume_needed, stock_well_id)
    #             total_volume += volume_needed
    #         except ValueError as e:
    #             self.logger.error(f"{e}")
    #             continue

    #     volume_water = well_volume - total_volume
    #     if volume_water < -safety_margin:
    #         self.logger.error(f"Total volume {total_volume:.2f} uL exceeds well capacity.")
    #         return

    #     volume_water = max(volume_water, 0.0)

    #     # Log formulation plan
    #     log_msg = f"Formulating well {well_id}:"
    #     for sol, (_, _, vol, _) in solution_plan.items():
    #         log_msg += f" {vol:.2f} uL of {sol},"
    #     log_msg += f" {volume_water:.2f} uL of water."
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
    #             f"Transferred {volume:.2f} uL of {solution} from {stock_well_id} to {well_id}."
    #         )

    #     self.logger.info("Finished formulating random single wells.")

    # def formulate_random_single_well(
    #     self,
    #     well_ids: list[str],
    #     concentrations_per_well: dict[str, dict[str, float]],
    #     well_volume: float = 150.0,
    # ):
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

    #     # Find all components to dispense
    #     all_solutions = set()
    #     for well_id in well_ids:
    #         all_solutions.update(concentrations_per_well.get(well_id, {}).keys())

    #     # solution_plan: well_id → solution → (target_conc, stock_conc, volume, stock_well_id)
    #     solution_plan = {well_id: {} for well_id in well_ids}
    #     total_volumes = {well_id: 0.0 for well_id in well_ids}

    #     for well_id in well_ids:
    #         concentrations = concentrations_per_well.get(well_id, {})
    #         for solution, target_conc in concentrations.items():
    #             if pd.isna(target_conc) or target_conc <= 0:
    #                 self.logger.warning(f"Skipping {solution} in {well_id} due to missing or invalid concentration.")
    #                 continue

    #             try:
    #                 suitable_wells = get_list_of_well_ids_concentration(
    #                     containers=self.containers,
    #                     solution=solution,
    #                     requested_concentration=target_conc,
    #                 )
    #                 stock_well_id = suitable_wells[0]
    #                 stock_conc = float(self.containers[stock_well_id].concentration)
    #                 volume_needed = (target_conc * well_volume) / stock_conc

    #                 if not (0 <= volume_needed <= well_volume):
    #                     raise ValueError(f"Calculated volume {volume_needed:.2f} uL for {solution} in {well_id} is out of range.")

    #                 solution_plan[well_id][solution] = (target_conc, stock_conc, volume_needed, stock_well_id)
    #                 total_volumes[well_id] += volume_needed
    #             except ValueError as e:
    #                 self.logger.error(f"{e}")
    #                 continue

    #     # Transfer each solution across all wells that need it
    #     for solution in all_solutions:
    #         for well_id in well_ids:
    #             if solution not in solution_plan[well_id]:
    #                 continue
    #             _, _, volume, stock_well_id = solution_plan[well_id][solution]

    #             self._transfer(
    #                 volume=volume,
    #                 source=self.containers[stock_well_id],
    #                 destination=self.containers[well_id],
    #                 solution=solution,
    #             )

    #             self.logger.debug(
    #                 f"Transferred {volume:.2f} uL of {solution} from {stock_well_id} to {well_id}."
    #             )

    #     # Transfer water last
    #     for well_id in well_ids:
    #         remaining = well_volume - total_volumes[well_id]
    #         if remaining < -safety_margin:
    #             self.logger.error(f"Total volume {total_volumes[well_id]:.2f} uL in {well_id} exceeds well capacity.")
    #             continue

    #         volume_water = max(remaining, 0.0)

    #         self._transfer(
    #             volume=volume_water,
    #             source=self.containers[well_id_water],
    #             destination=self.containers[well_id],
    #             solution="water",
    #         )

    #         self.logger.debug(
    #             f"Transferred {volume_water:.2f} uL of water to {well_id}."
    #         )

    #     self.logger.info("Finished formulating wells: " + ", ".join(well_ids))

    # def compute_stock_requirements(self, df, stock_conc_list, stock_label):
    #     stock_usage = {}
    #     for conc in stock_conc_list:
    #         used = df[df[f'stock_{stock_label}_used'] == conc][f'vol_stock_{stock_label}'].sum()
    #         # print(f"{stock_label}, {conc} mM, {used} uL.")
    #         # Add 1000 uL as buffer
    #         stock_usage[conc] = used + 1000
    #     return stock_usage

    # def make_dilution_plan(self, max_stocks, stock_list, main_label, stock_requirements):
    #     plan = {}
    #     for i, conc in enumerate(stock_list):
    #         if i == 0:
    #             # Use max_stocks as the source concentration for the first (highest) dilution
    #             plan[conc] = {
    #                 'source_conc': max_stocks[main_label],     # max stocks for this label
    #                 'source_label': f"main_{main_label}",
    #                 'target_conc': conc,
    #                 'total_vol': stock_requirements[conc],
    #                 'dilution_factor': max_stocks[main_label] / conc
    #             }
    #         else:
    #             prev = stock_list[i - 1]
    #             plan[conc] = {
    #                 'source_conc': prev,
    #                 'source_label': f"{main_label}_{prev}",
    #                 'target_conc': conc,
    #                 'total_vol': stock_requirements[conc],
    #                 'dilution_factor': prev / conc
    #             }
    #     return plan

    # def formulate_batches(self, batch_df: pd.DataFrame, well_volume: int):
    #     well_id_water = get_well_id_solution(
    #         containers=self.containers, solution_name="water"
    #     )

    #     print("Step 1: Pipette solution1 stocks (low → high to avoid contamination)")
    #     sol1_sort = batch_df.sort_values("dil1", ascending=True)
    #     self.right_pipette.pick_up_tip()
    #     for _, row in sol1_sort.iterrows():
    #         well = row["well_id"]
    #         src = f"x_{row['dil1']}"
    #         vol = row["vol1"]
    #         self._transfer(volume=vol, source=src, destination=well, drop_tip_behavior="keep", side="left")
    #     self.right_pipette.drop_tip()

    #     print("Step 2: Pipette solution2 stocks (low → high to avoid contamination)")
    #     sol2_sort = batch_df.sort_values("dil2", ascending=True)
    #     self.right_pipette.pick_up_tip()
    #     for _, row in sol2_sort.iterrows():
    #         well = row["well_id"]
    #         src = f"y_{row['dil2']}"
    #         vol = row["vol2"]
    #         self._transfer(volume=vol, source=src, destination=well, drop_tip_behavior="keep", side="right")
    #     self.right_pipette.drop_tip()

    #     print("Step 3: Add MilliQ and mix")
    #     for _, row in batch_df.iterrows():
    #         self.right_pipette.pick_up_tip()
    #         well = row["well_id"]
    #         vol = row["vol_water"]
    #         if vol == 0:
    #             self.right_pipette.mixing(container=well, mix=("after", ))
    #         else:
    #             self.right_pipette.transfer(
    #             volume=vol,
    #             source=well_id_water,
    #             destination=well,
    #             mix=("after", int(round(well_volume / 1.2, 0)), self.mixing_steps),
    #         )
    #         self.right_pipette.drop_tip()

    #     return batch_df["well_id"].tolist()
