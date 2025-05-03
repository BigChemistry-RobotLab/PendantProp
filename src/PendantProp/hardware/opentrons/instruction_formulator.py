import os
import pandas as pd
from PendantProp.hardware.opentrons.pipette import Pipette
from PendantProp.hardware.opentrons.containers import Container
from PendantProp.utils.logger import Logger
from PendantProp.utils.search_containers import (
    get_well_id_solution,
)
from PendantProp.utils.utils import get_well_id_from_index


class InstructionFormulator:
    """
    A formulator which can only make samples on instruction.
    """

    def __init__(
        self,
        left_pipette: Pipette,
        right_pipette: Pipette,
        stocks: dict,
        vessels: dict,
        labware: dict,
        wash_index=0,
        experiments_dir="experiments",
        experiment_name="example",
    ):
        """
        left_pipette: Pipette
            ?
        right_pipette: Pipette
            ?
        stocks: dict
            Stock solutions.
        vessels: dict
            Vessels which will hold formulations.
        labware: dict
            ?
        experiments_dir="experiments"
            Directory to which data will be written.
        """

        self.left_pipette = left_pipette
        self.right_pipette = right_pipette
        self.stocks = stocks
        self.vessels = vessels
        self.labware = labware
        self.wash_index = wash_index

        log_path = f"{experiments_dir}/{experiment_name}/meta_data"
        os.makedirs(log_path, exist_ok=True)
        self.logger = Logger(name="protocol", file_path=log_path)

    def transfer(
        self,
        volume: float,
        source: Container,
        destination: Container,
        mix=False,
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

    def execute_design(
        self,
        design: pd.DataFrame,
        volume_conversion_factor=1e6,
        mix=False,
        drop_tip=False,
        volume_token="/ L",
    ):
        """
        Perform an experimental design specified in a spreadsheet.

        design: pd.DataFrame
            Data frame with samples as the index and quantities of stock
            solutions in the columns.

        volume_conversion_factor: float
            Factor required to change the volumes given in the design to
            microlitres.

        mix: bool
            Whether the pipette will perform mixing for each solution transfer
            or not.

        drop_tip: bool
            Whether the pipette tip will be changed for each solution transfer
            or not.

        volume_token: str
            If this token is in a string, then it will be recognised as a
            stock key.
        """

        if design.shape[0] > len(self.vessels):
            self.logger.warning(
                "More samples specified in design than available vessels! \
                 Samples at the end of the design list will not be prepared"
            )

        # Get the columns of the design which correspond to the stock
        # quantities
        stock_columns = [x for x in design.columns if "/ L" in x]
        vessels = self.vessels.keys()

        # Iterate over the design and fill vessels, stock by stock
        for stock in stock_columns:
            stock_volumes = design[stock].to_list()

            source = self.stocks.get(stock.strip("/ L"))
            if source is None:
                self.logger.warning(
                    f"{stock} not in self.stocks. Skipping this component."
                )
                continue

            for vessel, volume in zip(vessels, stock_volumes):
                target = self.vessels[vessel]
                volume_in_ul = volume * volume_conversion_factor

                self.transfer(
                    volume_in_ul,
                    source,
                    target,
                    mix=mix,
                    drop_tip=drop_tip,
                )

            # Make sure tips are changed between stock solutions
            for p in self.pipettes:
                if p.has_tip:
                    p.drop_tip()

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
