# Imports

## Packages
import numpy as np
import os

## Custom code
from utils.logger import Logger


class Container:
    def __init__(
        self,
        labware_info: dict,
        well: str,
        initial_volume_mL: float = 0,
        content: dict = {"empty": "pure"},
        inner_diameter_mm: float = None,
    ):
        # Settings
        from utils.load_save_functions import load_settings

        self.settings = load_settings()

        # Constant attributes
        self.WELL = well
        self.LOCATION = labware_info["location"]
        self.WELL_ID = f"{self.LOCATION}{self.WELL}"
        self.INITIAL_VOLUME_ML = initial_volume_mL
        self.MAX_VOLUME = labware_info["max_volume"]
        self.LABWARE_ID = labware_info["labware_id"]
        self.LABWARE_NAME = labware_info["labware_name"]
        self.DEPTH = labware_info["depth"]
        self.INNER_DIAMETER_MM = inner_diameter_mm
        self.INITIAL_HEIGHT_MM = self.update_liquid_height(
            volume_mL=self.INITIAL_VOLUME_ML
        )
        self.WELL_DIAMETER = labware_info["well_diameter"]
        self.CONTAINER_TYPE = None

        # Variable attributes
        self.volume_mL = self.INITIAL_VOLUME_ML
        self.height_mm = self.INITIAL_HEIGHT_MM

        self.content = content

        # Create logger (container & protocol)
        os.makedirs(
            f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{self.LABWARE_NAME}",
            exist_ok=True,
        )
        os.makedirs(
            f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{self.LABWARE_NAME}/{self.WELL_ID}",
            exist_ok=True,
        )
        self.container_logger = Logger(
            name=self.WELL_ID,
            file_path=f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{self.LABWARE_NAME}/{self.WELL_ID}",
        )
        self.protocol_logger = Logger(
            name="protocol",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )

    def aspirate(self, volume: float, log=True):
        if self.volume_mL < (volume * 1e-3):
            self.protocol_logger.warning(
                "Aspiration volume is larger than container volume!"
            )
            return
        if self._check_if_container_is_empty(self.content):
            self.protocol_logger.warning(f"Source {self.WELL_ID} has no content")
            return

        self.volume_mL -= volume * 1e-3
        self.update_liquid_height(volume_mL=self.volume_mL)
        if log:
            content_str = ", ".join(
                [f"{compound}: {conc} mM" for compound, conc in self.content.items()]
            )
            self.container_logger.info(
                f"Container: aspirated {volume} uL from this container with content [{content_str}]."
            )

    def dispense(self, volume: float, source: "Container", log=True):
        if (self.volume_mL * 1e3) + volume > self.MAX_VOLUME:
            self.protocol_logger.warning("Overflowing of container!")
            return

        self.volume_mL += volume * 1e-3
        self.update_liquid_height(volume_mL=self.volume_mL)

        # case 1:the source is empty -> warning, no dispensing from an empty container
        if self._check_if_container_is_empty(content=source.content):
            self.protocol_logger.warning(
                f"Source {source.WELL_ID} of the dispensing step is empty."
            )
            self.container_logger.warning(
                f"Container: source of the dispensing step is empty."
            )
            return

        # case 2: container is empty and source contains either water or formulation (not empty)
        if self._check_if_container_is_empty(
            self.content
        ) and not self._check_if_container_is_empty(source.content):
            self.content = source.content

        # case 3: container contains water and formulation is added from source
        elif (
            self._check_water_in_content(self.content)
            and self._check_formulation_in_content(source.content)
        ):
            new_content = {}
            for compound, conc in source.content.items():
                n_source = float(conc) * volume * 1e-3
                c_compound = n_source / self.volume_mL
                new_content[compound] = c_compound
            self.content = new_content

        # case 4: container contains formulation and water is added from source
        elif (
            self._check_formulation_in_content(self.content)
            and self._check_water_in_content(source.content)
        ):
            # Dilute all compounds in self.content according to new total volume
            new_content = {}
            for compound, conc in self.content.items():
                n_compound = float(conc) * (self.volume_mL - volume * 1e-3)
                new_conc = n_compound / self.volume_mL
                new_content[compound] = new_conc
            self.content = new_content

        # case 5: container contains formulation and contains formulation as well
        elif self._check_formulation_in_content(self.content) and self._check_formulation_in_content(source.content):
            # Calculate total moles for each compound in both container and source
            total_moles = {}

            # Moles in container
            for compound, conc in self.content.items():
                n_container = float(conc) * (self.volume_mL - volume * 1e-3)  # volume before addition
                total_moles[compound] = n_container

            # Moles in source
            for compound, conc in source.content.items():
                n_source = float(conc) * (volume * 1e-3)
                if compound in total_moles:
                    total_moles[compound] += n_source
                else:
                    total_moles[compound] = n_source

            # Calculate new concentrations after mixing
            new_content = {}
            for compound, n_total in total_moles.items():
                new_conc = n_total / self.volume_mL
                new_content[compound] = new_conc

            self.content = new_content

        if log:
            content_source_str = ", ".join(
                [f"{compound}: {conc} mM" for compound, conc in source.content.items()]
            )
            content_str = ", ".join(
                [f"{compound}: {conc} mM" for compound, conc in self.content.items()]
            )
            self.container_logger.info(
                f"Container: dispensed {volume} uL into this container from source {source.WELL_ID} containing [{content_source_str}]. Current content container [{content_str}] with a total volume of {self.volume_mL} mL."
            )

    def _check_water_in_content(self, content: dict):
        contains_water = "water" in list(content.keys())
        if len(content.keys()) > 1 and contains_water:
            self.protocol_logger.warning(
                "Found content with water and more compounds. Something went wrong here."
            )
        if len(content.keys()) == 0:
            self.protocol_logger.warning(
                "Found content with length 0. Something went wrong here."
            )
        return contains_water

    def _check_if_container_is_empty(self, content: dict):
        is_empty = "empty" in list(content.keys())
        if len(content.keys()) > 1 and is_empty:
            self.protocol_logger.warning(
                "Found content empty and more compounds. Something went wrong here."
            )
        if len(content.keys()) == 0:
            self.protocol_logger.warning(
                "Found content with length 0. Something went wrong here."
            )
        return is_empty
    
    def _check_formulation_in_content(self, content: dict):
        contains_formulation = False
        if not self._check_water_in_content(content=content) and not self._check_if_container_is_empty(content=content):
            contains_formulation = True
        return contains_formulation
    

    def __str__(self):
        content_str = ", ".join(
            [f"{compound}: {conc} mM" for compound, conc in self.content.items()]
        )

        return f"""
        Container object

        Container type = {self.CONTAINER_TYPE}
        Well ID = {self.WELL_ID}
        Content: {content_str}
        Inner diameter: {self.INNER_DIAMETER_MM} mm
        Well: {self.WELL}
        Location: {self.LOCATION}
        Initial height: {self.INITIAL_HEIGHT_MM:.2f} mm
        Current height: {self.height_mm:.2f} mm
        Current volume: {self.volume_mL * 1e3:.0f} uL
        """


class Eppendorf(Container):
    """
    Container class for 1.5 mL eppendorf
    Accurate to 0.2 mL
    """

    def __init__(
        self, labware_info: dict, well: str, initial_volume_mL: float, content: dict
    ):
        super().__init__(
            labware_info,
            well,
            initial_volume_mL,
            content,
            inner_diameter_mm=10,
        )
        self.CONTAINER_TYPE = "Eppendorf 1.5 mL"

    def update_liquid_height(self, volume_mL):
        HEIGHT_CONE_MM = 18.6  # this matches for cone part of eppendorf tube of 1.5 mL
        INNER_DIAMETER_CONE_MM = 2  # this matches for the inner diameter of the truncated cone part of eppendorf tube of 1.5 mL
        VOLUME_CONE_ML = (
            1e-3
            * (1 / 12)
            * np.pi
            * HEIGHT_CONE_MM
            * (
                self.INNER_DIAMETER_MM**2
                + self.INNER_DIAMETER_MM * INNER_DIAMETER_CONE_MM
                + INNER_DIAMETER_CONE_MM**2
            )
        )
        GENERAL_OFFSET = 3.5  # due to dead volume
        if volume_mL < VOLUME_CONE_ML:
            self.height_mm = (
                1e3
                * 12
                * volume_mL
                / (
                    np.pi
                    * (
                        self.INNER_DIAMETER_MM**2
                        + self.INNER_DIAMETER_MM * INNER_DIAMETER_CONE_MM
                        + INNER_DIAMETER_CONE_MM**2
                    )
                )
                - GENERAL_OFFSET
            )
        else:
            self.height_mm = (
                1e3
                * (volume_mL - VOLUME_CONE_ML)
                / (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2)
                + HEIGHT_CONE_MM
                - GENERAL_OFFSET
            )
        return self.height_mm


class FalconTube15(Container):
    """
    Container class for 15 mL Falcon tube
    Accurate to 1 mL
    """

    def __init__(
        self, labware_info: dict, well: str, initial_volume_mL: float, content
    ):
        super().__init__(
            labware_info,
            well,
            initial_volume_mL,
            content,
            inner_diameter_mm=15.25,
        )
        self.CONTAINER_TYPE = "Falcon tube 15 mL"

    def update_liquid_height(self, volume_mL):
        dead_volume_mL = 1.0
        dispense_bottom_out_mm = 15
        self.height_mm = (
            (volume_mL - dead_volume_mL)
            * 1e3
            / (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2)
        ) + dispense_bottom_out_mm
        self.height_mm -= 3.0  # TODO: check if this is correct
        return self.height_mm


class FalconTube50(Container):
    """
    Container class for 15 mL Falcon tube
    Accurate to 5 mL
    """

    def __init__(
        self, labware_info: dict, well: str, initial_volume_mL: float, content
    ):
        super().__init__(
            labware_info,
            well,
            initial_volume_mL,
            content,
            inner_diameter_mm=28,
        )
        self.CONTAINER_TYPE = "Falcon tube 50 mL"

    def update_liquid_height(self, volume_mL):
        dead_volume_mL = 5
        dispense_bottom_out_mm = 21
        self.height_mm = (
            (volume_mL - dead_volume_mL)
            * 1e3
            / (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2)
        ) + dispense_bottom_out_mm
        return self.height_mm


class GlassVial(Container):
    """
    Container class for 5 mL glass vial
    Accurate to 0.5 mL
    """

    def __init__(
        self, labware_info: dict, well: str, initial_volume_mL: float, content
    ):
        super().__init__(
            labware_info,
            well,
            initial_volume_mL,
            content,
            inner_diameter_mm=18,
        )
        self.CONTAINER_TYPE = "Glass Vial"

    def update_liquid_height(self, volume_mL):
        self.height_mm = 1e3 * (volume_mL) / (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2)
        return self.height_mm - 1


class PlateWell(Container):
    """
    Container class for a well in a plate (max volume 400 uL)
    Accurate t0 40 uL
    """

    def __init__(
        self, labware_info: dict, well: str, initial_volume_mL: float, content
    ):
        super().__init__(
            labware_info,
            well,
            initial_volume_mL,
            content,
            inner_diameter_mm=6.96,
        )
        self.CONTAINER_TYPE = "Plate well"

    def update_liquid_height(self, volume_mL):
        # self.height_mm = 1e3 * (volume_mL) / (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2)
        self.height_mm = self.settings["WELL_DEPTH_MM"]  # static height for now
        return self.height_mm


class DropStage:
    """
    Drop stage representation as container
    """

    def __init__(self, labware_info):
        self.LABWARE_ID = labware_info["labware_id"]
        self.LABWARE_NAME = labware_info["labware_name"]
        self.LOCATION = labware_info["location"]
        self.CONTAINER_TYPE = "Cuvette"
        self.WELL = "A1"  # always 1 well in drop stage
        self.WELL_ID = f"{self.LOCATION}{self.WELL}"
        self.DEPTH = labware_info["depth"]
        self.height_mm = labware_info["depth"]
        self.MAX_VOLUME = labware_info["max_volume"]
        self.solution_name = "empty"
        self.concentration = "pure"

    def aspirate(self, volume, log=True):
        pass

    def dispense(self, volume, source: Container, log=True):
        self.solution_name = source.solution_name
        pass

    def __str__(self):
        return f"""
        Drop stage object

        Container type: {self.CONTAINER_TYPE}
        Well: {self.WELL}
        Location: {self.LOCATION}
        Drop height:  {self.height_mm:.2f} mm
        """


class LightHolder:
    """
    Light stage representation as container
    """

    def __init__(self, labware_info):
        self.LABWARE_ID = labware_info["labware_id"]
        self.LABWARE_NAME = labware_info["labware_name"]
        self.LOCATION = labware_info["location"]
        self.CONTAINER_TYPE = "Light holder"
        self.WELL = "A1"  # place holder
        self.WELL_ID = f"{self.LOCATION}{self.WELL}"
        self.DEPTH = labware_info["depth"]
        self.height_mm = labware_info["depth"]
        self.MAX_VOLUME = labware_info["max_volume"]

    def aspirate(self, volume):
        print("Attempted to aspirate from light holder. This should never be the case!")
        pass

    def dispense(self, volume, source: Container):
        print("Attempted to dispense from light holder. This should never be the case!")
        pass

    def __str__(self):
        return f"""
        Light holder object:

        Container type: {self.CONTAINER_TYPE}
        Location: {self.LOCATION}
        """


class Sponge:
    """
    Sponge representation as container
    """

    def __init__(self, labware_info):
        self.LABWARE_ID = labware_info["labware_id"]
        self.LABWARE_NAME = labware_info["labware_name"]
        self.LOCATION = labware_info["location"]
        self.CONTAINER_TYPE = "Sponge"
        self.ORDERING = labware_info["ordering"]
        self.DEPTH = labware_info["depth"]
        self.height_mm = labware_info["depth"]
        self.MAX_VOLUME = labware_info["max_volume"]

        self.index = 0
        self.well = self.ORDERING[self.index]

    def update_well(self):
        self.index += 1
        self.well = self.ORDERING[self.index]

    def aspirate(self, volume):
        print("Attempted to aspirate from sponge. This should never be the case!")
        pass

    def dispense(self, volume, source: Container):
        print("Attempted to dispense from sponge. This should never be the case!")
        pass

    def __str__(self):
        return f"""
        Sponge object:

        Container type: {self.CONTAINER_TYPE}
        Location: {self.LOCATION}
        Current well: {self.well}
        """
