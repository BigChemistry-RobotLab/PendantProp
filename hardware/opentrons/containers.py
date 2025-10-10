"""
TODO
----
- maybe i messed up by changing the version of aionotify (opentrons requires 0.2.0, but have pip upgraded to 0.3.1)
- Eppendorf child class
- Well diameter from labware_info, not hard coded.
"""

import numpy as np
import os
from utils.logger import Logger
from utils.load_save_functions import load_settings

# from hardware.opentrons.pipette import Pipette


class Container:
    def __init__(
        self,
        labware_info: dict,
        well: str,
        initial_volume_mL: float = 0,
        solutes: dict = None,
        inner_diameter_mm: float = None,
    ):
        # Settings
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
        self.solutes = solutes if solutes is not None else {}

        # Create logger (container & protocol)
        os.makedirs(f"experiments/{self.settings['EXPERIMENT_NAME']}/data", exist_ok=True)
        os.makedirs(
            f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{self.WELL_ID}",
            exist_ok=True,
        )
        self.container_logger = Logger(
            name=self.WELL_ID,
            file_path=f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{self.WELL_ID}",
        )
        self.protocol_logger = Logger(
            name="protocol",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )
        
    def get_concentration(self, solute_name: str = "all") -> float | list:
        """
        Calculates the concentration of a specific solute in the container.

        Args:
            solute_name (str): The name of the solute to check.

        Returns:
            float: The concentration of the solute in mM (millimolar), or 0 if not present.
            list: If all concentrations (mM) present are required.
        """
        if solute_name == "all":
            solute_concs = list(self.solutes.values())
            return [x / self.volume_mL for x in solute_concs]

        if self.volume_mL == 0 or solute_name not in self.solutes:
            return 0.0
        # Concentration (mM) = amount (µmol) / volume (mL)
        print("container debug", self.solutes[solute_name], self.volume_mL)
        return (self.solutes[solute_name] / self.volume_mL)
    
    def get_solution(self) -> str | list:
        solutes = list(self.solutes.keys())
        return solutes

    def get_contents_str(self) -> str:
        """Returns a formatted string of the container's contents for logging."""

        if not self.solutes and self.volume_mL == 0:
            return "empty"
        if not self.solutes:
            return "water"

        # Sort keys for consistent log output
        sorted_solutes = sorted(self.solutes.keys())

        contents_parts = []
        for solute in sorted_solutes:
            conc = self.get_concentration(solute)
            contents_parts.append(f"{conc:.3f} mM {solute}")
        return ", ".join(contents_parts)

    def aspirate(self, volume: float, log = True):
        volume_to_remove_mL = volume / 1000.0
        
        # 1. Check for sufficient volume
        if self.volume_mL < volume_to_remove_mL:
            self.protocol_logger.warning(
                "Aspiration volume is larger than container volume!"
            )
            return
        if len(self.solutes) == 1 and "water" in self.solutes:
            pass
        # 2. Calculate and remove the proportional solute amounts
        elif self.volume_mL > 0:
            # Calculate the fraction of liquid being removed
            removal_factor = volume_to_remove_mL / self.volume_mL
            
            # Iterate over a copy of keys to safely modify the dictionary
            solute_names_to_remove = list(self.solutes.keys())
            for solute_name in solute_names_to_remove:
                
                # Calculate the amount of solute removed
                amount_removed_umol = self.solutes[solute_name] * removal_factor
                
                # Update the remaining amount
                self.solutes[solute_name] -= amount_removed_umol
                
                # Clean up keys if the amount is minimal
                if abs(self.solutes[solute_name]) < 1e-9: 
                    del self.solutes[solute_name]
        else:
            self.container_logger.info(
                f"Edge case found. Container file, code 1."
            )

        # 3. Update total volume
        self.volume_mL -= volume_to_remove_mL
        self.update_liquid_height(volume_mL=self.volume_mL)

        # 4. Logging
        if log:
            contents_str = self.get_contents_str()
            self.container_logger.info(
                f"Aspirated {volume} uL from this container. Contents: {contents_str}."
            )

    def dispense(self, volume: float, source: "Container", log=True):
        volume_to_add_mL = volume / 1000.0

        # This single loop correctly handles all mixing scenarios (diluting, mixing same
        # or different solutions, adding to water, etc.) by tracking the amount of each solute.
        if source.volume_mL > 0:
            for solute_name, source_amount_umol in source.solutes.items():
                
                # Calculate concentration from source's internal tracking
                source_concentration_mM = source_amount_umol / source.volume_mL
                
                # Calculate the amount to transfer
                amount_to_transfer_umol = source_concentration_mM * volume_to_add_mL
                
                # Safely add the transferred amount to the current container
                self.solutes[solute_name] = self.solutes.get(solute_name, 0.0) + amount_to_transfer_umol

        # Update total volume and liquid height
        self.volume_mL += round(volume_to_add_mL,5)
        self.update_liquid_height(volume_mL=self.volume_mL)

        if log:
            source_contents_str = source.get_contents_str()
            dest_contents_str = self.get_contents_str()
            self.container_logger.info(
                f"Dispensed {volume} uL from {source.WELL_ID} ({source_contents_str}) in {self.WELL_ID} ({dest_contents_str}). "
            )
        
        if self.volume_mL > self.MAX_VOLUME: 
            self.protocol_logger.warning("Overflowing of container!")

    def __str__(self):
        return f"""
        Container object

        Container type = {self.CONTAINER_TYPE}
        Well ID = {self.WELL_ID}
        Contents: {self.get_contents_str()}
        Inner diameter: {self.INNER_DIAMETER_MM} mm
        Well: {self.WELL}
        Location: {self.LOCATION}
        Initial height: {self.INITIAL_HEIGHT_MM:.2f} mm
        Current height: {self.height_mm:.2f} mm
        Current volume: {self.volume_mL * 1e3:.0f} uL
        """


class FalconTube15(Container):
    def __init__(self, **kwargs):
        """Initializes a 15 mL Falcon Tube."""
        # Pass all keyword arguments to the parent, adding the specific diameter
        super().__init__(inner_diameter_mm=15.25, **kwargs)
        self.CONTAINER_TYPE = "Falcon tube 15 mL"

    def update_liquid_height(self, volume_mL):
        dead_volume_mL = 1.0
        dispense_bottom_out_mm = 15
        height = ((volume_mL - dead_volume_mL) * 1e3 / 
                  (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2))
        self.height_mm = height + dispense_bottom_out_mm
        self.height_mm -= 3.0  # Optional offset
        return self.height_mm

class FalconTube50(Container):
    def __init__(self, **kwargs):
        """Initializes a 50 mL Falcon Tube."""
        super().__init__(inner_diameter_mm=28.0, **kwargs)
        self.CONTAINER_TYPE = "Falcon tube 50 mL"

    def update_liquid_height(self, volume_mL):
        dead_volume_mL = 5.0
        dispense_bottom_out_mm = 21
        height = ((volume_mL - dead_volume_mL) * 1e3 / 
                  (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2))
        self.height_mm = height + dispense_bottom_out_mm
        return self.height_mm

class GlassVial(Container):
    def __init__(self, **kwargs):
        """Initializes a Glass Vial."""
        super().__init__(inner_diameter_mm=18.0, **kwargs)
        self.CONTAINER_TYPE = "Glass Vial"

    def update_liquid_height(self, volume_mL):
        height = 1e3 * (volume_mL) / (np.pi * (self.INNER_DIAMETER_MM / 2) ** 2)
        self.height_mm = height - 1.0  # Optional offset
        return self.height_mm

class PlateWell(Container):
    def __init__(self, **kwargs):
        """Initializes a well in a plate."""
        super().__init__(inner_diameter_mm=6.96, **kwargs)
        self.CONTAINER_TYPE = "Plate well"

    def update_liquid_height(self, volume_mL):
        # Using a static height as in your original code
        # self.height_mm = self.settings["WELL_DEPTH_MM"] 
        self.height_mm = 10.0 # Placeholder for your settings value
        self.height_mm = self.settings["WELL_DEPTH_MM"] # static height for now
        return self.height_mm


class DropStage:
    def __init__(self, labware_info):
        self.LABWARE_ID = labware_info["labware_id"]
        self.LABWARE_NAME = labware_info["labware_name"]
        self.LOCATION = labware_info["location"]
        self.CONTAINER_TYPE = "Cuvette"
        self.WELL = "A1"
        self.WELL_ID = f"{self.LOCATION}{self.WELL}"
        self.DEPTH = labware_info["depth"]
        self.height_mm = labware_info["depth"]
        self.MAX_VOLUME = labware_info["max_volume"]
        self.solutes = {}

    def aspirate(self, volume, log=True):
        pass

    def dispense(self, volume, source: Container, log=True):
        self.solutes = source.solutes.copy()

    def get_contents_str(self) -> str:
        """Helper to describe the drop's contents."""
        if not self.solutes:
            return "empty"
        # This part requires access to the total volume of the drop to calculate
        # concentration, which isn't tracked here. For now, we list the solutes.
        return ", ".join(self.solutes.keys())

    def __str__(self):
        return (f"Drop stage object\n\n"
                f"  Container type: {self.CONTAINER_TYPE}\n"
                f"  Well: {self.WELL}\n"
                f"  Location: {self.LOCATION}\n"
                f"  Current Content: {self.get_contents_str()}\n"
                f"  Drop height: {self.height_mm:.2f} mm")


class LightHolder:
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
        print(
            "Attempted to aspirate from light holder. This should never be the case!"
        )
        pass

    def dispense(self, volume, source: Container):
        print(
            "Attempted to dispense from light holder. This should never be the case!"
        )
        pass

    def __str__(self):
        return f"""
        Light holder object:

        Container type: {self.CONTAINER_TYPE}
        Location: {self.LOCATION}
        """


class Sponge:
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
        if self.index == len(self.ORDERING) - 1:
            print("Sponge is fully used! resetting index to 0.")
            self.index = 0

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
