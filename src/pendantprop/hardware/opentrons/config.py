# Imports from opentrons_api
from opentrons_api.config import Config as BaseConfig

# Import custom containers
from pendantprop.hardware.opentrons.special_containers import LightHolder, DropStage

# Import base containers that might be needed
from opentrons_api.containers import (
    FalconTube15,
    FalconTube50,
    Eppendorf,
    GlassVial,
    PlateWell
)

class Config(BaseConfig):
    """
    Extended Config class for PendantProp that adds support for custom container types.
    
    This class extends opentrons_api.config.Config to recognize PendantProp-specific
    labware like LightHolder and DropStage in addition to standard containers.
    """
    
    def _find_type(self, labware_name):
        """
        Determine the appropriate container class for a given labware name.
        
        This method extends the base opentrons_api _find_type to include
        PendantProp-specific container types (LightHolder, DropStage) while
        maintaining compatibility with all standard container types.
        
        Args:
            labware_name (str): Name of the labware from the layout CSV
            
        Returns:
            Container class or None if not found
            
        Supported PendantProp containers:
            - Light holder / lightholder -> LightHolder
            - Drop stage / dropstage / cuvette -> DropStage
            
        Supported base containers:
            - tube rack 15 mL -> FalconTube15
            - tube rack 50 mL -> FalconTube50
            - eppendorf rack -> Eppendorf
            - glass vial rack -> GlassVial
            - plate -> PlateWell
        """
        # Check for PendantProp-specific containers first
        labware_lower = labware_name.lower()
        
        # Light holder detection
        if ("light" in labware_lower and "holder" in labware_lower):
            return LightHolder
        
        # Drop stage
        if ("drop" in labware_lower and "stage" in labware_lower):
            return DropStage
        
        # Fall back to base opentrons_api container types
        if "tube rack 15 mL" in labware_name:
            return FalconTube15
        elif "glass vial rack" in labware_name:
            return GlassVial
        elif "tube rack 50 mL" in labware_name:
            return FalconTube50
        elif "eppendorf rack" in labware_name:
            return Eppendorf
        elif "plate" in labware_name:
            return PlateWell
        else:
            print("reached!")
            self.logger.warning(
                f"labware {labware_name} is container, but type is not found!"
            )
            return None


# Export for convenience
__all__ = ['Config']



