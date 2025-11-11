
class LightHolder:
    """
    Light stage representation as container
    """

    def __init__(self,
            labware_info: dict,
            well: str,
            settings: dict,
            initial_volume_mL: float = 0,
            content: dict = {"empty": "pure"},
            inner_diameter_mm: float = None,
            ):
        self.LABWARE_ID = labware_info["labware_id"]
        self.LABWARE_NAME = labware_info["labware_name"]
        self.LOCATION = labware_info["location"]
        self.CONTAINER_TYPE = "Light holder"
        self.WELL = "A1"  # place holder
        self.WELL_ID = f"{self.LOCATION}{self.WELL}"
        self.DEPTH = labware_info["depth"]
        self.height_mm = labware_info["depth"]
        self.MAX_VOLUME = labware_info["max_volume"]

    def __str__(self):
        return f"""
        Light holder object:

        Container type: {self.CONTAINER_TYPE}
        Location: {self.LOCATION}
        """

class DropStage:
    """
    Drop stage representation as container
    """

    def __init__(self,
            labware_info: dict,
            well: str,
            settings: dict,
            initial_volume_mL: float = 0,
            content: dict = {"empty": "pure"},
            inner_diameter_mm: float = None,
            ):
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

    def __str__(self):
        return f"""
        Drop stage object

        Container type: {self.CONTAINER_TYPE}
        Well: {self.WELL}
        Location: {self.LOCATION}
        Drop height:  {self.height_mm:.2f} mm
        """
