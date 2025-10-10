import numpy as np

def _get_pure_solute_info(container):
    """
    Helper to safely get the name and concentration of a pure solute.
    Returns (solute_name, concentration) or (None, None) if not a pure solute.
    """
    solute_names = list(container.solutes.keys())
    
    if len(solute_names) == 1:
        solute_name = solute_names[0]
        # Assuming concentration is stored in container.solutes[solute_name].concentration
        # If 'container.solutes' values are the solute objects, this is likely how to get it.
        # This part might need adjustment based on the exact structure of container.solutes values.
        # For now, we'll assume the concentration can be found directly on the solute object.
        try:
             # This is a best guess for the concentration attribute based on common patterns
            concentration = container.solutes[solute_name]/container.volume_mL  # TODO make this work properly instead of band-aid
        except AttributeError:
             # Fallback if the concentration is stored elsewhere or needs different access
            concentration = None 
        
        return solute_name, concentration
    return None, None


import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional

# NOTE: The following helper function is assumed to be defined elsewhere in the class/module:
# from your context, it seems to return the tube and a validity flag,
# and also that _get_pure_solute_info(container_obj) exists to get the name.
# def get_solution_and_conc(containers, container_type, exceptions, solution_name, conc): ...
# def _get_pure_solute_info(container_obj): ... 

def find_stock_tubes(
    containers: Dict[str, Any],
    stock_x: Optional[str] = None,
    stock_y: Optional[str] = None,
    tube_type: str = "tube 50"
) -> Union[List[str], Tuple[str, str]]:
    """
    Finds the container IDs for specified stock solutions (x and y) or,
    if none are specified, finds all non-water/non-trash stock tubes of the given type.

    Args:
        containers (dict): Dictionary mapping container IDs to container objects.
        stock_x (Optional[str]): The solution name for the X-component stock.
        stock_y (Optional[str]): The solution name for the Y-component stock.
        tube_type (str): The type of container to search (e.g., "tube 50").

    Returns:
        Union[List[str], Tuple[str, str]]:
            - If stock_x and stock_y are None: List of all valid stock tube IDs.
            - If stock_x and stock_y are specified: Tuple (well_id_x, well_id_y).
    """

    # --- Case 1: Find specific stock tubes ---
    if stock_x is not None or stock_y is not None:
        well_id_x = None
        well_id_y = None
        
        # Iterating through all known container IDs to find the match.
        # This assumes containers are keyed by WELL_ID, which is needed to return it.
        for tube_id, container_obj in containers.items():

            # dropstage triggers error otherwise
            if not hasattr(container_obj, 'container_type'):
                continue
            # Check if container is the correct type (optional, but good practice)
            if container_obj.container_type != tube_type:
                continue
                
            # Get the actual solution name and concentration from the container object
            solute_name, _ = _get_pure_solute_info(container_obj)
            
            # Check for match with stock_x
            if stock_x is not None and solute_name == stock_x:
                well_id_x = tube_id
                
            # Check for match with stock_y
            if stock_y is not None and solute_name == stock_y:
                well_id_y = tube_id
                
            # Once both are found, we can stop early
            if (stock_x is None or well_id_x is not None) and \
               (stock_y is None or well_id_y is not None):
                break
        
        # Must return two values: well_id_x (string) and well_id_y (string)
        if well_id_x is None:
             raise ValueError(f"Stock tube for solution '{stock_x}' not found.")
        if well_id_y is None:
             raise ValueError(f"Stock tube for solution '{stock_y}' not found.")

        return well_id_x, well_id_y


    # --- Case 2: Find all valid stock tubes (stock_x=None, stock_y=None) ---
    stock_tubes: List[str] = []
    checked_tubes: List[str] = []
    
    while True:
        try:
            # Try to find the next valid tube ID that hasn't been checked
            tube, valid = get_solution_and_conc(
                containers=containers,
                container_type=tube_type,
                exceptions=stock_tubes + checked_tubes, # Exclude already found/checked
                solution_name=None, # Search for ANY solution
                conc=None, 
            )
            
            # If get_solution_and_conc returns False for valid, we mark it checked
            if not valid:
                checked_tubes.append(tube)
                continue
            
            # Check the actual content of the found tube
            container_obj = containers[tube]
            solute_name, _ = _get_pure_solute_info(container_obj)
            
            # Check for "water" or "trash" in the actual solute name
            if solute_name and any(sub in solute_name.lower() for sub in ["water", "trash"]):
                checked_tubes.append(tube)
            else:
                # This is a valid stock tube
                stock_tubes.append(tube)

        except ValueError:
            # This exception signals that no more tubes of that type are available
            break
            
    # Per the requirement, we return the list of stock tube IDs (expected length 50)
    return stock_tubes

def find_container(
    containers: dict,
    content: str = "empty",
    type: str = "tube",
    amount: int = 0,
    conc = None,
    location = None,
    checked=None,
    match=None,
):
    x = True
    if checked is None:
        checked = []
    if match is None:
        match = []

    while x:
        exceptions = match + checked
        try:
            container, valid = get_solution_and_conc(
                containers=containers,
                solution_name=content, # Pass content as the solution_name filter
                container_type=type,
                conc=conc,
                exceptions=exceptions,
                location=location
            )
            if valid:
                match.append(container)
            else:
                checked.append(container)
        except ValueError:
            x = False  # No more tubes to check
    if amount == 0:
        return match
    else: 
        return match[:amount]

def get_well_id_solution(containers: dict, solution_name: str, sort: str = "volume") -> str:
    """
    Finds the well ID of a tube container holding a specific, pure solute.
    """
    search_names = [solution_name]
    found_names = []
    for container in containers.values():
        # Check if the container is a tube type
        if "tube" in container.CONTAINER_TYPE or "Falcon tube" in container.CONTAINER_TYPE:
            try:
                solute_names = list(container.solutes.keys())
                # Check if it contains exactly one solute and its name matches
                if len(solute_names) == 1 and solute_names[0] in search_names:
                    found_names.append(container)
            except AttributeError:
                 # Should not happen after fixing load_containers, but kept for robustness
                 continue 
    if found_names: 
        if sort == "volume":
            return max(found_names, key=lambda c: c.volume_mL).WELL_ID
        if sort == "concentration":
            return sorted(found_names, key=lambda c: c.get_concentration(), reverse=True).WELL_ID
        return 
    else:
        # If the loop finishes without finding a matching container
        if solution_name == "trash":
            return "2B3"
        raise ValueError(
        f"No container with type 'tube' and pure solution name '{solution_name}' found."
        )

def get_solution_and_conc(
    containers: dict,
    solution_name: str = None,
    container_type: str = "tube",
    volume: float = None,
    conc: float = None,
    location: int = None,
    exceptions: list = []
) -> tuple[str, bool]:
    containers_filtered = {k: v for k, v in containers.items() if k not in exceptions}

    for key, container in containers_filtered.items():
        if container_type in container.CONTAINER_TYPE:
            if solution_name == "empty" and container.get_solution() == [] and container.volume_mL == 0:
                return container.WELL_ID, True
            # Check location and volume first, as they are not solute-dependent
            location_match = location is None or container.LOCATION == location
            volume_match = volume is None or container.volume_mL == volume
            
            if location_match and volume_match:
                # Check solute information only if a tube (or other specified type) is found
                solute_name_found, concentration_found = _get_pure_solute_info(container)

                # If it's a pure solute, check name and concentration
                if solute_name_found is not None:
                    name_match = solution_name is None or solute_name_found == solution_name
                    
                    # Need to handle the concentration check, which requires the value from _get_pure_solute_info
                    # float(concentration) == conc or str(concentration) == conc is used in the original
                    conc_match = True
                    if conc is not None:
                        try:
                            # Use the concentration found from the solute object
                            if isinstance(conc, float) or isinstance(conc, int):
                                conc_match = float(concentration_found) == float(conc)
                            elif isinstance(conc, str):
                                conc_match = str(concentration_found) == conc
                            else:
                                conc_match = False # Safety fallback
                        except (ValueError, TypeError):
                            conc_match = False # Fails if concentration is None or not convertible
                    
                    if name_match and conc_match:
                        return container.WELL_ID, True  # Matching container
                    else:
                        return container.WELL_ID, False  # Tube, but doesn't match criteria
                else:
                    # Not a pure solute or solute info not found, but it is the right container type
                    # The original logic returns False (doesn't match criteria) for tubes that don't match criteria, 
                    # so we'll do the same for non-pure tubes to ensure the search continues.
                    return container.WELL_ID, False
            else:
                 # Right container type, but location or volume mismatch
                 return container.WELL_ID, False

    # If no tube found at all
    raise ValueError(
        f"No container with type '{container_type}' found after filtering. Searched for solution '{solution_name}' and conc {conc}."
    )

def get_well_id_concentration(containers: dict, solution: str, requested_concentration: float) -> str:
    differences = []
    well_ids = []

    # Collect differences and well IDs
    for key, container in containers.items():
        if "tube" in container.CONTAINER_TYPE or "Plate" in container.CONTAINER_TYPE:
            
            solute_name_found, concentration_found = _get_pure_solute_info(container)
            
            if solute_name_found == solution and concentration_found is not None:
                try:
                    # Use the concentration found from the solute object
                    container_concentration = float(concentration_found)
                    differences.append(container_concentration - requested_concentration)
                    well_ids.append(key)
                except (ValueError, TypeError):
                    # Skip if concentration is not a number
                    continue

    # Convert differences to a NumPy array and find positive differences
    differences_array = np.array(differences)
    positive_indices = np.where(differences_array > 0)[0]

    if positive_indices.size > 0:
        # Find the index of the smallest positive difference and get the corresponding well ID
        smallest_positive_index = positive_indices[
            np.argmin(differences_array[positive_indices])
        ]
        well_id = well_ids[smallest_positive_index]
        return well_id
    else:
        raise ValueError(
            f"No container found with a higher concentration than requested."
        )

def get_well_contents_dict(containers: dict, well_id: str) -> dict:
    """
    Returns the solutes and their concentrations for a given well as a dictionary.
    If the well is empty, returns {"water": 0.0}.
    
    Parameters:
        containers (dict): Dictionary of container instances keyed by well IDs.
        well_id (str): The well ID to query.
        
    Returns:
        dict: {solute_name: concentration} for all solutes in the well.
    """
    if well_id not in containers:
        raise ValueError(f"Well '{well_id}' not found in containers.")

    container = containers[well_id]

    if not container.solutes:
        if container.volume_mL < 0.00001:
            return{"empty": 0.0}
        else:
            return {"water": 0.0}

    solute_dict = {}
    for solute in container.solutes.keys():
        try:
            conc = container.get_concentration(solute)
        except AttributeError:
            conc = None  # Fallback if get_concentration is not available
        solute_dict[solute] = conc

    return solute_dict


def get_list_of_well_ids_concentration(
    containers: dict, solution: str, requested_concentration: float
) -> str:
    differences = []
    well_ids = []

    # Collect differences and well IDs
    for key, container in containers.items():
        if "tube" in container.CONTAINER_TYPE or "Plate" in container.CONTAINER_TYPE:
            
            solute_name_found, concentration_found = _get_pure_solute_info(container)

            if solute_name_found == solution and concentration_found is not None:
                try:
                    # Use the concentration found from the solute object
                    container_concentration = float(concentration_found)
                    differences.append(
                        container_concentration - requested_concentration
                    )
                    well_ids.append(key)
                except (ValueError, TypeError):
                    # Skip if concentration is not a number
                    continue

    if well_ids == []:
        raise ValueError(f"No container found with the requested solution {solution}.")

    # Convert differences to a NumPy array and find positive differences
    differences_array = np.array(differences)
    positive_indices = np.where(differences_array > 0)[0]
    well_ids = np.array(well_ids)
    # sort well_ids based on the differences
    sorted_indices = np.argsort(differences_array[positive_indices])
    positive_indices = positive_indices[sorted_indices]
    if positive_indices.size > 0:
        well_ids = well_ids[positive_indices]
        # Return as a list of strings (the original type hint is 'str' but 
        # it returns a NumPy array of strings in the original code, 
        # I'll keep the logic of returning the NumPy array)
        return well_ids
    else:
        raise ValueError(
            f"No container found with a higher concentration than the requested {requested_concentration} mM."
        )

def get_plate_ids(location: int):
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # letters = ["A", "B"]
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    well_ids = []
    for letter in letters:
        for number in numbers:
            well_id = f"{location}{letter}{number}"
            well_ids.append(well_id)
    return well_ids