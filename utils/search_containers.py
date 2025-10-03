import numpy as np

def find_stock_tubes(containers: dict, tube_type: str = "tube 50"):
    stock_tubes = []
    checked = []
    x = True
    while x:
        exceptions = stock_tubes + checked
        try:
            tube, valid = get_solution_and_conc(
                containers=containers,
                container_type=tube_type,
                exceptions=exceptions
            )
            if valid:
                if any(sub in containers[tube].solution_name for sub in ["water", "trash"]):
                    checked.append(tube)
                else:
                    stock_tubes.append(tube)
            else:
                checked.append(tube)
        except ValueError:
            x = False  # No more tubes to check
    return stock_tubes
   
def find_container(
    containers: dict,
    content: str = "empty",
    type: str = "tube",
    amount: int = 0,
    conc = None,
    location = None,
    checked=None,
    match=None
):
    x = True
    if checked is None:
        checked = []
    if match is None:
        match = []
# Does not give error when less containers than requested are present 
# Ask why this happens, 
    while x:
        exceptions = match + checked
        # print("exceptions: ",exceptions)
        try:
            container, valid = get_solution_and_conc(
                containers=containers,
                solution_name=content,
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


def get_well_id_solution(containers: dict, solution_name: str) -> str:
    for key, container in containers.items():
        if "tube 50" in container.CONTAINER_TYPE:
            if container.solution_name == solution_name:
                return container.WELL_ID
        else: 
            if "tube" in container.CONTAINER_TYPE:
                if container.solution_name == solution_name:
                    return container.WELL_ID
    raise ValueError(
        f"No container with type 'tube' and solution name '{solution_name}' found."
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
    containers = {k: v for k, v in containers.items() if k not in exceptions}

    for key, container in containers.items():
        if container_type in container.CONTAINER_TYPE:
            if (
                (solution_name is None or container.solution_name == solution_name)
                and (conc is None or float(container.concentration) == conc or str(container.concentration) == conc)
                and (location is None or container.LOCATION == location)
                and (volume is None or container.volume_mL == volume)
            ):
                return container.WELL_ID, True  # Matching container
            else:
                return container.WELL_ID, False  # It's a tube, but doesn't match criteria

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
            if container.solution_name == solution:
                differences.append(float(container.concentration) - requested_concentration)
                well_ids.append(key)

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


def get_list_of_well_ids_concentration(
    containers: dict, solution: str, requested_concentration: float
) -> str:
    differences = []
    well_ids = []

    # Collect differences and well IDs
    for key, container in containers.items():
        if "tube" in container.CONTAINER_TYPE or "Plate" in container.CONTAINER_TYPE:
            if container.solution_name == solution:
                differences.append(
                    float(container.concentration) - requested_concentration
                )
                well_ids.append(key)
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
