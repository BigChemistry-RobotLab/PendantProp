import numpy as np

def get_well_id(containers: dict, solution: str) -> str:
    for key, container in containers.items():
        if "tube" in container.CONTAINER_TYPE:
            if container.solution_name == solution:
                return container.WELL_ID
    raise ValueError(
        f"No container with type 'tube' and solution name '{solution}' found."
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


def get_plate_ids(location: int):
    # letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    letters = ["A", "B"]
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    well_ids = []
    for letter in letters:
        for number in numbers:
            well_id = f"{location}{letter}{number}"
            well_ids.append(well_id)
    return well_ids
