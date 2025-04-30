import numpy as np


def smooth_list(x: list, window_size):
    x_smooth = np.convolve(x, np.ones(window_size), "valid") / window_size
    return x_smooth


def calculate_average_in_column(x: list, column_index: int):
    x_column = [item[column_index] for item in x]
    return np.mean(x_column)


def calculate_equillibrium_value(x: list, n_eq_points: int, column_index: int):
    if len(x) > n_eq_points:
        x = x[-n_eq_points:]
    else:
        print(f"less than {n_eq_points} points.")
    return calculate_average_in_column(x=x, column_index=column_index)


def get_well_id_from_index(well_index: int, plate_location: int):
    """
    Assumes 96 well plate
    """
    list_of_wells = []
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        for i in range(1, 13):
            well_id = f"{plate_location}{letter}{i}"
            list_of_wells.append(well_id)
    return list_of_wells[well_index]


if __name__ == "__main__":
    print(get_well_id_from_index(12, 7))
