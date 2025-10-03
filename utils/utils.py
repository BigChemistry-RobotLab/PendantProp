import numpy as np
import pyttsx3

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

def get_well_id_from_index(well_index: int, plate_location: int, amount=0):
    """
    Assumes 96 well plate
    """
    list_of_wells = []
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        for i in range(1, 13):
            well_id = f"{plate_location}{letter}{i}"
            list_of_wells.append(well_id)
    amount_well=well_index+amount
    return list_of_wells[well_index:amount_well], amount

def get_wash_well_id_from_index(well_index: int, plate_location: int):
    """
    Assumes 96 well plate
    """
    list_of_wells = []
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        for i in range(1, 13):
            well_id = f"{plate_location}{letter}{i}"
            list_of_wells.append(well_id)
    return list_of_wells[well_index]

def get_tube_id_from_index(tube_index: int, rack_location: int):
    """
    Assumes a 5x3 tube rack
    """
    list_of_tubes = []
    for letter in ["A", "B", "C"]:
        for i in range(1, 6):
            tube_id = f"{rack_location}{letter}{i}"
            list_of_tubes.append(tube_id)
    return list_of_tubes[tube_index]

def play_sound(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    print(get_well_id_from_index(12, 7))


