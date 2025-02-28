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

def play_sound(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()



