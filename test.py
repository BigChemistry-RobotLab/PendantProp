import numpy as np
def update_liquid_height(volume_mL):
    dead_volume_mL = 5
    dispense_bottom_out_mm = 21
    height_mm = (
        (volume_mL - dead_volume_mL) * 1e3 / (np.pi * (28 / 2) ** 2)
    ) + dispense_bottom_out_mm
    return height_mm

print(update_liquid_height(volume_mL=50))
