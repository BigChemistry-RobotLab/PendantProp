import pandas as pd
import os
import glob
import threading
import shutil
import logging
import numpy as np
import cv2
import time

from PendantProp.utils.load_save_functions import (
    save_csv_file,
    load_settings,
    save_settings,
    save_settings_meta_data,
    load_commit_hash,
)
from PendantProp.hardware.cameras import OpentronCamera, PendantDropCamera
from PendantProp.hardware.opentrons.opentrons_api import OpentronsAPI
from PendantProp.hardware.sensor.sensor_api import SensorAPI
from PendantProp.hardware.opentrons.protocol import Protocol
from PendantProp.hardware.opentrons.instruction_formulator import InstructionFormulator


def main():
    # initialize APIs
    opentrons_api = OpentronsAPI()
    sensor_api = SensorAPI()
    pendant_drop_camera = PendantDropCamera()
    protocol = Protocol(
        opentrons_api=opentrons_api,
        sensor_api=sensor_api,
        pendant_drop_camera=pendant_drop_camera,
    )

    # Define stock solutions
    stocks = [
        Container(
            labware_info,
            well,
            initial_volume_mL,
            solution_name="empty",
            concentration=0.1,
            inner_diameter_mm=None,
            experiments_dir="experiments",
        )
    ]

    # Define vessels
    vessels = [
        Container(
            labware_info,
            well,
            initial_volume_mL,
            solution_name="empty",
            concentration=0.1,
            inner_diameter_mm=None,
            experiments_dir="experiments",
        )
    ]

    # define labware
    labware = {}
    for name in ["?"]:
        labware_info = opentrons_api.load_labware(
            labware_name=labware_name,
            labware_file=labware_file,
            location=position,
            custom_labware=custom_labware,
        )

        labware[name] = labware_info

    # define pipettes
    left_pipette = Pipette(
        opentrons_api,
        "mount",
        "pipette_name",
        "pipette_id",
        {"tips_info": "?"},
        {"containers": "?"},
        needle_info=None,
        experiments_dir="experiments",
    )

    right_pipette = Pipette(
        opentrons_api,
        "mount",
        "pipette_name",
        "pipette_id",
        {"tips_info": "?"},
        {"containers": "?"},
        needle_info=None,
        experiments_dir="experiments",
    )

    # Ideally update this with a Protocol.set_formulator() method.
    formulator = InstructionFormulator(
        left_pipette,
        right_pipette,
        stocks,
        vessels,
        labware,
        wash_index=0,
        experiments_dir="experiments",
        experiment_name="example",
    )

    settings = load_settings()

    has_opentrons_camera = settings["HAS_OPENTRONS_CAMERA"] == "True"
    if has_opentrons_camera:
        opentron_camera = OpentronCamera()

    design = pd.read_csv("experiment_design.csv")

    # Create samples
    protocol.formulater.execute_design(
        design,
        volume_conversion_factor=1,
        mix=False,
        drop_tip=False,
        volume_token="/ L",
    )

    # Measure pendant drop


if __name__ == "__main__":
    main()
