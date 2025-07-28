import time
import numpy as np
import pandas as pd

from utils.logger import Logger
from hardware.cameras import PendantDropCamera
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.containers import Container
from hardware.opentrons.pipette import Pipette
from analysis.plots import Plotter
from utils.load_save_functions import load_settings


class DropletManager:
    def __init__(
        self,
        left_pipette: Pipette,
        containers: dict,
        pendant_drop_camera: PendantDropCamera,
        opentrons_api: OpentronsAPI,
        plotter: Plotter,
    ):
        settings = load_settings()
        self.left_pipette = left_pipette
        self.containers = containers
        self.pendant_drop_camera = pendant_drop_camera
        self.opentrons_api = opentrons_api
        self.plotter = plotter
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{settings["EXPERIMENT_NAME"]}/meta_data',
        )
        self.MAX_RETRIES = int(settings["DROP_RETRIES"])
        self.DROP_VOLUME_DECREASE_AFTER_RETRY = float(
            settings["DROP_VOLUME_DECREASE_AFTER_RETRY"]
        )
        self.PENDANT_DROP_DEPTH_OFFSET = float(settings["PENDANT_DROP_DEPTH_OFFSET"])
        self.FLOW_RATE = float(settings["FLOW_RATE"])
        self.CHECK_TIME = settings["CHECK_TIME"]
        self.DROP_VOLUME_INCREASE_RESOLUTION = settings[
            "DROP_VOLUME_INCREASE_RESOLUTION"
        ]
        self.INITIAL_DROP_VOLUME = settings["INITIAL_DROP_VOLUME"]
        self.WORTINGTON_NUMBER_LIMIT = settings["WORTINGTON_NUMBER_LIMIT"]

    def measure_pendant_drop(self, source: Container, max_measure_time=60):
        """
        Measure pendant drop with retries if the initial measurement fails.
        """
        # initialize variables
        self.source = source
        drop_volume = self.INITIAL_DROP_VOLUME
        drop_volume_decrease = 0
        drop_time = 0
        valid_measurement = False
        dynamic_surface_tension = []
        self.logger.info(f"Start pendant drop measurement of {source.WELL_ID}.")

        for i in range(1, self.MAX_RETRIES + 1):
            self.drop_count = i
            self.logger.info(f"Attempt {self.drop_count} for pendant drop measurement.")
            # Prepare and dispense pendant drop
            self._prepare_pendant_drop()
            self._initialise_camera()
            valid_droplet, drop_volume = self._dispense_pendant_drop()
            if not valid_droplet:
                valid_measurement = False
                self.logger.warning(
                    f"No valid droplet was created for {self.source.WELL_ID}. Stoppped measurement for this well."
                )
                self._return_pendant_drop(drop_volume=drop_volume)
                dynamic_surface_tension = []  # failed measurement
                return dynamic_surface_tension, drop_volume, self.drop_count

            # reduce drop volume if retry
            if self.drop_count > 1:
                drop_volume_decrease = (self.drop_count-1) * self.DROP_VOLUME_DECREASE_AFTER_RETRY
                self._reduce_pendant_drop_volume(
                    drop_volume_decrease=drop_volume_decrease
                )
                drop_volume -= drop_volume_decrease
                self.logger.info(f"Waiting {drop_time:2f}s for droplet to reach lower surface tension, in order to achieve a pendant drop.")
                time.sleep(drop_time)

            # capture pendant drop measurement
            dynamic_surface_tension, valid_measurement, drop_time = self._capture(
                max_measure_time=max_measure_time
            )
            self._return_pendant_drop(drop_volume=drop_volume)

            if valid_measurement:
                self.logger.info(f"Successful measurement for {self.source.WELL_ID} on attempt {self.drop_count}.")
                break
            else:
                self.logger.warning(f"Measurement failed for {self.source.WELL_ID}. Retrying...")
                drop_volume_decrease += self.DROP_VOLUME_DECREASE_AFTER_RETRY

        # Log final result if all retries fail
        if not valid_measurement:
            self.logger.warning(
                f"No valid measurement was performed for {self.source.WELL_ID} after {self.MAX_RETRIES} attempts."
            )

        return dynamic_surface_tension, drop_volume, self.drop_count

    def _prepare_pendant_drop(self):

        self.logger.info("Preparing pendant drop.")
        # initialize left pipette
        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        self.left_pipette.mixing(container=self.source, mix=("before", 15, 3))
        self.left_pipette.aspirate(volume=17, source=self.source, flow_rate=15)
        self.left_pipette.air_gap(air_volume=3)
        self.left_pipette.clean_on_sponge()
        self.left_pipette.remove_air_gap(at_drop_stage=True)

    def _reduce_pendant_drop_volume(self, drop_volume_decrease: float):
        self.logger.info(
            f"Reducing pendant drop volume by {drop_volume_decrease}."
        )
        self.left_pipette.aspirate(
            volume=drop_volume_decrease,
            source=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            flow_rate=self.FLOW_RATE,
            log=False,
            update_info=False,
        )

    def _dispense_pendant_drop(self):
        wortington_number = 0

        self.logger.info(
            f"Dispensing initial pendant drop volume of {self.INITIAL_DROP_VOLUME}."
        )
        self.pendant_drop_camera.start_capture_before_measurement()
        self.left_pipette.dispense(
            volume=self.INITIAL_DROP_VOLUME,
            destination=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            flow_rate=self.FLOW_RATE,
            log=False,
            update_info=False,
        )

        drop_volume = self.INITIAL_DROP_VOLUME

        self.logger.info(
            "Starting dispensing pendant drop while checking Wortington number."
        )
        while (
            not (self.WORTINGTON_NUMBER_LIMIT <= wortington_number)
            and drop_volume < 14
        ):
            self.left_pipette.dispense(
                volume=self.DROP_VOLUME_INCREASE_RESOLUTION,
                destination=self.containers["drop_stage"],
                flow_rate=self.FLOW_RATE,
                depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                log=False,
                update_info=False,
            )
            drop_volume += self.DROP_VOLUME_INCREASE_RESOLUTION
            self.pendant_drop_camera.start_check(vol_droplet=drop_volume)
            time.sleep(self.CHECK_TIME)
            wortington_numbers = self.pendant_drop_camera.wortington_numbers
            self.pendant_drop_camera.stop_check()
            if len(wortington_numbers) > 1:
                wortington_number = np.mean(wortington_numbers)
            else:
                wortington_number = 0

            print(f"Wortington number: {wortington_number:2f}")

        if wortington_number < self.WORTINGTON_NUMBER_LIMIT:
            self.logger.warning(
                "No valid droplet was created. Wortington number below limit."
            )
            valid_droplet = False
        elif wortington_number > 1:
            self.logger.warning(
                "No valid droplet was created. Wortington number above theoritical limit."
            )
            valid_droplet = False
        else:
            self.logger.info(f"Valid droplet created with drop volume {drop_volume:2f}.")
            valid_droplet = True

        return valid_droplet, drop_volume

    def _return_pendant_drop(self, drop_volume: float):
        self.left_pipette.aspirate(
            volume=drop_volume,
            source=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            log=False,
            update_info=False,
        )  # aspirate drop in tip
        self.logger.info("Re-aspirated the pendant drop into the tip.")
        self.left_pipette.dispense(volume=17, destination=self.source)
        self.logger.info("Returned volume in needle to source.")
        self._close_camera()

    def _initialise_camera(self):
        self.pendant_drop_camera.initialize_measurement(
            well_id=self.source.WELL_ID, drop_count=self.drop_count
        )

    def _capture(self, max_measure_time: float):
        self.pendant_drop_camera.start_capture()
        start_time = time.time()
        prev_len_st = 0
        while time.time() - start_time < max_measure_time:
            time.sleep(10)
            dynamic_surface_tension = self.pendant_drop_camera.st_t
            self.plotter.plot_dynamic_surface_tension(
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=self.source.WELL_ID,
                drop_count=self.drop_count,
            )
            if dynamic_surface_tension:
                last_st = dynamic_surface_tension[-1][1]
            else:  # if no dynamic surface tension is measured, we set last_st to zero
                last_st = 0

            if prev_len_st == len(dynamic_surface_tension):
                self.logger.warning(
                    f"No new data was captured. length {len(dynamic_surface_tension)}.")

            prev_len_st = len(dynamic_surface_tension)
            if last_st < 25:
                drop_time = time.time() - start_time
                self.logger.warning(f"Droplet dropped. Drop time: {drop_time:2f} seconds.")
                valid_measurement = False
                self.pendant_drop_camera.stop_capture()
                return dynamic_surface_tension, valid_measurement, drop_time

        self.pendant_drop_camera.stop_capture()
        self.logger.info("Successful pendant drop measurement.")
        valid_measurement = True
        drop_time = 0
        return dynamic_surface_tension, valid_measurement, drop_time

    def _close_camera(self):
        self.pendant_drop_camera.stop_measurement()
