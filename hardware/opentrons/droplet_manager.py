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

    def measure_pendant_drop(self, source: Container, max_measure_time=60):

        # set attributes
        self.drop_count = 1
        self.source = source
        self.logger.info(
            f"Start pendant drop measurement of {source.WELL_ID}, drop count {self.drop_count}."
        )

        # initialize left pipette
        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        self._prepare_pendant_drop()
        self._initialise_camera()
        valid_droplet, drop_volume = self._dispense_pendant_drop()
        if valid_droplet:
            dynamic_surface_tension, valid_measurement = self._capture(
                max_measure_time=max_measure_time
            )
            self._return_pendant_drop(drop_volume=drop_volume)
        else:
            valid_measurement = False
            self.logger.warning(
                f"No valid droplet was created for {self.source.WELL_ID}."
            )

        # repeat measurement if droplet fell of the needle during first measurement
        while not valid_measurement and self.drop_count < self.MAX_RETRIES:
            drop_volume = (
                drop_volume - self.drop_count * self.DROP_VOLUME_DECREASE_AFTER_RETRY
            )
            self.drop_count += 1
            self._make_pendant_drop(drop_volume=drop_volume)
            self._initialise_camera()
            dynamic_surface_tension, valid_measurement = self._capture(
                max_measure_time=max_measure_time
            )
            self._return_pendant_drop(drop_volume=drop_volume)

        if not valid_measurement:
            self.logger.warning(
                f"No valid measurement was performed for {self.source.WELL_ID}"
            )

        return dynamic_surface_tension, drop_volume, self.drop_count

    def _make_pendant_drop(self, drop_volume: float):
        self._prepare_pendant_drop()
        self.left_pipette.dispense(
            volume=drop_volume,
            destination=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            flow_rate=self.FLOW_RATE,
            log=False,
            update_info=False,
        )
        time.sleep(10)  #!

    def _prepare_pendant_drop(self):
        self.left_pipette.mixing(container=self.source, mix=("before", 15, 3))
        self.left_pipette.aspirate(volume=17, source=self.source, flow_rate=15)
        self.left_pipette.air_gap(air_volume=3)
        self.left_pipette.clean_on_sponge()
        self.left_pipette.remove_air_gap(at_drop_stage=True)

    def _dispense_pendant_drop(self, check_time=1, volume_resolution=0.25):
        wortington_number = 0
        drop_volume = 0
        flow_rate = self.FLOW_RATE
        self.logger.info("Starting dispensing pendant drop while checking Wortington number.")
        while wortington_number < 0.7 and drop_volume < 17:
            self.left_pipette.dispense(
                volume=volume_resolution,
                destination=self.containers["drop_stage"],
                flow_rate=flow_rate,
                depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                log=False,
                update_info=False,
            )
            drop_volume += volume_resolution
            self.pendant_drop_camera.start_check(vol_droplet=drop_volume)
            time.sleep(check_time)
            wortington_numbers = self.pendant_drop_camera.wortington_numbers
            self.pendant_drop_camera.stop_check()
            if len(wortington_numbers) > 1:
                wortington_number = np.mean(wortington_numbers)
            else:
                wortington_number = 0

        if wortington_number < 0.7:
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
            self.logger.info(f"Valid droplet created with drop volume {drop_volume}.")
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
        self.logger.info("Returned volume in tip to source.")
        self._close_camera()

    def _initialise_camera(self):
        self.pendant_drop_camera.initialize_measurement(
            well_id=self.source.WELL_ID, drop_count=self.drop_count
        )

    def _capture(self, max_measure_time: float):
        self.pendant_drop_camera.start_capture()
        start_time = time.time()
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

            if last_st < 25:
                self.logger.warning("Droplet dropped.")
                valid_measurement = False
                self.pendant_drop_camera.stop_capture()
                return dynamic_surface_tension, valid_measurement

        self.pendant_drop_camera.stop_capture()
        self.logger.info("Successful pendant drop measurement.")
        valid_measurement = True
        return dynamic_surface_tension, valid_measurement

    def _close_camera(self):
        self.pendant_drop_camera.stop_measurement()
