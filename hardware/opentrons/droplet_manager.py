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
        self.max_volume_pendant_drop = int(settings["MAX_VOLUME_PENDANT_DROP"])
        self.delta_threshold = float(settings["DELTA_THRESHOLD"])
        self.PENDANT_DROP_DEPTH_OFFSET = float(settings["PENDANT_DROP_DEPTH_OFFSET"])
        self.FLOW_RATE = float(settings["FLOW_RATE"])
        self.CHECK_TIME = settings["CHECK_TIME"]
        self.DROP_VOLUME_INCREASE_RESOLUTION = settings[
            "DROP_VOLUME_INCREASE_RESOLUTION"
        ]
        self.INITIAL_DROP_VOLUME = settings["INITIAL_DROP_VOLUME"]
        self.WORTINGTON_NUMBER_LIMIT = settings["WORTINGTON_NUMBER_LIMIT"]
        self.WORTINGTON_NUMBER_STEP_SIZE = settings["WORTINGTON_NUMBER_STEP_SIZE"]

    def measure_pendant_drop(self, source: Container, max_measure_time=60):
        """
        Measure pendant drop with retries if the initial measurement fails.
        """
        # initialize variables
        self.source = source
        drop_volume = self.INITIAL_DROP_VOLUME
        drop_volume_decrease = 0
        # self.wt_threshold = 0.26
        valid_measurement = False
        dynamic_surface_tension = []
        self.logger.info(f"Start pendant drop measurement of {source.WELL_ID}.")

        for i in range(1, self.MAX_RETRIES + 1):
            self.drop_count = i
            self.wortington_number_limit = round(self.WORTINGTON_NUMBER_LIMIT - ((self.drop_count-1) * (0.03)), 2)
            
            self.current_check_time = self.CHECK_TIME + (0.7*(self.drop_count-1)) 
            self.current_deltaV = self.DROP_VOLUME_INCREASE_RESOLUTION - ((self.DROP_VOLUME_INCREASE_RESOLUTION/2)*(self.drop_count-1))
            self.logger.info(f"Attempt {self.drop_count} for pendant drop measurement.")
            # Prepare and dispense pendant drop
            self._prepare_pendant_drop()
            self._initialise_camera()
            valid_droplet, drop_volume, init_wt_number = self._dispense_pendant_drop()
            init_drop_volume = 0
            if not valid_droplet:
                valid_measurement = False
                self.logger.warning(
                    f"No valid droplet was created for {self.source.WELL_ID}. Stopped measurement for this well."
                )
                self._return_pendant_drop(drop_volume=drop_volume)
                dynamic_surface_tension = []  # failed measurement
                return dynamic_surface_tension, drop_volume, self.drop_count, 0, 0, 0
            init_drop_volume = drop_volume

            # reduce drop volume if retry
            if self.drop_count > 1:
                drop_volume_decrease = (self.drop_count-1) * self.DROP_VOLUME_DECREASE_AFTER_RETRY
                self._reduce_pendant_drop_volume(
                    drop_volume_decrease=drop_volume_decrease
                )
                drop_volume -= drop_volume_decrease

            # capture pendant drop measurement, might want to start capturing
            # when droplet is created to ease troubleshooting (or make it optional)
            dynamic_surface_tension, valid_measurement, drop_volume, measure_time, wt_number = self._capture(
                max_measure_time=max_measure_time, drop_volume=drop_volume, init_wt_number=init_wt_number
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
            dynamic_surface_tension = 0
        return dynamic_surface_tension, drop_volume, self.drop_count, measure_time, wt_number, init_drop_volume

    def _prepare_pendant_drop(self, ):

        self.logger.info("Preparing pendant drop.")
        # initialize left pipette
        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        self.left_pipette.mixing(container=self.source, mix=("before", 15, 5))
        self.left_pipette.aspirate(volume=self.max_volume_pendant_drop, source=self.source, flow_rate=15)
        # self.left_pipette.air_gap(air_volume=3)
        # self.left_pipette.clean_on_sponge()
        # self.left_pipette.remove_air_gap(at_drop_stage=True)
        self.left_pipette.move_to_well(
            container=self.containers["drop_stage"]
        )

    def _reduce_pendant_drop_volume(self, drop_volume_decrease: float):
        self.logger.info(
            f"Reduced pendant drop volume by {drop_volume_decrease}."
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
        if not hasattr(self, "wortington_number_limit"):
            self.wortington_number_limit = self.WORTINGTON_NUMBER_LIMIT

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
        
        print(f"Wortington number limit: {self.wortington_number_limit:2f}")
        self.logger.info(
            f"Starting dispensing pendant drop while checking for Wortington number {self.wortington_number_limit}."
        )
        break_after = False
        while (
            not (self.wortington_number_limit <= wortington_number <= 1)
            and drop_volume < (self.max_volume_pendant_drop)
        ):
            if (drop_volume + self.DROP_VOLUME_INCREASE_RESOLUTION >= 20):      # Limited by pipette size
                if (20-drop_volume > 0.1):
                    self.left_pipette.dispense(
                    volume=19.9-drop_volume,
                    destination=self.containers["drop_stage"],
                    flow_rate=self.FLOW_RATE,
                    depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                    log=False,
                    update_info=False,
                )
                    drop_volume += 19.9-drop_volume
                break_after = True
            else:
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
            if len(wortington_numbers) >= 1:
                wortington_number = np.mean(wortington_numbers)
            else:
                wortington_number = 0

            print(f"Wortington number: {wortington_number:2f}")
            if break_after:
                break

        if 0.55 < wortington_number < self.wortington_number_limit:
            self.logger.warning(
                "Poor quality droplet, continued to assess."
            )
            valid_droplet = True
        elif wortington_number < self.wortington_number_limit:
            self.logger.warning(
                f"No valid droplet was created. Wortington number below limit ({wortington_number})."
            )
            valid_droplet = False
        elif wortington_number > 1:
            self.logger.warning(
                f"No valid droplet was created. Wortington number above theoretical limit ({wortington_number})."
            )
            valid_droplet = False
        else:
            self.logger.info(f"Valid droplet created with drop volume {drop_volume:2f} and wortington number {wortington_number}.")
            valid_droplet = True

        return valid_droplet, drop_volume, wortington_number

    def _return_pendant_drop(self, drop_volume: float):
        self.left_pipette.aspirate(
            volume=drop_volume,
            source=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            log=False,
            update_info=False,
        )  # aspirate drop in tip
        self.logger.info("Re-aspirated the pendant drop into the tip.")
        self.left_pipette.dispense(volume=self.left_pipette.volume, destination=self.source)
        self.logger.info("Returned volume in needle to source.")
        self._close_camera()

    def _initialise_camera(self):
        self.pendant_drop_camera.initialize_measurement(
            well_id=self.source.WELL_ID, drop_count=self.drop_count
        )

    def _capture(self, max_measure_time: float, drop_volume: float, init_wt_number: float):
        self.pendant_drop_camera.start_capture()
        start_time = time.time()
        prev_len_st = 0
        delta = 1
        delta_count = 0
        base_upper_wt_limit = (self.wortington_number_limit+0.13) - (self.drop_count*0.010)
        base_lower_wt_limit = (self.wortington_number_limit-0.13)
        while time.time() - start_time < max_measure_time:
            time.sleep(3)
            dynamic_surface_tension = self.pendant_drop_camera.st_t
            self.plotter.plot_dynamic_surface_tension(
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=self.source.WELL_ID,
                drop_count=self.drop_count,
            )
            if dynamic_surface_tension:
                last_st = dynamic_surface_tension[-1][1]
            else: 
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
                return dynamic_surface_tension, valid_measurement, drop_volume, drop_time, 0
            
            if time.time() - start_time >= 150 and len(dynamic_surface_tension) >= 400:
                last_values = [pt[1] for pt in dynamic_surface_tension[-400:]]
                avg_last_400 = sum(last_values) / len(last_values)
                avg_last_10 = sum(last_values[-10:]) / len(last_values[-10:])
                delta = abs(avg_last_400 - avg_last_10)
                print(f"Delta: {delta:2f} mN/m")
                print(f"Average last 300 values: {avg_last_400:2f} mN/m")
                print(f"Last value: {avg_last_10:2f} mN/m")
                delta_count += 1
                if delta_count > 10:
                    self.logger.info(f"Delta: {delta:2f} mN/m")
                    self.logger.info(f"Worthington: {wt_number:5f}")
                    delta_count = 0

            if time.time() - start_time >= 30 and len(dynamic_surface_tension) >= 10:
                last_values = [pt[1] for pt in dynamic_surface_tension[-400:]]
                avg_last_10 = sum(last_values[-10:]) / len(last_values[-10:])
                wt_number = self.pendant_drop_camera.analyzer._calculate_wortington(
                            vol_droplet=drop_volume, st=avg_last_10)
                delta_count += 1
                if delta_count > 10:
                    print(f"Wortington number: {wt_number:5f}")
                    delta_count = 0
            
                # if not (0.94 * self.wortington_number_limit <= 
                #         wt_number <= 1.06 * self.wortington_number_limit):
                if not (base_lower_wt_limit * self.wortington_number_limit <= 
                    wt_number <= base_upper_wt_limit):
                    self.logger.info(f"Wortington number before correction: {wt_number:2f}")
                    while wt_number < (base_lower_wt_limit * self.wortington_number_limit):
                        if round(drop_volume + (self.DROP_VOLUME_INCREASE_RESOLUTION)/2,2) >= 20:
                            measure_time = round(time.time()-start_time,2)
                            drop_volume += ((self.DROP_VOLUME_INCREASE_RESOLUTION)/2)
                            return dynamic_surface_tension, False, drop_volume, measure_time, wt_number
                        # Wortington number is too low, dispense to increase drop volume
                        self.left_pipette.dispense(
                            volume=(self.DROP_VOLUME_INCREASE_RESOLUTION)/2,
                            destination=self.containers["drop_stage"],
                            flow_rate=self.FLOW_RATE,
                            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                            log=False,
                            update_info=False,
                        )
                        drop_volume += (self.DROP_VOLUME_INCREASE_RESOLUTION)/2
                        wt_number = self.pendant_drop_camera.analyzer._calculate_wortington(
                            vol_droplet=drop_volume, st=avg_last_10)
                        self.logger.info(f"Increased drop volume to {drop_volume:2f} ul for Wortington number of {wt_number}.")
                    
                    # while wt_number > self.wortington_number_limit:
                    while wt_number > base_upper_wt_limit:
                        if round(drop_volume - (self.DROP_VOLUME_INCREASE_RESOLUTION)/2,2) <= 0.00:
                            measure_time = round(time.time()-start_time,2)
                            drop_volume -= ((self.DROP_VOLUME_INCREASE_RESOLUTION)/2)
                            return dynamic_surface_tension, False, drop_volume, measure_time, wt_number
                        # Wortington number is too high, aspirate to decrease drop volume
                        self.left_pipette.aspirate(
                            volume=(self.DROP_VOLUME_INCREASE_RESOLUTION)/2,
                            source=self.containers["drop_stage"],
                            flow_rate=self.FLOW_RATE,
                            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                            log=False,
                            update_info=False,
                        )
                        drop_volume -= (self.DROP_VOLUME_INCREASE_RESOLUTION)/2
                        self.logger.info(f"Decreased drop volume to {drop_volume:2f} ul for Wortington number of {wt_number}.")
                        wt_number = self.pendant_drop_camera.analyzer._calculate_wortington(
                            vol_droplet=drop_volume, st=avg_last_10)
            if delta < self.delta_threshold:
                measure_time = round(time.time()-start_time,2)
                self.logger.info(
                    f"Dynamic surface tension stabilized at {last_st:2f} mN/m ({delta} mN/m) at {measure_time} seconds. Stopping capture."
                )
                break
        self.pendant_drop_camera.stop_capture()
        self.logger.info("Successful pendant drop measurement.")
        valid_measurement = True
        drop_time = 0   # Can I remove this?
        if 'measure_time' not in locals():  # Should only occur when max_measure_time is reached
            measure_time = max_measure_time
        if 'wt_number' not in locals():     # Not sure why this happened. Seems like the wt_number wasnt calculated before termination
            wt_number = init_wt_number
        return dynamic_surface_tension, valid_measurement, drop_volume, measure_time, wt_number

    def _close_camera(self):
        self.pendant_drop_camera.stop_measurement()
