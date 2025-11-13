## Packages
import time
import numpy as np
from typing import Dict

## Custom code
from opentrons_api.logger import Logger
from opentrons_api.pipette import Pipette
from opentrons_api.containers import Container
from pendantprop.hardware.cameras.pd_cam import PendantDropCamera
from pendantprop.hardware.cameras.pd_cam_mock import MockPendantDropCamera
from pendantprop.analysis.plots import Plotter


class DropletManager:
    def __init__(
        self,
        settings: dict,
        left_pipette: Pipette,
        containers: Dict[str, Container],
    ):
        self.settings = settings
        self.file_settings = settings["file_settings"]
        self.pendant_drop_settings = settings["pendant_drop_settings"]

        self.left_pipette = left_pipette
        self.containers = containers
        self.plotter = Plotter(settings=settings)

        self.logger = Logger(
            name="protocol",
            file_path=f'{self.file_settings["output_folder"]}/{self.file_settings["exp_tag"]}/{self.file_settings["meta_data_folder"]}',
        )

        # pendant drop settings
        self.MAX_RETRIES = int(self.pendant_drop_settings["max_retries"])
        self.DROP_VOLUME_DECREASE_AFTER_RETRY = float(
            self.pendant_drop_settings["drop_volume_decrease_after_retry"]
        )
        self.PENDANT_DROP_DEPTH_OFFSET = float(
            self.pendant_drop_settings["pendant_drop_depth_offset"]
        )
        self.FLOW_RATE = float(self.pendant_drop_settings["flow_rate"])
        self.CHECK_TIME = float(self.pendant_drop_settings["check_time"])
        self.DROP_VOLUME_INCREASE_RESOLUTION = float(self.pendant_drop_settings[
            "drop_volume_increase_resolution"
        ])
        self.INITIAL_DROP_VOLUME = float(self.pendant_drop_settings["initial_drop_volume"])
        self.WORTINGTON_NUMBER_LIMIT_LOWER = float(self.pendant_drop_settings[
            "worthington_limit_lower"
        ])
        self.MAX_MEASURE_TIME = float(self.pendant_drop_settings["max_measure_time"])
        self.WELL_ID_DROP_STAGE = self.pendant_drop_settings["well_id_drop_stage"]

        
        simulate = self.settings["general_settings"]["simulate"]
        if simulate:
            self.logger.info("Using Mock Pendant Drop Camera (simulate mode enabled)")
            self.pd_cam = MockPendantDropCamera(settings=settings)
            self.CHECK_TIME = 0.00001  # speed up checks in simulate mode
            self.MAX_MEASURE_TIME = 0.001  # speed up measurements in simulate mode
        else:
            self.logger.info("Using Real Pendant Drop Camera")
            self.pd_cam = PendantDropCamera(settings=settings)
    
    def measure_pendant_drop(self, source: Container):
        """
        Measure pendant drop with retries if the initial measurement fails.
        """
        # set sample id if not set
        if source.sample_id is None:
            source.sample_id = source.WELL_ID

        # log start
        content_str = ", ".join(
            [f"{compound}: {conc} mM" for compound, conc in source.content.items()]
        )
        self.logger.info(
            f"Start pendant drop measurement of {source.WELL_ID}, containing [{content_str}].\n"
        )
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
                max_measure_time=self.MAX_MEASURE_TIME
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
        return dynamic_surface_tension, drop_volume, self.MAX_MEASURE_TIME, self.drop_count #TODO later dynamic measure time

    def _prepare_pendant_drop(self):
        self.logger.info("Preparing pendant drop.")
        # initialize left pipette

        if not self.left_pipette.has_tip:
            self.left_pipette.pick_up_tip()

        self.left_pipette.mixing(container=self.source, volume_mix=15, repeat=3, touch_tip=False) #? in settings?
        self.left_pipette.aspirate(volume=17, source=self.source, flow_rate=15)
        self.left_pipette.air_gap(air_volume=3)
        self.left_pipette.remove_air_gap(container=self.containers[self.WELL_ID_DROP_STAGE])

    def _reduce_pendant_drop_volume(self, drop_volume_decrease: float):
        self.logger.info(
            f"Reducing pendant drop volume by {drop_volume_decrease}."
        )
        self.left_pipette.aspirate(
            volume=drop_volume_decrease,
            source=self.containers[self.WELL_ID_DROP_STAGE],
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
        self.pd_cam.start_capture_before_measurement()
        self.left_pipette.dispense(
            volume=self.INITIAL_DROP_VOLUME,
            destination=self.containers[self.WELL_ID_DROP_STAGE],
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
            not (self.WORTINGTON_NUMBER_LIMIT_LOWER <= wortington_number)
            and drop_volume < 17
        ):
            self.left_pipette.dispense(
                volume=self.DROP_VOLUME_INCREASE_RESOLUTION,
                destination=self.containers[self.WELL_ID_DROP_STAGE],
                flow_rate=self.FLOW_RATE,
                depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                log=False,
                update_info=False,
            )
            drop_volume += self.DROP_VOLUME_INCREASE_RESOLUTION
            self.pd_cam.start_check(vol_droplet=drop_volume)
            time.sleep(self.CHECK_TIME)
            wortington_numbers = self.pd_cam.wortington_numbers
            self.pd_cam.stop_check()
            if len(wortington_numbers) >= 1:
                wortington_number = np.mean(wortington_numbers)
            # else:
            #     wortington_number = 0

            # print(f"Wortington number: {wortington_number:2f}".ljust(30), end="\r")

        if wortington_number < self.WORTINGTON_NUMBER_LIMIT_LOWER:
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
            source=self.containers[self.WELL_ID_DROP_STAGE],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            log=False,
            update_info=False,
        )  # aspirate drop in tip
        self.logger.info("Re-aspirated the pendant drop into the tip.")
        self.left_pipette.dispense(volume=17, destination=self.source)
        self.logger.info("Returned volume in needle to source.")
        self._close_camera()

    def _initialise_camera(self):
        sample_id = f"{self.source.sample_id}"
        self.pd_cam.initialize_measurement(
            sample_id=sample_id, drop_count=self.drop_count
        )

    def _capture(self, max_measure_time: float):
        self.pd_cam.start_capture()
        start_time = time.time()
        prev_len_st = 0
        while time.time() - start_time < max_measure_time:
            time.sleep(10) #? in settings?
            dynamic_surface_tension = self.pd_cam.st_t
            self.plotter.plot_dynamic_surface_tension(
                dynamic_surface_tension=dynamic_surface_tension,
                container=self.source,
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
            if last_st < 20: #? in settings?
                drop_time = time.time() - start_time
                self.logger.warning(f"Droplet dropped. Drop time: {drop_time:2f} seconds.")
                valid_measurement = False
                self.pd_cam.stop_capture()
                return dynamic_surface_tension, valid_measurement, drop_time

        self.pd_cam.stop_capture()
        self.logger.info("Successful pendant drop measurement.")
        valid_measurement = True
        drop_time = 0
        return dynamic_surface_tension, valid_measurement, drop_time

    def _close_camera(self):
        self.pd_cam.stop_measurement()