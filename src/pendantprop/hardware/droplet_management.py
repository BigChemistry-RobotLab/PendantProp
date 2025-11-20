## Packages
import time
import numpy as np
from typing import Dict
import cv2
import os
from datetime import datetime

## Custom code
from opentrons_api.logger import Logger
from opentrons_api.pipette import Pipette
from opentrons_api.containers import Container
from pendantprop.hardware.cameras.pd_cam import PendantDropCamera
from pendantprop.hardware.cameras.pd_cam_mock import MockPendantDropCamera
from pendantprop.analysis.plots import Plotter
from pendantprop.analysis.image_analysis import PendantDropAnalysis

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
        self.analyzer = PendantDropAnalysis(settings=settings)
        self.logger = Logger(
            name="protocol",
            file_path=f'{self.file_settings["output_folder"]}/{self.file_settings["exp_tag"]}/{self.file_settings["meta_data_folder"]}',
        )
        self.save_root_img = f"{self.file_settings['output_folder_pictures']}/{self.file_settings['exp_tag']}/{self.file_settings['data_folder']}"


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
        self.WORTINGTON_NUMBER_LIMIT_UPPER = float(self.pendant_drop_settings[
            "worthington_limit_upper"
        ])
        self.MAX_MEASURE_TIME = float(self.pendant_drop_settings["max_measure_time"])
        self.EQUILIBRATION_TIME = float(self.pendant_drop_settings["equilibration_time"])
        self.EQUILIBRATION_SENSISTIVITY = float(self.pendant_drop_settings["equilibration_sensistivity"])
        self.WELL_ID_DROP_STAGE = self.pendant_drop_settings["well_id_drop_stage"]

        # Create camera instance
        simulate = self.settings["general_settings"]["simulate"]
        if simulate:
            self.logger.info("Using Mock Pendant Drop Camera (simulate mode enabled)")
            self.pd_cam = MockPendantDropCamera()
            self.pd_cam.start_capture()
            # self.CHECK_TIME = 1  # speed up checks in simulate mode
            # self.MAX_MEASURE_TIME = 5  # speed up measurements in simulate mode
        else:
            self.logger.info("Using Real Pendant Drop Camera")
            self.pd_cam = PendantDropCamera()
            self.pd_cam.start_capture()
    
    def measure(self, source: Container):
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
        self.sample_id = source.sample_id
        self.drop_volume = 0
        drop_volume_decrease = 0
        drop_time = 0
        valid_measurement = False
        dynamic_surface_tension = []
        self.logger.info(f"Start pendant drop measurement of {source.WELL_ID}.")

        for i in range(1, self.MAX_RETRIES + 1):
            self.drop_count = i
            self.logger.info(f"Attempt {self.drop_count} for pendant drop measurement.")
            self.drop_count = i
            self.prepare_pendant_drop()
            valid_droplet, self.drop_volume = self.dispense_pendant_drop()
            if not valid_droplet:
                return self.failed_measurement()
            # reduce drop volume if retry
            if self.drop_count > 1:
                drop_volume_decrease = (self.drop_count-1) * self.DROP_VOLUME_DECREASE_AFTER_RETRY
                self.reduce_pendant_drop_volume(
                    drop_volume_decrease=drop_volume_decrease
                )
                self.drop_volume -= drop_volume_decrease
                self.logger.info(f"Waiting {drop_time:2f}s for droplet to reach lower surface tension, in order to achieve a pendant drop.")
                time.sleep(drop_time)

            # measure pendant drop
            dynamic_surface_tension, valid_measurement, drop_time = self.measure_pendant_drop()
            self.return_pendant_drop()

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
        
        drop_parameters = {
            "drop_volume": self.drop_volume,
            "measure_time": self.MAX_MEASURE_TIME,  #TODO later dynamic measure time
            "drop_count": self.drop_count,
            "valid_measurement": valid_measurement,
        }
        
        return dynamic_surface_tension, drop_parameters

    def measure_pendant_drop(self):
        self.logger.info(f"Starting pendant drop measurement (sample ID: {self.sample_id}).")
        start_time = time.time()
        dynamic_surface_tension = []
        valid_measurement = False
        drop_time = 0
        equilibration_start_time = None
        last_st = None

        while time.time() - start_time < self.MAX_MEASURE_TIME:
            wo, st, img, analysis_img = self._average_wo_st_in_time_interval(vol_droplet=self.drop_volume, time_interval=self.CHECK_TIME)
            current_time = time.time() - start_time
            dynamic_surface_tension.append([current_time, st])
            print(f"Surface tension: {st:6.3f} mN/m | Wortington number: {wo:6.3f} | Time: {current_time:6.1f}s")
            
            # Save image for streaming and record
            self._save_img(img = img)
            self._save_img_for_stream(img=analysis_img)

            # Plot dynamic surface tension at intervals
            self.plotter.plot_dynamic_surface_tension(
                dynamic_surface_tension=dynamic_surface_tension,
                container=self.source,
                drop_count=self.drop_count
            )
            if wo < self.WORTINGTON_NUMBER_LIMIT_LOWER:
                self.logger.info(f"Wortington number {wo:6.3f} below lower limit. Increasing drop volume by {self.DROP_VOLUME_INCREASE_RESOLUTION} uL.")
                self.left_pipette.dispense(
                    volume=self.DROP_VOLUME_INCREASE_RESOLUTION,
                    destination=self.containers[self.WELL_ID_DROP_STAGE],
                    depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                    flow_rate=self.FLOW_RATE,
                    log=False,
                    update_info=False,
                )
                self.drop_volume += self.DROP_VOLUME_INCREASE_RESOLUTION
            elif wo > self.WORTINGTON_NUMBER_LIMIT_UPPER:
                self.logger.info(f"Wortington number {wo:6.3f} above upper limit. Decreasing drop volume by {self.DROP_VOLUME_INCREASE_RESOLUTION} uL.")
                self.left_pipette.aspirate(
                    volume=self.DROP_VOLUME_INCREASE_RESOLUTION,
                    source=self.containers[self.WELL_ID_DROP_STAGE],
                    depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
                    flow_rate=self.FLOW_RATE,
                    log=False,
                    update_info=False,
                )
                self.drop_volume -= self.DROP_VOLUME_INCREASE_RESOLUTION

            # Check if surface tension is too low
            if st < 15:
                valid_measurement = False
                drop_time = time.time() - start_time
                break
            
            # Check for equilibration (stable surface tension)
            if last_st is not None:
                st_change = abs(st - last_st)
                if st_change < self.EQUILIBRATION_SENSISTIVITY:  # Surface tension stable within sensitivity threshold
                    if equilibration_start_time is None:
                        equilibration_start_time = time.time()
                    elif time.time() - equilibration_start_time >= self.EQUILIBRATION_TIME:
                        self.logger.info(f"Surface tension equilibrated after {current_time:.1f}s")
                        valid_measurement = True
                        break
                else:
                    equilibration_start_time = None  # Reset if surface tension changes
            
            last_st = st
        
        if valid_measurement:
            self.logger.info("Successful pendant drop measurement.")
        else:
            # Only set valid if loop completed without breaking on low ST
            if not (valid_measurement == False and drop_time > 0):
                self.logger.info("Pendant drop measurement completed (max time reached).")
                valid_measurement = True
        
        return dynamic_surface_tension, valid_measurement, drop_time

    
    def prepare_pendant_drop(self):
        self.logger.info("Preparing pendant drop.")
        # initialize left pipette

        if not self.left_pipette.has_tip:
            self.left_pipette.pick_up_tip()

        self.left_pipette.mixing(container=self.source, volume_mix=15, repeat=3, touch_tip=False)
        self.left_pipette.aspirate(volume=20, source=self.source, flow_rate=10)
    
    def dispense_pendant_drop(self):
        wortington_number = 0

        self.logger.info(
            f"Dispensing initial pendant drop volume of {self.INITIAL_DROP_VOLUME}."
        )
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
            and drop_volume < 20
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
            wortington_number, surface_tension, img, analysis_img = self._average_wo_st_in_time_interval(vol_droplet=drop_volume, time_interval=self.CHECK_TIME)
            print(f"Current drop volume: {drop_volume:2f} uL | Wortington number: {wortington_number:6.3f}")
            self._save_img_before_measurement(img = img)
            self._save_img_for_stream(img = img)
        
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
    
    def reduce_pendant_drop_volume(self, drop_volume_decrease: float):
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
    
    def return_pendant_drop(self):
        self.left_pipette.aspirate(
            volume=self.drop_volume,
            source=self.containers[self.WELL_ID_DROP_STAGE],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            log=False,
            update_info=False,
        )  # aspirate drop in tip
        self.logger.info("Re-aspirated the pendant drop into the tip.")
        self.left_pipette.dispense(volume=20, destination=self.source)
        self.logger.info("Returned volume in needle to source.")
    
    def failed_measurement(self):
        """Handle failed measurement by returning drop and creating drop_parameters dict"""
        self.logger.warning(
            f"No valid droplet was created for {self.source.WELL_ID}. Stopped measurement for this well."
        )
        self.return_pendant_drop()
        dynamic_surface_tension = []  # failed measurement
        
        drop_parameters = {
            "drop_volume": self.drop_volume,
            "measure_time": self.MAX_MEASURE_TIME,
            "drop_count": self.drop_count,
            "valid_measurement": False,
        }
        
        return dynamic_surface_tension, drop_parameters
    
    def _average_wo_st_in_time_interval(self, vol_droplet: float, time_interval: float):
        
        start_time = time.time()
        wortington_numbers = []
        surface_tensions = []
        while time.time() - start_time < time_interval:
            try:
                st, wo, img, analysis_img = self._analyze_current_img(vol_droplet=vol_droplet)
                wortington_numbers.append(wo)
                surface_tensions.append(st)
            except Exception as e:
                pass
            time.sleep(0.01)
        if len(wortington_numbers) == 0:
            return 0, 0, None, None
        return np.mean(wortington_numbers), np.mean(surface_tensions), img, analysis_img
    
    def _save_img_before_measurement(self, img = None):
        if img is None:
            img = self.pd_cam.return_image()
        directory = f"{self.save_root_img}/{self.sample_id}/images/droplet_{self.drop_count}/before_measurement"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/{timestamp}.png"
        cv2.imwrite(filename, img)
    
    def _save_img(self, img = None):
        if img is None:
            img = self.pd_cam.return_image()
        directory = f"{self.save_root_img}/{self.sample_id}/images/droplet_{self.drop_count}"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/{timestamp}.png"
        cv2.imwrite(filename, img)
    
    def _save_img_for_stream(self, img = None):
        if img is None:
            img = self.pd_cam.return_image()
        filename = "pendant_drop_latest.png"
        filepath = self.file_settings["cache_images_folder"]
        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, filename)
        if img is not None:
            cv2.imwrite(full_path, img)

    def _analyze_current_img(self, vol_droplet: float):
        img = self.pd_cam.return_image()
        st, wo, analysis_img = self.analyzer.analyse_image(
            img=img,
            vol_droplet=vol_droplet
        )
        return st, wo, img, analysis_img

    
