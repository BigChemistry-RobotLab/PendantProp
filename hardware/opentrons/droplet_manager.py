import time
import numpy as np

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
        self.DROP_VOLUME_DECREASE_AFTER_RETRY = float(settings["DROP_VOLUME_DECREASE_AFTER_RETRY"])
        self.PENDANT_DROP_DEPTH_OFFSET = float(settings["PENDANT_DROP_DEPTH_OFFSET"])
        # TODO needed? self.depth_offset = -22

    def measure_pendant_drop(
        self, source: Container, drop_parameters: dict, calibrate=False
    ):
        drop_count = 1
        valid_droplet = False
        initial_volume = drop_parameters["drop_volume"]

        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        if not self.left_pipette.has_needle:
            self.left_pipette.pick_up_needle()

        while not valid_droplet and drop_count <= self.MAX_RETRIES:

            drop_parameters["drop_volume"] = (
                initial_volume - self.DROP_VOLUME_DECREASE_AFTER_RETRY * (drop_count - 1)
            )
            self.logger.info(
                f"Start measurment of pendant drop of {source.WELL_ID} with drop volume {drop_parameters['drop_volume']} uL and drop count {drop_count}."
            )
            self._make_pendant_drop(
                source=source,
                drop_volume=drop_parameters["drop_volume"],
                flow_rate=drop_parameters["flow_rate"],
                drop_count=drop_count,
            )
            # time.sleep(5)
            self.pendant_drop_camera.start_capture()

            start_time = time.time()
            while time.time() - start_time < drop_parameters["max_measure_time"]:
                time.sleep(10)
                dynamic_surface_tension = self.pendant_drop_camera.st_t
                self.plotter.plot_dynamic_surface_tension(
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=source.WELL_ID,
                    drop_count=drop_count,
                )
                if dynamic_surface_tension:
                    last_st = dynamic_surface_tension[-1][1]
                else:  # if no dynamic surface tension is measured, we set last_st to zero
                    last_st = 0

                if (
                    last_st < 25
                ):  # check if lower than 10 mN/m (not possible) or that the measure time becomes longer than the last recorded time of the droplet (i.e. no droplet is more found.)
                    drop_count += 1
                    self.pendant_drop_camera.stop_capture()
                    self._return_pendant_drop(
                        source=source, drop_volume=drop_parameters["drop_volume"]
                    )
                    if drop_count < self.MAX_RETRIES:
                        self.logger.warning(
                            f"Failed to create valid droplet for {source.WELL_ID} after {drop_count - 1} attempts. Will try again."
                        )
                        break

                    break
            else:
                valid_droplet = True

        if not valid_droplet:
            # no return?
            self.logger.warning(
                f"Failed to create valid droplet for {source.WELL_ID} after {self.MAX_RETRIES} attempts."
            )

        self.pendant_drop_camera.stop_capture()

        if valid_droplet:
            self._return_pendant_drop(
                source=source, drop_volume=drop_parameters["drop_volume"]
            )
        

        # update drop parameters
        drop_parameters["drop_count"] = drop_count

        if calibrate:
            self.logger.info("Done with calibration of PendantProp.")
            return self.pendant_drop_camera.scale_t, drop_parameters
        else:
            self.logger.info("Done with pendant drop measurement.")
            return self.pendant_drop_camera.st_t, drop_parameters

    def _make_pendant_drop(
        self, source: Container, drop_volume: float, flow_rate: float, drop_count: int
    ):

        self.left_pipette.mixing(container=source, mix=("before", 15, 3))
        self.left_pipette.aspirate(volume=17, source=source, flow_rate=15)
        self.left_pipette.air_gap(air_volume=3)
        self.left_pipette.clean_on_sponge()
        self.left_pipette.remove_air_gap(at_drop_stage=True)
        self.pendant_drop_camera.initialize_measurement(
            well_id=source.WELL_ID, drop_count=drop_count
        )
        self.logger.info("Dispensing pendant drop.")
        self.left_pipette.dispense(
            volume=drop_volume,
            destination=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,  # adjust if needed
            flow_rate=flow_rate,
            log=False,
            update_info=False,
        )
        # time.sleep(30)

    def _return_pendant_drop(self, source: Container, drop_volume: float):
        self.left_pipette.aspirate(
            volume=drop_volume,
            source=self.containers["drop_stage"],
            depth_offset=self.PENDANT_DROP_DEPTH_OFFSET,
            log=False,
            update_info=False,
        )  # aspirate drop in tip
        self.logger.info("Re-aspirated the pendant drop into the tip.")
        self.left_pipette.dispense(volume=17, destination=source)
        self.logger.info("Returned volume in tip to source.")

    def _initialise_camera(self, source: Container, drop_count: int):
        self.pendant_drop_camera.initialize_measurement(
            well_id=source.WELL_ID,
            drop_count=drop_count
        )
    
    def _prepare_pendant_drop(self, source: Container):
        self.left_pipette.mixing(container=source, mix=("before", 15, 3))
        self.left_pipette.aspirate(volume=17, source=source, flow_rate=15)
        self.left_pipette.air_gap(air_volume=3)
        self.left_pipette.clean_on_sponge()
        self.left_pipette.remove_air_gap(at_drop_stage=True)

    def _dispense_pendant_drop(self, flow_rate: float, check_time = 30):
        volume_resolution = 1
        wortington_number = 0
        drop_volume = 0
        while wortington_number < 0.7:
            self.left_pipette.dispense(
                volume = volume_resolution,
                destination=self.containers["drop_stage"],
                flow_rate=flow_rate
            )
            drop_volume += volume_resolution
            self.pendant_drop_camera.start_check(vol_droplet=drop_volume)
            time.sleep(check_time)
            self.pendant_drop_camera.stop_check()
            wortington_numbers = self.pendant_drop_camera.wortington_numbers

            if len(wortington_numbers) > 1:
                wortington_number = np.mean(wortington_numbers)

            else:
                wortington_number = 0
            
            print(wortington_number)
