import time

from utils.logger import Logger
from hardware.cameras import PendantDropCamera
from hardware.opentrons.http_communications import OpentronsAPI
from hardware.opentrons.containers import Container
from hardware.opentrons.pipette import Pipette
from analysis.plots import Plotter

class DropletManager:
    def __init__(self, left_pipette: Pipette, containers: dict, pendant_drop_camera: PendantDropCamera, opentrons_api: OpentronsAPI, plotter: Plotter,logger: Logger):
        self.left_pipette = left_pipette
        self.containers = containers
        self.pendant_drop_camera = pendant_drop_camera
        self.opentrons_api = opentrons_api
        self.plotter = plotter
        self.logger = logger
        self.max_retries = 2
        self.incremental_decrease_vol = 0.5

    def set_max_retries(self, retries: int):
        self.max_retries = retries

    def set_incremental_decrease_vol(self, delta_vol: float):
        self.incremental_decrease_vol = delta_vol

    def measure_pendant_drop(
        self, source: Container, drop_parameters: dict, calibrate=False
    ): 
        drop_count = 1
        valid_droplet = False
        initial_volume = drop_parameters["drop_volume"]
        self.left_pipette.pick_up_tip()

        while not valid_droplet and drop_count <= self.max_retries:

            drop_parameters["drop_volume"] = initial_volume - self.incremental_decrease_vol * (drop_count - 1)
            self.logger.info(f"Start measurment of pendant drop of {source.WELL_ID} with drop volume {drop_parameters['drop_volume']} uL and drop count {drop_count}.")
            self._make_pendant_drop(
                source=source,
                drop_volume=drop_parameters["drop_volume"],
                flow_rate=drop_parameters["flow_rate"],
                drop_count=drop_count,
            )
            self.pendant_drop_camera.start_capture()

            start_time = time.time()
            while time.time() - start_time < drop_parameters["max_measure_time"]:
                self.opentrons_api.delay(seconds=1)
                time.sleep(1)
                dynamic_surface_tension = self.pendant_drop_camera.st_t
                self.plotter.plot_dynamic_surface_tension(
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=source.WELL_ID,
                    drop_count=drop_count,
                )
                if dynamic_surface_tension:
                    last_st = dynamic_surface_tension[-1][1]
                else: #if no dynamic surface tension is measured, we set last_st to zero
                    last_st = 0

                if (
                    last_st < 25
                ):  # check if lower than 10 mN/m (not possible) or that the measure time becomes longer than the last recorded time of the droplet (i.e. no droplet is more found.)
                    self.logger.warning("No droplet detected, will remake droplet.")
                    drop_count += 1
                    self.pendant_drop_camera.stop_capture()
                    self._return_pendant_drop(
                        source=source, drop_volume=drop_parameters["drop_volume"]
                    )
                    break
            else:
                valid_droplet = True

        if not valid_droplet:
            self.logger.warning(
                f"Failed to create valid droplet for {source.WELL_ID} after {self.max_retries} attempts."
            )

        self.pendant_drop_camera.stop_capture()

        if valid_droplet:
            self._return_pendant_drop(
                source=source, drop_volume=drop_parameters["drop_volume"]
            )
        
        if self.left_pipette.has_tip:
            self.left_pipette.drop_tip()

        # update drop parameters
        drop_parameters["drop_count"] = drop_count
        if calibrate:
            self.logger.info("Done with calibration of PendantProp.")
            return self.pendant_drop_camera.scale_t
        else:
            self.logger.info("Done with pendant drop measurement.")
            return self.pendant_drop_camera.st_t, drop_parameters

    def _make_pendant_drop(
        self, source: Container, drop_volume: float, flow_rate: float, drop_count: int
    ):

        self.left_pipette.mixing(container=source, mix=("before", 15, 3))
        self.left_pipette.aspirate(volume=15, source=source, flow_rate=15)
        self.left_pipette.air_gap(air_volume=5)
        self.left_pipette.clean_tip()
        self.left_pipette.remove_air_gap(at_drop_stage=True)
        self.pendant_drop_camera.initialize_measurement(
            well_id=source.WELL_ID, drop_count=drop_count
        )
        self.left_pipette.dispense(
            volume=drop_volume,
            destination=self.containers["drop_stage"],
            depth_offset=-23.4,  # adjust if needed
            flow_rate=flow_rate,
            log=False,
            update_info=False,
        )

    def _return_pendant_drop(self, source: Container, drop_volume: float):
        self.left_pipette.aspirate(
            volume=drop_volume,
            source=self.containers["drop_stage"],
            depth_offset=-23.4,
            log=False,
            update_info=False,
        )  # aspirate drop in tip
        self.logger.info("Re-aspirated the pendant drop into the tip.")
        self.left_pipette.dispense(volume=15, destination=source)
        self.logger.info("Returned volume in tip to source.")
