# Imports

## Packages
import threading
from pypylon import pylon
from datetime import datetime
import time
import os
import cv2
import matplotlib

## Custom code
from utils.load_save_functions import load_settings
from utils.logger import Logger
from analysis.image_analysis import PendantDropAnalysis
from hardware.opentrons.containers import Container

## Change backend (otherwise error I dont understand) 
matplotlib.use('Agg')


class PendantDropCamera:
    def __init__(self):
        self._initialize_camera()
        self._initialize_attributes()
        self.lock = threading.Lock()  # Prevent race conditions

    def _initialize_camera(self):
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        except Exception as e:
            print(
                "Camera: Could not find pendant drop camera. Close camera software and check cables."
            )
            self.camera = None

    def _initialize_attributes(self):
        self.stop_stream_event = threading.Event()
        self.stop_process_event = threading.Event()
        self.stop_capture_before_measurement_event = threading.Event()
        self.stop_check_event = threading.Event()
        self.capturing = False
        self.capturing_before_measurement = False
        self.streaming = False
        self.checking = False
        self.current_image = None
        self.analysis_image = None
        self.stream_thread = None
        self.process_thread = None
        self.capture_before_measurement_thread = None
        self.check_thread = None
        self.well_id = None
        self.st_t = []  # Surface tension measurements
        self.wortington_numbers = []  # Wortington numbers

    def initialize_measurement(self, container: Container, drop_count: int):
        self.settings = load_settings()
        self.experiment_name = self.settings["EXPERIMENT_NAME"]
        self.analyzer = PendantDropAnalysis()
        self.logger = Logger(
            name="protocol",
            file_path=f"experiments/{self.experiment_name}/meta_data",
        )
        self.well_id = container.WELL_ID
        self.labware_name = container.LABWARE_NAME
        self.save_dir = f"experiments/{self.experiment_name}/data/{self.labware_name}"
        self.drop_count = drop_count
        self.start_stream()

    # Stream Management
    def start_stream(self):
        if not self.streaming:
            self.streaming = True
            self.stop_stream_event.clear()
            self.stream_thread = threading.Thread(target=self._stream, daemon=True)
            self.stream_thread.start()

    def stop_stream(self):
        self.streaming = False
        if self.stream_thread is not None:
            self.stop_stream_event.set()
            self.stream_thread.join()
        self.stream_thread = None
        self.current_image = None

    def _stream(self):
        if self.camera is None:
            return
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        last_cache_save = time.time()
        while self.streaming and self.camera.IsGrabbing() and not self.stop_stream_event.is_set():
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                frame = image.GetArray()
                with self.lock:
                    self.current_image = frame
                grabResult.Release()
                # Save cache image every second
                if time.time() - last_cache_save > 2:
                    self.save_latest_image_to_cache()
                    last_cache_save = time.time()
        self.camera.StopGrabbing()

    # Capture Management
    def start_capture(self):
        if not self.capturing:
            self.stop_capture_before_measurement()
            self.start_time = datetime.now()
            self.capturing = True
            self.stop_process_event.clear()
            self.process_thread = threading.Thread(
                target=self._process_thread, daemon=True
            )
            self.process_thread.start()
            self.logger.info(f"Camera: start capturing {self.well_id}.")

    def stop_capture(self):
        self.capturing = False
        if self.process_thread is not None:
            self.stop_process_event.set()
            self.process_thread.join()
            self.logger.info("Camera: stopped capturing")
        self.process_thread = None
        self.analysis_image = None
        self.current_image = None

    def _process_thread(self):
        last_save_time = time.time()
        while self.capturing and not self.stop_process_event.is_set():
            img = self.current_image
            if img is not None:
                if time.time() - last_save_time >= 1.0:
                    self._save_image(img)
                    last_save_time = time.time()
                self._analyze_image(img)
            time.sleep(0.01)
    
    def start_capture_before_measurement(self):
        if not self.capturing_before_measurement:
            self.start_time = datetime.now()
            self.capturing_before_measurement = True
            self.stop_capture_before_measurement_event.clear()
            self.capture_before_measurement_thread = threading.Thread(
                target=self._capture_before_measurement_thread, daemon=True
            )
            self.capture_before_measurement_thread.start()
            self.logger.info("Camera: capturing images before measument.")
    
    def stop_capture_before_measurement(self):
        self.capturing_before_measurement = False
        if self.capture_before_measurement_thread is not None:
            self.stop_capture_before_measurement_event.set()
            self.capture_before_measurement_thread.join()
            self.logger.info("Camera: stopped capturing before")
        self.capture_before_measurement_thread = None
        self.analysis_image = None
        self.current_image = None

    def _capture_before_measurement_thread(self):
        last_save_time = time.time()
        while self.capturing_before_measurement and not self.stop_capture_before_measurement_event.is_set():
            with self.lock:
                img = self.current_image.copy() if self.current_image is not None else None
            if img is not None:
                if time.time() - last_save_time >= 1.0:
                    self._save_image_before_capture(img)
                    last_save_time = time.time()
            time.sleep(0.1)

    # Check Management
    def start_check(self, vol_droplet):
        if not self.checking:
            self.start_time = datetime.now()
            self.checking = True
            self.stop_check_event.clear()
            self.check_thread = threading.Thread(
                target=self._check, args=(vol_droplet,), daemon=True
            )
            self.check_thread.start()
            # self.logger.info("Camera: checking started")

    def stop_check(self):
        self.checking = False
        if self.check_thread is not None:
            self.stop_check_event.set()
            self.check_thread.join()
        self.check_thread = None
        self.analysis_image = None
        self.current_image = None
        self.wortington_numbers = []
        # self.logger.info("Camera: stopped checking")

    def _check(self, vol_droplet):
        while self.checking:
            with self.lock:
                img = self.current_image.copy() if self.current_image is not None else None
            if img is not None:
                wortington_number = self._check_image(
                    img=img, vol_droplet=vol_droplet
                )
                if wortington_number is not None:
                    self.wortington_numbers.append(wortington_number)

    # Image Processing
    def _save_image(self, img):
        directory = f"{self.save_dir}/{self.well_id}/images/droplet_{self.drop_count}"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/{timestamp}.png"
        cv2.imwrite(filename, img)

    def _save_image_before_capture(self, img):
        directory = f"{self.save_dir}/{self.well_id}/images/droplet_{self.drop_count}/before_measurement"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/{timestamp}.png"
        cv2.imwrite(filename, img)

    def _analyze_image(self, img):
        try:
            time_stamp = datetime.now()
            relative_time = (time_stamp - self.start_time).total_seconds()
            st, analysis_image = self.analyzer.image2st(img)
            self.st_t.append([relative_time, st])
            with self.lock:
                self.analysis_image = analysis_image
        except Exception as e:
            with self.lock:
                self.analysis_image = None

    def _check_image(self, img, vol_droplet):
        try:
            return self.analyzer.image2wortington(img=img, vol_droplet=vol_droplet)
        except Exception:
            return None

    # # Frame Generation
    # def generate_frames(self):
    #     while True:
    #         with self.lock:
    #             image4feed = self.analysis_image if self.analysis_image is not None else self.current_image
    #         if image4feed is not None:
    #             ret, buffer = cv2.imencode(".jpg", image4feed)
    #             if ret:
    #                 frame = buffer.tobytes()
    #                 yield (
    #                     b"--frame\r\n"
    #                     b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    #                 )
    #         else:
    #             time.sleep(0.05)

    def save_latest_image_to_cache(self, cache_path="server/static/plots_cache/pendant_drop_latest.png"):
        with self.lock:
            image4cache = self.analysis_image if self.analysis_image is not None else self.current_image
            if image4cache is not None:
                cv2.imwrite(cache_path, image4cache)

    # Cleanup
    def stop_measurement(self):
        self.stop_capture_before_measurement()
        self.stop_capture()
        self.stop_check()
        self.stop_stream()
        self.st_t = []
        self.wortington_numbers = []