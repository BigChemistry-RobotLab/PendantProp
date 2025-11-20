import os
import cv2
import threading
from threading import Event
from datetime import datetime
from pypylon import pylon
import time

from pendantprop.analysis.image_analysis import PendantDropAnalysis
from opentrons_api.logger import Logger

class PendantDropCamera:
    def __init__(self, settings: dict):
        self.settings = settings
        self.file_settings = settings["file_settings"]
        self.camera_settings = settings["camera_settings"]
        self.capture_interval = float(self.camera_settings["capture_interval"])
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
        self.stop_background_threads = Event()
        self.capturing = False
        self.streaming = False
        self.checking = False
        self.current_image = None
        self.analysis_image = None
        self.stream_thread = None
        self.process_thread = None
        self.check_thread = None
        self.sample_id = None
        self.st_t = []  # Surface tension measurements
        self.wortington_numbers = []
        self.scales = []
        self.capturing_before_measurement = False
        self.capture_before_measurement_thread = None

    def initialize_measurement(self, sample_id: str, drop_count: int):
        self.save_root = f"{self.file_settings['output_folder_pictures']}/{self.file_settings['exp_tag']}/{self.file_settings['data_folder']}"
        self.analyzer = PendantDropAnalysis(settings=self.settings)
        self.logger = Logger(
            name="protocol",
            file_path=f'{self.file_settings["output_folder"]}/{self.file_settings["exp_tag"]}/{self.file_settings["meta_data_folder"]}',
        )
        self.sample_id = sample_id
        self.drop_count = drop_count
        self.start_stream()

    # Stream Management
    def start_stream(self):
        if not self.streaming:
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream, daemon=True)
            self.stream_thread.start()

    def stop_stream(self):
        self.streaming = False
        if self.stream_thread is not None:
            self.stop_background_threads.set()
            self.stream_thread.join()
        self.stream_thread = None
        self.current_image = None

    def _stream(self):
        if self.camera is None:
            return
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while self.streaming and self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                self.current_image = image.GetArray()
                grabResult.Release()
        self.camera.StopGrabbing()

    # Capture Management
    def start_capture(self):
        if not self.capturing:
            self.stop_capture_before_measurement()
            self.start_time = datetime.now()
            self.capturing = True
            self.process_thread = threading.Thread(
                target=self._process_thread, daemon=True
            )
            self.process_thread.start()
            self.logger.info(f"Camera: start capturing sample {self.sample_id}.")

    def stop_capture(self):
        self.capturing = False
        if self.process_thread is not None:
            self.stop_background_threads.set()
            self.process_thread.join()
            self.logger.info("Camera: stopped capturing")
        self.process_thread = None
        self.analysis_image = None
        self.current_image = None

    def _process_thread(self):
        last_save_time = time.time()
        while self.capturing:
            if self.current_image is not None:
                if time.time() - last_save_time >= self.capture_interval:
                    self._save_image(self.current_image)
                    self._save_img_for_stream(self.analysis_image)
                    last_save_time = time.time()
                with self.lock:
                    self._analyze_image(self.current_image)
            time.sleep(0.1)

    def start_capture_before_measurement(self):
        if not self.capturing_before_measurement:
            self.start_time = datetime.now()
            self.capturing_before_measurement = True
            self.capture_before_measurement_thread = threading.Thread(
                target=self._capture_before_measurement_thread, daemon=True
            )
            self.capture_before_measurement_thread.start()
            self.logger.info("Camera: capturing images before measument.")

    def stop_capture_before_measurement(self):
        self.capturing_before_measurement = False
        if self.capture_before_measurement_thread is not None:
            self.stop_background_threads.set()
            self.capture_before_measurement_thread.join()
            self.logger.info("Camera: stopped capturing before")
        self.capture_before_measurement_thread = None
        self.analysis_image = None
        self.current_image = None

    def _capture_before_measurement_thread(self):
        last_save_time = time.time()
        while self.capturing_before_measurement:
            if self.current_image is not None:
                if time.time() - last_save_time >= 1.0:
                    self._save_image_before_capture(self.current_image)
                    self._save_img_for_stream(self.current_image)
                    last_save_time = time.time()
            time.sleep(0.1)

    # Check Management
    def start_check(self, vol_droplet):
        if not self.checking:
            self.start_time = datetime.now()
            self.checking = True
            self.check_thread = threading.Thread(
                target=self._check, args=(vol_droplet,), daemon=True
            )
            self.check_thread.start()

    def stop_check(self):
        self.checking = False
        if self.check_thread is not None:
            self.check_thread.join()
        self.check_thread = None
        self.analysis_image = None
        self.current_image = None
        self.wortington_numbers = []
        # self.logger.info("Camera: stopped checking")

    def _check(self, vol_droplet):
        while self.checking:
            if self.current_image is not None:
                wortington_number = self._check_image(
                    img=self.current_image, vol_droplet=vol_droplet
                )
                if wortington_number is not None:
                    self.wortington_numbers.append(wortington_number)
                    return wortington_number

    # Image Processing
    def _save_image(self, img):
        directory = f"{self.save_root}/{self.sample_id}/images/droplet_{self.drop_count}"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/{timestamp}.png"
        cv2.imwrite(filename, img)

    def _save_image_before_capture(self, img):
        directory = f"{self.save_root}/{self.sample_id}/images/droplet_{self.drop_count}/before_measurement"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/{timestamp}.png"
        cv2.imwrite(filename, img)

    def _analyze_image(self, img):
        try:
            time_stamp = datetime.now()
            relative_time = (time_stamp - self.start_time).total_seconds()

            st, analysis_image = self.analyzer.img2st(img)

            self.st_t.append([relative_time, st])
            self.analysis_image = analysis_image

        except Exception as e:
            # print(f"Camera: Error analyzing image: {e}")
            self.analysis_image = None

    def _check_image(self, img, vol_droplet):
        try:
            return self.analyzer.img2wo(img=img, vol_droplet=vol_droplet)
        except Exception:
            return None
    
    def start_calibration(self):
        if not self.capturing:
            self.start_time = datetime.now()
            self.calibrating = True
            self.calibration_thread = threading.Thread(
                target=self._calibration_process_thread, daemon=True
            )
            self.calibration_thread.start()
            self.logger.info(f"Camera: start calibration sample {self.sample_id}.")

    def _calibration_process_thread(self):
        last_save_time = time.time()
        while self.calibrating:
            if self.current_image is not None:
                if time.time() - last_save_time >= self.capture_interval:
                    last_save_time = time.time()
                with self.lock:
                    scale = self._calibrate_image(self.current_image)
                    if scale is not None:
                        self.scales.append(scale)
            time.sleep(0.1)
    
    def stop_calibration(self):
        self.capturing = False
        if self.process_thread is not None:
            self.stop_background_threads.set()
            self.process_thread.join()
            self.logger.info("Camera: stopped calibration")
        self.process_thread = None
        self.analysis_image = None
        self.current_image = None

    def _calibrate_image(self, img):
        try:
            scale = self.analyzer.img2scale(img)
            return scale
        except Exception as e:
            return None
        
    def _save_img_for_stream(self, img):
        filename = "pendant_drop_latest.png"
        filepath = self.file_settings["cache_images_folder"]
        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, filename)
        if img is not None:
            cv2.imwrite(full_path, img)

    # Frame Generation
    def generate_frames(self):
        while True:
            if self.analysis_image is not None:
                image4feed = self.analysis_image
            else:
                image4feed = self.current_image

            if image4feed is not None:
                ret, buffer = cv2.imencode(".jpg", image4feed)
                if ret:
                    frame = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
            else:
                time.sleep(0.05)

    # Cleanup
    def stop_measurement(self):
        self.stop_capture()
        self.stop_capture_before_measurement()
        self.stop_check()
        self.stop_stream()
        self.st_t = []
        self.wortington_numbers = []
