import os
import cv2
import threading
from threading import Event
from datetime import datetime
import time
import numpy as np

from pendantprop.analysis.image_analysis import PendantDropAnalysis
from opentrons_api.logger import Logger


class MockPendantDropCamera:
    """
    Mock camera for simulating pendant drop experiments.
    Uses a static example image to simulate camera feed and analysis.
    """
    
    def __init__(self, settings: dict):
        self.settings = settings
        self.file_settings = settings["file_settings"]
        self.camera_settings = settings["camera_settings"]
        self.capture_interval = float(self.camera_settings["capture_interval"])
        self._initialize_camera()
        self._initialize_attributes()
        self.lock = threading.Lock()  # Prevent race conditions

    def _initialize_camera(self):
        """Load the example pendant drop image"""
        self.example_image_path = "docs/example_drop.png"
        try:
            self.mock_image = cv2.imread(self.example_image_path)
            if self.mock_image is None:
                print(f"[Mock Camera] Warning: Could not load {self.example_image_path}")
                # Create a blank image as fallback
                self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(self.mock_image, "Mock Camera - No Image", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                print(f"[Mock Camera] Loaded example image from {self.example_image_path}")
            self.camera = "mock"  # Mock camera object
        except Exception as e:
            print(f"[Mock Camera] Error loading image: {e}")
            self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.camera = "mock"

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
            print("[Mock Camera] Started streaming")

    def stop_stream(self):
        self.streaming = False
        if self.stream_thread is not None:
            self.stop_background_threads.set()
            self.stream_thread.join()
        self.stream_thread = None
        self.current_image = None
        print("[Mock Camera] Stopped streaming")

    def _stream(self):
        """Simulate camera stream by continuously providing the mock image"""
        while self.streaming:
            # Add slight random noise to simulate real camera
            noise = np.random.randint(-5, 5, self.mock_image.shape, dtype=np.int16)
            self.current_image = np.clip(self.mock_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            time.sleep(0.033)  # ~30 FPS

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
            self.logger.info(f"[Mock Camera] Started capturing sample {self.sample_id}.")

    def stop_capture(self):
        self.capturing = False
        if self.process_thread is not None:
            self.stop_background_threads.set()
            self.process_thread.join()
            self.logger.info("[Mock Camera] Stopped capturing")
        self.process_thread = None
        self.analysis_image = None
        self.current_image = None

    def _process_thread(self):
        last_save_time = time.time()
        while self.capturing:
            if self.current_image is not None:
                if time.time() - last_save_time >= self.capture_interval:
                    self._save_image(self.current_image)
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
            self.logger.info("[Mock Camera] Capturing images before measurement.")

    def stop_capture_before_measurement(self):
        self.capturing_before_measurement = False
        if self.capture_before_measurement_thread is not None:
            self.stop_background_threads.set()
            self.capture_before_measurement_thread.join()
            self.logger.info("[Mock Camera] Stopped capturing before")
        self.capture_before_measurement_thread = None
        self.analysis_image = None
        self.current_image = None

    def _capture_before_measurement_thread(self):
        last_save_time = time.time()
        while self.capturing_before_measurement:
            if self.current_image is not None:
                if time.time() - last_save_time >= 1.0:
                    self._save_image_before_capture(self.current_image)
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
            self.logger.error(f"[Mock Camera] Error analyzing image: {e}")
            self.analysis_image = None

    def _check_image(self, img, vol_droplet):
        try:
            return self.analyzer.img2wo(img=img, vol_droplet=vol_droplet)
        except Exception:
            return None

    # Frame Generation
    def generate_frames(self):
        """Generate frames for web streaming (MJPEG format)"""
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
        print("[Mock Camera] Measurement stopped")
