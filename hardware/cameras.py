import cv2
from pypylon import pylon
import threading
from threading import Thread, Event
import time
import os
from datetime import datetime
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend for non-GUI rendering

from utils.load_save_functions import load_settings
from utils.logger import Logger
from analysis.image_analysis import PendantDropAnalysis


class OpentronCamera:
    def __init__(self, width=640, height=480, fps=60):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.current_frame = None
        self.stop_background_threads = Event()

        if not self.camera.isOpened():
            print("Error: Could not open camera.")
        else:
            # Start the frame capture thread
            self.thread = Thread(target=self.capture_frames, daemon=True)
            self.thread.start()

    def capture_frames(self):
        while not self.stop_background_threads.is_set():
            success, frame = self.camera.read()
            if success:
                self.current_frame = frame
            else:
                print("Error: Could not read frame.")

    def generate_frames(self):
        while True:
            if self.current_frame is not None:
                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode(".jpg", self.current_frame)
                frame = buffer.tobytes()

                # Yield the frame in byte format
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def stop(self):
        self.stop_background_threads.set()
        self.thread.join()
        self.camera.release()


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
        except Exception:
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
        self.well_id = None
        self.st_t = []  # Surface tension measurements
        self.wortington_numbers = []  # Wortington numbers

    def initialize_measurement(
        self, well_id: str, drop_count: int, experiments_dir="experiments"
    ):
        self.settings = load_settings()
        self.experiment_name = self.settings["EXPERIMENT_NAME"]
        self.save_dir = f"{experiments_dir}/{self.experiment_name}/data"
        self.analyzer = PendantDropAnalysis()
        self.logger = Logger(
            name="protocol",
            file_path=f"{experiments_dir}/{self.experiment_name}/meta_data",
        )
        self.well_id = well_id
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
            self.start_time = datetime.now()
            self.capturing = True
            self.process_thread = threading.Thread(
                target=self._process_thread, daemon=True
            )
            self.process_thread.start()
            self.logger.info(f"Camera: start capturing {self.well_id}.")

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
                if time.time() - last_save_time >= 1.0:
                    self._save_image(self.current_image)
                    last_save_time = time.time()
                with self.lock:
                    self._analyze_image(self.current_image)
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
            # self.logger.info("Camera: checking started")

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

    # Image Processing
    def _save_image(self, img):
        directory = f"{self.save_dir}/{self.well_id}/images/droplet_{self.drop_count}"
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
            self.analysis_image = analysis_image

        except Exception:
            self.analysis_image = None

    def _check_image(self, img, vol_droplet):
        try:
            return self.analyzer.image2wortington(img=img, vol_droplet=vol_droplet)
        except Exception:
            return None

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
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
            else:
                time.sleep(0.05)

    # Cleanup
    def stop_measurement(self):
        self.stop_capture()
        self.stop_check()
        self.stop_stream()
        self.st_t = []
        self.wortington_numbers = []


if __name__ == "__main__":
    pass
