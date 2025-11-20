import cv2
import threading
import time

class MockPendantDropCamera:
    """
    Mock camera for simulating pendant drop video stream.
    Uses a static example image to simulate camera feed.
    """
    
    def __init__(self):
        self._initialize_camera()
        self.capture = False
        self.capture_thread = None
        self.current_image = None

    def _initialize_camera(self):
        """Load the example pendant drop image"""
        self.example_image_path = "docs/example_drop.png"
        self.mock_image = cv2.imread(self.example_image_path)
        if self.mock_image is None:
            raise FileNotFoundError(f"Example image not found at {self.example_image_path}")
        print(f"[Mock Camera] Loaded example image from {self.example_image_path}")
        self.camera = "mock"  # Mock camera object

    def open_camera(self):
        """Simulates opening the camera - no-op for mock"""
        print("[Mock Camera] Camera opened (simulated)")
        
    def close_camera(self):
        """Stop capturing and close mock camera"""
        if self.capture:
            self.stop_capture()
        print("[Mock Camera] Camera closed")

    def start_capture(self):
        """Start capturing frames in background thread"""
        self.current_image = self.mock_image


    def stop_capture(self):
        """Stop capturing frames"""
        pass

    def return_image(self):
        """Return the current frame"""
        return self.current_image
