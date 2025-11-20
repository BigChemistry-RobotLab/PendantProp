from pypylon import pylon
import threading

class PendantDropCamera:
    def __init__(self):
        self.open_camera()

    def open_camera(self):
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
            self.capture_thread = None
        
    def close_camera(self):
        if self.camera:
            self.stop_capture()

    # Stream Management
    def start_capture(self):
        self.capture = True
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.capture_thread.start()

    def stop_capture(self):
        self.capture = False
        if self.capture_thread is not None:
            self.capture_thread.join()
        self.capture_thread = None
        self.current_image = None

    def _capture(self):
        if self.camera is None:
            return
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while self.capture and self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                self.current_image = image.GetArray()
                grabResult.Release()
        self.camera.StopGrabbing()

    def return_image(self):
        return self.current_image

