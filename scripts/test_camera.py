import cv2
import time
from pendantprop.hardware.cameras.pd_cam import PendantDropCamera


# Create camera instance and start capturing
simulate = True  # Change to True to use mock camera
if simulate:
    from pendantprop.hardware.cameras.pd_cam_mock import MockPendantDropCamera
    print("Using Mock Camera (simulate mode enabled)")
    pd_cam = MockPendantDropCamera()
else:
    pd_cam = PendantDropCamera()

pd_cam.start_capture()
time.sleep(1)  # Wait for first frame
img = pd_cam.return_image()
#show img this particular moment
if img is not None:
    # Resize to 50% for display
    width = int(img.shape[1] * 0.5)
    height = int(img.shape[0] * 0.5)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Video Stream', resized)
    cv2.waitKey(0)  # Wait for a key press to close the window
pd_cam.close_camera()
cv2.destroyAllWindows()
