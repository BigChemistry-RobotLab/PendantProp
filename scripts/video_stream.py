import cv2
import time
from pendantprop.hardware.cameras.pd_cam import PendantDropCamera


print("=" * 70)
print("Test Video Stream")
print("=" * 70)
print("Press 'q' to quit")
print("=" * 70)

# Create camera instance and start capturing
simulate = True  # Change to True to use mock camera
if simulate:
    from pendantprop.hardware.cameras.pd_cam_mock import MockPendantDropCamera
    print("Using Mock Camera (simulate mode enabled)")
    pd_cam = MockPendantDropCamera()
else:
    pd_cam = PendantDropCamera()

if pd_cam.camera is None:
    print("Camera not found!")
    exit(1)

pd_cam.start_capture()
time.sleep(1)  # Wait for first frame
print("Camera started, streaming...")

try:
    while pd_cam.capture:
        if pd_cam.current_image is not None:
            frame = pd_cam.current_image.copy()
            
            # Resize to 50% for display
            width = int(frame.shape[1] * 0.5)
            height = int(frame.shape[0] * 0.5)
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow('Video Stream', resized)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.01)

except Exception as e:
    print(f"Error: {e}")
finally:
    pd_cam.close_camera()
    cv2.destroyAllWindows()
    print("Stream stopped")
