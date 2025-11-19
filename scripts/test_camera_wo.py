import cv2
import time
from pendantprop.hardware.cameras.pd_cam import PendantDropCamera
from pendantprop.hardware.cameras.pd_cam_mock import MockPendantDropCamera
from opentrons_api.load_save_functions import load_settings

print("=" * 70)
print("Test Camera - Simple Video Stream")
print("=" * 70)
print("Press 'q' to quit")
print("Press 'a' to toggle analysis view")
print("=" * 70)

settings = load_settings(file_path="config/settings.json")

# Use mock camera if simulate is enabled
if settings.get("general_settings", {}).get("simulate", False):
    print("Using Mock Camera (simulate mode enabled)")
    pd_cam = MockPendantDropCamera(settings=settings)
else:
    print("Using Real Camera")
    pd_cam = PendantDropCamera(settings=settings)

# Initialize camera and start streaming
pd_cam.initialize_measurement(sample_id="TestSample", drop_count=1)
vol_droplet = 19  # Volume in microliters for Worthington number calculation
pd_cam.start_check(vol_droplet=vol_droplet)

show_analysis = False
scale_percent = 50  # Reduce to 50% of original size
last_print_time = time.time()
print_interval = 2.0  # Print Worthington number every 2 seconds

try:
    while True:
        # Get current frame from camera
        if show_analysis and pd_cam.analysis_image is not None:
            frame = pd_cam.analysis_image
        elif pd_cam.current_image is not None:
            frame = pd_cam.current_image
        else:
            frame = None
        
        # Display the frame
        if frame is not None:
            # Resize frame
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow('Pendant Drop Camera', resized)
        
        # Print Worthington number periodically
        current_time = time.time()
        if current_time - last_print_time >= print_interval and pd_cam.wortington_numbers:
            last_wo = pd_cam.wortington_numbers[-1]
            elapsed = current_time - pd_cam.start_time.timestamp()
            print(f"t={elapsed:6.1f}s | Wo={last_wo:6.3f} | n={len(pd_cam.wortington_numbers)} samples")
            last_print_time = current_time
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            show_analysis = not show_analysis
            view = "Analysis" if show_analysis else "Raw"
            print(f"Switched to {view} view")
        
        time.sleep(0.01)  # Small delay to reduce CPU usage

finally:
    if pd_cam.wortington_numbers:
        print(f"Worthington number measurements: {len(pd_cam.wortington_numbers)} samples")
        print(f"Last Wo value: {pd_cam.wortington_numbers[-1]:.3f}")
    pd_cam.stop_measurement()
    cv2.destroyAllWindows()
    print("Camera stopped")