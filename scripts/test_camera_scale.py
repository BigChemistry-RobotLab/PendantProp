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
pd_cam.start_calibration()

show_analysis = False
scale_percent = 50  # Reduce to 50% of original size
last_print_time = time.time()
print_interval = 2.0  # Print surface tension every 2 seconds

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
    if pd_cam.scales:
        scale_values = pd_cam.scales
        avg_scale = sum(scale_values) / len(scale_values)
        print(f"\nScale measurements: {len(pd_cam.scales)} samples")
        print(f"Average scale: {avg_scale:.6f} px/mm")

    pd_cam.stop_calibration()
    cv2.destroyAllWindows()
    print("Camera stopped")