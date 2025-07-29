from hardware.cameras import PendantDropCamera
from utils.utils import calculate_equillibrium_value
import cv2
import numpy as np

pd_camera = PendantDropCamera()

# Initialize the camera
pd_camera._initialize_camera()

# Start stream camera
pd_camera.start_stream()

# Uncomment if you want to check if analysis scripts runs proper
pd_camera.initialize_measurement(well_id="7E3", drop_count=1)
# pd_camera.start_capture_before_measurement()
pd_camera.start_capture()


# # Test the frame generator
for frame_data in pd_camera.generate_frames():
    # Decode the JPEG frame
    frame = cv2.imdecode(np.frombuffer(frame_data.split(b'\r\n\r\n')[1], np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        # Zoom out 2x (resize to 50%)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Pendant Drop Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        st_t = pd_camera.st_t

        st_eq = calculate_equillibrium_value(x=st_t, n_eq_points=len(st_t), column_index=1)
        # uncomment if you want to see dynamic surface tension
        print(f"equilibrium surface tension: {st_eq}")
        
        pd_camera.stop_measurement()
        break

cv2.destroyAllWindows()
