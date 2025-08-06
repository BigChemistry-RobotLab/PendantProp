# Imports

## Packages
import cv2
import numpy as np

## Custom code
from hardware.cameras.pendant_drop_camera import PendantDropCamera
from utils.utils import calculate_equillibrium_value

# Initialisation
pd_camera = PendantDropCamera()
pd_camera._initialize_camera()
pd_camera.start_stream()

# Uncomment if you want to check if analysis scripts runs proper #TODO fix with containers
# pd_camera.initialize_measurement(well_id="7E3", drop_count=1)
# pd_camera.start_capture_before_measurement()
# pd_camera.start_capture()


# Generate frames
for frame_data in pd_camera.generate_frames():
    frame = cv2.imdecode(np.frombuffer(frame_data.split(b'\r\n\r\n')[1], np.uint8), cv2.IMREAD_COLOR) # Decode the JPEG frame
    if frame is not None:
        # Zoom out 2x (resize to 50%)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Pendant Drop Camera", frame)

    # Stop the stream if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        st_t = pd_camera.st_t
        st_eq = calculate_equillibrium_value(x=st_t, n_eq_points=len(st_t), column_index=1)
        # uncomment if you want to see dynamic surface tension
        print(f"equilibrium surface tension: {st_eq}")
        
        pd_camera.stop_measurement()
        break

cv2.destroyAllWindows()
