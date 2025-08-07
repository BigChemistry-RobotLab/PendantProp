# Imports

## Packages
import cv2
import numpy as np

## Custom code
from hardware.cameras.pendant_drop_camera import PendantDropCamera
from utils.utils import calculate_equillibrium_value
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.configuration import Configuration

# Initialisation
pd_camera = PendantDropCamera()
pd_camera._initialize_camera()
pd_camera.start_stream()

analysis = True
if analysis:
    opentrons_api = OpentronsAPI()
    opentrons_api.initialise()
    config = Configuration(opentrons_api=opentrons_api)
    labware = config.load_labware()
    containers = config.load_containers()
    container = containers["8H1"]
    pd_camera.initialize_measurement(container=container, drop_count=1)
    # pd_camera.start_capture_before_measurement()
    pd_camera.start_capture()


# Generate frames
for frame_data in pd_camera.generate_frames():
    frame = cv2.imdecode(np.frombuffer(frame_data.split(b'\r\n\r\n')[1], np.uint8), cv2.IMREAD_COLOR) # Decode the JPEG frame
    if frame is not None:
        # Zoom out 2x (resize to 50%)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Pendant Drop Camera", frame)

    # Stop the stream if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        if analysis:
            st_t = pd_camera.st_t
            st_eq = calculate_equillibrium_value(x=st_t, n_eq_points=len(st_t), column_index=1)
            print(f"equilibrium surface tension: {st_eq}")
        
        pd_camera.stop_measurement()
        break

cv2.destroyAllWindows()
