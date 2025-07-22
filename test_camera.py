from hardware.cameras import PendantDropCamera
import cv2
import numpy as np

pd_camera = PendantDropCamera()

# # Initialize the camera
# # pd_camera.stop_measurement()
pd_camera._initialize_camera()
pd_camera.initialize_measurement(well_id="7H1", drop_count=1)
# pd_camera.start_check(vol_droplet=12.700000)
pd_camera.start_capture()
pd_camera.start_stream()

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
        print(pd_camera.st_t)
        print(pd_camera.wortington_numbers)
        pd_camera.stop_measurement()
        break

cv2.destroyAllWindows()
