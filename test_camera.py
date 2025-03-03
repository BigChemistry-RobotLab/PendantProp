from hardware.cameras import PendantDropCamera
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd_cam = PendantDropCamera()
pd_cam.initialize_measurement(well_id="7A1", drop_count=1)
pd_cam.start_stream()
pd_cam.start_capture()

cv2.namedWindow("Pendant Drop Camera Feed", cv2.WINDOW_NORMAL)

try:
    buffer = b""
    for frame in pd_cam.generate_frames():
        buffer += frame
        start = buffer.find(b"\xff\xd8")  # JPEG start
        end = buffer.find(b"\xff\xd9")  # JPEG end

        if start != -1 and end != -1:
            jpg = buffer[start : end + 2]
            buffer = buffer[end + 2 :]

            # Decode the frame from bytes to a NumPy array
            np_arr = np.frombuffer(jpg, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                cv2.imshow("Pendant Drop Camera Feed", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    pd_cam.stop_stream()
    pd_cam.stop_capture()
    # cv2.destroyAllWindows()
    # scale_t = pd_cam.scale_t
    # st_t = pd_cam.st_t
    # t = [item[0] for item in scale_t]
    # scale = [item[1] for item in scale_t]
    # st = [item[1] for item in st_t]
    # # save the data in pd dataframe (scale and st)
    # df = pd.DataFrame(scale_t, columns=["time (s)", "scale (g)"])
    # df.to_csv("scale.csv")
    # df = pd.DataFrame(st_t, columns=["time (s)", "surface tension (mN/m)"])
    # df.to_csv("surface_tension.csv")

    # fig, ax = plt.subplots()
    # ax.plot(t, scale)
    # ax.set(xlabel="time (s)", ylabel="scale (g)", title="Scale vs Time")
    # ax.grid()
    # fig.savefig("test.png")

