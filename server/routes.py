import os
import glob
import threading
import shutil
import logging
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    Response,
    jsonify,
)

from utils.load_save_functions import (
    save_csv_file,
    load_settings,
    save_settings,
    save_settings_meta_data,
    load_commit_hash,
)
from hardware.cameras import OpentronCamera, PendantDropCamera
from hardware.opentrons.http_communications import OpentronsAPI
from hardware.sensor.sensor_api import SensorAPI
from hardware.opentrons.protocol import Protocol

# initialize the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")
app.config["UPLOAD_FOLDER"] = "experiments"

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Function to replace images in the static directory with placeholders
def replace_static_images_with_placeholders():
    image_files = glob.glob("server/static/plots_cache/*.png")
    placeholder_dir = "server/static/placehold_images"
    for file in image_files:
        os.remove(file)
        placeholder_file = os.path.join(placeholder_dir, os.path.basename(file))
        if os.path.exists(placeholder_file):
            shutil.copy(placeholder_file, "server/static/plots_cache")

replace_static_images_with_placeholders()

# initialize api's
opentrons_api = OpentronsAPI()
sensor_api = SensorAPI()
pendant_drop_camera = PendantDropCamera()
opentron_camera = OpentronCamera()

# Global variable to store the protocol instance
protocol = None

@app.route("/")
def index():
    file_path_results_plot = "plots_cache/results_plot.png"
    file_path_dynamic_surface_tension = "plots_cache/dynamic_surface_tension_plot.png"
    last_action = session.get("last_action", "None")
    return render_template(
        "index.html",
        last_action=last_action,
        file_path_results_plot=file_path_results_plot,
        file_path_dynamic_surface_tension=file_path_dynamic_surface_tension,
    )


@app.route("/input_settings", methods=["POST"])
def input_settings():
    return render_template("input_settings.html", settings=load_settings())


@app.route("/settings", methods=["POST"])
def settings():
    settings = load_settings()

    # update settings with form data
    ## general settings
    settings["ROBOT_IP"] = request.form.get("ROBOT_IP")
    settings["ROBOT_TYPE"] = request.form.get("ROBOT_TYPE")

    ## pendant drop settings
    settings["DROP_VOLUME"] = request.form.get("DROP_VOLUME")
    settings["EQUILIBRATION_TIME"] = request.form.get("EQUILIBRATION_TIME")
    settings["FLOW_RATE"] = request.form.get("FLOW_RATE")
    settings["DENSITY"] = request.form.get("DENSITY")
    settings["SCALE"] = request.form.get("SCALE")
    settings["DROP_RETRIES"] = request.form.get("DROP_RETRIES")
    settings["DROP_VOLUME_DECREASE_AFTER_RETRY"] = request.form.get(
        "DROP_VOLUME_DECREASE_AFTER_RETRY"
    )

    ## characterization settings
    settings["EXPLORE_POINTS"] = request.form.get("EXPLORE_POINTS")
    settings["EXPLOIT_POINTS"] = request.form.get("EXPLOIT_POINTS")
    settings["WELL_VOLUME"] = request.form.get("WELL_VOLUME")

    save_settings(settings)
    session["last_action"] = "Settings updated"
    return redirect(url_for("index"))


@app.route("/reset_settings", methods=["POST"])
def reset_settings():
    settings = load_settings(file_name="default_settings.json")
    save_settings(settings)
    session["last_action"] = "Settings reset"
    return redirect(url_for("index"))


@app.route("/input_initialisation", methods=["POST"])
def input_initialisation():
    return render_template("input_initialisation.html")


def initialize_protocol():
    global protocol
    protocol = Protocol(
        opentrons_api=opentrons_api,
        sensor_api=sensor_api,
        pendant_drop_camera=pendant_drop_camera,
    )


@app.route("/initialisation", methods=["POST"])
def initialisation():
    exp_name = request.form.get("exp_name")
    csv_file = request.files.get("csv_file")

    settings = load_settings()
    settings["EXPERIMENT_NAME"] = exp_name
    settings["GIT_COMMIT_HASH"] = load_commit_hash()
    settings["CONFIG_FILENAME"] = csv_file.filename
    sub_dir = "meta_data"

    # save settings and layout file
    save_csv_file(exp_name, sub_dir, csv_file, app)
    save_settings_meta_data(settings=settings)
    save_settings(settings)

    # start the initialization of protocol in a separate thread
    thread = threading.Thread(target=initialize_protocol)
    thread.daemon = True
    thread.start()

    session["last_action"] = "Initialisation"
    return redirect(url_for("index"))


@app.route("/input_calibration", methods=["POST"])
def input_calibration():
    return render_template("input_calibration.html")


@app.route("/calibrate", methods=["POST"])
def calibrate():
    global protocol
    if protocol is None:
        session["last_action"] = "Protocol not initialized"
        return redirect(url_for("index"))
    thread = threading.Thread(target=protocol.calibrate)
    thread.daemon = True
    thread.start()
    session["last_action"] = "Calibration"
    return redirect(url_for("index"))


@app.route("/input_measure_wells", methods=["POST"])
def input_measure_wells():
    return render_template("input_measure_wells.html")


@app.route("/measure_wells", methods=["POST"])
def measure_wells():
    global protocol
    if protocol is None:
        session["last_action"] = "Protocol not initialized"
        return redirect(url_for("index"))

    settings = load_settings()
    csv_file = request.files.get("csv_file")
    settings["WELL_INFO_FILENAME"] = csv_file.filename
    save_settings_meta_data(settings=settings)
    save_settings(settings)
    save_csv_file(
        exp_name=settings["EXPERIMENT_NAME"],
        subdir_name="meta_data",
        csv_file=csv_file,
        app=app,
    )
    thread = threading.Thread(target=protocol.measure_wells)
    thread.daemon = True
    thread.start()
    session["last_action"] = "Measuring wells"
    return redirect(url_for("index"))


@app.route("/input_surfactant_characterization", methods=["POST"])
def input_surfactant_characterization():
    return render_template("input_surfactant_characterization.html")


@app.route("/characterize", methods=["POST"])
def characterize():
    global protocol
    if protocol is None:
        session["last_action"] = "Protocol not initialized"
        return redirect(url_for("index"))
    settings = load_settings()
    csv_file = request.files.get("csv_file")
    settings["CHARACTERIZATION_INFO_FILENAME"] = csv_file.filename
    save_settings_meta_data(settings=settings)
    save_settings(settings)
    save_csv_file(
        exp_name=settings["EXPERIMENT_NAME"],
        subdir_name="meta_data",
        csv_file=csv_file,
        app=app,
    )
    thread = threading.Thread(target=protocol.characterize_surfactant)
    thread.daemon = True
    thread.start()
    session["last_action"] = "Surfactant characterization"
    return redirect(url_for("index"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/opentron_video_feed")
def opentron_video_feed():
    return Response(
        opentron_camera.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/pendant_drop_video_feed")
def pendant_drop_video_feed():
    return Response(
        pendant_drop_camera.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/toggle_pendant_drop_camera", methods=["POST"])
def toggle_pendant_drop_camera():
    if pendant_drop_camera.streaming:
        pendant_drop_camera.stop_stream()
        session["last_action"] = "toggle off pendant drop camera"
    else:
        pendant_drop_camera.start_stream()
        session["last_action"] = "toggle on pendant drop camera"

    return redirect(url_for("index"))


@app.route("/status", methods=["POST"])
def status():
    data = request.get_json()
    status = data.get("status")
    print(f"Status: {status}")
    return jsonify({"status": status})


@app.route("/pendant_drop_plot_feed")
def pendant_drop_plot_feed():
    pendant_drop_camera.start_plot_frame_thread()
    return Response(
        pendant_drop_camera.generate_plot_frame(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
