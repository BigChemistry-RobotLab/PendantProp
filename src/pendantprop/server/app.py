"""
PendantProp Flask Server
Simplified server for pendant drop measurements
"""

import os
import glob
import shutil
import threading
import logging
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
)

from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.protocol import Protocol

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "pendantprop_secret_key_2025")
app.config["UPLOAD_FOLDER"] = "config"

# Suppress Flask development server logs (optional)
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

# Global variables
protocol = None
settings = None


def replace_plots_with_placeholders():
    """Replace plot images with placeholders on server start"""
    plot_files = glob.glob("src/pendantprop/server/static/cache_images/*.png")
    placeholder_dir = "src/pendantprop/server/static/placehold_images"
    
    for file in plot_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Copy placeholders if they exist
    if os.path.exists(placeholder_dir):
        for placeholder in glob.glob(f"{placeholder_dir}/*.png"):
            filename = os.path.basename(placeholder)
            dest = f"src/pendantprop/server/static/cache_images/{filename}"
            if os.path.exists(placeholder):
                shutil.copy(placeholder, dest)


# Initialize on startup
replace_plots_with_placeholders()


@app.route("/")
def index():
    """Main page"""
    protocol_status = "Initialized" if protocol is not None else "Not Initialized"
    
    # Load settings to get refresh_rate
    current_settings = load_settings(file_path="config/settings.json")
    refresh_rate = current_settings.get("general_settings", {}).get("refresh_rate", 1.0)
    
    return render_template(
        "index.html",
        protocol_status=protocol_status,
        refresh_rate=refresh_rate,
    )


@app.route("/input_settings", methods=["POST"])
def input_settings():
    """Show settings form with current values"""
    current_settings = load_settings(file_path="config/settings.json")
    return render_template("input_settings.html", settings=current_settings)


@app.route("/update_settings", methods=["POST"])
def update_settings():
    """Update settings from form submission"""
    current_settings = load_settings(file_path="config/settings.json")
    
    # Update general settings
    current_settings["general_settings"]["simulate"] = request.form.get("simulate") == "true"
    current_settings["general_settings"]["refresh_rate"] = float(request.form.get("refresh_rate"))
    
    # Update robot settings
    current_settings["robot_settings"]["robot_ip"] = request.form.get("robot_ip")
    current_settings["robot_settings"]["robot_type"] = request.form.get("robot_type")
    current_settings["robot_settings"]["left_pipette_name"] = request.form.get("left_pipette_name")
    current_settings["robot_settings"]["right_pipette_name"] = request.form.get("right_pipette_name")
    current_settings["robot_settings"]["well_depth_mm"] = float(request.form.get("well_depth_mm"))
    
    # Update wash settings
    current_settings["wash_settings"]["wash_volume_ul"] = int(request.form.get("wash_volume_ul"))
    current_settings["wash_settings"]["wash_repeats"] = int(request.form.get("wash_repeats"))
    current_settings["wash_settings"]["mixing_volume_ul"] = int(request.form.get("mixing_volume_ul"))
    current_settings["wash_settings"]["mix_repeats"] = int(request.form.get("mix_repeats"))
    
    # Update sensor settings
    current_settings["sensor_settings"]["serial_port"] = request.form.get("serial_port")
    
    # Update pendant drop settings
    current_settings["pendant_drop_settings"]["explore_points"] = int(request.form.get("explore_points"))
    current_settings["pendant_drop_settings"]["worthington_limit_lower"] = float(request.form.get("worthington_limit_lower"))
    current_settings["pendant_drop_settings"]["worthington_limit_upper"] = float(request.form.get("worthington_limit_upper"))
    current_settings["pendant_drop_settings"]["initial_drop_volume"] = float(request.form.get("initial_drop_volume"))
    current_settings["pendant_drop_settings"]["check_time"] = float(request.form.get("check_time"))
    current_settings["pendant_drop_settings"]["equilibration_time"] = float(request.form.get("equilibration_time"))
    current_settings["pendant_drop_settings"]["max_measure_time"] = float(request.form.get("max_measure_time"))
    current_settings["pendant_drop_settings"]["max_retries"] = int(request.form.get("max_retries"))
    current_settings["pendant_drop_settings"]["drop_volume_decrease_after_retry"] = float(request.form.get("drop_volume_decrease_after_retry"))
    current_settings["pendant_drop_settings"]["drop_volume_increase_resolution"] = float(request.form.get("drop_volume_increase_resolution"))
    current_settings["pendant_drop_settings"]["flow_rate"] = float(request.form.get("flow_rate"))
    current_settings["pendant_drop_settings"]["pendant_drop_depth_offset"] = float(request.form.get("pendant_drop_depth_offset"))
    current_settings["pendant_drop_settings"]["well_id_drop_stage"] = request.form.get("well_id_drop_stage")
    current_settings["pendant_drop_settings"]["n_equilibration_points"] = int(request.form.get("n_equilibration_points"))
    
    # Update image analysis settings
    current_settings["image_analysis_settings"]["scale"] = float(request.form.get("scale"))
    current_settings["image_analysis_settings"]["diameter_needle_px"] = float(request.form.get("diameter_needle_px"))
    current_settings["image_analysis_settings"]["diameter_tolerance_percent"] = float(request.form.get("diameter_tolerance_percent"))
    current_settings["image_analysis_settings"]["st_water"] = float(request.form.get("st_water"))
    current_settings["image_analysis_settings"]["density"] = float(request.form.get("density"))
    current_settings["image_analysis_settings"]["gravity_constant"] = float(request.form.get("gravity_constant"))
    
    # Save updated settings
    save_settings(current_settings, file_path="config/settings.json")
    
    return redirect(url_for("index"))


@app.route("/input_initialisation", methods=["POST"])
def input_initialisation():
    """Show initialization form"""
    return render_template("input_initialisation.html")


def initialize_protocol_thread(layout_csv_path):
    """Initialize protocol in background thread"""
    global protocol, settings
    
    try:
        # Reload settings to get updated config
        settings = load_settings(file_path="config/settings.json")
        
        # Create protocol instance
        protocol = Protocol()
        
        print("[Server] Protocol initialized successfully")
        
    except Exception as e:
        print(f"[Server] Protocol initialization failed: {e}")
        import traceback
        traceback.print_exc()


@app.route("/initialisation", methods=["POST"])
def initialisation():
    """Handle initialization form submission"""
    exp_tag = request.form.get("exp_tag")
    csv_file = request.files.get("csv_file")
    
    if not exp_tag or not csv_file:
        return redirect(url_for("index"))
    
    # Save layout CSV
    layout_dir = "config/layouts"
    os.makedirs(layout_dir, exist_ok=True)
    layout_path = os.path.join(layout_dir, csv_file.filename).replace("\\", "/")
    csv_file.save(layout_path)
    
    # Update settings
    settings = load_settings(file_path="config/settings.json")
    settings["file_settings"]["exp_tag"] = exp_tag
    settings["file_settings"]["config_filepath"] = layout_path
    save_settings(settings, file_path="config/settings.json")
    
    # Start initialization in background thread
    thread = threading.Thread(target=initialize_protocol_thread, args=(layout_path,))
    thread.daemon = True
    thread.start()
    
    return redirect(url_for("index"))


@app.route("/input_measure_wells", methods=["POST"])
def input_measure_wells():
    """Show measure wells form"""
    return render_template("input_measure_wells.html")


@app.route("/input_characterise_solution", methods=["POST"])
def input_characterise_solution():
    """Show characterise solution form"""
    return render_template("input_characterise_solution.html")


def measure_wells_thread(sample_csv_path):
    """Run measure wells in background thread"""
    global protocol
    
    try:
        if protocol is None:
            print("[Server] Error: Protocol not initialized")
            return
        
        # Update settings with sample info path
        settings = load_settings(file_path="config/settings.json")
        settings["file_settings"]["sample_info_filepath"] = sample_csv_path
        save_settings(settings, file_path="config/settings.json")
        
        # Run measurement protocol
        protocol.measure_wells()
        
        print("[Server] Measurements completed successfully")
        
    except Exception as e:
        print(f"[Server] Measurement failed: {e}")
        import traceback
        traceback.print_exc()


def characterise_solution_thread(sample_csv_path):
    """Run characterise solution in background thread"""
    global protocol
    
    try:
        if protocol is None:
            print("[Server] Error: Protocol not initialized")
            return
        
        # Update settings with sample info path
        settings = load_settings(file_path="config/settings.json")
        settings["file_settings"]["sample_info_filepath"] = sample_csv_path
        save_settings(settings, file_path="config/settings.json")
        
        # Run characterisation protocol
        protocol.characterise_solution()
        
        print("[Server] Characterisation completed successfully")
        
    except Exception as e:
        print(f"[Server] Characterisation failed: {e}")
        import traceback
        traceback.print_exc()


@app.route("/measure_wells", methods=["POST"])
def measure_wells():
    """Handle measure wells form submission"""
    global protocol
    
    if protocol is None:
        return redirect(url_for("index"))
    
    csv_file = request.files.get("csv_file")
    
    if not csv_file:
        return redirect(url_for("index"))
    
    # Save sample info CSV
    sample_dir = "config/info"
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, csv_file.filename).replace("\\", "/")
    csv_file.save(sample_path)
    
    # Start measurement in background thread
    thread = threading.Thread(target=measure_wells_thread, args=(sample_path,))
    thread.daemon = True
    thread.start()
    
    return redirect(url_for("index"))


@app.route("/characterise_solution", methods=["POST"])
def characterise_solution():
    """Handle characterise solution form submission"""
    global protocol
    
    if protocol is None:
        return redirect(url_for("index"))
    
    csv_file = request.files.get("csv_file")
    
    if not csv_file:
        return redirect(url_for("index"))
    
    # Save sample info CSV
    sample_dir = "config/info"
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, csv_file.filename).replace("\\", "/")
    csv_file.save(sample_path)
    
    # Start characterisation in background thread
    thread = threading.Thread(target=characterise_solution_thread, args=(sample_path,))
    thread.daemon = True
    thread.start()
    
    return redirect(url_for("index"))




@app.route("/plots/<plot_name>")
def get_plot(plot_name):
    """Serve plot images from cache with no-cache headers"""
    response = send_from_directory(
        "static/cache_images",
        plot_name,
        mimetype="image/png",
    )
    # Prevent browser caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")


if __name__ == "__main__":
    print("=" * 70)
    print("PendantProp Server Starting...")
    print("=" * 70)
    print("Access the server at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
