"""
sensor_api.py

Compact API for the environmental sensor (OT712).
Measures temperature, pressure, and humidity.

Supports two modes:
- Real mode: Reads from serial sensor and serves via HTTP
- Mock mode: Returns simulated standard lab conditions

Classes:
    SensorAPI: Interface for retrieving sensor data (real or mock)
"""

import serial
import threading
import time
from datetime import datetime
from flask import Flask, jsonify

from opentrons_api.logger import Logger


# Mock sensor data - standard lab conditions
MOCK_DATA = {
    'Temperature (C)': 23.5,    # Room temperature
    'Pressure (Pa)': 101325.0,   # 1 atm
    'Humidity (%)': 45.0,        # Typical indoor humidity
    'date & time': None
}


class SensorAPI:
    """
    Environmental Sensor API
    
    Args:
        settings (dict): Settings dictionary with sensor configuration.
                        Expected keys:
                        - 'SIMULATE': bool (use mock data if True)
                        - 'SENSOR_SETTINGS': dict with:
                            - 'serial_port': str (e.g., 'COM3')
                            - 'baud_rate': int (e.g., 9600)
                            - 'flask_port': int (e.g., 5001)
        
    Alternative Args (for backward compatibility):
        simulate (bool): If True, uses mock data instead of real sensor
        serial_port (str): COM port for sensor (e.g., 'COM3')
        baud_rate (int): Serial baud rate (default: 9600)
        flask_port (int): Flask server port (default: 5001)
    """
    
    _server_started = False
    _flask_port = None
    _latest_data = {
        'Temperature (C)': None,
        'Pressure (Pa)': None,
        'Humidity (%)': None,
        'date & time': None
    }
    
    def __init__(
        self, 
        settings: dict
    ):
        # Handle settings dict (preferred method)
        file_settings = settings["file_settings"]
        self.simulate = settings["general_settings"]["simulate"]
        sensor_settings = settings['sensor_settings']
        self.serial_port = sensor_settings.get('serial_port')
        self.baud_rate = sensor_settings.get('baud_rate')
        self.flask_port = sensor_settings.get('flask_port')

        # Set class-level flask port for server
        SensorAPI._flask_port = self.flask_port

        self.logger = Logger(
            name="protocol",
            file_path=f'{file_settings["output_folder"]}/{file_settings["exp_tag"]}/{file_settings["meta_data_folder"]}',
        )
        
        if self.simulate:
            self.logger.info("[Sensor API] Running in MOCK mode - using simulated data")
        else:
            # Only start server once for real mode
            if not SensorAPI._server_started:
                self._start_server()
                SensorAPI._server_started = True
                time.sleep(1)  # Give server time to start
            self.logger.info(f"[Sensor API] Connected to real sensor on {self.serial_port}")
    
    def capture_sensor_data(self):
        """
        Capture current sensor readings.
        
        Returns:
            dict: Sensor data with keys 'Temperature (C)', 'Pressure (Pa)', 
                  'Humidity (%)', and 'date & time'
        """
        if self.simulate:
            # Return mock data with current timestamp
            data = MOCK_DATA.copy()
            data['date & time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return data
        else:
            # Return real sensor data from server
            try:
                import requests
                response = requests.get(f"http://localhost:{self.flask_port}/data", timeout=5)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.error(f"[Sensor API Error] Could not fetch data: {e}")
                # Return default values on error
                return {
                    "Temperature (C)": 0, 
                    "Humidity (%)": 0, 
                    "Pressure (Pa)": 0,
                    "date & time": None
                }
    
    def _start_server(self):
        """Start the sensor server for reading real sensor data"""
        # Start serial reading in a daemon thread
        threading.Thread(target=self._read_serial, daemon=True).start()
        # Start Flask server in another daemon thread
        threading.Thread(target=self._run_flask, daemon=True).start()
        self.logger.info(f"[Sensor Server] Started on port {self.flask_port}")
    
    def _read_serial(self):
        """Read data from serial sensor"""
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            self.logger.info(f"[Sensor Server] Connected to {self.serial_port}")
            while True:
                line = ser.readline().decode('utf-8').strip()
                parts = line.split("\t")
                if len(parts) == 3:
                    try:
                        temp, pressure, humidity = map(float, parts)
                        SensorAPI._latest_data['date & time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        SensorAPI._latest_data['Temperature (C)'] = temp
                        SensorAPI._latest_data['Pressure (Pa)'] = pressure
                        SensorAPI._latest_data['Humidity (%)'] = humidity
                    except ValueError:
                        self.logger.warning(f"[Sensor Server] Could not parse: {line}")
        except serial.SerialException as e:
            self.logger.error(f"[Sensor Server Error] {e}")
            self.logger.error(f"[Sensor Server] Could not connect to {self.serial_port}")
    
    def _run_flask(self):
        """Run Flask server to serve sensor data"""
        app = Flask(__name__)
        
        @app.route('/data', methods=['GET'])
        def get_data():
            return jsonify(SensorAPI._latest_data)
        
        # Prevent Flask reloader from spawning extra processes
        app.run(debug=False, host='0.0.0.0', port=SensorAPI._flask_port, use_reloader=False)

