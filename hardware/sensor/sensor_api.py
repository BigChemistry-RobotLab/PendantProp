import pandas as pd
import requests
from hardware.sensor.sensor_server import start_server

class SensorAPI:
    def __init__(self):
        self.sensor_name = "OT712 T, P, humidity sensor"
        start_server()  # This will return immediately (non-blocking)

    def capture_sensor_data(self):
        try:
            response = requests.get("http://localhost:5001/data", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[Error] Could not fetch data: {e}")
            return {
                "Temperature (C)": 0,
                "Humidity (%)": 0,
                "Pressure (Pa)": 0
            }