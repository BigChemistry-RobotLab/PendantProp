"""
sensor_api.py

This file provides the API for the sensor (OT712), which measures temperature, pressure and humidity.
It starts a local sensor server and provides a method to fetch the latest sensor data via HTTP.

Classes:
    SensorAPI: Interface for starting the sensor server and retrieving sensor data.
"""

# Imports

## Packages
import requests

## Custom code
from hardware.sensor.sensor_server import start_server


class SensorAPI:
    """
    
    """
    def __init__(self):
        start_server()  # Start the sensor server where the sensor data is printed to

    def capture_sensor_data(self):
        """ """
        try:
            response = requests.get("http://localhost:5001/data", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[Error] Could not fetch data: {e}")
            return {"Temperature (C)": 0, "Humidity (%)": 0, "Pressure (Pa)": 0}
