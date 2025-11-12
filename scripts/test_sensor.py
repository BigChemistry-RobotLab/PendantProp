"""
Example: Using the Sensor API in your protocols

Shows how to initialize and use the sensor in both real and mock modes.
"""

from pendantprop.hardware.sensor.sensor_api import SensorAPI
from opentrons_api.load_save_functions import load_settings

# Example 1: Real sensor (default)
print("=" * 70)
print("Test sensor")
print("=" * 70)
settings = load_settings(file_path="config/settings.json")
sensor_real = SensorAPI(settings=settings)
data = sensor_real.capture_sensor_data()

print(f"Temperature: {data['Temperature (C)']} °C")
print(f"Pressure: {data['Pressure (Pa)']} Pa")
print(f"Humidity: {data['Humidity (%)']} %")
print(f"Timestamp: {data['date & time']}")