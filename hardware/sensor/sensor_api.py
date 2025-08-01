import pandas as pd
import requests

class SensorAPI:
    def __init__(self):
        self.sensor_name = "OT712 T, P, humidity sensor" 

    def capture_sensor_data(self):
        try:
            response = requests.get("http://localhost:5001/data", timeout=5)
            response.raise_for_status()  # Raise error for bad status
            data = response.json()
            df = pd.DataFrame([data])
            return df
        except requests.RequestException as e:
            print(f"[Error] Could not fetch data: {e}")
            empty_data = {
            "Temperature (C)": 0,
            "Humidity (%)": 0,
            "Pressure (Pa)": 0
            }
            return empty_data

    
    def __str__(self):
        f"""
        Sensor instance.

        sensor name: {self.sensor_name}
        last sensor data: {self.capture_sensor_data()}
        """

