import os
import pandas as pd


class SensorAPI:
    def __init__(self):
        self.sensor_name = "OT712 T, P, humidity sensor" 

    def capture_sensor_data(self):
        file_path = "hardware/sensor/sensor_data.txt"
        with open(file_path, "r") as file:
            data = file.read()

        # Replace double newlines with a unique separator
        data = data.replace("\n\n", "<NEW_ROW>")

        # Split the data into rows
        rows = data.split("<NEW_ROW>")

        # Create a DataFrame from the rows
        df = pd.DataFrame([row.split("\t") for row in rows])

        # Set the column names
        df.columns = ["date & time", "Temperature (C)", "Pressure (Pa)", "Humidity (%)"]

        # Retrieve the latest sensor data (second last row)
        last_sensor_data = df.iloc[-2]

        return last_sensor_data
    
    def __str__(self):
        f"""
        Sensor instance.

        sensor name: {self.sensor_name}
        last sensor data: {self.capture_sensor_data()}
        """

