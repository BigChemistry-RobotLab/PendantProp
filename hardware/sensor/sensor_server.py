# server.py
import serial
import threading
from datetime import datetime
from flask import Flask, jsonify

SERIAL_PORT = 'COM6'
BAUD_RATE = 9600

latest_data = {
    'Temperature (C)': None,
    'Pressure (Pa)': None,
    'Humidity (%)': None,
    'date & time': None
}

def read_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[Sensor server] Connected to {SERIAL_PORT}")
        while True:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split("\t")
            if len(parts) == 3:
                try:
                    temp, pressure, humidity = map(float, parts)
                    latest_data['date & time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    latest_data['Temperature (C)'] = temp
                    latest_data['Pressure (Pa)'] = pressure
                    latest_data['Humidity (%)'] = humidity
                    # print(f"[Serial] Updated: {latest_data}")
                except ValueError:
                    print(f"[Serial] Could not parse: {line}")
    except serial.SerialException as e:
        print(f"[Sensor server error] {e}")

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(latest_data)

def _run_flask():
    # Prevent Flask reloader from spawning extra processes
    app.run(host='0.0.0.0', port=5001, use_reloader=False)

def start_server():
    # Start serial reading in a daemon thread
    threading.Thread(target=read_serial, daemon=True).start()
    # Start Flask server in another daemon thread
    threading.Thread(target=_run_flask, daemon=True).start()
    print("[Sensor server] Sensor server started in background")

if __name__ == '__main__':
    # When running standalone, just call the function
    start_server()
    # Keep the main thread alive (only for standalone mode)
    while True:
        pass
