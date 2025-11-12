"""
Test script for environmental sensor (OT712)
Tests temperature, pressure, and humidity sensor functionality
"""

import time
# import sys

# Add project root to path if needed
# sys.path.insert(0, r'c:\Users\BigChemistry\Documents\repositories\PendantProp\src')

from pendantprop.hardware.sensor_api import SensorAPI

def test_sensor_connection():
    """Test if sensor server starts and can be connected to"""
    print("=" * 70)
    print("SENSOR CONNECTION TEST")
    print("=" * 70)
    
    try:
        sensor = SensorAPI()
        print("✓ Sensor API initialized")
        print("✓ Sensor server started")
        
        # Give server time to start and collect first reading
        print("\nWaiting for initial sensor readings...")
        time.sleep(3)
        
        return sensor
        
    except Exception as e:
        print(f"✗ Failed to initialize sensor: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_reading(sensor):
    """Test a single sensor reading"""
    print("\n" + "=" * 70)
    print("SINGLE READING TEST")
    print("=" * 70)
    
    try:
        data = sensor.capture_sensor_data()
        
        print(f"\nSensor Data:")
        print(f"  Temperature: {data.get('Temperature (C)', 'N/A')} °C")
        print(f"  Pressure:    {data.get('Pressure (Pa)', 'N/A')} Pa")
        print(f"  Humidity:    {data.get('Humidity (%)', 'N/A')} %")
        print(f"  Timestamp:   {data.get('date & time', 'N/A')}")
        
        # Check if data is valid (not None or 0)
        if data.get('Temperature (C)') and data.get('Temperature (C)') != 0:
            print("\n✓ Valid sensor data received")
            return True
        else:
            print("\n⚠ Warning: Sensor data may not be valid (check sensor connection)")
            return False
            
    except Exception as e:
        print(f"\n✗ Error reading sensor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continuous_readings(sensor, duration_seconds=10, interval_seconds=1):
    """Test continuous sensor readings over time"""
    print("\n" + "=" * 70)
    print(f"CONTINUOUS READING TEST ({duration_seconds}s)")
    print("=" * 70)
    
    readings = []
    
    try:
        print(f"\nCollecting readings every {interval_seconds}s...\n")
        print(f"{'Time':<10} {'Temp (°C)':<12} {'Press (Pa)':<15} {'Humid (%)':<12}")
        print("-" * 70)
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            data = sensor.capture_sensor_data()
            
            temp = data.get('Temperature (C)', 0)
            pressure = data.get('Pressure (Pa)', 0)
            humidity = data.get('Humidity (%)', 0)
            
            elapsed = time.time() - start_time
            print(f"{elapsed:>6.1f}s    {temp:>8.2f}     {pressure:>11.1f}     {humidity:>8.2f}")
            
            readings.append({
                'time': elapsed,
                'temperature': temp,
                'pressure': pressure,
                'humidity': humidity
            })
            
            time.sleep(interval_seconds)
        
        print("\n" + "-" * 70)
        
        # Calculate statistics
        if readings:
            temps = [r['temperature'] for r in readings if r['temperature'] != 0]
            pressures = [r['pressure'] for r in readings if r['pressure'] != 0]
            humidities = [r['humidity'] for r in readings if r['humidity'] != 0]
            
            if temps:
                print(f"\nStatistics over {len(readings)} readings:")
                print(f"  Temperature: {min(temps):.2f} - {max(temps):.2f} °C (avg: {sum(temps)/len(temps):.2f})")
                print(f"  Pressure:    {min(pressures):.1f} - {max(pressures):.1f} Pa (avg: {sum(pressures)/len(pressures):.1f})")
                print(f"  Humidity:    {min(humidities):.2f} - {max(humidities):.2f} % (avg: {sum(humidities)/len(humidities):.2f})")
                print("\n✓ Continuous reading test passed")
                return True
            else:
                print("\n⚠ Warning: No valid readings collected")
                return False
        else:
            print("\n✗ No readings collected")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error during continuous reading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensor_stability(sensor, num_readings=5):
    """Test sensor stability - check if readings are consistent"""
    print("\n" + "=" * 70)
    print("SENSOR STABILITY TEST")
    print("=" * 70)
    
    try:
        temps = []
        
        print(f"\nCollecting {num_readings} readings to check stability...")
        
        for i in range(num_readings):
            data = sensor.capture_sensor_data()
            temp = data.get('Temperature (C)', 0)
            temps.append(temp)
            print(f"  Reading {i+1}: {temp:.2f} °C")
            time.sleep(1)
        
        # Check stability
        valid_temps = [t for t in temps if t != 0]
        if valid_temps:
            temp_range = max(valid_temps) - min(valid_temps)
            avg_temp = sum(valid_temps) / len(valid_temps)
            
            print(f"\nTemperature Range: {temp_range:.3f} °C")
            print(f"Average: {avg_temp:.2f} °C")
            
            if temp_range < 1.0:  # Less than 1°C variation
                print("✓ Sensor readings are stable")
                return True
            else:
                print("⚠ Warning: Large temperature variation detected")
                return False
        else:
            print("✗ No valid readings collected")
            return False
            
    except Exception as e:
        print(f"✗ Error during stability test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all sensor tests"""
    print("\n" + "=" * 70)
    print("ENVIRONMENTAL SENSOR TEST SUITE")
    print("OT712 - Temperature, Pressure, Humidity")
    print("=" * 70 + "\n")
    
    # Test 1: Connection
    sensor = test_sensor_connection()
    if not sensor:
        print("\n" + "=" * 70)
        print("TESTS FAILED - Could not initialize sensor")
        print("=" * 70)
        return
    
    # Test 2: Single reading
    test_single_reading(sensor)
    
    # Test 3: Stability
    test_sensor_stability(sensor)
    
    # Test 4: Continuous readings
    test_continuous_readings(sensor, duration_seconds=10, interval_seconds=2)
    
    print("\n" + "=" * 70)
    print("ALL SENSOR TESTS COMPLETE")
    print("=" * 70)
    print("\nNOTE: Make sure the sensor is connected to COM6")
    print("      Check sensor_server.py if you need to change the port")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
