"""
Run All Tests Script
This script runs all test scripts in simulation mode to verify the system functionality.
IMPORTANT: This script should ONLY be run in simulation mode.
"""

import subprocess
import sys
from pathlib import Path
from opentrons_api.load_save_functions import load_settings

# Define the test scripts to run
TEST_SCRIPTS = [
    "test_config.py",
    "test_protocol.py",
    "test_droplet_manager.py",
    "test_washing.py",
    "test_camera.py",  
    "test_sensor.py",  
]

def check_simulation_mode():
    """Check if simulation mode is enabled in settings."""
    try:
        settings = load_settings(file_path="config/settings.json")
        simulate = settings.get('general_settings', {}).get('simulate', False)
        return simulate
    except Exception as e:
        print(f"Error loading settings: {e}")
        return False

def run_test(script_name):
    """Run a single test script."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}")  # Show first 500 chars
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            if result.stdout:
                print(f"Standard output:\n{result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ {script_name} raised an exception: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("="*60)
    print("PENDANT PROP - ALL TESTS RUNNER")
    print("="*60)
    
    # Safety check: Ensure simulation mode is enabled
    if not check_simulation_mode():
        print("\n⚠️  ERROR: Simulation mode is NOT enabled!")
        print("This script should ONLY be run in simulation mode.")
        print("Please enable 'simulate: true' in config/settings.json")
        sys.exit(1)
    
    print("✓ Simulation mode is enabled\n")
    
    # Run all tests
    results = {}
    for script in TEST_SCRIPTS:
        results[script] = run_test(script)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for script, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {script}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
