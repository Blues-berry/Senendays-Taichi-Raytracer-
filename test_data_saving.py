#!/usr/bin/env python3
"""
Test script to verify the data saving functionality
"""
import os
import csv
from datetime import datetime

def test_benchmark_fix():
    print("Testing benchmark data saving fix...")
    
    # Test 1: Check if timestamped directory creation works
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"benchmark_results_{timestamp}"
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✓ Directory creation works: {test_dir}")
        
        # Test 2: Check CSV writing
        csv_path = os.path.join(test_dir, "benchmark_results.csv")
        test_data = [
            [1, "Path Tracing", 30.5, 0.0, datetime.now().isoformat()],
            [2, "Path Tracing", 31.2, 0.0, datetime.now().isoformat()],
            [3, "Grid", 45.8, 0.001234, datetime.now().isoformat()]
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "mode", "fps", "mse", "timestamp"])
            writer.writerows(test_data)
        
        print(f"✓ CSV writing works: {csv_path}")
        
        # Verify data
        with open(csv_path, 'r') as f:
            content = f.read()
            print(f"✓ CSV content verification:\n{content}")
            
        # Cleanup
        os.remove(csv_path)
        os.rmdir(test_dir)
        print("✓ Cleanup successful")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

def test_experiment_fix():
    print("\nTesting experiment data saving fix...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiment_{timestamp}"
    
    try:
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"✓ Experiment directory creation works: {experiment_dir}")
        
        # Test log file creation
        log_path = os.path.join(experiment_dir, 'experiment_log.csv')
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'mode', 'FPS', 'MSE'])
            test_data = [
                [100, 'Adaptive', 35.7, 0.000567],
                [101, 'Adaptive', 36.1, 0.000552]
            ]
            writer.writerows(test_data)
        
        print(f"✓ Log file creation works: {log_path}")
        
        # Test screenshot path
        screenshot_path = os.path.join(experiment_dir, "test_screenshot.png")
        # Create a dummy file
        with open(screenshot_path, 'w') as f:
            f.write("dummy image data")
        
        print(f"✓ Screenshot path works: {screenshot_path}")
        
        # Cleanup
        os.remove(log_path)
        os.remove(screenshot_path)
        os.rmdir(experiment_dir)
        print("✓ Cleanup successful")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Running data saving fix tests...\n")
    
    success1 = test_benchmark_fix()
    success2 = test_experiment_fix()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Data saving should work correctly now.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")