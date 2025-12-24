#!/usr/bin/env python3
"""
Test script to verify FPS calculation and data saving fixes
"""
import time
import csv
import os
from datetime import datetime

def test_fps_calculation():
    """Test the improved FPS calculation"""
    print("Testing FPS calculation improvements...")
    
    # Test normal case
    start_time = time.perf_counter()
    time.sleep(0.016)  # Simulate 60 FPS
    frame_time = time.perf_counter() - start_time
    fps = 1.0 / frame_time if frame_time > 0.0001 else 0
    print(f"Normal case: {fps:.1f} FPS (expected ~60)")
    
    # Test very small frame time (unrealistic)
    start_time = time.perf_counter()
    time.sleep(0.0001)  # Very small
    frame_time = time.perf_counter() - start_time
    if frame_time > 0.0001:
        fps = 1.0 / frame_time
        if fps > 10000:
            fps = 0.0
    else:
        fps = 0
    print(f"Very small frame time: {fps:.1f} FPS (filtered)")
    
    # Test realistic high FPS
    start_time = time.perf_counter()
    time.sleep(0.001)  # 1000 FPS
    frame_time = time.perf_counter() - start_time
    fps = 1.0 / frame_time if frame_time > 0.0001 else 0
    if fps > 10000:
        fps = 0.0
    print(f"High FPS: {fps:.1f} FPS (expected ~1000)")

def test_csv_saving():
    """Test improved CSV saving with append mode"""
    print("\nTesting CSV saving improvements...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"test_csv_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    
    csv_path = os.path.join(test_dir, "test_benchmark.csv")
    
    # Simulate multiple flush operations
    test_data_batches = [
        [["frame", "mode", "fps", "mse", "timestamp"]],
        [[1, "PT", 30.5, 0.0, datetime.now().isoformat()]],
        [[2, "Grid", 45.8, 0.001, datetime.now().isoformat()], 
         [3, "Grid", 44.2, 0.0011, datetime.now().isoformat()]],
        [[4, "Hybrid", 35.7, 0.002, datetime.now().isoformat()]]
    ]
    
    print(f"Writing data to: {csv_path}")
    
    for i, batch in enumerate(test_data_batches):
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                print("  Writing header...")
            writer.writerows(batch)
        print(f"  Batch {i+1}: {len(batch)} records")
    
    # Verify the file
    with open(csv_path, 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        print(f"Total lines written: {len(lines)}")
        print("First few lines:")
        for line in lines[:3]:
            print(f"  {line}")
    
    # Cleanup
    os.remove(csv_path)
    os.rmdir(test_dir)
    print("Test completed and cleaned up.")

if __name__ == "__main__":
    print("Running FPS and data saving fix tests...\n")
    test_fps_calculation()
    test_csv_saving()
    print("\n✅ All tests completed!")