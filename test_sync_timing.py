#!/usr/bin/env python3
"""
Test script to verify GPU synchronization timing fix
"""
import taichi as ti
import time

# Initialize Taichi
ti.init(arch=ti.gpu)

# Create field outside kernel
result = ti.field(dtype=ti.f32, shape=(1000, 1000))

@ti.kernel
def dummy_computation():
    # Simulate some GPU work
    for i, j in result:
        result[i, j] = ti.sqrt(i * i + j * j) * 0.001

def test_timing():
    print("Testing GPU synchronization timing...")
    
    # Test without synchronization
    print("\n--- Without synchronization ---")
    times_no_sync = []
    for i in range(5):
        start = time.perf_counter()
        dummy_computation()
        end = time.perf_counter()
        elapsed = end - start
        times_no_sync.append(elapsed)
        print(f"Run {i+1}: {elapsed*1000:.3f}ms")
    
    # Test with synchronization
    print("\n--- With synchronization ---")
    times_with_sync = []
    for i in range(5):
        ti.sync()  # Sync before timing
        start = time.perf_counter()
        dummy_computation()
        ti.sync()  # sync after timing
        end = time.perf_counter()
        elapsed = end - start
        times_with_sync.append(elapsed)
        print(f"Run {i+1}: {elapsed*1000:.3f}ms")
    
    # Compare results
    avg_no_sync = sum(times_no_sync) / len(times_no_sync)
    avg_with_sync = sum(times_with_sync) / len(times_with_sync)
    
    print(f"\n--- Comparison ---")
    print(f"Average without sync: {avg_no_sync*1000:.3f}ms")
    print(f"Average with sync: {avg_with_sync*1000:.3f}ms")
    print(f"Ratio: {avg_no_sync/avg_with_sync:.2f}x")
    
    if avg_no_sync < avg_with_sync * 0.5:
        print("✓ Without sync is suspiciously fast - async execution detected")
        print("✓ With sync gives more realistic timing")
    else:
        print("? Both timings similar - sync may not be critical here")

if __name__ == "__main__":
    test_timing()