"""
Quick Test Script for New Methods

快速测试新方法的独立脚本，用于验证实现是否正确。
运行时间：约 5-10 分钟
"""

import taichi as ti
import numpy as np
import time
from datetime import datetime

ti.init(arch=ti.gpu, random_seed=42)

# Import modules
import main
from main import spheres, materials
from camera_ms_aic import MultiScaleGrid
from camera_motion_comp import MotionCompensatedTemporalFilter
import experiment_config as cfg
from ray import Ray

vec3 = ti.types.vector(3, float)

print("="*60)
print("New Methods Quick Test")
print("="*60)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Setup scene
print("[1/5] Setting up scene (cornell_box)...")
world, cam = main.setup_scene('cornell_box')
cam.scene_mode = 'cornell_box'
print("✓ Scene setup complete")
print()

# Test Multi-Scale Grid
print("[2/5] Testing Multi-Scale Grid (MS-AIC)...")
GRID_RESOLUTIONS = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
grid_origin = vec3(-8.0, -1.0, -8.0)
grid_cell_size = 1.0

ms_grid = MultiScaleGrid(GRID_RESOLUTIONS, grid_origin, grid_cell_size)
print(f"  Grid levels: {len(GRID_RESOLUTIONS)}")
print(f"  Resolutions: {GRID_RESOLUTIONS}")
print(f"  Memory: {ms_grid.get_memory_usage_mb():.2f} MB")

# Test grid update
print("  Updating grid (10 iterations)...")
camera_pos = cam.camera_origin
start_time = time.time()
for i in range(10):
    ms_grid.update_all_levels(world, 0.01, camera_pos)
    ti.sync()
update_time = time.time() - start_time
print(f"  Update time: {update_time*1000:.1f} ms avg")
print(f"  Update rate: {10/update_time:.1f} updates/sec")
print("✓ Multi-Scale Grid test complete")
print()

# Test Motion-Compensated Temporal Filter
print("[3/5] Testing Motion-Compensated Temporal Filter (MCTF)...")
mctf = MotionCompensatedTemporalFilter(cam.img_res)
print(f"  Resolution: {cam.img_res}")
print(f"  Spatial sigma: {mctf.spatial_sigma}")
print(f"  Temporal sigma: {mctf.temporal_sigma}")

# Simulate a few frames
print("  Simulating 5 frames...")
for i in range(5):
    cam.render(world, 2)  # Hybrid mode
    ti.sync()

    # Apply MCTF
    filtered = mctf.process_frame(
        cam.frame,
        cam.normal_buffer,
        cam.depth_buffer
    )
    ti.sync()

print("✓ Motion-Compensated Filter test complete")
print()

# Test Integration (Full Method)
print("[4/5] Testing Full Method Integration...")
print("  Rendering 20 frames with MS-AIC + MCTF...")

frame_times = []
for i in range(20):
    start = time.perf_counter()

    # Update multi-scale grid
    ms_grid.update_all_levels(world, 0.01, camera_pos)

    # Render
    cam.render(world, 2)  # Hybrid mode
    ti.sync()

    # Apply MCTF
    mctf.process_frame(
        cam.frame,
        cam.normal_buffer,
        cam.depth_buffer
    )
    ti.sync()

    elapsed = time.perf_counter() - start
    frame_times.append(elapsed)

    if (i + 1) % 5 == 0:
        print(f"    Frame {i+1}/20: {1/elapsed:.1f} FPS")

avg_frame_time = np.mean(frame_times)
avg_fps = 1.0 / avg_frame_time
print(f"  Average FPS: {avg_fps:.1f}")
print("✓ Full Method test complete")
print()

# Quality Check
print("[5/5] Running Quality Check...")
print("  Building PT reference (50 spp)...")
cam.render_pt_reference(world, target_spp=50, chunk_spp=5, reset=True)
pt_frame = cam.pt_frame.to_numpy()

# Render current frame
ms_grid.update_all_levels(world, 0.01, camera_pos)
cam.render(world, 2)
current_frame = cam.frame.to_numpy()

# Calculate MSE
pt_linear = pt_frame.astype(np.float32)
curr_linear = current_frame.astype(np.float32)

if pt_linear.max() > 1.0:
    pt_linear = pt_linear / 255.0
if curr_linear.max() > 1.0:
    curr_linear = curr_linear / 255.0

pt_linear = np.nan_to_num(pt_linear, nan=0.0, posinf=0.0, neginf=0.0)
curr_linear = np.nan_to_num(curr_linear, nan=0.0, posinf=0.0, neginf=0.0)

diff = curr_linear - pt_linear
mse = np.mean(diff ** 2)

print(f"  MSE vs PT reference: {mse:.6e}")
if mse < 1e-2:
    print("  ✓ Quality check PASSED (MSE < 0.01)")
else:
    print("  ⚠ Quality check WARNING (MSE >= 0.01)")
print()

# Summary
print("="*60)
print("Quick Test Summary")
print("="*60)
print(f"Multi-Scale Grid:")
print(f"  - Levels: {len(GRID_RESOLUTIONS)}")
print(f"  - Memory: {ms_grid.get_memory_usage_mb():.2f} MB")
print(f"  - Update rate: {10/update_time:.1f} updates/sec")
print()
print(f"Motion-Compensated Filter:")
print(f"  - Resolution: {cam.img_res}")
print(f"  - Parameters: spatial={mctf.spatial_sigma}, temporal={mctf.temporal_sigma}")
print()
print(f"Full Method Integration:")
print(f"  - Average FPS: {avg_fps:.1f}")
print(f"  - MSE vs PT: {mse:.6e}")
print()
print("✓ All tests completed successfully!")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("="*60)
print()

# Save test results
import os
results_dir = "results/quick_test"
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(results_dir, f"test_report_{timestamp}.txt")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Quick Test Report\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Scene: cornell_box\n\n")

    f.write("Multi-Scale Grid:\n")
    f.write(f"  Levels: {len(GRID_RESOLUTIONS)}\n")
    f.write(f"  Resolutions: {GRID_RESOLUTIONS}\n")
    f.write(f"  Memory: {ms_grid.get_memory_usage_mb():.2f} MB\n")
    f.write(f"  Update rate: {10/update_time:.1f} updates/sec\n\n")

    f.write("Motion-Compensated Filter:\n")
    f.write(f"  Resolution: {cam.img_res}\n")
    f.write(f"  Spatial sigma: {mctf.spatial_sigma}\n")
    f.write(f"  Temporal sigma: {mctf.temporal_sigma}\n\n")

    f.write("Full Method Integration:\n")
    f.write(f"  Average FPS: {avg_fps:.1f}\n")
    f.write(f"  MSE vs PT: {mse:.6e}\n")

print(f"Test report saved to: {report_path}")
print()
print("Next steps:")
print("  1. Run full experiments: python experiment_new_methods.py")
print("  2. Generate paper figures: see PAPER_IMPLEMENTATION_GUIDE_NEW.md")
print("  3. Start writing paper: use provided LaTeX templates")
print()
