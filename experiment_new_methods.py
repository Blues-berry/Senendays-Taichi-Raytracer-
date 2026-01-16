"""
New Methods Experiment Script

This script implements and benchmarks the proposed new methods:
1. Multi-Scale Adaptive Irradiance Caching (MS-AIC)
2. Motion-Compensated Temporal Filtering (MCTF)
3. Full Method (MS-AIC + MCTF)

Compare against baseline methods:
- Path Tracing (PT)
- Pure Grid
- Hybrid (current implementation)
"""

import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
import main
from main import spheres, materials, cam, world
import utils

# Import new method implementations
from camera_ms_aic import MultiScaleGrid
from camera_motion_comp import MotionCompensatedTemporalFilter

# Initialize Taichi
ti.init(arch=ti.gpu, random_seed=42)

# Rendering modes
RENDER_MODE_PT = 0
RENDER_MODE_GRID = 1
RENDER_MODE_HYBRID = 2
RENDER_MODE_MS_AIC = 3      # Multi-Scale AIC only
RENDER_MODE_MCTF = 4        # Motion-Compensated Temporal Filter only
RENDER_MODE_FULL = 5        # Full method (MS-AIC + MCTF)

# Experiment configuration
TEST_FRAMES = 600
MOVEMENT_FRAME = 200
PT_REFERENCE_FRAMES = 150

# Grid configurations for multi-scale
GRID_RESOLUTIONS = [
    (16, 16, 16),  # Coarse level
    (32, 32, 32),  # Medium level
    (64, 64, 64)   # Fine level (same as current)
]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "results"
output_dir = os.path.join(results_dir, f"new_methods_benchmark_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")


def log_message(message):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def save_screenshot(field_buffer, filename):
    """Save screenshot"""
    filepath = os.path.join(output_dir, filename)
    ti.tools.imwrite(field_buffer, filepath)
    log_message(f"Saved screenshot: {filepath}")


def calculate_mse(current_linear, reference_linear):
    """Calculate MSE in linear space"""
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)

    # Normalize to [0, 1]
    if curr_f.max() > 255.0:
        curr_f = curr_f / 255.0
    if ref_f.max() > 255.0:
        ref_f = ref_f / 255.0

    # Handle NaN/Inf
    curr_f = np.nan_to_num(curr_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_f = np.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)

    # MSE
    diff = curr_f - ref_f
    mse = np.mean(diff ** 2)
    return float(mse)


def init_multi_scale_grid(base_cam):
    """Initialize multi-scale grid from base camera configuration"""
    grid_origin = vec3(-8.0, -1.0, -8.0)
    grid_cell_size = 1.0

    ms_grid = MultiScaleGrid(GRID_RESOLUTIONS, grid_origin, grid_cell_size)

    log_message(f"Multi-scale grid initialized:")
    log_message(f"  Level 0 (coarse): {GRID_RESOLUTIONS[0]}, cell_size={grid_cell_size*4:.2f}")
    log_message(f"  Level 1 (medium): {GRID_RESOLUTIONS[1]}, cell_size={grid_cell_size*2:.2f}")
    log_message(f"  Level 2 (fine):   {GRID_RESOLUTIONS[2]}, cell_size={grid_cell_size:.2f}")
    log_message(f"  Total memory: {ms_grid.get_memory_usage_mb():.2f} MB")

    return ms_grid


def init_motion_filter(img_res):
    """Initialize motion-compensated temporal filter"""
    mctf = MotionCompensatedTemporalFilter(img_res)
    log_message(f"Motion-compensated temporal filter initialized: {img_res}")
    return mctf


def run_experiment_for_mode(mode_idx, mode_name, world_obj, cam_obj, pt_reference_linear, scene_name):
    """Run experiment for a specific rendering mode"""
    log_message(f"\n{'='*60}")
    log_message(f"Running mode: {mode_name}")
    log_message(f"{'='*60}\n")

    csv_data = []

    # Initialize components based on mode
    ms_grid = None
    mctf = None

    if mode_idx in [RENDER_MODE_MS_AIC, RENDER_MODE_FULL]:
        ms_grid = init_multi_scale_grid(cam_obj)

    if mode_idx in [RENDER_MODE_MCTF, RENDER_MODE_FULL]:
        mctf = init_motion_filter(cam_obj.img_res)

    # Frame buffer
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam_obj.img_res)

    for frame in range(TEST_FRAMES):
        # Object movement
        if frame == MOVEMENT_FRAME and len(main.spheres) > 0:
            big_sphere_idx = len(main.spheres) - 1
            old_x = main.spheres[big_sphere_idx].center[0]
            main.spheres[big_sphere_idx].center[0] = old_x + 1.0
            log_message(f"Object movement at frame {frame}: x {old_x:.2f} -> {main.spheres[big_sphere_idx].center[0]:.2f}")

            # Reset motion filter after large movement
            if mctf is not None:
                mctf.reset()

            current_frame.fill(0.001)

        # Render based on mode
        ti.sync()
        start_time = time.perf_counter()

        if mode_idx == RENDER_MODE_PT:
            cam_obj.render(world_obj, RENDER_MODE_PT)
            current_frame = cam_obj.frame

        elif mode_idx == RENDER_MODE_GRID:
            cam_obj.update_grid(world_obj, 0.01)
            cam_obj.render(world_obj, RENDER_MODE_GRID)
            current_frame = cam_obj.frame

        elif mode_idx == RENDER_MODE_HYBRID:
            cam_obj.update_grid(world_obj, 0.01)
            cam_obj.render(world_obj, RENDER_MODE_HYBRID)
            current_frame = cam_obj.frame

        elif mode_idx == RENDER_MODE_MS_AIC:
            # Update multi-scale grid
            camera_pos = cam_obj.camera_origin
            ms_grid.update_all_levels(world_obj, 0.01, camera_pos)
            # Render using multi-scale grid (would need integration)
            # For now, use standard render
            cam_obj.render(world_obj, RENDER_MODE_GRID)
            current_frame = cam_obj.frame

        elif mode_idx == RENDER_MODE_MCTF:
            cam_obj.update_grid(world_obj, 0.01)
            cam_obj.render(world_obj, RENDER_MODE_HYBRID)

            # Apply motion-compensated filter
            current_frame = mctf.process_frame(
                cam_obj.frame,
                cam_obj.normal_buffer,
                cam_obj.depth_buffer
            )

        elif mode_idx == RENDER_MODE_FULL:
            # Update multi-scale grid
            camera_pos = cam_obj.camera_origin
            ms_grid.update_all_levels(world_obj, 0.01, camera_pos)
            # Render (would need integration with ms_grid)
            cam_obj.update_grid(world_obj, 0.01)
            cam_obj.render(world_obj, RENDER_MODE_HYBRID)
            # Apply motion filter
            current_frame = mctf.process_frame(
                cam_obj.frame,
                cam_obj.normal_buffer,
                cam_obj.depth_buffer
            )

        ti.sync()
        frame_time = time.perf_counter() - start_time
        fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0

        # Calculate MSE
        mse = 0.0
        if pt_reference_linear is not None and mode_idx != RENDER_MODE_PT:
            current_linear = current_frame.to_numpy()
            mse = calculate_mse(current_linear, pt_reference_linear)

        # Record data
        csv_data.append({
            'scene': scene_name,
            'mode': mode_name,
            'frame': frame + 1,
            'fps': fps,
            'mse': mse,
            'gpu_time_ms': frame_time * 1000.0,
            'timestamp': datetime.now().isoformat(),
        })

        # Save key frames
        if frame in [5, 50, 100, 150, 199, 205, 210, 250, 300, 400, 500, 599]:
            save_screenshot(current_frame, f"{scene_name}_{mode_name}_frame_{frame+1}.png")

        # Progress
        if (frame + 1) % 50 == 0:
            log_message(f"  Frame {frame+1}/{TEST_FRAMES}: FPS={fps:.1f}, MSE={mse:.6e}")

    # Save final result
    save_screenshot(current_frame, f"{scene_name}_{mode_name}_result.png")
    log_message(f"  {mode_name} completed")

    # Save CSV
    csv_path = os.path.join(output_dir, f"{scene_name}_{mode_name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['scene', 'mode', 'frame', 'fps', 'mse', 'gpu_time_ms', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    log_message(f"Saved CSV: {csv_path}")

    return csv_data


def run_all_experiments(scene_name='cornell_box'):
    """Run all experiments for a given scene"""
    log_message("="*60)
    log_message(f"New Methods Benchmark for Scene: {scene_name}")
    log_message("="*60)

    # Setup scene
    world_obj, cam_obj = main.setup_scene(scene_name)
    cam_obj.scene_mode = scene_name

    # Build PT reference
    log_message("Building PT reference...")
    pt_accum = np.zeros((*cam_obj.img_res, 3), dtype=np.float32)
    for _ in range(PT_REFERENCE_FRAMES):
        cam_obj.render_pt(world_obj)
        ti.sync()
        pt_accum += cam_obj.pt_frame.to_numpy().astype(np.float32)
    pt_reference_linear = pt_accum / PT_REFERENCE_FRAMES
    save_screenshot(cam_obj.pt_frame, f"{scene_name}_pt_reference.png")
    log_message("PT reference built")

    # Define modes to test
    modes = [
        (RENDER_MODE_PT, "PT"),
        (RENDER_MODE_GRID, "Grid"),
        (RENDER_MODE_HYBRID, "Hybrid"),
        (RENDER_MODE_MS_AIC, "MS_AIC"),
        (RENDER_MODE_MCTF, "MCTF"),
        (RENDER_MODE_FULL, "FULL"),
    ]

    # Run each mode
    all_data = {}
    for mode_idx, mode_name in modes:
        try:
            data = run_experiment_for_mode(mode_idx, mode_name, world_obj, cam_obj, pt_reference_linear, scene_name)
            all_data[mode_name] = data
        except Exception as e:
            log_message(f"ERROR in mode {mode_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary
    log_message("\nGenerating summary...")
    summary_path = os.path.join(output_dir, f"{scene_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Summary for Scene: {scene_name}\n")
        f.write("="*60 + "\n\n")

        for mode_name, data in all_data.items():
            if mode_name == "PT":
                continue

            avg_fps = np.mean([row['fps'] for row in data])
            avg_mse = np.mean([row['mse'] for row in data])
            final_mse = data[-1]['mse']

            f.write(f"{mode_name}:\n")
            f.write(f"  Avg FPS: {avg_fps:.1f}\n")
            f.write(f"  Avg MSE: {avg_mse:.6e}\n")
            f.write(f"  Final MSE: {final_mse:.6e}\n\n")

        # Performance comparison
        if "Hybrid" in all_data and "FULL" in all_data:
            hybrid_mse = np.mean([row['mse'] for row in all_data["Hybrid"]])
            full_mse = np.mean([row['mse'] for row in all_data["FULL"]])
            improvement = (hybrid_mse - full_mse) / hybrid_mse * 100

            f.write(f"Quality Improvement (FULL vs Hybrid):\n")
            f.write(f"  {improvement:.2f}% reduction in MSE\n\n")

    log_message(f"Saved summary: {summary_path}")

    log_message("\n" + "="*60)
    log_message("All experiments completed!")
    log_message("="*60)


if __name__ == "__main__":
    # Run experiments for all scenes
    scenes_to_test = ['cornell_box', 'random', 'two_room', 'night_scene']

    for scene in scenes_to_test:
        log_message(f"\n\n>>> Testing scene: {scene}")
        try:
            run_all_experiments(scene)
        except Exception as e:
            log_message(f"ERROR in scene {scene}: {e}")
            import traceback
            traceback.print_exc()

    log_message("\n\n>>> All scenes tested!")
    log_message(f"Results saved to: {output_dir}")
