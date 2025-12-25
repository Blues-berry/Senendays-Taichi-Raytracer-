import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
from main import main, spheres, cam, world
import utils

# Note: `main` already initializes Taichi when imported, avoid re-initializing here.

# Rendering modes
RENDER_MODE_PT = 0      # Path Tracing (Ground Truth)
RENDER_MODE_GRID = 1    # Pure Grid
RENDER_MODE_HYBRID = 2  # Hybrid Adaptive

# Create results directory and timestamped output subdirectory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "results"
output_dir = os.path.join(results_dir, f"benchmark_results_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Global variables
render_mode = RENDER_MODE_PT
frame_count = 0
current_mode_frames = 0
mode_start_time = 0
benchmark_data = []
pt_reference = None  # Will store PT reference for MSE calculation (Linear Space)
pt_reference_linear = None  # Store linear space version for accurate MSE

# Initialize GUI
gui = ti.GUI('Raytracing Benchmark', cam.img_res, fast_gui=True)
current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
pt_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_screenshot(gui, filename):
    """Save a screenshot with the given filename"""
    filepath = os.path.join(output_dir, filename)
    ti.tools.imwrite(current_frame, filepath)
    log_message(f"Saved screenshot: {filepath}")

def calculate_accurate_mse(current_linear, reference_linear):
    """Calculate MSE in linear space for accurate photometric comparison"""
    # Ensure both are numpy arrays of type float32
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)
    
    # Normalize to [0, 1] range if needed
    if curr_f.max() > 1.1:
        curr_f = curr_f / 255.0
    if ref_f.max() > 1.1:
        ref_f = ref_f / 255.0
    
    # Calculate MSE in linear space
    mse = np.mean((curr_f - ref_f) ** 2)
    return float(mse)

def calculate_mse(img1, img2):
    """Legacy MSE function - kept for compatibility"""
    return calculate_accurate_mse(img1, img2)

def save_benchmark_results():
    """Final save of any remaining benchmark data to CSV"""
    if benchmark_data:
        # Flush any remaining data
        flush_benchmark_data()
        log_message("Final benchmark data save completed")
    else:
        log_message("No remaining data to save")

def switch_mode(new_mode):
    """Switch to a new rendering mode"""
    global render_mode, current_mode_frames, mode_start_time
    render_mode = new_mode
    current_mode_frames = 0
    mode_start_time = time.time()
    
    # Clear the frame when switching modes
    current_frame.fill(0)
    
    mode_names = {
        RENDER_MODE_PT: "Path Tracing",
        RENDER_MODE_GRID: "Pure Grid",
        RENDER_MODE_HYBRID: "Hybrid"
    }
    log_message(f"Switched to {mode_names[render_mode]} mode")

def get_mode_name(mode):
    """Get the name of a rendering mode"""
    names = {
        RENDER_MODE_PT: "Path Tracing",
        RENDER_MODE_GRID: "Pure Grid",
        RENDER_MODE_HYBRID: "Hybrid"
    }
    return names.get(mode, "Unknown")

def run_benchmark():
    global frame_count, current_mode_frames, pt_reference, pt_reference_linear
    
    mode_frames = 450  # Number of frames to run per mode
    modes = [RENDER_MODE_PT, RENDER_MODE_GRID, RENDER_MODE_HYBRID]
    current_mode_idx = 0
    
    # Track if displacement has occurred in current mode
    displacement_occurred = False
    displacement_frame_in_mode = -1  # Track frames after displacement
    
    # Initialize grid for Grid and Hybrid modes
    cam.adapt_grid_to_scene(spheres, verbose=True)
    log_message("Grid initialized for benchmark")
    
    # Initialize current_frame with a small value to avoid pure black
    current_frame.fill(0.001)
    
    switch_mode(modes[current_mode_idx])
    
    # Do a warm-up render to initialize everything
    ti.sync()
    if render_mode == RENDER_MODE_GRID or render_mode == RENDER_MODE_HYBRID:
        cam.update_grid(world, 0.01)
    cam.render(world, render_mode)
    ti.sync()
    log_message("Warm-up render completed")
    
    # Add timing validation
    last_frame_time = time.perf_counter()
    
    # Main rendering loop
    while gui.running and current_mode_idx < len(modes):
        # Handle key events
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
        
        # Render frame with proper GPU synchronization timing
        current_time = time.perf_counter()
        expected_gap = current_time - last_frame_time
        
        # Force synchronization before timing to ensure accurate measurement
        ti.sync()
        start_time = time.perf_counter()
        
        # Select rendering method based on current mode
        # Use existing render method: mode 0=PT, 1=Grid, 2=Hybrid
        if render_mode == RENDER_MODE_PT:
            # Pure path tracing (no grid updates)
            cam.render(world, render_mode)
        elif render_mode == RENDER_MODE_GRID:
            # Grid-only with reduced base update (1%) to improve performance
            cam.update_grid(world, 0.01)
            cam.render(world, render_mode)
        else:
            # Adaptive hybrid: apply reduced base update (1%)
            cam.update_grid(world, 0.01)
            cam.render(world, render_mode)
        
        # Force synchronization after rendering to ensure completion
        ti.sync()
        frame_time = time.perf_counter() - start_time
        
        # Debug: log detailed timing for first few frames
        if frame_count < 5:
            log_message(f"Frame {frame_count}: expected_gap={expected_gap:.6f}s, render_time={frame_time:.6f}s")
        
        # Use the actual render time for FPS calculation
        fps = 1.0 / frame_time if frame_time > 0.001 else 0  # More reasonable threshold
        
        # Apply mode-specific FPS caps to filter outliers
        max_fps = {
            RENDER_MODE_PT: 200,     # Path Tracing is slow
            RENDER_MODE_GRID: 2000,  # Grid method is fast
            RENDER_MODE_HYBRID: 500   # Hybrid is medium
        }
        
        if fps > max_fps.get(render_mode, 500):
            old_fps = fps
            fps = 0.0
            if frame_count < 10:  # Only log first few warnings
                log_message(f"FPS capped: {old_fps:.1f} -> 0.0 (max for {get_mode_name(render_mode)}: {max_fps.get(render_mode, 500)})")
        
        # Update last frame time
        last_frame_time = current_time
        
        # Update frame buffer with progressive rendering
        weight = 1.0 / (current_mode_frames + 1)
        average_frames(current_frame, cam.frame, weight)
        
        # Debug: check frame content for first few frames
        if frame_count < 3:
            frame_min = float(current_frame.to_numpy().min())
            frame_max = float(current_frame.to_numpy().max())
            log_message(f"Frame {frame_count} content: min={frame_min:.6f}, max={frame_max:.6f}")
        
        # Store PT reference for MSE calculation (at frame 149, before displacement at frame 150)
        if render_mode == RENDER_MODE_PT and current_mode_frames == 149:
            # Store both gamma and linear space versions
            pt_reference = current_frame.to_numpy()  # Gamma space for display compatibility
            pt_reference_linear = cam.frame.to_numpy()  # Linear space for accurate MSE
            log_message("PT reference frame stored for MSE comparison")
        
        # Calculate MSE if we have a PT reference and we're not in PT mode
        mse = 0.0
        if pt_reference_linear is not None and render_mode != RENDER_MODE_PT:
            # Use linear space frames for accurate MSE calculation
            current_linear = cam.frame.to_numpy()  # Current linear frame
            mse = calculate_accurate_mse(current_linear, pt_reference_linear)
        
        # Update benchmark data
        benchmark_data.append([
            frame_count,
            get_mode_name(render_mode),
            fps,
            mse,
            datetime.now().isoformat()
        ])
        
        # Flush data to file every 10 frames to prevent data loss
        if len(benchmark_data) >= 10:
            flush_benchmark_data()
        
        # Display
        gui.set_image(current_frame)
        
        # Display stats
        gui.text(f"Mode: {get_mode_name(render_mode)}", (0.05, 0.95))
        gui.text(f"FPS: {fps:.1f}", (0.05, 0.90))
        if render_mode != RENDER_MODE_PT:
            gui.text(f"MSE vs PT: {mse:.6f}", (0.05, 0.85))
        else:
            gui.text("MSE vs PT: N/A (Reference)", (0.05, 0.85))
        gui.text(f"Frame: {current_mode_frames + 1}/{mode_frames}", (0.05, 0.80))
        gui.text(f"Data: {len(benchmark_data)} records", (0.05, 0.75))
        
        gui.show()
        
        # Dynamic displacement trigger: at frame 150, move the light source (only once per mode)
        if current_mode_frames == 150 and not displacement_occurred and len(spheres) > 0:
            log_message(f"DISPLACEMENT TRIGGERED - Total frames: {frame_count}, Mode frames: {current_mode_frames}")
            # Find the light source (the last sphere added, with high albedo)
            light_index = len(spheres) - 1  # The top light we added
            old_pos = spheres[light_index].center[0]
            spheres[light_index].center[0] = old_pos + 1.0
            log_message(f"Light source displaced from X={old_pos:.1f} to X={spheres[light_index].center[0]:.1f}")

            # Mark displacement as occurred
            displacement_occurred = True
            displacement_frame_in_mode = 0  # we'll count frames after displacement from 0

            # Reset frame buffer and frame counter for ALL modes.
            # This is critical to observe MSE spike at the displacement moment and subsequent convergence.
            current_frame.fill(0)
            current_mode_frames = 0
            log_message(f"Frame buffer and counter reset for {get_mode_name(render_mode)} mode after displacement")

            # Re-adapt grid for Grid and Hybrid modes
            if render_mode == RENDER_MODE_GRID or render_mode == RENDER_MODE_HYBRID:
                cam.adapt_grid_to_scene(spheres, verbose=False)
                log_message("Grid re-adapted after light source movement")

            # Force sync so the movement + reset are visible immediately
            ti.sync()

        # Screenshot: the 10th frame AFTER displacement (only once per mode)
        if displacement_occurred and displacement_frame_in_mode == 10:
            mode_name = get_mode_name(render_mode).lower().replace(" ", "_")
            save_screenshot(gui, f"after_displacement_{mode_name}_frame_10.png")
            # Prevent repeated saves if counters get reset unexpectedly
            displacement_frame_in_mode = -999999

        # Update counters
        frame_count += 1
        current_mode_frames += 1
        if displacement_occurred and displacement_frame_in_mode >= 0:
            displacement_frame_in_mode += 1
        
        # Save screenshot at specified frames: 5, 50, 100 (before displacement), then after displacement: 200, 250, 300, 350, 400, 450
        screenshot_frames = [5, 50, 100, 200, 250, 300, 350, 400, 450]
        if current_mode_frames in screenshot_frames:
            mode_name = get_mode_name(render_mode).lower().replace(" ", "_")
            filename = f"{mode_name}_frame_{current_mode_frames}.png"
            save_screenshot(gui, filename)
        
        # Save screenshot at the last frame of each mode
        if current_mode_frames == mode_frames:
            log_message(f"MODE COMPLETE - Total frames: {frame_count}, Mode frames: {current_mode_frames}")
            # Save screenshot
            mode_name = get_mode_name(render_mode).lower().replace(" ", "_")
            save_screenshot(gui, f"result_{mode_name}.png")
            
            # Switch to next mode
            current_mode_idx += 1
            if current_mode_idx < len(modes):
                switch_mode(modes[current_mode_idx])
                # Reset displacement flag for new mode
                displacement_occurred = False
                displacement_frame_in_mode = -1
            else:
                # Benchmark complete
                save_benchmark_results()
                log_message("Benchmark completed successfully!")
                break

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    """Average frames for progressive rendering"""
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])

def flush_benchmark_data():
    """Flush any pending benchmark data to CSV immediately"""
    if benchmark_data:
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        try:
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # Write header if file is new
                if not file_exists:
                    writer.writerow(["frame", "mode", "fps", "mse", "timestamp"])
                writer.writerows(benchmark_data)
            log_message(f"Flushed {len(benchmark_data)} records to {csv_path}")
            benchmark_data.clear()
        except Exception as e:
            log_message(f"Failed to flush benchmark data: {e}")

# Use existing render method from camera.py
# mode: 0=PT, 1=Grid, 2=Hybrid

if __name__ == "__main__":
    log_message("Starting benchmark...")
    try:
        run_benchmark()
    except KeyboardInterrupt:
        log_message("Benchmark interrupted by user")
        flush_benchmark_data()
    except Exception as e:
        log_message(f"Benchmark error: {e}")
        flush_benchmark_data()
        raise
    finally:
        # Ensure any remaining data is saved
        flush_benchmark_data()
