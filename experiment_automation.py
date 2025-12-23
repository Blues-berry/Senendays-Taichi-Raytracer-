import taichi as ti
import numpy as np
import time
import os
from datetime import datetime
from main import main, spheres, cam, world
import experiment_config as config

ti.init(arch=ti.gpu)

# Global variables for experiment control
use_adaptive_logic = True  # Toggle for adaptive logic
experiment_dir = config.OUTPUT_DIRECTORY
frame_count = 0
fps_history = []
convergence_data = []
current_experiment = ""
start_time = 0
last_frame_time = 0
last_big_sphere_move = 0
convergence_start_frame = 0
is_converging = False

# Create experiment directory if it doesn't exist
os.makedirs(experiment_dir, exist_ok=True)

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def calculate_grid_memory_usage():
    """Calculate the memory usage of the irradiance grid in MB"""
    grid_size = np.prod(cam.grid_res) * 3 * 4  # 3 floats (RGB) * 4 bytes per float
    return grid_size / (1024 * 1024)  # Convert to MB

def save_frame_info(frame_num, fps, memory_usage, is_keyframe=False, convergence_frames=None):
    """Save frame information to a log file"""
    log_file = os.path.join(experiment_dir, f"{current_experiment}_log.txt")
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"Frame {frame_num}: FPS={fps:.2f}, Grid Memory={memory_usage:.2f}MB"
        if is_keyframe:
            line += f", Keyframe at frame {frame_num}"
        if convergence_frames is not None:
            line += f", Converged in {convergence_frames} frames"
        line += "\n"
        f.write(f"[{timestamp}] {line}")

def save_screenshot(gui, frame_num, prefix=""):
    """Save a screenshot with the given prefix and frame number"""
    filename = os.path.join(experiment_dir, f"{current_experiment}_{prefix}frame_{frame_num:04d}.png")
    gui.show(filename)
    log_message(f"Saved screenshot: {filename}")

def move_big_sphere():
    """Move the first big sphere to a new random position"""
    global last_big_sphere_move, convergence_start_frame, is_converging
    
    # Find the first big sphere (not the floor)
    big_sphere_idx = -1
    for i, s in enumerate(spheres):
        if s.radius > 0.9 and i > 0:  # Skip the floor
            big_sphere_idx = i
            break
    
    if big_sphere_idx != -1:
        # Save current position for convergence check
        prev_pos = spheres[big_sphere_idx].center
        
        # Move to a new random position
        new_x = np.random.uniform(-8, 8)
        new_z = np.random.uniform(-8, 8)
        spheres[big_sphere_idx].center = ti.Vector([new_x, 1.0, new_z])
        
        # Update tracking variables
        last_big_sphere_move = frame_count
        convergence_start_frame = frame_count
        is_converging = True
        
        log_message(f"Moved big sphere from {prev_pos} to {spheres[big_sphere_idx].center}")
        return True
    return False

def check_convergence():
    """Check if the image has converged after a sphere move"""
    global is_converging, convergence_data
    
    if not is_converging:
        return None
    
    # Simple convergence check: if we've waited at least 5 frames after the move
    frames_since_move = frame_count - convergence_start_frame
    if frames_since_move >= 5:  # Minimum frames to wait before checking
        convergence_frames = frames_since_move
        convergence_data.append(convergence_frames)
        log_message(f"Convergence: {convergence_frames} frames to stabilize")
        is_converging = False
        return convergence_frames
    
    return None

def experiment_main():
    global frame_count, current_experiment, start_time, last_frame_time
    
    # Setup experiment
    current_experiment = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_message(f"Starting experiment: {current_experiment}")
    log_message(f"Adaptive logic: {'ENABLED' if use_adaptive_logic else 'DISABLED'}")
    
    # Initialize GUI and frame buffer
    gui = ti.GUI('Taichi Raytracing Experiment', cam.img_res, fast_gui=True)
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
    
    start_time = time.time()
    last_frame_time = start_time
    
    while gui.running and frame_count < 1000:  # Run for up to 1000 frames
        frame_count += 1
        current_time = time.time()
        
        # Move big sphere every 200 frames
        if frame_count % 200 == 0 and frame_count > 0:
            move_big_sphere()
        
        # Update grid with adaptive logic if enabled
        if use_adaptive_logic:
            # Reset weights to 1.0
            cam.grid_update_weight.fill(1.0)
            
            # Boost weights near big spheres
            for s in spheres:
                if s.radius > 0.9:  # Big sphere
                    center = s.center
                    influence = s.radius * 3.0
                    for i in range(cam.grid_res[0]):
                        for j in range(cam.grid_res[1]):
                            for k in range(cam.grid_res[2]):
                                posx = cam.grid_origin[0] + (i + 0.5) * cam.grid_cell_size
                                posy = cam.grid_origin[1] + (j + 0.5) * cam.grid_cell_size
                                posz = cam.grid_origin[2] + (k + 0.5) * cam.grid_cell_size
                                dx = posx - center[0]
                                dy = posy - center[1]
                                dz = posz - center[2]
                                if dx*dx + dy*dy + dz*dz <= influence*influence:
                                    cam.grid_update_weight[i, j, k] = 3.0
            
            # Smooth weights
            cam.blur_update_weights()
        
        # Update grid (5% base update rate)
        cam.update_grid(world, 0.05)
        
        # Render frame (mode: adaptive if enabled, else grid)
        mode_int = 2 if use_adaptive_logic else 1
        cam.render(world, mode_int)
        
        # Update frame buffer
        weight = 1.0 / (frame_count + 1)
        average_frames(current_frame, cam.frame, weight)
        
        # Display
        gui.set_image(current_frame)
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - last_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(fps)
        last_frame_time = current_time
        
        # Calculate memory usage
        memory_usage = calculate_grid_memory_usage()
        
        # Log every 100 frames or on key events
        if frame_count % 100 == 0 or frame_count == 1 or is_converging:
            log_message(f"Frame {frame_count}: FPS={fps:.2f}, Grid Memory={memory_usage:.2f}MB")
        
        # Check for convergence after sphere movement
        convergence_result = check_convergence()
        
        # Save screenshots at key frames
        if is_converging:
            frames_since_move = frame_count - convergence_start_frame
            if frames_since_move == 5:  # Right after movement stabilizes
                save_screenshot(gui, frame_count, "after_move_")
                save_frame_info(frame_count, fps, memory_usage, is_keyframe=True)
            elif frames_since_move == 50:  # After some convergence
                save_screenshot(gui, frame_count, "converged_")
                save_frame_info(frame_count, fps, memory_usage, is_keyframe=True, 
                              convergence_frames=convergence_result)
        
        gui.show()
    
    # Save final results
    save_results()
    log_message(f"Experiment {current_experiment} completed after {frame_count} frames")

def save_results():
    """Save experiment results to files"""
    # Save FPS data
    if fps_history:
        np.savetxt(os.path.join(experiment_dir, f"{current_experiment}_fps.csv"), 
                  np.array(fps_history), delimiter=",")
    
    # Save convergence data
    if convergence_data:
        np.savetxt(os.path.join(experiment_dir, f"{current_experiment}_convergence.csv"),
                  np.array(convergence_data), delimiter=",")
    
    # Save experiment summary
    summary = f"""Experiment Summary
================
Experiment ID: {current_experiment}
Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
Duration: {time.time() - start_time:.2f} seconds
Total Frames: {frame_count}
Average FPS: {np.mean(fps_history):.2f}
Max FPS: {np.max(fps_history):.2f}
Min FPS: {np.min(fps_history):.2f}
Grid Memory Usage: {calculate_grid_memory_usage():.2f} MB
Adaptive Logic: {'ENABLED' if use_adaptive_logic else 'DISABLED'}
"""
    
    if convergence_data:
        summary += f"\nConvergence Statistics:\n"
        summary += f"  Total Convergence Events: {len(convergence_data)}\n"
        summary += f"  Average Frames to Converge: {np.mean(convergence_data):.2f} frames\n"
        summary += f"  Fastest Convergence: {np.min(convergence_data)} frames\n"
        summary += f"  Slowest Convergence: {np.max(convergence_data)} frames\n"
    
    with open(os.path.join(experiment_dir, f"{current_experiment}_summary.txt"), "w") as f:
        f.write(summary)
    
    log_message("Results saved to experiment_results/")

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])

if __name__ == "__main__":
    # Run with adaptive logic enabled
    use_adaptive_logic = True
    experiment_main()
    
    # Optionally, run again with adaptive logic disabled for comparison
    # use_adaptive_logic = False
    # experiment_main()
