import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
from main import main, spheres, cam, world

# Initialize Taichi
ti.init(arch=ti.gpu)

# Rendering modes
RENDER_MODE_PT = 0      # Path Tracing (Ground Truth)
RENDER_MODE_GRID = 1    # Pure Grid
RENDER_MODE_HYBRID = 2  # Hybrid Adaptive

# Global variables
render_mode = RENDER_MODE_PT
frame_count = 0
current_mode_frames = 0
mode_start_time = 0
benchmark_data = []
pt_reference = None  # Will store PT reference for MSE calculation

# Create output directory
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)

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
    gui.show(filepath)
    log_message(f"Saved screenshot: {filepath}")

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images"""
    diff = img1 - img2
    return float((diff * diff).mean())

def save_benchmark_results():
    """Save benchmark results to CSV"""
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["frame", "mode", "fps", "mse", "timestamp"])
        # Write data
        for data in benchmark_data:
            writer.writerow(data)
    log_message(f"Benchmark results saved to {csv_path}")

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
    global frame_count, current_mode_frames, pt_reference
    
    mode_frames = 200  # Number of frames to run per mode
    modes = [RENDER_MODE_PT, RENDER_MODE_GRID, RENDER_MODE_HYBRID]
    current_mode_idx = 0
    switch_mode(modes[current_mode_idx])
    
    # Main rendering loop
    while gui.running and current_mode_idx < len(modes):
        # Handle key events
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
        
        # Render frame
        start_time = time.time()
        
        # Select rendering method based on current mode
        if render_mode == RENDER_MODE_PT:
            cam.render_pt(world)
        elif render_mode == RENDER_MODE_GRID:
            cam.render_grid(world)
        else:  # RENDER_MODE_HYBRID
            cam.render_hybrid(world)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # Update frame buffer with progressive rendering
        weight = 1.0 / (current_mode_frames + 1)
        average_frames(current_frame, cam.frame, weight)
        
        # Store PT reference for MSE calculation
        if render_mode == RENDER_MODE_PT:
            pt_reference = current_frame.to_numpy()
        
        # Calculate MSE if we have a PT reference
        mse = 0.0
        if pt_reference is not None and render_mode != RENDER_MODE_PT:
            current_img = current_frame.to_numpy()
            mse = calculate_mse(pt_reference, current_img)
        
        # Update benchmark data
        benchmark_data.append([
            frame_count,
            get_mode_name(render_mode),
            fps,
            mse,
            datetime.now().isoformat()
        ])
        
        # Display
        gui.set_image(current_frame)
        
        # Display stats
        gui.text(f"Mode: {get_mode_name(render_mode)}", (0.05, 0.95))
        gui.text(f"FPS: {fps:.1f}", (0.05, 0.90))
        if render_mode != RENDER_MODE_PT:
            gui.text(f"MSE vs PT: {mse:.6f}", (0.05, 0.85))
        gui.text(f"Frame: {current_mode_frames + 1}/{mode_frames}", (0.05, 0.80)
        
        gui.show()
        
        # Update counters
        frame_count += 1
        current_mode_frames += 1
        
        # Save screenshot at the last frame of each mode
        if current_mode_frames == mode_frames:
            # Save screenshot
            mode_name = get_mode_name(render_mode).lower().replace(" ", "_")
            save_screenshot(gui, f"result_{mode_name}.png")
            
            # Switch to next mode
            current_mode_idx += 1
            if current_mode_idx < len(modes):
                switch_mode(modes[current_mode_idx])
            else:
                # Benchmark complete
                save_benchmark_results()
                log_message("Benchmark completed successfully!")
                break

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    """Average frames for progressive rendering"""
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * new_frame[i, j]

if __name__ == "__main__":
    # Add new render methods to the Camera class
    @ti.kernel
    def render_pt(self, world: ti.template()):
        """Render using path tracing (ground truth)"""
        for x, y in self.frame:
            pixel_color = vec3(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                view_ray = self.get_ray(x, y)
                pixel_color += self.get_ray_color_pt(view_ray, world) / self.samples_per_pixel
            self.frame[x, y] = pixel_color

    @ti.kernel
    def render_grid(self, world: ti.template()):
        """Render using only the irradiance grid"""
        for x, y in self.frame:
            pixel_color = vec3(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                view_ray = self.get_ray(x, y)
                pixel_color += self.get_ray_color_grid(view_ray, world) / self.samples_per_pixel
            self.frame[x, y] = pixel_color

    @ti.kernel
    def render_hybrid(self, world: ti.template()):
        """Render using hybrid adaptive method"""
        for x, y in self.frame:
            pixel_color = vec3(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                view_ray = self.get_ray(x, y)
                pixel_color += self.get_ray_color_hybrid(view_ray, world) / self.samples_per_pixel
            self.frame[x, y] = pixel_color

    # Add the new methods to the Camera class
    from camera import Camera
    Camera.render_pt = render_pt
    Camera.render_grid = render_grid
    Camera.render_hybrid = render_hybrid

    log_message("Starting benchmark...")
    run_benchmark()
