"""
Quick Test Script - Minimal benchmark for testing
快速测试脚本 - 运行精简版的消融实验
"""

import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
import sys

# Add parent directory to path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import spheres, cam, world

# Initialize Taichi (already done in main, but keeping for clarity)

# Simple experiment groups - reduced to 2 for quick testing
EXPERIMENT_GROUPS = [
    {
        "name": "Baseline",
        "interpolation_on": False,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
    },
    {
        "name": "Full_Hybrid",
        "interpolation_on": True,
        "importance_sampling_on": True,
        "adaptive_logic_on": True,
    },
]

RENDER_MODE_PT = 0
RENDER_MODE_GRID = 1
RENDER_MODE_HYBRID = 2
COMPARE_RENDER_MODE = RENDER_MODE_HYBRID

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("test", f"test_results_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

gui = ti.GUI('Quick Test', cam.img_res, fast_gui=True)
current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)

def log_message(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")

def save_screenshot(filename):
    filepath = os.path.join(output_dir, filename)
    ti.tools.imwrite(current_frame, filepath)
    log_message(f"Saved: {filepath}")

def calculate_mse(current_linear, reference_linear):
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)
    if curr_f.max() > 1.1:
        curr_f = curr_f / 255.0
    if ref_f.max() > 1.1:
        ref_f = ref_f / 255.0
    mse = np.mean((curr_f - ref_f) ** 2)
    return float(mse)

def _apply_ablation_toggles(group_cfg):
    cam.interpolate_grid_sampling = bool(group_cfg.get("interpolation_on", False))
    cam.enable_light_guided_probes = bool(group_cfg.get("importance_sampling_on", False))

def _trigger_object_movement(frame_idx, trigger_frame=50):
    if frame_idx != trigger_frame:
        return False
    if len(spheres) <= 0:
        return False
    light_index = len(spheres) - 1
    old_x = spheres[light_index].center[0]
    spheres[light_index].center[0] = old_x + 1.0
    log_message(f"Object Movement @ frame {frame_idx}: x {old_x:.2f} -> {spheres[light_index].center[0]:.2f}")
    return True

def _write_group_csv(group_name, rows):
    csv_path = os.path.join(output_dir, f"{group_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "mse", "fps", "gpu_time_ms"])
        w.writeheader()
        w.writerows(rows)
    log_message(f"Saved CSV: {csv_path}")

def run_quick_test(scene_mode='cornell_box', test_frames=100):
    global world, cam, spheres
    
    print("\n" + "="*60)
    print("QUICK TEST - Ablation Study")
    print("="*60)
    
    for gi, g in enumerate(EXPERIMENT_GROUPS):
        group_name = g["name"]
        log_message(f"\n=== Group {gi+1}/{len(EXPERIMENT_GROUPS)}: {group_name} ===")
        
        # Setup scene
        world, cam = main.setup_scene(scene_mode)
        cam.scene_mode = scene_mode
        _apply_ablation_toggles(g)
        
        try:
            cam.set_light_sources(spheres, globals().get("materials", None))
        except:
            pass
        
        cam.adapt_grid_to_scene(spheres, verbose=False)
        current_frame.fill(0.001)
        
        # Warm-up
        ti.sync()
        cam.update_grid(world, 0.01)
        cam.render(world, COMPARE_RENDER_MODE)
        if COMPARE_RENDER_MODE == RENDER_MODE_HYBRID:
            cam.asvgf_filter()
        ti.sync()
        
        # PT reference (50 frames for quick test)
        pt_ref_frames = 50
        pt_accum = np.zeros((*cam.img_res, 3), dtype=np.float32)
        for _ in range(pt_ref_frames):
            cam.render_pt(world)
            ti.sync()
            pt_accum += cam.pt_frame.to_numpy().astype(np.float32)
        pt_reference_linear = pt_accum / float(pt_ref_frames)
        log_message(f"PT reference ready ({pt_ref_frames} spp)")
        
        # Main loop
        movement_frame = 50
        group_rows = []
        
        for f in range(test_frames):
            moved_this_frame = _trigger_object_movement(f, movement_frame)
            
            if moved_this_frame and COMPARE_RENDER_MODE in (RENDER_MODE_GRID, RENDER_MODE_HYBRID):
                cam.adapt_grid_to_scene(spheres, verbose=False)
            
            ti.sync()
            start_time = time.perf_counter()
            
            if COMPARE_RENDER_MODE in (RENDER_MODE_GRID, RENDER_MODE_HYBRID):
                cam.update_grid(world, 0.01)
            cam.render(world, COMPARE_RENDER_MODE)
            if COMPARE_RENDER_MODE == RENDER_MODE_HYBRID:
                cam.asvgf_filter()
            
            if bool(g.get("adaptive_logic_on", False)):
                import experiment_config as cfg
                cam.compute_adaptive_weights(
                    cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
                    cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
                    cfg.ADAPTIVE_MAX_MULTIPLIER
                )
            
            ti.sync()
            frame_time = time.perf_counter() - start_time
            fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
            gpu_time_ms = frame_time * 1000.0
            
            current_linear = cam.frame.to_numpy()
            mse = calculate_mse(current_linear, pt_reference_linear)
            
            group_rows.append({
                "frame": int(f),
                "mse": float(mse),
                "fps": float(fps),
                "gpu_time_ms": float(gpu_time_ms)
            })
            
            # GUI update every 10 frames
            if f % 10 == 0 and gui.running:
                weight = 1.0 / (f + 1)
                main.average_frames(current_frame, cam.frame, weight)
                gui.set_image(current_frame)
                gui.text(f"Group: {group_name}", (0.05, 0.92))
                gui.text(f"Frame: {f+1}/{test_frames}", (0.05, 0.88))
                gui.text(f"MSE: {mse:.4e}", (0.05, 0.84))
                gui.show()
        
        _write_group_csv(group_name, group_rows)
        save_screenshot(f"result_{group_name}.png")
        
        # Statistics
        mses = [r["mse"] for r in group_rows if r["mse"] > 0]
        fpss = [r["fps"] for r in group_rows if r["fps"] > 0]
        if mses:
            log_message(f"  MSE - Min: {min(mses):.4e}, Max: {max(mses):.4e}, Mean: {np.mean(mses):.4e}")
        if fpss:
            log_message(f"  FPS - Min: {min(fpss):.1f}, Max: {max(fpss):.1f}, Mean: {np.mean(fpss):.1f}")
    
    log_message("\n" + "="*60)
    log_message("QUICK TEST COMPLETE")
    log_message("="*60)

if __name__ == "__main__":
    try:
        run_quick_test(scene_mode='cornell_box', test_frames=100)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
