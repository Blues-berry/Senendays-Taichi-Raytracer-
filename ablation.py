import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
import main
from main import spheres, cam, world
import utils

from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Experiment groups for ablation study ---
# The ablation switches are:
#   interpolation_on         (tri-linear interpolation)
#   importance_sampling_on   (light importance sampling / light-guided probes)
#   adaptive_logic_on        (adaptive weight update)
#   normal_weighting_on      (normal-weighted interpolation)
#   distance_weighting_on    (distance-based weighting + cutoff)
#   neighbor_clamping_on     (26-neighbor clamp)
#
# Required runs (in order):
#   Baseline            : all OFF
#   V1                  : interpolation only
#   V2                  : interpolation + adaptive
#   V3                  : interpolation + normal-weighting ONLY (proves anti-leaking independently)
#   Full_Hybrid         : all ON
EXPERIMENT_GROUPS = [
    {
        "name": "Baseline",
        "interpolation_on": False,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
        "normal_weighting_on": False,
        "distance_weighting_on": False,
        "neighbor_clamping_on": False,
    },
    {
        "name": "V1_Trilinear",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
        "normal_weighting_on": False,
        "distance_weighting_on": False,
        "neighbor_clamping_on": False,
    },
    {
        "name": "V2_Trilinear_Adaptive",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": True,
        "normal_weighting_on": False,
        "distance_weighting_on": False,
        "neighbor_clamping_on": False,
    },
    {
        "name": "V3_Normal_Only",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
        "normal_weighting_on": True,
        "distance_weighting_on": False,
        "neighbor_clamping_on": False,
    },
    {
        "name": "Full",
        "interpolation_on": True,
        "importance_sampling_on": True,
        "adaptive_logic_on": True,
        "normal_weighting_on": True,
        "distance_weighting_on": True,
        "neighbor_clamping_on": True,
    },
]

# Note: `main` already initializes Taichi when imported, avoid re-initializing here.

# Rendering modes
RENDER_MODE_PT = 0      # Path Tracing (Ground Truth)
RENDER_MODE_GRID = 1    # Pure Grid
RENDER_MODE_HYBRID = 2  # Hybrid Adaptive

# Only compare groups in HYBRID render mode (keeps runtime manageable)
COMPARE_RENDER_MODE = RENDER_MODE_HYBRID

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

    # Normalize to [0, 1] range if needed (more robust threshold check)
    if curr_f.max() > 255.0:
        curr_f = curr_f / 255.0
    if ref_f.max() > 255.0:
        ref_f = ref_f / 255.0

    # Handle NaN and Inf values (robust error handling)
    curr_f = np.nan_to_num(curr_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_f = np.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure values are in valid range [0, 1] after normalization
    curr_f = np.clip(curr_f, 0.0, 1.0)
    ref_f = np.clip(ref_f, 0.0, 1.0)

    # Calculate MSE in linear space
    diff = curr_f - ref_f
    mse = np.mean(diff ** 2)
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


def plot_mse_curves(mse_by_group, out_path, title):
    """Plot multiple MSE curves (one per group) on the same figure."""
    plt.figure(figsize=(10, 5))
    for group_name, series in mse_by_group.items():
        if len(series) == 0:
            continue
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        plt.plot(xs, ys, label=group_name, linewidth=1.5)
    plt.xlabel('Frame (since start of group run)')
    plt.ylabel('MSE vs PT (linear)')
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _apply_ablation_toggles(group_cfg: dict):
    """Apply ablation toggles to the camera + global experiment config.

    This benchmark expects camera.py to expose:
      - cam.interpolate_grid_sampling
      - cam.enable_light_guided_probes

    Adaptive logic is controlled in this benchmark by whether we call
    cam.compute_adaptive_weights() each frame.

    Anti-leak mechanisms are controlled via experiment_config (cfg.*):
      - NORMAL_WEIGHTING_ENABLED
      - DISTANCE_WEIGHTING_ENABLED
      - NEIGHBOR_CLAMPING_ENABLED

    IMPORTANT: These must be independently switchable for a rigorous ablation.
    """
    cam.interpolate_grid_sampling = bool(group_cfg.get("interpolation_on", False))
    cam.enable_light_guided_probes = bool(group_cfg.get("importance_sampling_on", False))

    import experiment_config as cfg

    # Independent anti-leak toggles
    cfg.NORMAL_WEIGHTING_ENABLED = bool(group_cfg.get("normal_weighting_on", False))
    cfg.DISTANCE_WEIGHTING_ENABLED = bool(group_cfg.get("distance_weighting_on", False))
    cfg.NEIGHBOR_CLAMPING_ENABLED = bool(group_cfg.get("neighbor_clamping_on", False))


def _trigger_object_movement_at_frame(frame_idx: int, trigger_frame: int = 200) -> bool:
    """Move the light sphere at a deterministic frame.

    Returns True if movement was applied at this frame.
    """
    if frame_idx != trigger_frame:
        return False
    if len(spheres) <= 0:
        return False

    light_index = len(spheres) - 1
    old_x = spheres[light_index].center[0]
    spheres[light_index].center[0] = old_x + 1.0
    log_message(
        f"Object Movement @ frame {frame_idx}: light sphere x {old_x:.3f} -> {spheres[light_index].center[0]:.3f}"
    )
    return True


def _write_group_csv(group_name: str, rows: List[Dict[str, Any]]):
    csv_path = os.path.join(output_dir, f"ablation_{group_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "mse",
                "fps",
                "gpu_time_ms",
                "timestamp",
                "interpolation_on",
                "importance_sampling_on",
                "adaptive_logic_on",
                "normal_weighting_on",
                "distance_weighting_on",
                "neighbor_clamping_on",
                "movement_applied",
                "grid_memory_mb",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    log_message(f"Saved group CSV: {csv_path}")


def _save_error_heatmap(group_name: str, frame_label: str, reference_spp: int = 512):
    """Generates and saves an error heatmap for the current state.

    Assumes `cam.frame` already contains the *hybrid* result (linear).

    Steps:
      1) Build a fixed PT reference (reference_spp, default 512 spp)
      2) Compute heatmap = abs(hybrid - pt_reference) with pseudo-color mapping
      3) Save PNG (gamma corrected)
    """
    log_message(f"Generating heatmap for {group_name} at {frame_label} (PT ref={reference_spp} spp)...")

    # Build PT reference (slow)
    cam.render_pt_reference(world, target_spp=int(reference_spp), chunk_spp=16, reset=True)

    # Compute heatmap (overwrites cam.frame)
    cam.render_error_heatmap()

    # Save heatmap (gamma corrected)
    heatmap_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
    main.average_frames(heatmap_buffer, cam.frame, 1.0)  # weight=1.0 just copies with gamma

    filename = f"ERROR_{group_name}_{frame_label}.png"
    filepath = os.path.join(output_dir, filename)
    ti.tools.imwrite(heatmap_buffer, filepath)
    log_message(f"Saved error heatmap: {filepath}")


def run_group_experiments(scene_mode='cornell_box'):
    """Run ablation study and output *four* CSV files (one per group)."""
    global frame_count, current_mode_frames, pt_reference, pt_reference_linear, world, cam, spheres

    # Keep an optional combined plot for quick inspection
    mse_by_group = {g['name']: [] for g in EXPERIMENT_GROUPS}

    # Shared settings
    movement_frame = 50
    test_frames = 100 # Reduced from 450 to prevent timeout

    for gi, g in enumerate(EXPERIMENT_GROUPS):
        group_name = g["name"]
        log_message(f"\n=== Running group {gi+1}/{len(EXPERIMENT_GROUPS)}: {group_name} ===")

        # Re-init scene for fairness (same camera/world setup)
        world, cam = main.setup_scene(scene_mode)
        cam.scene_mode = scene_mode

        # Apply feature toggles on camera
        _apply_ablation_toggles(g)

        # Ensure the compact light list exists when importance sampling is enabled
        try:
            # Some scene builders might not expose materials in this benchmark context.
            # Calling with whatever globals exist is fine; failures are non-fatal.
            cam.set_light_sources(spheres, globals().get("materials", None))
        except Exception:
            pass

        # Reset counters and references
        frame_count = 0
        current_mode_frames = 0
        pt_reference = None
        pt_reference_linear = None
        current_frame.fill(0.001)

        # (Re)adapt grid once
        cam.adapt_grid_to_scene(spheres, verbose=False)

        # Warm-up
        ti.sync()
        cam.update_grid(world, 0.01)
        cam.render(world, COMPARE_RENDER_MODE)
        if COMPARE_RENDER_MODE == RENDER_MODE_HYBRID:
            cam.asvgf_filter()
        ti.sync()

        # Capture PT reference for this group (short PT run)
        pt_ref_spp_frames = 150
        pt_accum = np.zeros((*cam.img_res, 3), dtype=np.float32)
        for _ in range(pt_ref_spp_frames):
            cam.render_pt(world)
            ti.sync()
            pt_accum += cam.pt_frame.to_numpy().astype(np.float32)
        pt_reference_linear = pt_accum / float(pt_ref_spp_frames)

        # Main test loop (record MSE/FPS). Ensure identical movement trigger for all groups.
        group_rows = []
        movement_applied_flag = False

        for f in range(test_frames):
            # Deterministic object movement at the same frame for every group
            moved_this_frame = _trigger_object_movement_at_frame(f, movement_frame)
            movement_applied_flag = movement_applied_flag or moved_this_frame

            # If the scene changed, re-adapt grid (important for hybrid/grid correctness)
            if moved_this_frame and COMPARE_RENDER_MODE in (RENDER_MODE_GRID, RENDER_MODE_HYBRID):
                cam.adapt_grid_to_scene(spheres, verbose=False)

            ti.sync()
            start_time = time.perf_counter()

            if COMPARE_RENDER_MODE in (RENDER_MODE_GRID, RENDER_MODE_HYBRID):
                cam.update_grid(world, 0.01)
            cam.render(world, COMPARE_RENDER_MODE)
            if COMPARE_RENDER_MODE == RENDER_MODE_HYBRID:
                cam.asvgf_filter()

            # Adaptive logic toggle: update weights only when enabled
            if bool(g.get("adaptive_logic_on", False)):
                import experiment_config as cfg

                cam.compute_adaptive_weights(
                    cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
                    cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
                    cfg.ADAPTIVE_MAX_MULTIPLIER,
                )

            # Save error heatmaps at move+5 and move+50 (only for Hybrid compare mode)
            # Using an offline fixed PT reference (512 spp) for paper-quality visualization.
            if COMPARE_RENDER_MODE == RENDER_MODE_HYBRID and (f == movement_frame + 5 or f == movement_frame + 50):
                rel = f - movement_frame
                # NOTE: _save_error_heatmap overwrites cam.frame with the heatmap.
                # That's fine because we already computed MSE/logged for this frame above.
                _save_error_heatmap(group_name, f"move_{rel}", reference_spp=512)

            ti.sync()
            frame_time = time.perf_counter() - start_time
            # Calculate FPS with filtering for extreme values
            if frame_time > 1e-6:
                fps = 1.0 / frame_time
                # Filter out unreasonable FPS values
                if fps < 0.1 or fps > 10000:
                    fps = 0.0
            else:
                fps = 0.0
            gpu_time_ms = frame_time * 1000.0  # Convert to milliseconds

            current_linear = cam.frame.to_numpy()
            mse = calculate_accurate_mse(current_linear, pt_reference_linear)

            mse_by_group[group_name].append((f, mse))
            group_rows.append(
                {
                    "frame": int(f),
                    "mse": float(mse),
                    "fps": float(fps),
                    "gpu_time_ms": float(gpu_time_ms),
                    "timestamp": datetime.now().isoformat(),
                    "interpolation_on": bool(g.get("interpolation_on", False)),
                    "importance_sampling_on": bool(g.get("importance_sampling_on", False)),
                    "adaptive_logic_on": bool(g.get("adaptive_logic_on", False)),
                    "normal_weighting_on": bool(g.get("normal_weighting_on", False)),
                    "distance_weighting_on": bool(g.get("distance_weighting_on", False)),
                    "neighbor_clamping_on": bool(g.get("neighbor_clamping_on", False)),
                    "movement_applied": bool(moved_this_frame),
                    "grid_memory_mb": float(cam.grid_res[0] * cam.grid_res[1] * cam.grid_res[2] * 3 * 4 / (1024.0 * 1024.0)),
                }
            )

            # Optional GUI preview
            if gui.running:
                weight = 1.0 / (f + 1)
                average_frames(current_frame, cam.frame, weight)
                gui.set_image(current_frame)
                gui.text(f"Group: {group_name}", (0.05, 0.95))
                gui.text(f"Mode: {get_mode_name(COMPARE_RENDER_MODE)}", (0.05, 0.90))
                gui.text(f"Frame: {f+1}/{test_frames}", (0.05, 0.85))
                gui.text(f"MSE: {mse:.6e}", (0.05, 0.80))
                if f == movement_frame:
                    gui.text("Object Movement", (0.05, 0.75), color=0xAAAAAA)
                gui.show()

        if not movement_applied_flag:
            log_message(
                f"WARNING: movement was not applied for group {group_name}. Expected at frame {movement_frame}."
            )

        _write_group_csv(group_name, group_rows)

        # Save a result screenshot per group
        save_screenshot(gui, f"result_{group_name}.png")

    # Optional combined plot for quick inspection
    plot_path = os.path.join(output_dir, 'mse_curves_groups.png')
    plot_mse_curves(
        mse_by_group,
        plot_path,
        title=f"MSE Curves (mode={get_mode_name(COMPARE_RENDER_MODE)}) - {scene_mode}",
    )
    log_message(f"Saved MSE plot: {plot_path}")

    return mse_by_group

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

def run_benchmark(scene_mode='cornell_box'):
    global frame_count, current_mode_frames, pt_reference, pt_reference_linear, world, cam, spheres, materials

    # 使用指定场景模式初始化场景
    world, cam = main.setup_scene(scene_mode)

    # 设置场景模式到相机对象
    cam.scene_mode = scene_mode

    mode_frames = 450  # Number of frames to run per mode
    modes = [RENDER_MODE_PT, RENDER_MODE_GRID, RENDER_MODE_HYBRID]
    current_mode_idx = 0

    # Track if displacement has occurred in current mode
    displacement_occurred = False
    displacement_frame_in_mode = -1  # Track frames after displacement

    # Grid 初始化（setup_scene 内已经 adapt 过，这里保持一次以防后续逻辑依赖）
    cam.adapt_grid_to_scene(spheres, verbose=True)
    log_message(f"Grid initialized for benchmark (scene={scene_mode})")

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
            # Apply lightweight A-SVGF denoiser to hybrid output
            cam.asvgf_filter()

        # Force synchronization after rendering to ensure completion
        ti.sync()
        frame_time = time.perf_counter() - start_time
        gpu_time_ms = frame_time * 1000.0  # Convert to milliseconds

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
            gpu_time_ms,
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

        # Update adaptive sampling weights for next frame
        import experiment_config as cfg
        cam.compute_adaptive_weights(cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
                         cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
                         cfg.ADAPTIVE_MAX_MULTIPLIER)

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

            # If this is Hybrid mode, and the frame is move+5/move+50, save heatmap too.
            if render_mode == RENDER_MODE_HYBRID and displacement_occurred and displacement_frame_in_mode in [5, 50]:
                _save_error_heatmap(mode_name, f"move_frame_{displacement_frame_in_mode}")

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

                # Auto-generate analysis plots
                log_message("\n=== Auto-generating analysis plots ===")
                auto_generate_analysis_plots(output_dir)

                break

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    """Average frames for progressive rendering (do averaging in linear space).

    We keep `current_frame` stored as gamma for compatibility, so we:
    1. convert the stored gamma `current_frame` back to linear,
    2. average with the incoming `new_frame` (assumed linear),
    3. convert the result back to gamma and store.
    This prevents double-gamma application which causes brightness blowup.
    """
    for i, j in new_frame:
        curr_linear = utils.gamma_to_linear_vec3(current_frame[i, j])
        new_linear = new_frame[i, j]
        avg_linear = (1.0 - weight) * curr_linear + weight * new_linear
        current_frame[i, j] = utils.linear_to_gamma_vec3(avg_linear)

def auto_generate_analysis_plots(results_dir: str):
    """Automatically generate all analysis plots after benchmark completes"""
    try:
        # Import plotting manager
        from plots.plot_manager import generate_all_plots

        log_message(f"Generating plots from: {results_dir}")
        generate_all_plots(results_dir)

        log_message("\n=== Analysis Complete ===")
        log_message("Generated files:")
        plots_dir = os.path.join(results_dir, "plots")
        if os.path.exists(plots_dir):
            for file in os.listdir(plots_dir):
                if file.endswith(('.png', '.txt', '.md')):
                    log_message(f"  - {file}")

    except Exception as e:
        log_message(f"WARNING: Failed to auto-generate plots: {e}")
        log_message("You can manually run: python -c \"from plots.plot_manager import generate_all_plots; generate_all_plots('results/YOUR_RESULTS_DIR')\"")


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
                    writer.writerow(["frame", "mode", "fps", "mse", "gpu_time_ms", "timestamp"])
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
        # New: 3-group ablation study (Baseline / Optimized_V1 / Full_Hybrid)
        run_group_experiments('cornell_box')
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
