import taichi as ti
from hittable import Sphere
from world import World
from camera import Camera
import material
import utils
import random
import time
import os
import csv
from datetime import datetime
import experiment_config as cfg

# Initialize Taichi with fixed RNG seed for reproducibility
ti.init(arch=ti.gpu, random_seed=42)
# Seed Python RNG for deterministic scene generation
random.seed(42)

vec3 = ti.types.vector(3, float)
spheres = []
materials = []

# 当前场景模式：'random', 'cornell_box', 'night_scene', 'two_room', 'classroom', 'bathroom', 'veach_mis'
# NOTE: Do NOT initialize the scene at import time; benchmarks import main.py.
# The scene must be constructed explicitly in __main__ or benchmark runners.
DEFAULT_SCENE = 'cornell_box'

def setup_scene(
    mode: str = 'random',
    *,
    interpolation_on=None,
    importance_sampling_on=None,
    adaptive_logic_on=None,
    normal_weighting_on=None,
    distance_weighting_on=None,
    neighbor_clamping_on=None,
):
    """根据 mode 构造场景，并返回 (world, cam)。

    支持：
    - 'random'：原来的随机小球场景（默认）
    - 'cornell_box'：五面墙 + 顶部强光 + 金属球/玻璃球
    - 'night_scene'：黑色背景 + 多彩高亮点光源 + 高反射金属球
    - 'two_room'：两室一门场景（测试窄缝漏光）
    - 'classroom'：教室场景（黑板 + 窗户光）
    - 'bathroom'：浴室场景（白瓷砖 + 镜面 + caustics）
    - 'veach_mis'：Veach MIS 场景（多强度光源测试重要性采样）
    """
    global spheres, materials, world, cam, big_indices, prev_centers

    # 重新生成场景时，保持可重复性
    random.seed(42)

    spheres = []
    materials = []

    # 使用统一场景配置
    from scenes.scene_configs import get_scene
    spheres, materials, cam_params = get_scene(mode)

    world = World(spheres, materials)
    # Prepare ablation-specific camera parameters, filtering out any that were not passed
    ablation_params = {
        "interpolation_on": interpolation_on,
        "importance_sampling_on": importance_sampling_on,
        "adaptive_logic_on": adaptive_logic_on,
        "normal_weighting_on": normal_weighting_on,
        "distance_weighting_on": distance_weighting_on,
        "neighbor_clamping_on": neighbor_clamping_on,
    }
    # Only include parameters that were explicitly passed to setup_scene, allowing Camera defaults
    final_ablation_params = {k: v for k, v in ablation_params.items() if v is not None}

    # Merge scene parameters with ablation parameters
    # Ablation parameters will override any conflicting keys from scene_params
    final_cam_params = {**cam_params, **final_ablation_params}

    cam = Camera(world, **final_cam_params)

    # Store scene_bounds on camera for downstream scripts (ablation/benchmark) to reuse
    if 'scene_bounds' in cam_params:
        cam.scene_bounds = cam_params['scene_bounds']

    print(f"Initializing scene: {mode}")
    # Prefer scene-provided logical bounds to avoid huge AABB from "wall spheres".
    scene_bounds = cam_params.get('scene_bounds', None)
    cam.adapt_grid_to_scene(spheres, verbose=True, scene_bounds=scene_bounds)
    # Populate camera-side light source list for importance sampling (NEE)
    cam.set_light_sources(spheres, materials)

    # Identify large spheres (to monitor for movement) and store previous centers
    big_indices = []
    prev_centers = []
    for i, s in enumerate(spheres):
        if s.radius > 0.9:
            big_indices.append(i)
            prev_centers.append(s.center)

    return world, cam


# IMPORTANT: Scene must NOT be initialized at import time.
# `world` / `cam` will be created in __main__ or by benchmark scripts.
world = None
cam = None

# Experiment control
# options: 'PT', 'Grid', 'Adaptive', 'ERROR'
render_mode = 'ERROR'
mode_map = {'PT': 0, 'Grid': 1, 'Adaptive': 2, 'ERROR': 3}

# Create results directory and timestamped experiment subdirectory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "results"
experiment_dir = os.path.join(results_dir, f"experiment_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)
log_path = os.path.join(experiment_dir, 'experiment_log.csv')

# Initialize log file with header
with open(log_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame_index', 'mode', 'FPS', 'MSE'])

print(f"Created results directory: {results_dir}")
print(f"Created experiment directory: {experiment_dir}")
print(f"Experiment log will be saved to: {log_path}")

def main():
    # start_time = time.perf_counter()
    # print(f"Rendering time: {time.perf_counter() - start_time:.2f}s")

    gui = ti.GUI('Taichi Raytracing', cam.img_res, fast_gui=True)
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
    rendered_frames = 0

    # Monitoring / automation variables
    fps_tick_start = time.perf_counter()
    fps_tick_count = 0
    last_frame_time = time.perf_counter()
    move_count = 0
    last_move_frame = -1000
    recovery_monitoring = False
    recovery_start_frame = 0
    consecutive_small = 0
    last_mse = None
    # buffered log writes to avoid per-frame IO
    logs_buffer = []

    # Run indefinitely (or until GUI closed)
    while gui.running:
        weight = 1.0 / (rendered_frames + 1)
        # Detect large-sphere movement and boost nearby grid update probability if moved
        moved_indices = []
        for idx_i, idx in enumerate(big_indices):
            cur_center = spheres[idx].center
            prev = prev_centers[idx_i]
            # compare components; if changed, mark moved
            moved = False
            for t in range(3):
                # only consider meaningful movement above threshold to avoid noisy triggers
                if abs(cur_center[t] - prev[t]) > 0.01:
                    moved = True
                    break
            if moved:
                moved_indices.append(idx)
                prev_centers[idx_i] = cur_center

            # If any big sphere moved, clear grids and re-adapt bounds
            if len(moved_indices) > 0:
                print(f"Frame {rendered_frames}: Large sphere moved, clearing grids and re-adapting...")
                cam.clear_grid_data()
                # Optional: re-adapt if AABB changed significantly
                cam.adapt_grid_to_scene(spheres, verbose=True)

        # Reset weights to 1.0 using Taichi field fill (avoids expensive Python loops)
        cam.grid_update_weight.fill(1.0)

        # Boost weights in regions near moved big spheres via Taichi kernel (GPU-side)
        for midx in moved_indices:
            center = spheres[midx].center
            influence = spheres[midx].radius * 3.0
            cam.boost_weights_region([float(center[0]), float(center[1]), float(center[2])], influence, cfg.ADAPTIVE_BOOST_MULTIPLIER)

        # After boosting, smooth weights to avoid hard seams
        if len(moved_indices) > 0:
            cam.blur_update_weights()

        # Mode-specific behavior
        mode_int = mode_map.get(render_mode, 2)

        if render_mode == 'PT':
            # Pure path tracing (no grid updates)
            cam.render(world, mode_int)
        elif render_mode == 'Grid':
            # Grid-only with reduced base update (1%) to improve performance
            cam.update_grid(world, 0.01)
            cam.render(world, mode_int)
        elif render_mode == 'ERROR':
            # Error heatmap (real-time): compare Hybrid vs an incrementally-accumulated PT reference.
            # This keeps the GUI responsive while PT reference converges over time.
            # 1) produce a hybrid frame
            cam.update_grid(world, 0.01)
            cam.render(world, mode_map['Adaptive'])
            cam.asvgf_filter()

            # 2) incrementally accumulate PT reference (a few spp per frame)
            if rendered_frames == 0:
                cam.clear_pt_reference()
            # tune this: 1~4 for interactivity
            cam.accumulate_pt_reference(world, 2)

            # 3) overwrite cam.frame with heatmap visualization
            cam.render_error_heatmap()
        else:
            # Adaptive hybrid: we updated weights above, apply reduced base update (1%)
            cam.update_grid(world, 0.01)
            cam.render(world, mode_int)
            # Apply lightweight A-SVGF to hybrid frame to reduce noise while preserving edges
            cam.asvgf_filter()

        # Average into display buffer (gamma conversion handled inside)
        average_frames(current_frame, cam.frame, weight)
        gui.set_image(current_frame)
        gui.show()

        # Update adaptive sampling weights (computed from current frame) for next frame
        cam.compute_adaptive_weights(cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
                         cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
                         cfg.ADAPTIVE_MAX_MULTIPLIER)

        # --- Experiment automation bookkeeping ---
        rendered_frames += 1
        fps_tick_count += 1

        # instantaneous FPS measurement
        now_frame = time.perf_counter()
        frame_time = now_frame - last_frame_time
        fps = (1.0 / frame_time) if frame_time > 0.0 else 0.0
        last_frame_time = now_frame

        # Every 60 frames, print average FPS, grid memory usage, and PT reference spp
        if fps_tick_count >= 60:
            now = time.perf_counter()
            elapsed = now - fps_tick_start
            avg_fps = fps_tick_count / elapsed if elapsed > 0 else 0.0
            nx, ny, nz = cam.grid_res
            grid_mem_mb = float(nx * ny * nz * 3 * 4) / (1024.0 * 1024.0)
            print(f"[Stats] Frame {rendered_frames}: "
                  f"FPS={avg_fps:.2f}, "
                  f"Grid={grid_mem_mb:.2f} MB, "
                  f"PT ref={cam.pt_spp_count[None]} spp")
            fps_tick_start = now
            fps_tick_count = 0

        # Dynamic displacement every 200 frames: move primary large sphere along X by +0.5
        if rendered_frames % 200 == 0:
            if len(big_indices) > 0:
                idx = big_indices[0]
                spheres[idx].center[0] = spheres[idx].center[0] + 0.5
                prev_centers[0] = spheres[idx].center
                print(f"[Move] Frame {rendered_frames}: moved big sphere {idx} by +0.5 on X")
                cam.clear_grid_data()
                cam.adapt_grid_to_scene(spheres, verbose=True)
                move_count += 1
                last_move_frame = rendered_frames
                # start monitoring recovery
                recovery_monitoring = True
                recovery_start_frame = rendered_frames
                consecutive_small = 0
                last_mse = None

        # Render PT baseline for MSE comparison (single-sample PT)
        cam.render_pt(world)
        mse = cam.compute_mse()

        # Recovery detection: after a move, look for 5 consecutive frames with <0.1% relative MSE change
        if recovery_monitoring:
            if last_mse is None:
                last_mse = mse
                consecutive_small = 0
            else:
                rel = abs(mse - last_mse) / (abs(last_mse) + 1e-12)
                if rel < 0.001:
                    consecutive_small += 1
                else:
                    consecutive_small = 0
                last_mse = mse
                if consecutive_small >= 5:
                    recovery_frames = rendered_frames - recovery_start_frame
                    print(f"[Recovery] move #{move_count} recovered in {recovery_frames} frames")
                    recovery_monitoring = False

        # Timed screenshots at move+5 and move+50
        if last_move_frame >= 0:
            rel_frame = rendered_frames - last_move_frame
            if rel_frame == 5 or rel_frame == 50:
                # 1) save current displayed buffer
                filename = f"{render_mode}_move_{move_count}_frame_{rel_frame}.png"
                filepath = os.path.join(experiment_dir, filename)
                ti.tools.imwrite(current_frame, filepath)
                print(f"Saved screenshot: {filepath}")

                # 2) additionally save error heatmap (Hybrid vs PT reference)
                #    for paper visualization of convergence around shadow edges.
                # For interactive ERROR mode, PT reference is accumulated incrementally.
                # If you want an offline 1024-spp reference for the saved images, switch
                # back to: cam.render_pt_reference(world, target_spp=1024, chunk_spp=8, reset=True)
                cam.render_error_heatmap()
                heatmap = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
                average_frames(heatmap, cam.frame, 1.0)
                heatmap_name = f"ERROR_move_{move_count}_frame_{rel_frame}.png"
                heatmap_path = os.path.join(experiment_dir, heatmap_name)
                ti.tools.imwrite(heatmap, heatmap_path)
                print(f"Saved error heatmap: {heatmap_path}")

        # Buffer per-frame data and write in batches to avoid per-frame IO
        logs_buffer.append([rendered_frames, render_mode, f"{fps:.2f}", f"{mse:.8e}"])
        if len(logs_buffer) >= 10:
            try:
                with open(log_path, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(logs_buffer)
                logs_buffer.clear()
            except Exception as e:
                print(f"Failed to flush logs: {e}")

    # Flush remaining logs on exit
    if logs_buffer:
        try:
            with open(log_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerows(logs_buffer)
        except Exception as e:
            print(f"Failed to flush logs on exit: {e}")

    # Save last displayed image on exit
    output_filename = f"output_{render_mode}_{timestamp}.png"
    output_filepath = os.path.join(experiment_dir, output_filename)
    ti.tools.imwrite(current_frame, output_filepath)
    print(f"Saved final output: {output_filepath}")

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])


if __name__ == '__main__':
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Taichi Hybrid Path Tracer')
    parser.add_argument('--scene', type=str, default=DEFAULT_SCENE,
                       help='Scene to render (cornell_box, two_room, night_scene, random, classroom, bathroom, veach_mis)')
    parser.add_argument('--mode', type=str, default='ERROR',
                       help='Rendering mode (PT, Grid, Adaptive, ERROR)')

    args = parser.parse_args()

    # Initialize scene with specified parameters
    world, cam = setup_scene(args.scene)

    # Set render mode if specified
    if args.mode in mode_map:
        render_mode = args.mode

    main()
