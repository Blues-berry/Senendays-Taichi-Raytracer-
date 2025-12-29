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

# 当前场景模式：'random', 'cornell_box', 'night_scene'
CURRENT_SCENE = 'random'

def setup_scene(mode: str = 'random'):
    """根据 mode 构造场景，并返回 (world, cam)。

    支持：
    - 'random'：原来的随机小球场景（默认）
    - 'cornell_box'：五面墙 + 顶部强光 + 金属球/玻璃球
    - 'night_scene'：黑色背景 + 多彩高亮点光源 + 高反射金属球
    """
    global spheres, materials, world, cam, big_indices, prev_centers

    # 重新生成场景时，保持可重复性
    random.seed(42)

    spheres = []
    materials = []

    # 默认相机参数（不同场景会覆盖）
    cam_params = dict(
        lookfrom=vec3(13, 2, 3),
        lookat=vec3(0, 0, 0),
        vup=vec3(0, 1, 0),
        vfov=20.0,
        defocus_angle=0.6,
        focus_dist=10.0,
        scene_mode=mode,
    )

    if mode == 'random':
        floor = Sphere(center=vec3(0, -1000, -1), radius=1000)
        floor_mat = material.Lambert(vec3(0.5, 0.5, 0.5))
        spheres.append(floor)
        materials.append(floor_mat)

        # Small spheres grid (deterministic via seeded RNG above)
        for a in range(-11, 11):
            for b in range(-11, 11):
                choose_mat = random.random()
                center = vec3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
                if (center - vec3(4, 0.2, 0)).norm() > 0.9:
                    if choose_mat < 0.8:
                        spheres.append(Sphere(center=center, radius=0.2))
                        materials.append(material.Lambert(utils.rand_vec(0, 1) * utils.rand_vec(0, 1)))
                    elif choose_mat < 0.95:
                        spheres.append(Sphere(center=center, radius=0.2))
                        materials.append(material.Metal(utils.rand_vec(0.5, 1), 0.5 * random.random()))
                    else:
                        spheres.append(Sphere(center=center, radius=0.2))
                        materials.append(material.Dielectric(1.5))

        sph_1 = Sphere(center=vec3(0, 1, 0), radius=1)
        spheres.append(sph_1)
        materials.append(material.Dielectric(1.5))

        sph_2 = Sphere(center=vec3(-4, 1, 0), radius=1)
        spheres.append(sph_2)
        materials.append(material.Lambert(vec3(0.4, 0.2, 0.1)))

        sph_3 = Sphere(center=vec3(4, 1, 0), radius=1)
        spheres.append(sph_3)
        materials.append(material.Metal(vec3(0.7, 0.6, 0.5), 0.0))

        top_light = Sphere(center=vec3(0, 5, 0), radius=0.5)
        spheres.append(top_light)
        materials.append(material.DiffuseLight(vec3(20, 20, 20)))

    elif mode == 'cornell_box':
        # 标准 Cornell Box（五面墙：左红、右绿、其余白；顶部强发光面光源；两球：金属+玻璃）
        # 注：当前项目只有 Sphere 作为几何体，因此墙体/面光源用“大球近似平面”的方式实现。
        white = vec3(0.73, 0.73, 0.73)
        left_red = vec3(0.65, 0.05, 0.05)
        right_green = vec3(0.12, 0.45, 0.15)

        # Cornell Box 内部空间范围（近似）：x,y,z ∈ [-half, half]，相机位于盒子外部朝 -z 看
        half = 2.75
        R = 1000.0  # 大球半径（越大越接近平面）

        # === 五面墙（没有前墙，便于相机观察） ===
        # 左墙（红）/ 右墙（绿）
        spheres.append(Sphere(center=vec3(-(R + half), 0.0, 0.0), radius=R))
        materials.append(material.Lambert(left_red))
        spheres.append(Sphere(center=vec3((R + half), 0.0, 0.0), radius=R))
        materials.append(material.Lambert(right_green))

        # 地面 / 天花板 / 后墙（白）
        spheres.append(Sphere(center=vec3(0.0, -(R + half), 0.0), radius=R))
        materials.append(material.Lambert(white))
        spheres.append(Sphere(center=vec3(0.0, (R + half), 0.0), radius=R))
        materials.append(material.Lambert(white))
        spheres.append(Sphere(center=vec3(0.0, 0.0, -(R + half)), radius=R))
        materials.append(material.Lambert(white))

        # === 顶部面光源（用一个较大的发光球近似平面光源） ===
        # 位置靠近天花板中央，略向后，半径增大以模拟“面光源”面积。
        spheres.append(Sphere(center=vec3(0.0, half - 0.15, -1.0), radius=0.85))
        materials.append(material.DiffuseLight(vec3(25.0, 25.0, 25.0)))

        # === 盒内两球 ===
        # 高反射金属球（低 fuzz）
        spheres.append(Sphere(center=vec3(-0.85, -half + 0.70, -1.65), radius=0.70))
        materials.append(material.Metal(vec3(0.93, 0.93, 0.93), 0.01))

        # 折射玻璃球
        spheres.append(Sphere(center=vec3(0.95, -half + 0.70, -0.95), radius=0.70))
        materials.append(material.Dielectric(1.5))

        # Cornell Box 相机建议：在盒子外面往里看（禁用景深）
        cam_params.update(
            lookfrom=vec3(0.0, 0.0, 8.5),
            lookat=vec3(0.0, -0.2, -1.3),
            vfov=40.0,
            defocus_angle=0.0,
            focus_dist=8.5,
        )

    elif mode == 'night_scene':
        # 暗色地面
        spheres.append(Sphere(center=vec3(0, -1000, 0), radius=1000))
        materials.append(material.Lambert(vec3(0.08, 0.08, 0.09)))

        # 5 个不同颜色的高亮度点光源
        light_positions = [
            vec3(-6, 4, -2),
            vec3(-2, 3, 2),
            vec3(2, 3.5, -1),
            vec3(6, 4, 2),
            vec3(0, 5, 6),
        ]
        light_colors = [
            vec3(15, 9, 4),   # 暖橘
            vec3(4, 10, 15),  # 冰蓝
            vec3(14, 4, 10),  # 洋红
            vec3(6, 15, 6),   # 绿
            vec3(12, 12, 16), # 冷白偏蓝
        ]
        for p, c in zip(light_positions, light_colors):
            spheres.append(Sphere(center=p, radius=0.35))
            materials.append(material.DiffuseLight(c))

        # 随机散布高反射金属球
        for _ in range(25):
            x = random.uniform(-7.0, 7.0)
            z = random.uniform(-7.0, 7.0)
            r = random.uniform(0.25, 0.6)
            y = r
            spheres.append(Sphere(center=vec3(x, y, z), radius=r))
            # 高反射：低 fuzz
            base = utils.rand_vec(0.7, 1.0)
            materials.append(material.Metal(base, random.uniform(0.0, 0.08)))

        cam_params.update(
            lookfrom=vec3(0, 3, 12),
            lookat=vec3(0, 1, 0),
            vfov=35.0,
            defocus_angle=0.0,
            focus_dist=12.0,
        )

    else:
        raise ValueError(f"未知场景模式: {mode}")

    world = World(spheres, materials)
    cam = Camera(world, **cam_params)

    print(f"Initializing scene: {mode}")
    cam.adapt_grid_to_scene(spheres, verbose=True)
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


# 初始化场景
world, cam = setup_scene(CURRENT_SCENE)

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

            # If any big sphere moved, re-adapt grid to scene bounds
            if len(moved_indices) > 0:
                print(f"Frame {rendered_frames}: Large sphere moved, re-adapting grid...")
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
    main()
