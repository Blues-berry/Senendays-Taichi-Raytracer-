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

# Initialize Taichi with fixed RNG seed for reproducibility
ti.init(arch=ti.gpu, random_seed=42)
# Seed Python RNG for deterministic scene generation
random.seed(42)

vec3 = ti.types.vector(3, float)
spheres = []
materials = []

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
                # diffuse
                spheres.append(Sphere(center=center, radius=0.2))
                materials.append(material.Lambert(utils.rand_vec(0,1) * utils.rand_vec(0,1)))
            elif choose_mat < 0.95:
                # metal
                spheres.append(Sphere(center=center, radius=0.2))
                materials.append(material.Metal(utils.rand_vec(0.5, 1), 0.5 * random.random()))
            else:
                # glass
                spheres.append(Sphere(center=center, radius=0.2))
                materials.append(material.Dielectric(1.5))

sph_1 = Sphere(center=vec3(0, 1, 0), radius=1)
sph_1_mat = material.Dielectric(1.5)
spheres.append(sph_1)
materials.append(sph_1_mat)

sph_2 = Sphere(center=vec3(-4, 1, 0), radius=1)
sph_2_mat = material.Lambert(vec3(0.4, 0.2, 0.1))
spheres.append(sph_2)
materials.append(sph_2_mat)

sph_3 = Sphere(center=vec3(4, 1, 0), radius=1)
sph_3_mat = material.Metal(vec3(0.7, 0.6, 0.5), 0.0)
spheres.append(sph_3)
materials.append(sph_3_mat)

world = World(spheres, materials)
cam = Camera(world)
# Adapt grid to scene AABB with verbose output
print("Initializing irradiance grid...")
cam.adapt_grid_to_scene(spheres, verbose=True)

# Identify large spheres (to monitor for movement) and store previous centers
big_indices = []
prev_centers = []
for i, s in enumerate(spheres):
    if s.radius > 0.9:
        big_indices.append(i)
        prev_centers.append(s.center)

# Experiment control
render_mode = 'Adaptive'  # options: 'PT', 'Grid', 'Adaptive'
mode_map = {'PT': 0, 'Grid': 1, 'Adaptive': 2}

# Ensure experiment log exists and has header
log_path = 'experiment_log.csv'
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_index', 'mode', 'FPS', 'MSE'])

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
            cam.boost_weights_region([float(center[0]), float(center[1]), float(center[2])], influence, 3.0)

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
        else:
            # Adaptive hybrid: we updated weights above, apply reduced base update (1%)
            cam.update_grid(world, 0.01)
            cam.render(world, mode_int)

        # Average into display buffer (gamma conversion handled inside)
        average_frames(current_frame, cam.frame, weight)
        gui.set_image(current_frame)
        gui.show()

        # --- Experiment automation bookkeeping ---
        rendered_frames += 1
        fps_tick_count += 1

        # instantaneous FPS measurement
        now_frame = time.perf_counter()
        frame_time = now_frame - last_frame_time
        fps = (1.0 / frame_time) if frame_time > 0.0 else 0.0
        last_frame_time = now_frame

        # Every 100 frames, print average FPS and grid memory usage
        if fps_tick_count >= 100:
            now = time.perf_counter()
            elapsed = now - fps_tick_start
            avg_fps = fps_tick_count / elapsed if elapsed > 0 else 0.0
            nx, ny, nz = cam.grid_res
            grid_mem_mb = float(nx * ny * nz * 3 * 4) / (1024.0 * 1024.0)
            print(f"[Stats] Frame {rendered_frames}: Avg FPS={avg_fps:.5f}, Grid mem={grid_mem_mb:.5f} MB")
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
                filename = f"{render_mode}_move_{move_count}_frame_{rel_frame}.png"
                ti.tools.imwrite(current_frame, filename)
                print(f"Saved screenshot: {filename}")

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
    ti.tools.imwrite(current_frame, "output(32x32x32)caizhifenliu.png")

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])


if __name__ == '__main__':
    main()
