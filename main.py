import taichi as ti
from hittable import Sphere
from world import World
from camera import Camera
import material
import utils
import random
import time

ti.init(arch=ti.gpu)

vec3 = ti.types.vector(3, float)
spheres = []
materials = []

floor = Sphere(center=vec3(0, -1000, -1), radius=1000)
floor_mat = material.Lambert(vec3(0.5, 0.5, 0.5))
spheres.append(floor)
materials.append(floor_mat)

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

def main():
    # start_time = time.perf_counter()
    # print(f"Rendering time: {time.perf_counter() - start_time:.2f}s")

    gui = ti.GUI('Taichi Raytracing', cam.img_res, fast_gui=True)
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
    rendered_frames = 0
    while gui.running and rendered_frames < 200:
        weight = 1.0 / (rendered_frames + 1)
        # Detect large-sphere movement and boost nearby grid update probability if moved
        moved_indices = []
        for idx_i, idx in enumerate(big_indices):
            cur_center = spheres[idx].center
            prev = prev_centers[idx_i]
            # compare components; if changed, mark moved
            moved = False
            for t in range(3):
                if abs(cur_center[t] - prev[t]) > 1e-5:
                    moved = True
                    break
            if moved:
                moved_indices.append(idx)
                prev_centers[idx_i] = cur_center

            # If any big sphere moved, re-adapt grid to scene bounds
            if len(moved_indices) > 0:
                print(f"Frame {rendered_frames}: Large sphere moved, re-adapting grid...")
                cam.adapt_grid_to_scene(spheres, verbose=True)

        # Reset weights to 1.0
        for i in range(cam.grid_res[0]):
            for j in range(cam.grid_res[1]):
                for k in range(cam.grid_res[2]):
                    cam.grid_update_weight[i, j, k] = 1.0

        # Boost weights in regions near moved big spheres
        for midx in moved_indices:
            center = spheres[midx].center
            influence = spheres[midx].radius * 3.0
            for i in range(cam.grid_res[0]):
                for j in range(cam.grid_res[1]):
                    for k in range(cam.grid_res[2]):
                        # compute cell center in Python floats
                        posx = cam.grid_origin[0] + (i + 0.5) * cam.grid_cell_size
                        posy = cam.grid_origin[1] + (j + 0.5) * cam.grid_cell_size
                        posz = cam.grid_origin[2] + (k + 0.5) * cam.grid_cell_size
                        cx = float(center[0])
                        cy = float(center[1])
                        cz = float(center[2])
                        dx = posx - cx
                        dy = posy - cy
                        dz = posz - cz
                        if dx * dx + dy * dy + dz * dz <= influence * influence:
                            cam.grid_update_weight[i, j, k] = cam.grid_update_weight[i, j, k] * 3.0

        # After boosting, smooth weights to avoid hard seams
        if len(moved_indices) > 0:
            cam.blur_update_weights()

        # Perform adaptive grid updates (base 5% per frame)
        cam.update_grid(world, 0.05)

        cam.render(world)
        average_frames(current_frame, cam.frame, weight)
        gui.set_image(current_frame)
        gui.show()
        rendered_frames += 1
    ti.tools.imwrite(current_frame, "output(32x32x32)caizhifenliu.png")

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])


if __name__ == '__main__':
    main()
