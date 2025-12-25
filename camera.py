import taichi as ti
import taichi.math as tm
import math
import utils
from ray import Ray

vec3 = ti.types.vector(3, float)

ray_return = ti.types.struct(hit_surface=bool, resulting_ray=Ray, color=vec3)

@ti.data_oriented
class Camera:
    @ti.func
    def get_background_color(self, direction: vec3) -> vec3:
        """根据场景模式返回背景色"""
        if ti.static(self.scene_mode in ['night_scene', 'cornell_box']):
            # 夜间场景和Cornell Box使用纯黑背景
            return vec3(0.0, 0.0, 0.0)
        else:
            # 默认的渐变天空色
            unit_direction = direction.normalized()
            a = 0.5 * (unit_direction[1] + 1.0)
            return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)

    def __init__(
        self,
        world: ti.template(),
        lookfrom=vec3(13, 2, 3),
        lookat=vec3(0, 0, 0),
        vup=vec3(0, 1, 0),
        vfov=20.0,
        defocus_angle=0.6,
        focus_dist=10.0,
        scene_mode: str = 'random',
    ):
        # 记录当前场景模式（用于背景色等逻辑）
        self.scene_mode = scene_mode
        aspect_ratio = 16.0 / 9.0
        width = 1200
        height = int(width / aspect_ratio)
        self.img_res = (width, height)

        self.samples_per_pixel = 5
        self.max_ray_depth = 500

        # Use parameters passed into constructor (do not overwrite)
        # `vfov`, `lookfrom`, `lookat`, `vup`, `defocus_angle`, `focus_dist`
        # are provided by the caller (e.g. `setup_scene`) and should not
        # be replaced by hard-coded defaults here.
        self.defocus_angle = defocus_angle

        # Virtual rectangle in scene that camera sends rays through
        theta = math.radians(vfov)
        h = tm.tan(theta / 2)
        viewport_height = 2.0 * h * focus_dist
        viewport_width = viewport_height * width / height

        w = utils.normalize(lookfrom - lookat)
        u = utils.normalize(vup.cross(w))
        v = w.cross(u)

        self.camera_origin = lookfrom

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        viewport_u = u * viewport_width
        viewport_v = -v * viewport_height

        # Calculate per-pixel deltas
        self.px_delta_u = viewport_u / width
        self.px_delta_v = viewport_v / height

        # Calculate the upper-left corner of the viewport
        viewport_ul = self.camera_origin - (focus_dist * w) - (viewport_u / 2) + (viewport_v / 2)
        self.first_px = viewport_ul + self.px_delta_u / 2 - self.px_delta_v / 2

        defocus_radius = focus_dist * math.tan(math.radians(self.defocus_angle / 2))
        self.defocus_disk_u = u * defocus_radius
        self.defocus_disk_v = v * defocus_radius

        self.frame = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        # store a PT baseline frame for MSE comparisons
        self.pt_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        # accumulator for MSE kernel
        self._mse_acc = ti.field(dtype=ti.f32, shape=())

        # Irradiance grid for storing spatial lighting information
        self.grid_res = (32, 32, 32)
        # Define grid origin and cell size (world-space AABB)
        self.grid_origin = vec3(-8.0, -1.0, -8.0)
        self.grid_cell_size = 1.0
        self.irradiance_grid = ti.Vector.field(n=3, dtype=ti.f32, shape=self.grid_res)
        # Per-cell weight multiplier for update probability (default 1.0)
        self.grid_update_weight = ti.field(dtype=ti.f32, shape=self.grid_res)
        # Temporary buffer for weight blur
        self.grid_update_weight_tmp = ti.field(dtype=ti.f32, shape=self.grid_res)
        # initialize weights to 1.0
        self.grid_update_weight.fill(1.0)

    @ti.kernel
    def render(self, world: ti.template(), mode: ti.i32):
        # Main rendering loop, mode: 0=PT, 1=Grid, 2=Adaptive
        for x, y in self.frame:
            pixel_color = vec3(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                view_ray = self.get_ray(x, y)
                if mode == 0:
                    pixel_color += self.get_ray_color_pt(view_ray, world) / self.samples_per_pixel
                elif mode == 1:
                    pixel_color += self.get_ray_color_grid(view_ray, world) / self.samples_per_pixel
                else:
                    pixel_color += self.get_ray_color_hybrid(view_ray, world) / self.samples_per_pixel
            self.frame[x, y] = pixel_color

    @ti.kernel
    def render_pt(self, world: ti.template()):
        # Render a single-sample PT baseline into pt_frame
        for x, y in self.pt_frame:
            self.pt_frame[x, y] = self.get_ray_color_pt(self.get_ray(x, y), world)

    @ti.kernel
    def compute_mse(self) -> ti.f32:
        # Computes MSE between current `frame` and `pt_frame` (linear values)
        self._mse_acc[None] = 0.0
        for i, j in self.frame:
            a = self.frame[i, j]
            b = self.pt_frame[i, j]
            for c in ti.static(range(3)):
                diff = a[c] - b[c]
                # atomic add to accumulator
                self._mse_acc[None] += diff * diff
        # normalize
        denom = float(self.img_res[0] * self.img_res[1] * 3)
        return self._mse_acc[None] / denom

    @ti.func
    def defocus_disk_sample(self):
        p = utils.random_in_unit_disc()
        return self.camera_origin + self.defocus_disk_u * p[0] + self.defocus_disk_v * p[1]

    @ti.func
    def sample_irradiance_grid(self, p: vec3, world: ti.template(), fallback_id: int) -> vec3:
        # Map world position to grid-space continuous coordinates
        local = p - self.grid_origin
        fx = local[0] / self.grid_cell_size
        fy = local[1] / self.grid_cell_size
        fz = local[2] / self.grid_cell_size

        # integer base indices
        ix0 = int(ti.floor(fx))
        iy0 = int(ti.floor(fy))
        iz0 = int(ti.floor(fz))

        # fractional part
        tx = fx - ix0
        ty = fy - iy0
        tz = fz - iz0

        # clamp base indices so ix1 = ix0+1 is valid
        nx = self.grid_res[0]
        ny = self.grid_res[1]
        nz = self.grid_res[2]
        color = vec3(0.0)
        if ix0 < 0 or ix0 >= nx - 1 or iy0 < 0 or iy0 >= ny - 1 or iz0 < 0 or iz0 >= nz - 1:
            # Outside or on the border: fallback to material albedo
            color = world.materials.albedo[fallback_id]
        else:
            ix1 = ix0 + 1
            iy1 = iy0 + 1
            iz1 = iz0 + 1

            # fetch the 8 neighbors
            c000 = self.irradiance_grid[ix0, iy0, iz0]
            c100 = self.irradiance_grid[ix1, iy0, iz0]
            c010 = self.irradiance_grid[ix0, iy1, iz0]
            c110 = self.irradiance_grid[ix1, iy1, iz0]
            c001 = self.irradiance_grid[ix0, iy0, iz1]
            c101 = self.irradiance_grid[ix1, iy0, iz1]
            c011 = self.irradiance_grid[ix0, iy1, iz1]
            c111 = self.irradiance_grid[ix1, iy1, iz1]

            # trilinear interpolation
            c00 = c000 * (1.0 - tx) + c100 * tx
            c10 = c010 * (1.0 - tx) + c110 * tx
            c01 = c001 * (1.0 - tx) + c101 * tx
            c11 = c011 * (1.0 - tx) + c111 * tx

            c0 = c00 * (1.0 - ty) + c10 * ty
            c1 = c01 * (1.0 - ty) + c11 * ty

            color = c0 * (1.0 - tz) + c1 * tz
        return color

    # === Path Tracing (Ground-Truth) ===
    @ti.func
    def get_ray_color_pt(self, ray: Ray, world: ti.template()) -> vec3:
        """经典递归/迭代 Path Tracing，不使用光照网格，用作真值。"""
        attenuation = vec3(1.0)
        current_ray = ray
        color = vec3(0.0)
        for _ in range(self.max_ray_depth):
            hit = world.hit_world(current_ray, 0.001, tm.inf)
            if hit.did_hit:
                scatter_ret = world.materials.scatter(current_ray, hit.record)
                if scatter_ret.did_scatter:
                    attenuation *= scatter_ret.attenuation
                    current_ray = Ray(origin=scatter_ret.scattered.origin + tm.normalize(scatter_ret.scattered.direction) * 0.0002,
                                      direction=scatter_ret.scattered.direction)
                else:
                    # 发光材质或不散射材质，返回发光颜色
                    color = attenuation * scatter_ret.attenuation
                    break
            else:
                # 抵达背景
                env = self.get_background_color(current_ray.direction)
                color = attenuation * env
                break
        return color

    # === 纯网格模式 ===
    @ti.func
    def get_ray_color_grid(self, ray: Ray, world: ti.template()) -> vec3:
        hit = world.hit_world(ray, 0.001, tm.inf)
        color = vec3(0.0)
        if hit.did_hit:
            # Check if this is a light source
            mat_idx = world.materials.mat_index[hit.record.id]
            if mat_idx == world.materials.DIFFUSE_LIGHT:
                # Direct light emission
                color = world.materials.albedo[hit.record.id]
            else:
                # Sample irradiance grid for non-light sources
                color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id)
        else:
            color = self.get_background_color(ray.direction)
        return color

    # === 混合模式（自适应） ===
    @ti.func
    def get_ray_color_hybrid(self, ray: Ray, world: ti.template()) -> vec3:
        color = vec3(0.0, 0.0, 0.0)
        hit = world.hit_world(ray, 0.001, tm.inf)

        if hit.did_hit:
            mat_idx = world.materials.mat_index[hit.record.id]

            # Handle light sources directly
            if mat_idx == world.materials.DIFFUSE_LIGHT:
                # Direct light emission
                color = world.materials.albedo[hit.record.id]
            else:
                # Sample grid for ambient term or primary color
                grid_color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id)

                if mat_idx == world.materials.LAMBERT:
                    # For Lambertian, the color is primarily from the grid
                    color = grid_color
                else:  # Metal or Dielectric
                    # For specular materials, trace further and add ambient term
                    bounced_color = vec3(0.0)
                    attenuation = vec3(1.0)
                    current_ray = ray
                    current_hit_record = hit.record

                    # Perform 1-2 bounces for reflections
                    for i in range(2):
                        scatter_ret = world.materials.scatter(current_ray, current_hit_record)
                        if scatter_ret.did_scatter:
                            attenuation *= scatter_ret.attenuation
                            current_ray = scatter_ret.scattered
                            bounce_hit = world.hit_world(current_ray, 0.001, tm.inf)
                            if bounce_hit.did_hit:
                                current_hit_record = bounce_hit.record
                                # If the bounce hits a diffuse surface, sample grid and terminate
                                if world.materials.mat_index[bounce_hit.record.id] == world.materials.LAMBERT:
                                    bounced_color = attenuation * self.sample_irradiance_grid(bounce_hit.record.p, world, bounce_hit.record.id)
                                    break
                            else:
                                # Ray escaped to background
                                background = self.get_background_color(current_ray.direction)
                                bounced_color = attenuation * background
                                break
                        else:
                            # Absorbed
                            bounced_color = vec3(0.0)
                            break

                    # Final color is the traced reflection plus a small ambient contribution from the grid
                    ambient_contribution = 0.15
                    color = bounced_color + grid_color * ambient_contribution
        else:
            # Background color
            color = self.get_background_color(ray.direction)

        return color

    @ti.func
    def get_ray(self, x: int, y: int) -> Ray:
        # Generate a random offset within the pixel
        u_offset = ti.random(ti.f32) - 0.5
        v_offset = ti.random(ti.f32) - 0.5

        # Calculate the target point on the viewport
        pixel_sample = self.first_px + (x + u_offset) * self.px_delta_u - (y + v_offset) * self.px_delta_v

        ray_origin = self.camera_origin if self.defocus_angle <= 0 else self.defocus_disk_sample()

        # Calculate the direction of the ray
        direction = pixel_sample - ray_origin

        # Return the ray
        return Ray(origin=ray_origin, direction=direction)

    @ti.func
    def step_ray(self, ray: Ray, world: ti.template()) -> ray_return:
        color = vec3(0, 0, 0)
        hit = world.hit_world(ray, 0.00, tm.inf)
        resulting_ray = Ray()

        if hit.did_hit:
            # If the ray hits an object, scatter it
            scatter_ret = world.materials.scatter(ray, hit.record)
            if scatter_ret.did_scatter:
                color = scatter_ret.attenuation
                origin = scatter_ret.scattered.origin + tm.normalize(scatter_ret.scattered.direction) * .0002
                resulting_ray = Ray(origin=origin, direction=scatter_ret.scattered.direction)
            else:
                color = vec3(0, 0, 0)
        else:
            # Background color
            color = self.get_background_color(ray.direction)
        return ray_return(hit_surface=hit.did_hit, resulting_ray=resulting_ray, color=color)

    @ti.kernel
    def update_grid(self, world: ti.template(), base_prob: float):
        # Randomly update ~base_prob of the grid, scaled by per-cell weight
        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            p = ti.random(ti.f32)
            prob = base_prob * self.grid_update_weight[i, j, k]
            if p < prob:
                # world position at cell center
                pos = self.grid_origin + vec3((i + 0.5) * self.grid_cell_size,
                                             (j + 0.5) * self.grid_cell_size,
                                             (k + 0.5) * self.grid_cell_size)
                # cast a random probe ray and allow one additional bounce (2-bounce probe)
                dir = utils.random_unit_vector()
                r = Ray(origin=pos, direction=dir)
                hit = world.hit_world(r, 0.001, tm.inf)
                col = vec3(0.0, 0.0, 0.0)
                if hit.did_hit:
                    # first-hit material
                    scatter_ret = world.materials.scatter(r, hit.record)
                    if scatter_ret.did_scatter:
                        # second ray from scattered direction
                        origin2 = scatter_ret.scattered.origin + tm.normalize(scatter_ret.scattered.direction) * 0.0002
                        r2 = Ray(origin=origin2, direction=scatter_ret.scattered.direction)
                        hit2 = world.hit_world(r2, 0.001, tm.inf)
                        if hit2.did_hit:
                            col = scatter_ret.attenuation * world.materials.albedo[hit2.record.id]
                        else:
                            # second ray missed -> use environment color attenuated
                            unit_direction = r2.direction.normalized()
                            a = 0.5 * (unit_direction[1] + 1.0)
                            env = self.get_background_color(r2.direction)
                            col = scatter_ret.attenuation * env
                    else:
                        # no scatter: use albedo (local) only
                        col = world.materials.albedo[hit.record.id]
                    self.irradiance_grid[i, j, k] = col
                else:
                    # sample environment color
                    unit_direction = dir.normalized()
                    a = 0.5 * (unit_direction[1] + 1.0)
                    self.irradiance_grid[i, j, k] = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)

    @ti.kernel
    def boost_weights_region(self, cx: vec3, influence: float, multiplier: float):
        # Boost grid_update_weight for cells within `influence` radius of `cx` by `multiplier`.
        inf2 = influence * influence
        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            # compute cell center in grid/world space
            posx = self.grid_origin[0] + (i + 0.5) * self.grid_cell_size
            posy = self.grid_origin[1] + (j + 0.5) * self.grid_cell_size
            posz = self.grid_origin[2] + (k + 0.5) * self.grid_cell_size
            dx = posx - cx[0]
            dy = posy - cx[1]
            dz = posz - cx[2]
            if dx * dx + dy * dy + dz * dz <= inf2:
                self.grid_update_weight[i, j, k] = self.grid_update_weight[i, j, k] * multiplier

    @ti.kernel
    def blur_update_weights(self):
        # Gaussian blur with 3x3x3 kernel for smoother weight transitions
        # Using approximate Gaussian weights:
        # - Center: 0.4
        # - Face neighbors (6): 0.06 each
        # - Edge neighbors (12): 0.02 each
        # - Corner neighbors (8): 0.01 each
        # Total ≈ 1.0 for normalization

        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            s = 0.0
            total_weight = 0.0

            # 3x3x3 neighborhood
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    for dk in ti.static(range(-1, 2)):
                        ni = i + di
                        nj = j + dj
                        nk = k + dk

                        # Check bounds
                        if 0 <= ni < self.grid_res[0] and 0 <= nj < self.grid_res[1] and 0 <= nk < self.grid_res[2]:
                            # Calculate Gaussian weight based on distance
                            dist_sq = di * di + dj * dj + dk * dk
                            w = 0.0
                            if dist_sq == 0:
                                # Center
                                w = 0.4
                            elif dist_sq == 1:
                                # Face neighbors (6 total)
                                w = 0.06
                            elif dist_sq == 2:
                                # Edge neighbors (12 total)
                                w = 0.02
                            elif dist_sq == 3:
                                # Corner neighbors (8 total)
                                w = 0.01

                            s += w * self.grid_update_weight[ni, nj, nk]
                            total_weight += w

            # Normalize to prevent weight drift
            if total_weight > 0.0:
                self.grid_update_weight_tmp[i, j, k] = s / total_weight
            else:
                self.grid_update_weight_tmp[i, j, k] = self.grid_update_weight[i, j, k]

        # Copy back
        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            self.grid_update_weight[i, j, k] = self.grid_update_weight_tmp[i, j, k]

    def adapt_grid_to_scene(self, spheres: ti.template(), verbose: bool = False):
        """
        Automatically compute and adapt the irradiance grid to tightly fit the scene AABB.

        Args:
            spheres: List of Sphere objects in the scene
            verbose: If True, print diagnostic information about the grid adaptation
        """
        # Compute scene AABB from sphere centers and radii (Python-side)
        # spheres is a Python list of Sphere objects
        # find min and max
        first = True
        min_x = 0.0
        min_y = 0.0
        min_z = 0.0
        max_x = 0.0
        max_y = 0.0
        max_z = 0.0
        for s in spheres:
            c = s.center
            r = s.radius
            cx = float(c[0])
            cy = float(c[1])
            cz = float(c[2])
            if first:
                min_x = cx - r
                min_y = cy - r
                min_z = cz - r
                max_x = cx + r
                max_y = cy + r
                max_z = cz + r
                first = False
            else:
                if cx - r < min_x:
                    min_x = cx - r
                if cy - r < min_y:
                    min_y = cy - r
                if cz - r < min_z:
                    min_z = cz - r
                if cx + r > max_x:
                    max_x = cx + r
                if cy + r > max_y:
                    max_y = cy + r
                if cz + r > max_z:
                    max_z = cz + r

        # pad slightly to avoid boundary issues
        pad = 0.5  # Increased padding for better coverage
        min_x -= pad
        min_y -= pad
        min_z -= pad
        max_x += pad
        max_y += pad
        max_z += pad

        dx = max_x - min_x
        dy = max_y - min_y
        dz = max_z - min_z

        # choose cell size so that largest dimension fits grid resolution
        nx = self.grid_res[0]
        ny = self.grid_res[1]
        nz = self.grid_res[2]
        max_dim = dx
        if dy > max_dim:
            max_dim = dy
        if dz > max_dim:
            max_dim = dz

        cell = max_dim / float(max(nx, ny, nz))

        # set origin to min corner
        self.grid_origin = vec3(min_x, min_y, min_z)
        self.grid_cell_size = cell

        if verbose:
            print(f"\n=== Grid Adaptation ===")
            print(f"Scene AABB: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f}) to ({max_x:.2f}, {max_y:.2f}, {max_z:.2f})")
            print(f"Scene dimensions: {dx:.2f} x {dy:.2f} x {dz:.2f}")
            print(f"Grid resolution: {nx} x {ny} x {nz}")
            print(f"Grid origin: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f})")
            print(f"Cell size: {cell:.4f}")
            print(f"Grid coverage: {nx*cell:.2f} x {ny*cell:.2f} x {nz*cell:.2f}")
            print(f"Total spheres: {len(spheres)}")
            print("=====================\n")