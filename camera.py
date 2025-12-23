import taichi as ti
import taichi.math as tm
import math
import utils
from ray import Ray

vec3 = ti.types.vector(3, float)

ray_return = ti.types.struct(hit_surface=bool, resulting_ray=Ray, color=vec3)

@ti.data_oriented
class Camera:
    def __init__(self, world: ti.template()):
        aspect_ratio = 16.0 / 9.0
        width = 1200
        height = int(width / aspect_ratio)
        self.img_res = (width, height)

        self.samples_per_pixel = 5
        self.max_ray_depth = 500

        vfov = 20
        lookfrom = vec3(13,2,3)
        lookat = vec3(0,0,0)
        vup = vec3(0,1,0)
        self.defocus_angle = 0.6
        focus_dist = 10

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

        # Irradiance grid for storing spatial lighting information
        self.grid_res = (16, 16, 16)
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
    def render(self, world: ti.template()):
        # Main rendering loop
        for x, y in self.frame:
            pixel_color = vec3(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                view_ray = self.get_ray(x, y)
                pixel_color += self.get_ray_color(view_ray, world) / self.samples_per_pixel
            self.frame[x, y] = pixel_color

    @ti.func
    def defocus_disk_sample(self):
        p = utils.random_in_unit_disc()
        return self.camera_origin + self.defocus_disk_u * p[0] + self.defocus_disk_v * p[1]

    @ti.func
    def get_ray_color(self, ray: Ray, world: ti.template()) -> vec3:
        # Single-step: trace to first surface and sample irradiance grid
        hit = world.hit_world(ray, 0.001, tm.inf)
        color = vec3(0.0, 0.0, 0.0)
        if hit.did_hit:
            p = hit.record.p
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
            if ix0 < 0:
                ix0 = 0
                tx = 0.0
            if iy0 < 0:
                iy0 = 0
                ty = 0.0
            if iz0 < 0:
                iz0 = 0
                tz = 0.0
            if ix0 >= nx - 1 or iy0 >= ny - 1 or iz0 >= nz - 1:
                # Outside or on the border: fallback to material albedo
                color = world.materials.albedo[hit.record.id]
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
        else:
            unit_direction = ray.direction.normalized()
            a = 0.5 * (unit_direction[1] + 1.0)
            color = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)

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
            unit_direction = ray.direction.normalized()
            a = 0.5 * (unit_direction[1] + 1.0)
            color = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)
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
                            env = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)
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
    def blur_update_weights(self):
        # Simple 1-step neighbor blur: center 0.5, 6-axis neighbors 0.08333 each
        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            center = self.grid_update_weight[i, j, k]
            s = 0.5 * center
            w = 0.08333333333333333
            # accumulate axis-aligned neighbors
            if i - 1 >= 0:
                s += w * self.grid_update_weight[i - 1, j, k]
            if i + 1 < self.grid_res[0]:
                s += w * self.grid_update_weight[i + 1, j, k]
            if j - 1 >= 0:
                s += w * self.grid_update_weight[i, j - 1, k]
            if j + 1 < self.grid_res[1]:
                s += w * self.grid_update_weight[i, j + 1, k]
            if k - 1 >= 0:
                s += w * self.grid_update_weight[i, j, k - 1]
            if k + 1 < self.grid_res[2]:
                s += w * self.grid_update_weight[i, j, k + 1]
            self.grid_update_weight_tmp[i, j, k] = s

        # copy back
        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            self.grid_update_weight[i, j, k] = self.grid_update_weight_tmp[i, j, k]

    def adapt_grid_to_scene(self, spheres: ti.template()):
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

        # pad slightly
        pad = 0.1
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