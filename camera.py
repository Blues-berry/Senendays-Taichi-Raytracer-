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
            # compute grid index
            local = p - self.grid_origin
            ix = int(local[0] // self.grid_cell_size)
            iy = int(local[1] // self.grid_cell_size)
            iz = int(local[2] // self.grid_cell_size)
            # clamp indices - if outside grid, use material albedo
            if ix < 0 or ix >= self.grid_res[0] or iy < 0 or iy >= self.grid_res[1] or iz < 0 or iz >= self.grid_res[2]:
                color = world.materials.albedo[hit.record.id]
            else:
                color = self.irradiance_grid[ix, iy, iz]
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
                # cast a single probe ray in a random direction
                dir = utils.random_unit_vector()
                r = Ray(origin=pos, direction=dir)
                hit = world.hit_world(r, 0.001, tm.inf)
                if hit.did_hit:
                    # approximate irradiance by albedo of hit material
                    col = world.materials.albedo[hit.record.id]
                    self.irradiance_grid[i, j, k] = col
                else:
                    # sample environment color
                    unit_direction = dir.normalized()
                    a = 0.5 * (unit_direction[1] + 1.0)
                    self.irradiance_grid[i, j, k] = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)