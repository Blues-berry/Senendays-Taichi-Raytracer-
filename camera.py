import taichi as ti
import taichi.math as tm
import math
import utils
import experiment_config as cfg
import material as mat
from ray import Ray

vec3 = ti.types.vector(3, float)

ray_return = ti.types.struct(hit_surface=bool, resulting_ray=Ray, color=vec3)
probe_return = ti.types.struct(color=vec3, weight=ti.f32)

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

        # Grid / probe tuning (configurable)
        self.grid_res = cfg.GRID_RESOLUTION
        self.grid_samples_per_update = 4
        self.grid_probe_depth = 3
        self.grid_update_alpha = 0.5
        self.gaussian_blur_enabled = cfg.GAUSSIAN_BLUR_ENABLED

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
        # store a PT reference frame for MSE/heatmap comparisons
        # - pt_frame: current reference (typically averaged over N spp)
        # - pt_accum: running sum accumulator to build a high-spp reference
        self.pt_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        self.pt_accum = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        self.pt_spp_count = ti.field(dtype=ti.i32, shape=())
        self.pt_spp_count[None] = 0
        self.pt_accum.fill(0.0)
        # accumulator for MSE kernel
        self._mse_acc = ti.field(dtype=ti.f32, shape=())

        # G-Buffer: normals (vec3) and depth (float)
        self.normal_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        self.depth_buffer = ti.field(dtype=ti.f32, shape=self.img_res)
        # denoised output buffer for A-SVGF
        self.denoised_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)

        # Temporal accumulation (EMA) buffers for reducing noise across frames
        self.accum_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        # previous-frame G-buffer for motion detection
        self.prev_normal_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)
        self.prev_depth_buffer = ti.field(dtype=ti.f32, shape=self.img_res)
        # EMA tuning (per-pixel alpha selection)
        # When pixel is static -> small alpha (strong history). When moving -> large alpha (fast update).
        self.accum_alpha_static = 0.10
        self.accum_alpha_moving = 0.80
        # initialize accumulators
        self.accum_frame.fill(0.0)
        self.prev_normal_buffer.fill(0.0)
        self.prev_depth_buffer.fill(1e9)

        # Adaptive sampling weight map (>=1.0). Starts neutral (1.0)
        self.adaptive_weight_map = ti.field(dtype=ti.f32, shape=self.img_res)
        self.adaptive_weight_map.fill(1.0)

        # sampling configuration
        self.base_samples = self.samples_per_pixel
        self.max_samples_per_pixel = max(8, int(self.base_samples * 2))

        # Irradiance grid for storing spatial lighting information
        # `self.grid_res` set above from config
        # Define grid origin and cell size (world-space AABB)
        self.grid_origin = vec3(-8.0, -1.0, -8.0)
        self.grid_cell_size = 1.0
        self.irradiance_grid = ti.Vector.field(n=3, dtype=ti.f32, shape=self.grid_res)
        # Per-cell weight multiplier for update probability (default 1.0)
        self.grid_update_weight = ti.field(dtype=ti.f32, shape=self.grid_res)
        # Per-cell luminance mean and variance for variance-guided updates
        self.irradiance_mean_lum = ti.field(dtype=ti.f32, shape=self.grid_res)
        self.irradiance_variance = ti.field(dtype=ti.f32, shape=self.grid_res)
        # Per-cell mean hit distance for leak detection
        self.grid_mean_distance = ti.field(dtype=ti.f32, shape=self.grid_res)
        # Temporary buffer for weight blur
        self.grid_update_weight_tmp = ti.field(dtype=ti.f32, shape=self.grid_res)
        # initialize weights to 1.0
        self.grid_update_weight.fill(1.0)
        # initialize variance/mean/distance defaults
        self.irradiance_mean_lum.fill(0.0)
        self.irradiance_variance.fill(0.0)
        # large sentinel distance (means "no geometry recorded yet")
        self.grid_mean_distance.fill(1e9)

        # Light source list for importance sampling (populated from Python)
        self.max_light_sources = 64
        self.light_count = ti.field(dtype=ti.i32, shape=())
        self.light_centers = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_light_sources)
        self.light_radii = ti.field(dtype=ti.f32, shape=self.max_light_sources)
        self.light_count[None] = 0

# Feature toggles (controlled by benchmark experiments)
        # - interpolate_grid_sampling: trilinear (8-probe) sampling vs nearest-probe sampling
        # - enable_light_guided_probes: occasionally aim probe rays at light sources during grid updates
        self.interpolate_grid_sampling = True
        self.enable_light_guided_probes = True

    @ti.kernel
    def render(self, world: ti.template(), mode: ti.i32):
        # Main rendering loop, supports adaptive per-pixel sample counts
        for x, y in self.frame:
            acc = vec3(0.0)
            # determine local samples from adaptive weight map
            local_w = self.adaptive_weight_map[x, y]
            local_samples = int(self.base_samples * local_w)
            if local_samples <= 0:
                local_samples = 1
            if local_samples > self.max_samples_per_pixel:
                local_samples = self.max_samples_per_pixel

            # perform up to max_samples_per_pixel probes, only accumulate active ones
            for s in range(self.max_samples_per_pixel):
                if s < local_samples:
                    view_ray = self.get_ray(x, y)
                    if mode == 0:
                        acc += self.get_ray_color_pt(view_ray, world, x, y)
                    elif mode == 1:
                        acc += self.get_ray_color_grid(view_ray, world, x, y)
                    else:
                        acc += self.get_ray_color_hybrid(view_ray, world, x, y)

            # Store averaged color
            cur_col = acc / float(local_samples)

            # --- Temporal EMA accumulation per-pixel ---
            # motion detection using previous G-buffer
            n_cur = self.normal_buffer[x, y]
            d_cur = self.depth_buffer[x, y]
            n_prev = self.prev_normal_buffer[x, y]
            d_prev = self.prev_depth_buffer[x, y]

            # default assume static
            alpha = self.accum_alpha_static

            # determine motion: consider depth relative change and normal alignment
            is_moving = False
            if d_prev > 1e8 and d_cur > 1e8:
                is_moving = False
            else:
                # relative depth change
                depth_rel = 0.0
                if d_prev > 1e-6:
                    depth_rel = d_cur - d_prev
                    if depth_rel < 0.0:
                        depth_rel = -depth_rel
                    depth_rel = depth_rel / d_prev
                else:
                    depth_rel = d_cur - d_prev
                    if depth_rel < 0.0:
                        depth_rel = -depth_rel

                ndot = n_cur.dot(n_prev)
                if depth_rel > 0.02 or ndot < 0.98:
                    is_moving = True

            if is_moving:
                alpha = self.accum_alpha_moving
            else:
                alpha = self.accum_alpha_static

            old = self.accum_frame[x, y]
            new_acc = old * (1.0 - alpha) + cur_col * alpha
            self.accum_frame[x, y] = new_acc

            # write out accumulated result and update prev G-buffer for next frame
            self.frame[x, y] = new_acc
            self.prev_normal_buffer[x, y] = n_cur
            self.prev_depth_buffer[x, y] = d_cur

    @ti.kernel
    def clear_pt_reference(self):
        # Reset high-spp PT reference accumulation
        for x, y in self.pt_accum:
            self.pt_accum[x, y] = vec3(0.0)
        self.pt_spp_count[None] = 0

    @ti.kernel
    def accumulate_pt_reference(self, world: ti.template(), spp: ti.i32):
        """Accumulate `spp` additional PT samples into pt_accum, then update pt_frame.

        This builds a much more stable PT reference for error heatmaps / metrics.
        """
        for x, y in self.pt_accum:
            # accumulate spp samples for this pixel
            for s in range(spp):
                self.pt_accum[x, y] += self.get_ray_color_pt(self.get_ray(x, y), world, x, y)

        # update global spp count
        self.pt_spp_count[None] += spp

        # update averaged reference frame
        inv = 1.0 / float(self.pt_spp_count[None])
        for x, y in self.pt_frame:
            self.pt_frame[x, y] = self.pt_accum[x, y] * inv

    def render_pt_reference(self, world, target_spp: int = 512, chunk_spp: int = 8, reset: bool = True):
        """Python helper to build a PT reference of `target_spp` samples per pixel.

        Args:
            target_spp: total spp to accumulate (e.g., 1024)
            chunk_spp: how many spp to accumulate per kernel call (controls responsiveness)
            reset: if True, clears previous accumulation first
        """
        if reset:
            self.clear_pt_reference()

        # accumulate in chunks to avoid extremely long single kernel
        remaining = int(target_spp)
        while remaining > 0:
            step = int(chunk_spp) if remaining > int(chunk_spp) else int(remaining)
            self.accumulate_pt_reference(world, step)
            remaining -= step

    @ti.kernel
    def render_pt(self, world: ti.template()):
        # Backward-compatible: single-sample PT into pt_frame (does NOT touch pt_accum)
        for x, y in self.pt_frame:
            self.pt_frame[x, y] = self.get_ray_color_pt(self.get_ray(x, y), world, x, y)

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

    @ti.kernel
    def render_error_heatmap(self):
        """Render error heatmap into `self.frame`.

        Error = abs(hybrid - pt_reference). Then map scalar error to a pseudo-color:
        low error -> cold (blue), high error -> hot (red).

        Notes:
        - Assumes `self.frame` currently contains the hybrid result (linear).
        - Assumes `self.pt_frame` contains the PT reference (linear).
        """
        for x, y in self.frame:
            a = self.frame[x, y]
            b = self.pt_frame[x, y]
            # L1 error magnitude (mean abs per channel)
            err = (ti.abs(a[0] - b[0]) + ti.abs(a[1] - b[1]) + ti.abs(a[2] - b[2])) / 3.0

            # Simple blue->cyan->green->yellow->red ramp.
            # Normalize using a tunable scale (bigger => more blue overall).
            # This is a visualization; not used for metrics.
            scale = 0.75
            t = err / scale
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0

            c = vec3(0.0)
            if t < 0.25:
                # blue -> cyan
                u = t / 0.25
                c = vec3(0.0, u, 1.0)
            elif t < 0.5:
                # cyan -> green
                u = (t - 0.25) / 0.25
                c = vec3(0.0, 1.0, 1.0 - u)
            elif t < 0.75:
                # green -> yellow
                u = (t - 0.5) / 0.25
                c = vec3(u, 1.0, 0.0)
            else:
                # yellow -> red
                u = (t - 0.75) / 0.25
                c = vec3(1.0, 1.0 - u, 0.0)

            self.frame[x, y] = c

    @ti.func
    def defocus_disk_sample(self):
        p = utils.random_in_unit_disc()
        return self.camera_origin + self.defocus_disk_u * p[0] + self.defocus_disk_v * p[1]

    @ti.func
    def _probe_contrib(self, p: vec3, ix: ti.i32, iy: ti.i32, iz: ti.i32, w: ti.f32) -> probe_return:
        result_color = vec3(0.0)
        result_weight = 0.0
        if w > 0.0:
            cell_center = self.grid_origin + vec3((ix + 0.5) * self.grid_cell_size,
                                                 (iy + 0.5) * self.grid_cell_size,
                                                 (iz + 0.5) * self.grid_cell_size)
            actual_d = (p - cell_center).norm()
            mean_d = self.grid_mean_distance[ix, iy, iz]

            use_probe = True
            if not (mean_d > 1e8 or mean_d <= 1e-6):
                rel = ti.abs(actual_d - mean_d) / mean_d
                if rel > 0.20:
                    use_probe = False

            if use_probe:
                result_color = self.irradiance_grid[ix, iy, iz] * w
                result_weight = w

        return probe_return(color=result_color, weight=result_weight)

    @ti.func
    def sample_irradiance_grid(self, p: vec3, world: ti.template(), fallback_id: int, normal: vec3) -> vec3:
        """Sample the irradiance grid.

        When `self.interpolate_grid_sampling` is True: trilinear interpolation over 8 probes.
        When False: nearest-probe sampling (baseline).
        """
        # Map world position to grid-space continuous coordinates
        local = p - self.grid_origin
        fx = local[0] / self.grid_cell_size
        fy = local[1] / self.grid_cell_size
        fz = local[2] / self.grid_cell_size

        nx = self.grid_res[0]
        ny = self.grid_res[1]
        nz = self.grid_res[2]

        color = vec3(0.0)

        if self.interpolate_grid_sampling:
            # integer base indices (cell corner)
            ix0f = ti.floor(fx)
            iy0f = ti.floor(fy)
            iz0f = ti.floor(fz)
            ix0 = int(ix0f)
            iy0 = int(iy0f)
            iz0 = int(iz0f)

            # fractional offset inside the cell
            tx = fx - ix0f
            ty = fy - iy0f
            tz = fz - iz0f

            # Default fallback (outside grid / border)
            if ix0 < 0 or ix0 >= nx - 1 or iy0 < 0 or iy0 >= ny - 1 or iz0 < 0 or iz0 >= nz - 1:
                color = world.materials.albedo[fallback_id]
            else:
                ix1 = ix0 + 1
                iy1 = iy0 + 1
                iz1 = iz0 + 1

                # fetch 8 neighbors
                c000 = self.irradiance_grid[ix0, iy0, iz0]
                c100 = self.irradiance_grid[ix1, iy0, iz0]
                c010 = self.irradiance_grid[ix0, iy1, iz0]
                c110 = self.irradiance_grid[ix1, iy1, iz0]
                c001 = self.irradiance_grid[ix0, iy0, iz1]
                c101 = self.irradiance_grid[ix1, iy0, iz1]
                c011 = self.irradiance_grid[ix0, iy1, iz1]
                c111 = self.irradiance_grid[ix1, iy1, iz1]

                # linear weights (explicit 8-corner blend)
                w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
                w100 = tx * (1.0 - ty) * (1.0 - tz)
                w010 = (1.0 - tx) * ty * (1.0 - tz)
                w110 = tx * ty * (1.0 - tz)
                w001 = (1.0 - tx) * (1.0 - ty) * tz
                w101 = tx * (1.0 - ty) * tz
                w011 = (1.0 - tx) * ty * tz
                w111 = tx * ty * tz

                # For each of the 8 probe corners, check mean-distance mismatch.
                # If a probe's recorded mean distance deviates from the actual distance by >20%,
                # zero its interpolation weight (treat it as occluded).

                ret0 = self._probe_contrib(p, ix0, iy0, iz0, w000)
                ret1 = self._probe_contrib(p, ix1, iy0, iz0, w100)
                ret2 = self._probe_contrib(p, ix0, iy1, iz0, w010)
                ret3 = self._probe_contrib(p, ix1, iy1, iz0, w110)
                ret4 = self._probe_contrib(p, ix0, iy0, iz1, w001)
                ret5 = self._probe_contrib(p, ix1, iy0, iz1, w101)
                ret6 = self._probe_contrib(p, ix0, iy1, iz1, w011)
                ret7 = self._probe_contrib(p, ix1, iy1, iz1, w111)

                sum_w = ret0.weight + ret1.weight + ret2.weight + ret3.weight + ret4.weight + ret5.weight + ret6.weight + ret7.weight

                if sum_w <= 0.0:
                    # all probes invalid -> fallback
                    color = world.materials.albedo[fallback_id]
                else:
                    total_color = ret0.color + ret1.color + ret2.color + ret3.color + ret4.color + ret5.color + ret6.color + ret7.color
                    color = total_color / sum_w
        else:
            # Baseline: nearest-probe sampling
            ix = int(ti.floor(fx + 0.5))
            iy = int(ti.floor(fy + 0.5))
            iz = int(ti.floor(fz + 0.5))
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                color = world.materials.albedo[fallback_id]
            else:
                # nearest probe: perform percentage depth mismatch check (20%)
                mean_d = self.grid_mean_distance[ix, iy, iz]
                cell_center = self.grid_origin + vec3((ix + 0.5) * self.grid_cell_size,
                                                     (iy + 0.5) * self.grid_cell_size,
                                                     (iz + 0.5) * self.grid_cell_size)
                actual_d = (p - cell_center).norm()
                if mean_d > 1e8:
                    color = self.irradiance_grid[ix, iy, iz]
                else:
                    if mean_d <= 1e-6:
                        color = self.irradiance_grid[ix, iy, iz]
                    else:
                        rel = (actual_d - mean_d)
                        if rel < 0.0:
                            rel = -rel
                        rel = rel / mean_d
                        if rel <= 0.20:
                            color = self.irradiance_grid[ix, iy, iz]
                        else:
                            color = world.materials.albedo[fallback_id]

        return color

    # === Path Tracing (Ground-Truth) ===
    @ti.func
    def get_ray_color_pt(self, ray: Ray, world: ti.template(), px: ti.i32, py: ti.i32) -> vec3:
        """经典递归/迭代 Path Tracing，不使用光照网格，用作真值。"""
        attenuation = vec3(1.0)
        current_ray = ray
        color = vec3(0.0)
        for _ in range(self.max_ray_depth):
            hit = world.hit_world(current_ray, 0.001, tm.inf)
            if hit.did_hit:
                # store G-buffer info for the pixel
                self.normal_buffer[px, py] = hit.record.normal
                self.depth_buffer[px, py] = hit.record.t
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
                # background: mark depth as large and normal as zero
                self.normal_buffer[px, py] = vec3(0.0, 0.0, 0.0)
                self.depth_buffer[px, py] = 1e9
                break
        return color

    # === 纯网格模式 ===
    @ti.func
    def get_ray_color_grid(self, ray: Ray, world: ti.template(), px: ti.i32, py: ti.i32) -> vec3:
        hit = world.hit_world(ray, 0.001, tm.inf)
        color = vec3(0.0)
        if hit.did_hit:
            # store G-buffer
            self.normal_buffer[px, py] = hit.record.normal
            self.depth_buffer[px, py] = hit.record.t
            # Check if this is a light source
            mat_idx = world.materials.mat_index[hit.record.id]

            # ensure grid_color is always defined for Taichi static analysis
            grid_color = vec3(0.0)
            if mat_idx == world.materials.DIFFUSE_LIGHT:
                # Direct light emission
                color = world.materials.albedo[hit.record.id]
            else:
                # Sample irradiance grid for non-light sources (use surface normal)
                # Perform distance mismatch check to avoid light-leaking from other side of occluders
                local = hit.record.p - self.grid_origin
                fx = local[0] / self.grid_cell_size
                fy = local[1] / self.grid_cell_size
                fz = local[2] / self.grid_cell_size
                ix0 = int(ti.floor(fx))
                iy0 = int(ti.floor(fy))
                iz0 = int(ti.floor(fz))

                if ix0 < 0 or ix0 >= self.grid_res[0] or iy0 < 0 or iy0 >= self.grid_res[1] or iz0 < 0 or iz0 >= self.grid_res[2]:
                    color = world.materials.albedo[hit.record.id]
                else:
                    # compute distance from surface point to the nearest cell center
                    cell_center = self.grid_origin + vec3((ix0 + 0.5) * self.grid_cell_size,
                                                         (iy0 + 0.5) * self.grid_cell_size,
                                                         (iz0 + 0.5) * self.grid_cell_size)
                    actual_dist = (hit.record.p - cell_center).norm()
                    mean_d = self.grid_mean_distance[ix0, iy0, iz0]
                    # if no geometry recorded allow grid; otherwise use 20% relative threshold
                    if mean_d > 1e8:
                        color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                    else:
                        if mean_d <= 1e-6:
                            color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                        else:
                            rel = actual_dist - mean_d
                            if rel < 0.0:
                                rel = -rel
                            rel = rel / mean_d
                            if rel <= 0.20:
                                color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                            else:
                                # mismatch -> likely occluded; fallback to material albedo
                                color = world.materials.albedo[hit.record.id]
        else:
            color = self.get_background_color(ray.direction)
            self.normal_buffer[px, py] = vec3(0.0, 0.0, 0.0)
            self.depth_buffer[px, py] = 1e9
        return color

    # === 混合模式（自适应） ===
    @ti.func
    def get_ray_color_hybrid(self, ray: Ray, world: ti.template(), px: ti.i32, py: ti.i32):
        color = vec3(0.0, 0.0, 0.0)
        hit = world.hit_world(ray, 0.001, tm.inf)

        if hit.did_hit:
            # store G-buffer
            self.normal_buffer[px, py] = hit.record.normal
            self.depth_buffer[px, py] = hit.record.t
            mat_idx = world.materials.mat_index[hit.record.id]
            # Ensure grid_color exists for all branches
            grid_color = vec3(0.0)

            # Handle light sources directly
            if mat_idx == world.materials.DIFFUSE_LIGHT:
                # Direct light emission
                color = world.materials.albedo[hit.record.id]
            else:
                # Sample grid for ambient term or primary color
                # Distance-mismatch check to avoid using grid values coming from a different visible side
                local_g = hit.record.p - self.grid_origin
                gfx = local_g[0] / self.grid_cell_size
                gfy = local_g[1] / self.grid_cell_size
                gfz = local_g[2] / self.grid_cell_size
                gx = int(ti.floor(gfx))
                gy = int(ti.floor(gfy))
                gz = int(ti.floor(gfz))
                if gx < 0 or gx >= self.grid_res[0] or gy < 0 or gy >= self.grid_res[1] or gz < 0 or gz >= self.grid_res[2]:
                    grid_color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                else:
                    cell_center = self.grid_origin + vec3((gx + 0.5) * self.grid_cell_size,
                                                          (gy + 0.5) * self.grid_cell_size,
                                                          (gz + 0.5) * self.grid_cell_size)
                    actual_d = (hit.record.p - cell_center).norm()
                    mean_d = self.grid_mean_distance[gx, gy, gz]
                    if mean_d > 1e8:
                        grid_color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                    else:
                        if mean_d <= 1e-6:
                            grid_color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                        else:
                            rel = actual_d - mean_d
                            if rel < 0.0:
                                rel = -rel
                            rel = rel / mean_d
                            if rel <= 0.20:
                                grid_color = self.sample_irradiance_grid(hit.record.p, world, hit.record.id, hit.record.normal)
                            else:
                                grid_color = vec3(0.0)

                # Direct illumination: cast shadow rays to each emissive entity
                direct = vec3(0.0)
                # iterate over world.entities to find lights
                for li in range(world.entities.shape[0]):
                    lm_idx = world.materials.mat_index[li]
                    if lm_idx == world.materials.DIFFUSE_LIGHT:
                        light_center = world.entities[li].center
                        light_radius = world.entities[li].radius
                        to_light = light_center - hit.record.p
                        dist2 = to_light.dot(to_light)
                        dist = tm.sqrt(dist2)
                        if dist < 1e-6:
                            continue
                        ldir = to_light / dist

                        # shadow test: if any geometry intersects between point and light center
                        shadow_ray = Ray(origin=hit.record.p + ldir * 0.0005, direction=ldir)
                        shadow_hit = world.hit_world(shadow_ray, 0.001, dist - light_radius)
                        if shadow_hit.did_hit:
                            # occluded
                            pass
                        else:
                            # simple lambert + inverse-square approx
                            cos = hit.record.normal.dot(ldir)
                            if cos > 0.0:
                                emission = world.materials.albedo[li]
                                # approximate geometric falloff by inverse-square and scale by light size
                                att = cos / (dist2 + 1e-6)
                                direct += emission * att

                # For lambertian surfaces, prefer direct + indirect composition
                if mat_idx == world.materials.LAMBERT:
                    # Combine direct (sharp shadows/highlights) with indirect (from grid)
                    color = direct + grid_color
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
                                    # distance mismatch check for bounce-hit before using grid
                                    local_b = bounce_hit.record.p - self.grid_origin
                                    bfx = local_b[0] / self.grid_cell_size
                                    bfy = local_b[1] / self.grid_cell_size
                                    bfz = local_b[2] / self.grid_cell_size
                                    bix = int(ti.floor(bfx))
                                    biy = int(ti.floor(bfy))
                                    biz = int(ti.floor(bfz))
                                    use_grid = True
                                    if bix < 0 or bix >= self.grid_res[0] or biy < 0 or biy >= self.grid_res[1] or biz < 0 or biz >= self.grid_res[2]:
                                        use_grid = True
                                    else:
                                        cell_center_b = self.grid_origin + vec3((bix + 0.5) * self.grid_cell_size,
                                                                                (biy + 0.5) * self.grid_cell_size,
                                                                                (biz + 0.5) * self.grid_cell_size)
                                        actual_db = (bounce_hit.record.p - cell_center_b).norm()
                                        mean_db = self.grid_mean_distance[bix, biy, biz]
                                        if mean_db <= 1e8 and abs(actual_db - mean_db) > cfg.DISTANCE_MISMATCH_THRESHOLD * self.grid_cell_size:
                                            # use percentage-based mismatch (20%)
                                            if mean_db <= 1e-6:
                                                use_grid = True
                                            else:
                                                relb = actual_db - mean_db
                                                if relb < 0.0:
                                                    relb = -relb
                                                relb = relb / mean_db
                                                if relb > 0.20:
                                                    use_grid = False

                                    if use_grid:
                                        bounced_color = attenuation * self.sample_irradiance_grid(bounce_hit.record.p, world, bounce_hit.record.id, bounce_hit.record.normal)
                                    else:
                                        bounced_color = attenuation * world.materials.albedo[bounce_hit.record.id]
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
            self.normal_buffer[px, py] = vec3(0.0, 0.0, 0.0)
            self.depth_buffer[px, py] = 1e9

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

    @ti.func
    def make_ray(self, origin: vec3, direction: vec3) -> Ray:
        return Ray(origin=origin, direction=direction)

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
                # variance-guided number of probes per update (per-cell)
                # compute dynamic samples: base * (1 + variance * scale), clamped to MAX_PROBE_SAMPLES
                samples_f = float(self.grid_samples_per_update) * (1.0 + self.irradiance_variance[i, j, k] * cfg.VARIANCE_SAMPLING_SCALE)
                samples = int(ti.min(samples_f, float(cfg.MAX_PROBE_SAMPLES)))
                if samples <= 0:
                    samples = 1

                acc_col = vec3(0.0)
                # accumulate distances for mean_distance update
                sum_dist = 0.0
                hit_count = 0
                for s in range(ti.static(cfg.MAX_PROBE_SAMPLES)):
                    if s >= samples:
                        # skip remaining static iterations
                        continue
                    # Half of BASE_UPDATE_PROBABILITY: explicitly shoot rays toward emissive objects.
                    # Otherwise, keep purely-random probing.
                    d = vec3(0.0)

                    use_light = False
                    if self.enable_light_guided_probes and self.light_count[None] > 0:
                        # desired probability = base_prob/2
                        light_prob = base_prob * 0.5
                        if ti.random(ti.f32) < light_prob:
                            use_light = True

                    if use_light:
                        # pick a random light from the compact list
                        li = int(ti.floor(ti.random(ti.f32) * self.light_count[None]))
                        lc = self.light_centers[li]
                        lr = self.light_radii[li]

                        # sample a target point on the light sphere surface
                        target = vec3(lc[0], lc[1], lc[2]) + lr * utils.random_unit_vector()
                        dir_vec = target - pos
                        dist2 = dir_vec.dot(dir_vec)
                        if dist2 < 1e-12:
                            d = utils.random_unit_vector()
                        else:
                            d = dir_vec / tm.sqrt(dist2)
                    else:
                        # purely random probe direction
                        d = utils.random_unit_vector()

                    r = self.make_ray(pos, d)
                    col = vec3(0.0, 0.0, 0.0)
                    # probe with limited depth
                    cur_ray = r
                    att = vec3(1.0)
                    sample_dist = 1e9
                    for depth in range(ti.static(self.grid_probe_depth)):
                        hit = world.hit_world(cur_ray, 0.001, tm.inf)
                        if hit.did_hit:
                            scatter_ret = world.materials.scatter(cur_ray, hit.record)
                            if scatter_ret.did_scatter:
                                att = att * scatter_ret.attenuation
                                cur_ray = self.make_ray(scatter_ret.scattered.origin + tm.normalize(scatter_ret.scattered.direction) * 0.0002,
                                                       scatter_ret.scattered.direction)
                                # continue probing
                                continue
                            else:
                                # hit emissive or absorbing material
                                midx = world.materials.mat_index[hit.record.id]
                                emitted = scatter_ret.attenuation
                                if midx == world.materials.DIFFUSE_LIGHT:
                                    col = att * emitted * cfg.LIGHT_IMPORTANCE_SCALE
                                else:
                                    col = att * emitted
                                # record distance from probe pos to hit point
                                diff = hit.record.p - pos
                                sample_dist = tm.sqrt(diff.dot(diff))
                                hit_count += 1
                                break
                        else:
                            # environment
                            env = self.get_background_color(cur_ray.direction)
                            col = att * env
                            sample_dist = 1e9
                            break
                    acc_col += col
                    sum_dist += sample_dist

                # average over performed samples
                new_col = acc_col / float(samples)
                # exponential smoothing with alpha
                old = self.irradiance_grid[i, j, k]
                a = self.grid_update_alpha
                self.irradiance_grid[i, j, k] = old * (1.0 - a) + new_col * a
                # update luminance mean & variance (exponential moving)
                lum_new = 0.2126 * new_col[0] + 0.7152 * new_col[1] + 0.0722 * new_col[2]
                old_lum = self.irradiance_mean_lum[i, j, k]
                old_var = self.irradiance_variance[i, j, k]
                delta = lum_new - old_lum
                self.irradiance_mean_lum[i, j, k] = old_lum * (1.0 - a) + lum_new * a
                self.irradiance_variance[i, j, k] = old_var * (1.0 - a) + (delta * delta) * a

                # update mean distance (use average of distances recorded this update)
                avg_dist = 1e9
                if hit_count > 0:
                    avg_dist = sum_dist / float(hit_count)
                oldd = self.grid_mean_distance[i, j, k]
                self.grid_mean_distance[i, j, k] = oldd * (1.0 - a) + avg_dist * a

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

    def blur_update_weights(self):
        # Python wrapper: only run kernel if blur enabled in config
        if not self.gaussian_blur_enabled:
            return
        self._blur_update_weights_kernel()

    @ti.kernel
    def _blur_update_weights_kernel(self):
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

    # --- Lightweight A-SVGF spatial filter (uses normals & depth to preserve edges) ---
    @ti.kernel
    def asvgf_filter(self):
        # small 3x3 bilateral-like filter guided by normal/depth
        for i, j in self.frame:
            c0 = self.frame[i, j]
            n0 = self.normal_buffer[i, j]
            d0 = self.depth_buffer[i, j]
            sum_c = vec3(0.0)
            sum_w = 0.0

            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < self.img_res[0] and 0 <= nj < self.img_res[1]:
                        c1 = self.frame[ni, nj]
                        n1 = self.normal_buffer[ni, nj]
                        d1 = self.depth_buffer[ni, nj]

                        # spatial weight (based on Manhattan distance)
                        dist2 = float(di * di + dj * dj)
                        spatial_w = tm.exp(-dist2 * 0.5)

                        # normal alignment weight (preserve edges)
                        n_dot = n0.dot(n1)
                        if n_dot < 0.0:
                            n_dot = 0.0

                        # depth similarity weight
                        depth_diff = d0 - d1
                        depth_w = tm.exp(- (depth_diff * depth_diff) * 50.0)

                        # color similarity (luminance) weight
                        lum0 = 0.2126 * c0[0] + 0.7152 * c0[1] + 0.0722 * c0[2]
                        lum1 = 0.2126 * c1[0] + 0.7152 * c1[1] + 0.0722 * c1[2]
                        col_diff = lum0 - lum1
                        color_w = tm.exp(- (col_diff * col_diff) * 200.0)

                        w = spatial_w * n_dot * depth_w * color_w
                        sum_c += c1 * w
                        sum_w += w

            if sum_w > 0.0:
                self.denoised_frame[i, j] = sum_c / sum_w
            else:
                self.denoised_frame[i, j] = c0

        # copy back
        for i, j in self.frame:
            self.frame[i, j] = self.denoised_frame[i, j]

    @ti.kernel
    def compute_adaptive_weights(self, threshold: ti.f32, multiplier: ti.f32, max_mul: ti.f32):
        # Simple local contrast-based adaptive weight map
        for i, j in self.frame:
            c0 = self.frame[i, j]
            lum0 = 0.2126 * c0[0] + 0.7152 * c0[1] + 0.0722 * c0[2]
            max_l = lum0
            min_l = lum0
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < self.img_res[0] and 0 <= nj < self.img_res[1]:
                        cn = self.frame[ni, nj]
                        l = 0.2126 * cn[0] + 0.7152 * cn[1] + 0.0722 * cn[2]
                        if l > max_l:
                            max_l = l
                        if l < min_l:
                            min_l = l
            contrast = max_l - min_l
            # initialize weight then increase when contrast exceeds threshold
            w = 1.0
            if contrast > threshold:
                w = w + multiplier
            # clamp
            if w > max_mul:
                w = max_mul
            self.adaptive_weight_map[i, j] = w

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

    def set_light_sources(self, spheres: ti.template(), materials_list: ti.template()):
        """
        Build a compact list of emissive scene objects (Python-side).
        `spheres` and `materials_list` are the same Python lists used to construct the World.
        This populates `light_centers`, `light_radii` and `light_count` fields so
        that Taichi kernels can perform Next-Event Estimation (NEE).
        """
        count = 0
        maxl = int(self.max_light_sources)
        for i, s in enumerate(spheres):
            if count >= maxl:
                break
            m = materials_list[i]
            # Detect emissive materials by comparing material index
            if hasattr(m, 'index') and m.index == mat.Materials.DIFFUSE_LIGHT:
                # store center and radius (as floats)
                c = s.center
                self.light_centers[count] = [float(c[0]), float(c[1]), float(c[2])]
                self.light_radii[count] = float(s.radius)
                count += 1

        # write back the count
        self.light_count[None] = count