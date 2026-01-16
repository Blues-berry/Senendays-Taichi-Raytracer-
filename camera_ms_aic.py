"""
Multi-Scale Adaptive Irradiance Caching (MS-AIC) Implementation

This module implements a multi-resolution irradiance caching system with
adaptive level selection based on distance and dynamic regions.
"""

import taichi as ti
import taichi.math as tm
import math
import utils
import experiment_config as cfg
from ray import Ray

vec3 = ti.types.vector(3, float)

probe_return = ti.types.struct(color=vec3, weight=ti.f32)


@ti.data_oriented
class MultiScaleGrid:
    """Multi-resolution irradiance caching system"""

    def __init__(self, grid_resolutions: list, grid_origin: vec3, grid_cell_size: float):
        """
        Initialize multi-scale grid system

        Args:
            grid_resolutions: List of (nx, ny, nz) for each level, e.g., [(16,16,16), (32,32,32), (64,64,64)]
            grid_origin: World-space origin of grid
            grid_cell_size: Size of grid cell for the highest resolution level
        """
        self.num_levels = len(grid_resolutions)
        self.grid_resolutions = grid_resolutions
        self.grid_origin = grid_origin
        self.base_cell_size = grid_cell_size

        # Per-level grids
        self.irradiance_grids = []
        self.normal_grids = []
        self.mean_distance_grids = []
        self.update_weight_grids = []

        for level_idx, res in enumerate(grid_resolutions):
            # Calculate cell size for this level (each level is 2x coarser)
            level_cell_size = grid_cell_size * (2 ** (self.num_levels - 1 - level_idx))

            # Irradiance grid
            irrad = ti.Vector.field(n=3, dtype=ti.f32, shape=res)
            irrad.fill(0.0)
            self.irradiance_grids.append(irrad)

            # Normal grid (for anti-leaking)
            normal = ti.Vector.field(n=3, dtype=ti.f32, shape=res)
            normal.fill(0.0)
            self.normal_grids.append(normal)

            # Mean distance grid (for depth-based occlusion)
            mean_dist = ti.field(dtype=ti.f32, shape=res)
            mean_dist.fill(1e9)  # Sentinel: no geometry recorded yet
            self.mean_distance_grids.append(mean_dist)

            # Update weight grid (adaptive update probability)
            update_weight = ti.field(dtype=ti.f32, shape=res)
            update_weight.fill(1.0)
            self.update_weight_grids.append(update_weight)

        # Selection map (stores which level to use for each pixel)
        self.img_res = (1200, 675)  # Will be set from camera
        self.level_selection_map = ti.field(dtype=ti.i32, shape=self.img_res)
        self.level_selection_map.fill(2)  # Default: use highest resolution

    @ti.func
    def select_grid_level(self, world_pos: vec3, camera_pos: vec3, is_dynamic: bool) -> ti.i32:
        """
        Select appropriate grid level based on world position and scene dynamics

        Args:
            world_pos: World-space position to sample
            camera_pos: Camera position
            is_dynamic: Whether the region is detected as dynamic

        Returns:
            Selected level index (0 = coarsest, num_levels-1 = finest)
        """
        # Distance from camera
        dist_to_camera = (world_pos - camera_pos).norm()

        # Distance thresholds for level selection (tunable)
        # Far regions use coarse level, near regions use fine level
        dist_thresholds = [50.0, 25.0]  # For 3 levels: >50 use L0, 25-50 use L1, <25 use L2

        # Base level from distance
        selected_level = self.num_levels - 1  # Default to finest

        for i in range(self.num_levels - 1):
            if dist_to_camera > dist_thresholds[i]:
                selected_level = i
                break

        # Upgrade level for dynamic regions (use higher resolution for moving objects)
        if is_dynamic:
            selected_level = ti.min(selected_level + 1, self.num_levels - 1)

        return selected_level

    @ti.func
    def _probe_contrib_multiscale(
        self,
        level_idx: ti.i32,
        p: vec3,
        ix: ti.i32,
        iy: ti.i32,
        iz: ti.i32,
        w: ti.f32,
        query_normal: vec3,
        cell_size: float
    ) -> probe_return:
        """Compute probe contribution for a specific grid level"""
        result_color = vec3(0.0)
        result_weight = 0.0

        if w > 0.0:
            res = self.grid_resolutions[level_idx]
            cell_center = self.grid_origin + vec3(
                (ix + 0.5) * cell_size,
                (iy + 0.5) * cell_size,
                (iz + 0.5) * cell_size
            )
            actual_d = (p - cell_center).norm()
            mean_d = self.mean_distance_grids[level_idx][ix, iy, iz]

            # Distance mismatch check (20% threshold)
            use_probe = True
            if not (mean_d > 1e8 or mean_d <= 1e-6):
                rel = ti.abs(actual_d - mean_d) / mean_d
                if rel > 0.20:
                    use_probe = False

            # Normal-weighted interpolation (anti-leaking)
            if ti.static(cfg.NORMAL_WEIGHTING_ENABLED) and use_probe:
                stored_normal = self.normal_grids[level_idx][ix, iy, iz]
                if stored_normal.norm() > 1e-6 and query_normal.norm() > 1e-6:
                    n_stored = stored_normal.normalized()
                    qn = query_normal.normalized()
                    n_dot = tm.max(0.0, qn.dot(n_stored))
                    normal_weight = tm.pow(n_dot, cfg.NORMAL_POWER)
                    w *= normal_weight
                else:
                    use_probe = False

            # Distance-based weighting
            if ti.static(cfg.DISTANCE_WEIGHTING_ENABLED) and use_probe:
                dist_sq = actual_d * actual_d
                dist_weight = 1.0 / (dist_sq + 1e-4)
                cutoff = cfg.DISTANCE_CUTOFF_MULTIPLIER * cell_size
                if actual_d > cutoff:
                    dist_weight = 0.0
                w *= dist_weight

            if use_probe:
                result_color = self.irradiance_grids[level_idx][ix, iy, iz] * w
                result_weight = w

        return probe_return(color=result_color, weight=result_weight)

    @ti.func
    def sample_multiscale_grid(
        self,
        p: vec3,
        query_normal: vec3,
        camera_pos: vec3,
        is_dynamic: bool,
        fallback_id: ti.i32,
        world: ti.template()
    ) -> vec3:
        """
        Sample irradiance from multi-scale grid with adaptive level selection

        Args:
            p: World-space position to sample
            query_normal: Surface normal at sample point
            camera_pos: Camera position
            is_dynamic: Whether region is dynamic
            fallback_id: Material ID to use as fallback color
            world: World structure for accessing materials

        Returns:
            Sampled irradiance color
        """
        # Select appropriate level
        level = self.select_grid_level(p, camera_pos, is_dynamic)

        # Get grid parameters for selected level
        res = self.grid_resolutions[level]
        cell_size = self.base_cell_size * (2 ** (self.num_levels - 1 - level))

        # Map world position to grid coordinates
        local = p - self.grid_origin
        fx = local[0] / cell_size
        fy = local[1] / cell_size
        fz = local[2] / cell_size

        # Base cell corner
        ix0f = ti.floor(fx)
        iy0f = ti.floor(fy)
        iz0f = ti.floor(fz)
        ix0 = int(ix0f)
        iy0 = int(iy0f)
        iz0 = int(iz0f)

        # Boundary check
        nx, ny, nz = res[0], res[1], res[2]
        outside = (ix0 < 0 or ix0 >= nx - 1 or iy0 < 0 or iy0 >= ny - 1 or iz0 < 0 or iz0 >= nz - 1)

        if outside:
            return world.materials.albedo[fallback_id]

        # Trilinear interpolation weights
        tx = fx - ix0f
        ty = fy - iy0f
        tz = fz - iz0f

        # Sample 8 corners with per-probe validity and weighting
        total_color = vec3(0.0)
        total_w = 0.0

        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    ix = ix0 + di
                    iy = iy0 + dj
                    iz = iz0 + dk

                    wx = tx if di == 1 else (1.0 - tx)
                    wy = ty if dj == 1 else (1.0 - ty)
                    wz = tz if dk == 1 else (1.0 - tz)
                    w = wx * wy * wz

                    pr = self._probe_contrib_multiscale(
                        level, p, ix, iy, iz, w, query_normal, cell_size
                    )
                    total_color += pr.color
                    total_w += pr.weight

        # Normalize if valid samples found
        if total_w > 1e-8:
            color = total_color / total_w

            # Neighbor clamping (anti-bleeding)
            if ti.static(cfg.NEIGHBOR_CLAMPING_ENABLED):
                max_neighbor_irradiance = vec3(0.0)
                for di in ti.static(range(-1, 2)):
                    for dj in ti.static(range(-1, 2)):
                        for dk in ti.static(range(-1, 2)):
                            ni, nj, nk = ix0 + di, iy0 + dj, iz0 + dk
                            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                                max_neighbor_irradiance = tm.max(
                                    max_neighbor_irradiance,
                                    self.irradiance_grids[level][ni, nj, nk]
                                )
                color = tm.min(color, max_neighbor_irradiance)
        else:
            color = vec3(0.0)

        return color

    @ti.kernel
    def update_all_levels(
        self,
        world: ti.template(),
        base_prob: float,
        camera_pos: vec3
    ):
        """Update all grid levels with adaptive probability scaling"""
        for level_idx in range(self.num_levels):
            res = self.grid_resolutions[level_idx]
            cell_size = self.base_cell_size * (2 ** (self.num_levels - 1 - level_idx))

            for i, j, k in ti.ndrange(res[0], res[1], res[2]):
                # Update probability: scale by level (coarser levels update less frequently)
                # Level 0 (coarse): 0.5x, Level 1: 1.0x, Level 2 (fine): 2.0x
                level_scale = ti.cast(level_idx + 1, float) / float(self.num_levels)
                prob = base_prob * self.update_weight_grids[level_idx][i, j, k] * level_scale

                p = ti.random(ti.f32)
                if p < prob:
                    # Probe from this cell center
                    pos = self.grid_origin + vec3(
                        (i + 0.5) * cell_size,
                        (j + 0.5) * cell_size,
                        (k + 0.5) * cell_size
                    )

                    # Cast probe rays
                    col = vec3(0.0)
                    cur_ray = self.make_ray(pos, utils.random_unit_vector())
                    att = vec3(1.0)
                    sample_dist = 1e9

                    for depth in ti.static(cfg.GRID_PROBE_DEPTH):
                        hit = world.hit_world(cur_ray, 0.001, tm.inf)
                        if hit.did_hit:
                            scatter_ret = world.materials.scatter(cur_ray, hit.record)
                            if scatter_ret.did_scatter:
                                att *= scatter_ret.attenuation
                                cur_ray = self.make_ray(
                                    scatter_ret.scattered.origin +
                                    tm.normalize(scatter_ret.scattered.direction) * 0.0002,
                                    scatter_ret.scattered.direction
                                )
                            else:
                                # Hit emissive material
                                midx = world.materials.mat_index[hit.record.id]
                                emitted = scatter_ret.attenuation
                                if midx == world.materials.DIFFUSE_LIGHT:
                                    col = att * emitted * cfg.LIGHT_IMPORTANCE_SCALE
                                else:
                                    col = att * emitted

                                diff = hit.record.p - pos
                                sample_dist = tm.sqrt(diff.dot(diff))
                                break
                        else:
                            # Environment
                            env = self.get_background_color(cur_ray.direction)
                            col = att * env
                            sample_dist = 1e9
                            break

                    # EMA update
                    old = self.irradiance_grids[level_idx][i, j, k]
                    alpha = cfg.GRID_UPDATE_ALPHA
                    self.irradiance_grids[level_idx][i, j, k] = old * (1.0 - alpha) + col * alpha

                    # Update mean distance
                    old_dist = self.mean_distance_grids[level_idx][i, j, k]
                    if old_dist > 1e8:
                        self.mean_distance_grids[level_idx][i, j, k] = sample_dist
                    else:
                        # EMA for distance
                        self.mean_distance_grids[level_idx][i, j, k] = old_dist * 0.9 + sample_dist * 0.1

    @ti.func
    def make_ray(self, origin: vec3, direction: vec3) -> Ray:
        return Ray(origin=origin, direction=direction)

    @ti.func
    def get_background_color(self, direction: vec3) -> vec3:
        unit_direction = direction.normalized()
        a = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)

    def get_memory_usage_mb(self) -> float:
        """Calculate total memory usage in MB"""
        total_cells = 0
        for res in self.grid_resolutions:
            total_cells += res[0] * res[1] * res[2]

        # Each level has 4 fields: irradiance (3xfloat), normal (3xfloat), distance (1xfloat), weight (1xfloat)
        bytes_per_cell = (3 + 3 + 1 + 1) * 4  # 8 floats * 4 bytes
        total_bytes = total_cells * bytes_per_cell

        return total_bytes / (1024.0 * 1024.0)
