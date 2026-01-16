"""
Motion-Compensated Temporal Filtering Implementation

This module implements a temporal filtering system with motion compensation
to reduce temporal artifacts and improve stability in dynamic scenes.
"""

import taichi as ti
import taichi.math as tm
import numpy as np
import utils
import experiment_config as cfg

vec3 = ti.types.vector(3, float)


@ti.data_oriented
class MotionCompensatedTemporalFilter:
    """Temporal filter with motion compensation for dynamic scenes"""

    def __init__(self, img_res: tuple):
        """
        Initialize motion-compensated temporal filter

        Args:
            img_res: Image resolution (width, height)
        """
        self.img_res = img_res
        self.width, self.height = img_res

        # Current frame buffers
        self.current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=img_res)
        self.current_normal = ti.Vector.field(n=3, dtype=ti.f32, shape=img_res)
        self.current_depth = ti.field(dtype=ti.f32, shape=img_res)

        # Previous frame buffers
        self.prev_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=img_res)
        self.prev_normal = ti.Vector.field(n=3, dtype=ti.f32, shape=img_res)
        self.prev_depth = ti.field(dtype=ti.f32, shape=img_res)

        # Accumulated history
        self.accum_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=img_res)
        self.history_weight = ti.field(dtype=ti.f32, shape=img_res)

        # Motion vectors (estimated from G-buffer)
        self.motion_x = ti.field(dtype=ti.f32, shape=img_res)
        self.motion_y = ti.field(dtype=ti.f32, shape=img_res)

        # Confidence map (how reliable is the temporal accumulation)
        self.confidence = ti.field(dtype=ti.f32, shape=img_res)

        # Initialize
        self.current_frame.fill(0.0)
        self.current_normal.fill(0.0)
        self.current_depth.fill(1e9)
        self.prev_frame.fill(0.0)
        self.prev_normal.fill(0.0)
        self.prev_depth.fill(1e9)
        self.accum_frame.fill(0.0)
        self.history_weight.fill(0.0)
        self.motion_x.fill(0.0)
        self.motion_y.fill(0.0)
        self.confidence.fill(1.0)

        # Filter parameters
        self.spatial_sigma = 1.5      # Spatial kernel sigma (pixels)
        self.temporal_sigma = 3.0     # Temporal similarity sigma
        self.alpha_static = 0.05      # Accumulation factor for static regions
        self.alpha_dynamic = 0.50      # Accumulation factor for dynamic regions
        self.max_history = 20.0       # Maximum history weight

    @ti.func
    def estimate_motion_vector(
        self,
        x: ti.i32,
        y: ti.i32
    ) -> tuple:
        """
        Estimate 2D motion vector from G-buffer (depth and normal changes)

        Args:
            x, y: Pixel coordinates

        Returns:
            (motion_x, motion_y): Estimated motion in screen space
        """
        # Depth change
        depth_curr = self.current_depth[x, y]
        depth_prev = self.prev_depth[x, y]

        # Normal change
        normal_curr = self.current_normal[x, y]
        normal_prev = self.prev_normal[x, y]

        # Motion is zero if depth/normal don't exist
        if depth_curr > 1e8 or depth_prev > 1e8:
            return 0.0, 0.0

        # Check local region for better motion estimate (3x3 window)
        best_dx = 0.0
        best_dy = 0.0
        best_score = -1e9

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    depth_prev_n = self.prev_depth[nx, ny]
                    normal_prev_n = self.prev_normal[nx, ny]

                    if depth_prev_n > 1e8:
                        continue

                    # Score: depth similarity + normal alignment
                    depth_sim = 1.0 / (tm.abs(depth_curr - depth_prev_n) + 1e-6)
                    normal_sim = 1.0

                    if normal_curr.norm() > 1e-6 and normal_prev_n.norm() > 1e-6:
                        n_curr = normal_curr.normalized()
                        n_prev = normal_prev_n.normalized()
                        ndot = tm.max(0.0, n_curr.dot(n_prev))
                        normal_sim = ndot

                    score = depth_sim * 0.7 + normal_sim * 0.3

                    if score > best_score:
                        best_score = score
                        best_dx = float(dx)
                        best_dy = float(dy)

        # Clamp motion to reasonable range
        max_motion = 5.0  # pixels
        best_dx = tm.clamp(best_dx, -max_motion, max_motion)
        best_dy = tm.clamp(best_dy, -max_motion, max_motion)

        return best_dx, best_dy

    @ti.func
    def compute_temporal_weight(
        self,
        x: ti.i32,
        y: ti.i32,
        prev_x: ti.f32,
        prev_y: ti.f32
    ) -> ti.f32:
        """
        Compute temporal similarity weight with motion compensation

        Args:
            x, y: Current pixel coordinates
            prev_x, prev_y: Motion-compensated previous coordinates

        Returns:
            Temporal weight [0, 1]
        """
        # Sample with bilinear interpolation from prev frame
        ix = ti.floor(prev_x)
        iy = ti.floor(prev_y)
        fx = prev_x - ix
        fy = prev_y - iy

        if ix < 0 or ix >= self.width - 1 or iy < 0 or iy >= self.height - 1:
            return 0.0

        # Bilinear interpolation
        c00 = self.prev_frame[int(ix), int(iy)]
        c10 = self.prev_frame[int(ix) + 1, int(iy)]
        c01 = self.prev_frame[int(ix), int(iy) + 1]
        c11 = self.prev_frame[int(ix) + 1, int(iy) + 1]

        prev_color = (1.0 - fx) * (1.0 - fy) * c00 + \
                     fx * (1.0 - fy) * c10 + \
                     (1.0 - fx) * fy * c01 + \
                     fx * fy * c11

        curr_color = self.current_frame[x, y]

        # Color difference (L2 norm)
        color_diff = (curr_color - prev_color).norm()

        # Temporal weight (bilateral: color + motion)
        motion_mag = tm.sqrt((prev_x - float(x))**2 + (prev_y - float(y))**2)

        temporal_weight = tm.exp(-color_diff**2 / (2.0 * self.temporal_sigma**2))
        motion_weight = tm.exp(-motion_mag**2 / (2.0 * self.temporal_sigma**2))

        return temporal_weight * motion_weight

    @ti.func
    def compute_spatial_weight(
        self,
        x0: ti.i32,
        y0: ti.i32,
        x1: ti.i32,
        y1: ti.i32
    ) -> ti.f32:
        """
        Compute spatial bilateral weight (color + normal + depth)

        Args:
            x0, y0: Center pixel
            x1, y1: Neighbor pixel

        Returns:
            Spatial weight [0, 1]
        """
        if not (0 <= x1 < self.width and 0 <= y1 < self.height):
            return 0.0

        # Color similarity
        c0 = self.current_frame[x0, y0]
        c1 = self.current_frame[x1, y1]
        color_diff = (c0 - c1).norm()
        color_weight = tm.exp(-color_diff**2 / (2.0 * self.spatial_sigma**2))

        # Normal similarity
        n0 = self.current_normal[x0, y0]
        n1 = self.current_normal[x1, y1]

        normal_weight = 1.0
        if n0.norm() > 1e-6 and n1.norm() > 1e-6:
            n0_norm = n0.normalized()
            n1_norm = n1.normalized()
            ndot = tm.max(0.0, n0_norm.dot(n1_norm))
            normal_weight = tm.pow(ndot, 2.0)  # Power for sharper falloff

        # Depth similarity
        d0 = self.current_depth[x0, y0]
        d1 = self.current_depth[x1, y1]

        depth_weight = 1.0
        if d0 < 1e8 and d1 < 1e8:
            rel_depth = tm.abs(d0 - d1) / (0.5 * (d0 + d1) + 1e-6)
            depth_weight = tm.exp(-rel_depth**2 / (2.0 * self.spatial_sigma**2))

        return color_weight * normal_weight * depth_weight

    @ti.kernel
    def update_motion_vectors(self):
        """Estimate motion vectors from current and previous G-buffer"""
        for x, y in self.current_frame:
            dx, dy = self.estimate_motion_vector(x, y)
            self.motion_x[x, y] = dx
            self.motion_y[x, y] = dy

    @ti.kernel
    def apply_motion_compensated_filter(self):
        """
        Apply motion-compensated temporal filtering

        This implements a bilateral filter that combines:
        1. Temporal accumulation with motion compensation
        2. Spatial bilateral filtering for denoising
        3. Adaptive history based on motion confidence
        """
        for x, y in self.current_frame:
            # Motion-compensated previous position
            prev_x = float(x) + self.motion_x[x, y]
            prev_y = float(y) + self.motion_y[x, y]

            # Compute temporal weight with motion compensation
            temporal_w = self.compute_temporal_weight(x, y, prev_x, prev_y)

            # Compute adaptive alpha based on motion and temporal weight
            # More motion + lower temporal weight -> higher alpha (less history)
            motion_mag = tm.sqrt(self.motion_x[x, y]**2 + self.motion_y[x, y]**2)
            is_moving = motion_mag > 1.0 or temporal_w < 0.5

            alpha = self.alpha_dynamic if is_moving else self.alpha_static

            # Spatial bilateral filtering (3x3 window)
            sum_color = vec3(0.0)
            sum_weight = 0.0

            radius = 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        spatial_w = self.compute_spatial_weight(x, y, nx, ny)

                        # Combine spatial and temporal weights
                        combined_w = spatial_w * (1.0 + temporal_w)

                        sum_color += self.current_frame[nx, ny] * combined_w
                        sum_weight += combined_w

            # Spatially filtered current frame
            if sum_weight > 1e-6:
                filtered_curr = sum_color / sum_weight
            else:
                filtered_curr = self.current_frame[x, y]

            # Temporal accumulation with motion compensation
            # Sample from previous frame with motion compensation
            prev_x_int = int(ti.floor(prev_x))
            prev_y_int = int(ti.floor(prev_y))

            if 0 <= prev_x_int < self.width and 0 <= prev_y_int < self.height:
                prev_color = self.accum_frame[prev_x_int, prev_y_int]

                # Adaptive history weight based on confidence
                # High temporal weight -> more history, Low temporal weight -> less history
                history_weight = self.history_weight[x, y]
                new_history_weight = history_weight * temporal_w + 1.0
                new_history_weight = tm.min(new_history_weight, self.max_history)

                # Temporal EMA
                effective_alpha = alpha / tm.sqrt(new_history_weight + 1.0)
                effective_alpha = tm.clamp(effective_alpha, 0.02, 0.50)

                accumulated_color = prev_color * (1.0 - effective_alpha) + filtered_curr * effective_alpha

                # Store results
                self.accum_frame[x, y] = accumulated_color
                self.history_weight[x, y] = new_history_weight
                self.confidence[x, y] = temporal_w
            else:
                # Motion-compensated position is out of bounds, reset
                self.accum_frame[x, y] = filtered_curr
                self.history_weight[x, y] = 1.0
                self.confidence[x, y] = 0.0

    @ti.kernel
    def update_prev_buffers(self):
        """Copy current buffers to previous for next frame"""
        for x, y in self.current_frame:
            self.prev_frame[x, y] = self.current_frame[x, y]
            self.prev_normal[x, y] = self.current_normal[x, y]
            self.prev_depth[x, y] = self.current_depth[x, y]

    @ti.kernel
    def reset(self):
        """Reset all buffers (use after scene changes or large movements)"""
        for x, y in self.current_frame:
            self.accum_frame[x, y] = self.current_frame[x, y]
            self.history_weight[x, y] = 1.0
            self.confidence[x, y] = 1.0
            self.motion_x[x, y] = 0.0
            self.motion_y[x, y] = 0.0

    def process_frame(
        self,
        current_linear: ti.Vector.field,
        current_normal: ti.Vector.field,
        current_depth: ti.field
    ) -> ti.Vector.field:
        """
        Process a frame through motion-compensated temporal filter

        Args:
            current_linear: Current frame color (linear space)
            current_normal: Current frame normals
            current_depth: Current frame depth

        Returns:
            Filtered frame
        """
        # Copy current buffers
        for x, y in ti.ndrange(self.width, self.height):
            self.current_frame[x, y] = current_linear[x, y]
            self.current_normal[x, y] = current_normal[x, y]
            self.current_depth[x, y] = current_depth[x, y]

        # Estimate motion vectors
        self.update_motion_vectors()

        # Apply filter
        self.apply_motion_compensated_filter()

        # Update previous buffers
        self.update_prev_buffers()

        return self.accum_frame

    def get_confidence_map(self) -> ti.field:
        """Get temporal confidence map (for visualization)"""
        return self.confidence

    def get_motion_map(self) -> tuple:
        """Get motion vector map (for visualization)"""
        return self.motion_x, self.motion_y

    def set_parameters(
        self,
        spatial_sigma: float = None,
        temporal_sigma: float = None,
        alpha_static: float = None,
        alpha_dynamic: float = None,
        max_history: float = None
    ):
        """Update filter parameters"""
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
        if temporal_sigma is not None:
            self.temporal_sigma = temporal_sigma
        if alpha_static is not None:
            self.alpha_static = alpha_static
        if alpha_dynamic is not None:
            self.alpha_dynamic = alpha_dynamic
        if max_history is not None:
            self.max_history = max_history
