import taichi as ti

vec3 = ti.types.vector(3, float)
@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t: ti.f32) -> ti.math.vec3:
        return self.origin + t * self.direction