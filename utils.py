import taichi as ti
import taichi.math as tm
import random

vec3 = ti.types.vector(3, float)


@ti.func
def random_on_hemisphere(normal: vec3) -> vec3:
    vec = random_unit_vector()
    if vec.dot(normal) < 0.0:
        vec = -vec
    return vec


@ti.func
def near_zero(v: vec3) -> bool:
    s = 1e-8
    return (v[0] < s) and (v[1] < s) and (v[2] < s)


def length(v: vec3) -> float:
    return tm.sqrt(v.dot(v))


def normalize(v: vec3) -> vec3:
    return v / length(v)


@ti.func
def random_unit_vector() -> vec3:
    p = vec3(0, 0, 0)
    while True:
        p = 2.0 * vec3(ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)) - vec3(1, 1, 1)
        if p.dot(p) <= 1.0:
            p = tm.normalize(p)
            break
    return p


@ti.func
def random_in_unit_disc() -> vec3:
    p = vec3(0, 0, 0)
    while True:
        p = 2.0 * vec3(ti.random(ti.f32), ti.random(ti.f32), 0) - vec3(1, 1, 0)
        if p.norm_sqr() <= 1.0:
            break
    return p


@ti.func
def linear_to_gamma(x: float) -> float:
    return tm.pow(x, 1 / 2.2) if x > 0 else 0


@ti.func
def reflectance(cosine: float, ref_idx: float) -> float:
    # Schlick's approximation
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * tm.pow(1 - cosine, 5)


def rand_vec(min, max):
    return vec3(random.random() * (max - min) + min, random.random() * (max - min) + min,
                random.random() * (max - min) + min)

@ti.func
def reflect(vec: vec3, norm: vec3) -> vec3:
    return vec - 2 * vec.dot(norm) * norm

@ti.func
def refract(uv: vec3, n: vec3, etai_over_etat: float) -> vec3:
    cos_theta = min((-uv).dot(n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -tm.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * n
    return r_out_perp + r_out_parallel

@ti.func
def linear_to_gamma_vec3(x: vec3) -> vec3:
    return vec3(linear_to_gamma(x[0]), linear_to_gamma(x[1]), linear_to_gamma(x[2]))


@ti.func
def gamma_to_linear(x: float) -> float:
    return tm.pow(x, 2.2) if x > 0 else 0


@ti.func
def gamma_to_linear_vec3(x: vec3) -> vec3:
    return vec3(gamma_to_linear(x[0]), gamma_to_linear(x[1]), gamma_to_linear(x[2]))
