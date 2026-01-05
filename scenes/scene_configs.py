"""
场景配置模块 (Scene Configurations)
包含不同测试场景的定义
"""

import taichi as ti
from hittable import Sphere
import material
import utils
import random

vec3 = ti.types.vector(3, float)


def create_cornell_box_scene():
    """
    标准 Cornell Box 场景
    
    五面墙：左红、右绿、其余白
    顶部强发光面光源
    两球：金属 + 玻璃
    """
    spheres = []
    materials = []

    white = vec3(0.73, 0.73, 0.73)
    left_red = vec3(0.65, 0.05, 0.05)
    right_green = vec3(0.12, 0.45, 0.15)

    # Cornell Box 参数
    half = 2.75
    R = 1000.0  # 大球半径（越接近平面）

    # === 五面墙（没有前墙，便于相机观察） ===
    # 左墙（红）/ 右墙（绿）
    spheres.append(Sphere(center=vec3(-(R + half), 0.0, 0.0), radius=R))
    materials.append(material.Lambert(left_red))
    spheres.append(Sphere(center=vec3((R + half), 0.0, 0.0), radius=R))
    materials.append(material.Lambert(right_green))

    # 地面 / 天花板 / 后墙（白）
    spheres.append(Sphere(center=vec3(0.0, -(R + half), 0.0), radius=R))
    materials.append(material.Lambert(white))
    spheres.append(Sphere(center=vec3(0.0, (R + half), 0.0), radius=R))
    materials.append(material.Lambert(white))
    spheres.append(Sphere(center=vec3(0.0, 0.0, -(R + half)), radius=R))
    materials.append(material.Lambert(white))

    # === 顶部面光源（用一个较大的发光球近似平面光源） ===
    spheres.append(Sphere(center=vec3(0.0, half - 0.15, -1.0), radius=0.85))
    materials.append(material.DiffuseLight(vec3(25.0, 25.0, 25.0)))

    # === 盒内两球 ===
    # 高反射金属球（低 fuzz）
    spheres.append(Sphere(center=vec3(-0.85, -half + 0.70, -1.65), radius=0.70))
    materials.append(material.Metal(vec3(0.93, 0.93, 0.93), 0.01))

    # 折射玻璃球
    spheres.append(Sphere(center=vec3(0.95, -half + 0.70, -0.95), radius=0.70))
    materials.append(material.Dielectric(1.5))

    # 相机参数
    cam_params = dict(
        lookfrom=vec3(0.0, 0.0, 8.5),
        lookat=vec3(0.0, -0.2, -1.3),
        vup=vec3(0, 1, 0),
        vfov=40.0,
        defocus_angle=0.0,
        focus_dist=8.5,
        scene_mode='cornell_box'
    )

    return spheres, materials, cam_params


def create_two_room_scene():
    """
    两室一门场景 (Two-Room-One-Door Scene)
    
    用于测试复杂遮挡和"窄缝漏光"场景
    - 房间A：有光源
    - 房间B：仅通过门缝接受间接光
    - 测试：深度遮挡检测能否阻止光线"穿墙"
    """
    spheres = []
    materials = []

    white = vec3(0.73, 0.73, 0.73)
    left_red = vec3(0.65, 0.05, 0.05)
    right_green = vec3(0.12, 0.45, 0.15)

    # 房间尺寸
    room_width = 4.0
    room_depth = 4.0
    room_height = 3.0
    wall_thickness = 0.3
    R = 1000.0

    # === 房间A (Room A, 有光源，z < 0) ===
    # 地面
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, -R, -room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 天花板
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, room_height + R, -room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 左墙（红）
    spheres.append(Sphere(
        center=vec3(-room_width - R, room_height/2, -room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(left_red))
    
    # 右墙（绿）
    spheres.append(Sphere(
        center=vec3(0 - R, room_height/2, -room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(right_green))
    
    # 后墙
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, room_height/2, -room_depth - R),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 前墙（带门缝）
    # 使用多个大球组合形成带缝的墙
    wall_left_x = -room_width - R
    wall_right_x = 0 - R
    wall_y_center = room_height / 2
    wall_z = 0
    
    # 墙的左边部分（门左）
    spheres.append(Sphere(
        center=vec3(wall_left_x, wall_y_center, wall_z - R),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 墙的右边部分（门右）
    spheres.append(Sphere(
        center=vec3(wall_right_x, wall_y_center, wall_z - R),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 墙的顶部（门上）
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, room_height + R, wall_z - R),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 墙的底部（门下）
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, -R, wall_z - R),
        radius=R
    ))
    materials.append(material.Lambert(white))

    # === 房间B (Room B, 无光源，z > 0) ===
    # 地面
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, -R, room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 天花板
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, room_height + R, room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 左墙（红）
    spheres.append(Sphere(
        center=vec3(-room_width - R, room_height/2, room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(left_red))
    
    # 右墙（绿）
    spheres.append(Sphere(
        center=vec3(0 - R, room_height/2, room_depth/2),
        radius=R
    ))
    materials.append(material.Lambert(right_green))
    
    # 后墙
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, room_height/2, room_depth + R),
        radius=R
    ))
    materials.append(material.Lambert(white))
    
    # 前墙
    spheres.append(Sphere(
        center=vec3(-room_width/2 - R, room_height/2, room_depth*2 - R),
        radius=R
    ))
    materials.append(material.Lambert(white))

    # === 光源（仅在房间A） ===
    spheres.append(Sphere(
        center=vec3(-room_width/2, room_height - 0.3, -room_depth/2),
        radius=0.4
    ))
    materials.append(material.DiffuseLight(vec3(20.0, 20.0, 20.0)))

    # === 场景内物体 ===
    # 房间A内的小球
    spheres.append(Sphere(
        center=vec3(-room_width/2 + 1.0, 0.5, -room_depth/2 + 1.0),
        radius=0.3
    ))
    materials.append(material.Metal(vec3(0.8, 0.8, 0.9), 0.1))
    
    # 房间B内的小球（用于观察间接光效果）
    spheres.append(Sphere(
        center=vec3(-room_width/2 + 1.0, 0.5, room_depth/2 - 1.0),
        radius=0.3
    ))
    materials.append(material.Metal(vec3(0.9, 0.7, 0.5), 0.1))

    # 相机参数（从房间B朝向房间A，观察门缝效果）
    cam_params = dict(
        lookfrom=vec3(-room_width/2, room_height/2, room_depth + 2),
        lookat=vec3(-room_width/2, room_height/2, -room_depth/2),
        vup=vec3(0, 1, 0),
        vfov=60.0,
        defocus_angle=0.0,
        focus_dist=10.0,
        scene_mode='two_room'
    )

    return spheres, materials, cam_params


def create_night_scene():
    """
    夜间场景
    
    黑色背景 + 多彩高亮点光源 + 高反射金属球
    """
    spheres = []
    materials = []

    # 暗色地面
    spheres.append(Sphere(center=vec3(0, -1000, 0), radius=1000))
    materials.append(material.Lambert(vec3(0.08, 0.08, 0.09)))

    # 5 个不同颜色的高亮度点光源
    light_positions = [
        vec3(-6, 4, -2),
        vec3(-2, 3, 2),
        vec3(2, 3.5, -1),
        vec3(6, 4, 2),
        vec3(0, 5, 6),
    ]
    light_colors = [
        vec3(15, 9, 4),   # 暖橘
        vec3(4, 10, 15),  # 冰蓝
        vec3(14, 4, 10),  # 洋红
        vec3(6, 15, 6),   # 绿
        vec3(12, 12, 16), # 冷白偏蓝
    ]
    
    # 固定随机种子确保可重现
    random.seed(42)
    
    for p, c in zip(light_positions, light_colors):
        spheres.append(Sphere(center=p, radius=0.35))
        materials.append(material.DiffuseLight(c))

    # 随机散布高反射金属球
    for _ in range(25):
        x = random.uniform(-7.0, 7.0)
        z = random.uniform(-7.0, 7.0)
        r = random.uniform(0.25, 0.6)
        y = r
        spheres.append(Sphere(center=vec3(x, y, z), radius=r))
        base = utils.rand_vec(0.7, 1.0)
        materials.append(material.Metal(base, random.uniform(0.0, 0.08)))

    cam_params = dict(
        lookfrom=vec3(0, 3, 12),
        lookat=vec3(0, 1, 0),
        vup=vec3(0, 1, 0),
        vfov=35.0,
        defocus_angle=0.0,
        focus_dist=12.0,
        scene_mode='night_scene'
    )

    return spheres, materials, cam_params


def create_random_scene():
    """
    随机小球场景（默认场景）
    
    地面网格 + 随机材质小球 + 三个大球 + 顶部光源
    """
    spheres = []
    materials = []

    # 地面
    floor = Sphere(center=vec3(0, -1000, -1), radius=1000)
    floor_mat = material.Lambert(vec3(0.5, 0.5, 0.5))
    spheres.append(floor)
    materials.append(floor_mat)

    # 小球网格（固定种子确保可重现）
    random.seed(42)
    
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = vec3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
            if (center - vec3(4, 0.2, 0)).norm() > 0.9:
                if choose_mat < 0.8:
                    spheres.append(Sphere(center=center, radius=0.2))
                    materials.append(material.Lambert(utils.rand_vec(0, 1) * utils.rand_vec(0, 1)))
                elif choose_mat < 0.95:
                    spheres.append(Sphere(center=center, radius=0.2))
                    materials.append(material.Metal(utils.rand_vec(0.5, 1), 0.5 * random.random()))
                else:
                    spheres.append(Sphere(center=center, radius=0.2))
                    materials.append(material.Dielectric(1.5))

    # 三个大球
    sph_1 = Sphere(center=vec3(0, 1, 0), radius=1)
    spheres.append(sph_1)
    materials.append(material.Dielectric(1.5))

    sph_2 = Sphere(center=vec3(-4, 1, 0), radius=1)
    spheres.append(sph_2)
    materials.append(material.Lambert(vec3(0.4, 0.2, 0.1)))

    sph_3 = Sphere(center=vec3(4, 1, 0), radius=1)
    spheres.append(sph_3)
    materials.append(material.Metal(vec3(0.7, 0.6, 0.5), 0.0))

    # 顶部光源
    top_light = Sphere(center=vec3(0, 5, 0), radius=0.5)
    spheres.append(top_light)
    materials.append(material.DiffuseLight(vec3(20, 20, 20)))

    cam_params = dict(
        lookfrom=vec3(13, 2, 3),
        lookat=vec3(0, 0, 0),
        vup=vec3(0, 1, 0),
        vfov=20.0,
        defocus_angle=0.6,
        focus_dist=10.0,
        scene_mode='random'
    )

    return spheres, materials, cam_params


# 场景字典
SCENES = {
    'cornell_box': create_cornell_box_scene,
    'two_room': create_two_room_scene,
    'night_scene': create_night_scene,
    'random': create_random_scene,
}


def get_scene(scene_name='random'):
    """
    获取指定场景的配置
    
    Args:
        scene_name: 场景名称 ('cornell_box', 'two_room', 'night_scene', 'random')
    
    Returns:
        (spheres, materials, cam_params)
    """
    if scene_name not in SCENES:
        raise ValueError(f"Unknown scene: {scene_name}. Available: {list(SCENES.keys())}")
    
    return SCENES[scene_name]()
