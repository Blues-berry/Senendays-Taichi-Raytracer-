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

    # Cornell Box 参数（标准单位尺度，避免 AABB 过大导致 grid 失效）
    # Room extends roughly to [-half, +half] in X/Y and [-2*half, 0] in Z.
    half = 1.0
    wall_radius = 100.0  # 足够大近似平面，但不会让 AABB 膨胀到几千

    # === 五面墙（没有前墙，便于相机观察） ===
    # We build planes via large spheres placed outside the box.
    # Left wall (red): x = -half
    spheres.append(Sphere(center=vec3(-(wall_radius + half), 0.0, -half), radius=wall_radius))
    materials.append(material.Lambert(left_red))

    # Right wall (green): x = +half
    spheres.append(Sphere(center=vec3((wall_radius + half), 0.0, -half), radius=wall_radius))
    materials.append(material.Lambert(right_green))

    # Floor: y = -half
    spheres.append(Sphere(center=vec3(0.0, -(wall_radius + half), -half), radius=wall_radius))
    materials.append(material.Lambert(white))

    # Ceiling: y = +half
    spheres.append(Sphere(center=vec3(0.0, (wall_radius + half), -half), radius=wall_radius))
    materials.append(material.Lambert(white))

    # Back wall: z = -2*half
    spheres.append(Sphere(center=vec3(0.0, 0.0, -(wall_radius + 2.0 * half)), radius=wall_radius))
    materials.append(material.Lambert(white))

    # === 顶部面光源（小发光球近似面光源，置于天花板下方） ===
    spheres.append(Sphere(center=vec3(0.0, half - 0.05, -half - 0.35), radius=0.25))
    materials.append(material.DiffuseLight(vec3(25.0, 25.0, 25.0)))

    # === 盒内两球（标准大小） ===
    spheres.append(Sphere(center=vec3(-0.35, -half + 0.30, -half - 0.95), radius=0.30))
    materials.append(material.Metal(vec3(0.93, 0.93, 0.93), 0.01))

    spheres.append(Sphere(center=vec3(0.35, -half + 0.30, -half - 0.55), radius=0.30))
    materials.append(material.Dielectric(1.5))

    # 相机参数（贴近 Cornell Box 常用视角）
    # Provide a *logical* scene AABB for grid adaptation (avoid huge AABB from wall spheres).
    # Room approx: x,y in [-half, half], z in [-2*half, 0]
    pad = 0.1
    cam_params = dict(
        lookfrom=vec3(0.0, 0.0, 3.2),
        lookat=vec3(0.0, 0.0, -1.0),
        vup=vec3(0, 1, 0),
        vfov=40.0,
        defocus_angle=0.0,
        focus_dist=3.2,
        scene_mode='cornell_box',
        scene_bounds=(-half - pad, -half - pad, -2.0 * half - pad, half + pad, half + pad, 0.0 + pad),
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


def create_classroom_scene():
    """
    教室场景 (Classroom Scene)
    
    黑板 + 窗户光（环境光主导）
    测试目标：环境光照和间接光传播
    """
    spheres = []
    materials = []

    # 材质
    white = vec3(0.85, 0.85, 0.85)      # 白墙
    blackboard = vec3(0.05, 0.05, 0.08) # 深色黑板
    floor_color = vec3(0.6, 0.55, 0.45)  # 木地板
    sky_blue = vec3(0.6, 0.75, 0.95)   # 窗外天空色
    desk_color = vec3(0.45, 0.35, 0.25)  # 课桌

    R = 1000.0
    room_width = 8.0
    room_height = 4.0
    room_depth = 6.0

    # === 教室墙壁 ===
    # 地面
    spheres.append(Sphere(center=vec3(0, -R, 0), radius=R))
    materials.append(material.Lambert(floor_color))

    # 天花板
    spheres.append(Sphere(center=vec3(0, room_height + R, 0), radius=R))
    materials.append(material.Lambert(white))

    # 前墙（带黑板）
    spheres.append(Sphere(center=vec3(0, room_height/2, room_depth/2 + R), radius=R))
    materials.append(material.Lambert(white))

    # 黑板（大椭圆球近似）
    blackboard_center = vec3(0, room_height/2, room_depth/2 - 0.1)
    blackboard_radius = 2.5
    spheres.append(Sphere(center=blackboard_center, radius=blackboard_radius))
    materials.append(material.Lambert(blackboard))

    # 后墙（窗户所在）
    spheres.append(Sphere(center=vec3(0, room_height/2, -room_depth/2 - R), radius=R))
    materials.append(material.Lambert(white))

    # 左墙
    spheres.append(Sphere(center=vec3(-room_width/2 - R, room_height/2, 0), radius=R))
    materials.append(material.Lambert(white))

    # 右墙
    spheres.append(Sphere(center=vec3(room_width/2 + R, room_height/2, 0), radius=R))
    materials.append(material.Lambert(white))

    # === 窗户（发光大球模拟） ===
    # 两个窗户
    window_positions = [
        vec3(-room_width/2 + 0.8, room_height/2, -room_depth/2 + 0.1),
        vec3(room_width/2 - 0.8, room_height/2, -room_depth/2 + 0.1),
    ]
    for pos in window_positions:
        spheres.append(Sphere(center=pos, radius=0.8))
        materials.append(material.DiffuseLight(sky_blue * 8.0))

    # === 讲台和课桌 ===
    # 讲台
    spheres.append(Sphere(center=vec3(0, 0.6, room_depth/2 - 1.5), radius=0.5))
    materials.append(material.Metal(vec3(0.5, 0.45, 0.4), 0.1))

    # 课桌（两排）
    desk_positions = [
        (-2.5, -1.5), (0, -1.5), (2.5, -1.5),   # 后排
        (-2.5, -3.5), (0, -3.5), (2.5, -3.5),   # 前排
    ]
    for dx, dz in desk_positions:
        spheres.append(Sphere(center=vec3(dx, 0.4, dz), radius=0.3))
        materials.append(material.Lambert(desk_color))

    # === 教室内的物体 ===
    # 讲师桌上的物体
    spheres.append(Sphere(center=vec3(-0.5, 1.1, room_depth/2 - 1.5), radius=0.15))
    materials.append(material.Dielectric(1.5))

    spheres.append(Sphere(center=vec3(0.5, 1.1, room_depth/2 - 1.5), radius=0.15))
    materials.append(material.Metal(vec3(0.8, 0.7, 0.6), 0.05))

    # === 辅助光照（室内灯光） ===
    # 教室灯（天花板）
    spheres.append(Sphere(center=vec3(0, room_height - 0.2, 0), radius=0.4))
    materials.append(material.DiffuseLight(vec3(5, 5, 5)))

    # 相机参数
    cam_params = dict(
        lookfrom=vec3(0, room_height/2 + 0.5, room_depth + 2),
        lookat=vec3(0, room_height/2, 0),
        vup=vec3(0, 1, 0),
        vfov=55.0,
        defocus_angle=0.0,
        focus_dist=8.0,
        scene_mode='classroom'
    )

    return spheres, materials, cam_params


def create_bathroom_scene():
    """
    浴室场景 (Bathroom Scene)
    
    白瓷砖 + 镜面/玻璃（高反射+caustics）
    测试目标：镜面反射和焦散效果
    """
    spheres = []
    materials = []

    # 材质
    white_tile = vec3(0.92, 0.92, 0.95)  # 白瓷砖
    mirror_color = vec3(0.98, 0.98, 1.0)  # 镜子
    water_color = vec3(0.7, 0.85, 0.95)   # 水
    warm_light = vec3(1.0, 0.9, 0.75)    # 暖光

    R = 1000.0
    room_width = 3.5
    room_height = 3.0
    room_depth = 4.0

    # === 浴室墙壁（白瓷砖） ===
    # 地面
    spheres.append(Sphere(center=vec3(0, -R, 0), radius=R))
    materials.append(material.Lambert(white_tile))

    # 天花板
    spheres.append(Sphere(center=vec3(0, room_height + R, 0), radius=R))
    materials.append(material.Lambert(white_tile))

    # 后墙
    spheres.append(Sphere(center=vec3(0, room_height/2, room_depth/2 + R), radius=R))
    materials.append(material.Lambert(white_tile))

    # 前墙（门）
    spheres.append(Sphere(center=vec3(0, room_height/2, -room_depth/2 - R), radius=R))
    materials.append(material.Lambert(white_tile))

    # 左墙
    spheres.append(Sphere(center=vec3(-room_width/2 - R, room_height/2, 0), radius=R))
    materials.append(material.Lambert(white_tile))

    # 右墙
    spheres.append(Sphere(center=vec3(room_width/2 + R, room_height/2, 0), radius=R))
    materials.append(material.Lambert(white_tile))

    # === 洗手台（带镜子） ===
    # 洗手台体
    spheres.append(Sphere(center=vec3(0, 0.5, room_depth/2 - 0.8), radius=0.5))
    materials.append(material.Lambert(vec3(0.6, 0.5, 0.4)))

    # 镜子（高反射金属模拟）
    spheres.append(Sphere(center=vec3(0, 1.5, room_depth/2 - 1.2), radius=0.6))
    materials.append(material.Metal(mirror_color, 0.0))

    # === 浴缸/淋浴区 ===
    # 浴缸
    tub_center = vec3(-room_width/2 + 1.0, 0.4, -room_depth/2 + 1.5)
    spheres.append(Sphere(center=tub_center, radius=0.6))
    materials.append(material.Dielectric(1.5))

    # 水面（透明球）
    spheres.append(Sphere(center=vec3(tub_center[0], 0.7, tub_center[2]), radius=0.4))
    materials.append(material.Dielectric(1.33))

    # === 高反射球体（测试caustics） ===
    # 玻璃杯
    spheres.append(Sphere(center=vec3(0.8, 0.8, room_depth/2 - 0.8), radius=0.2))
    materials.append(material.Dielectric(1.5))

    # 镜面球
    spheres.append(Sphere(center=vec3(-0.8, 0.8, room_depth/2 - 0.8), radius=0.2))
    materials.append(material.Metal(vec3(0.95, 0.95, 1.0), 0.0))

    # === 光源 ===
    # 顶部灯光（暖光）
    spheres.append(Sphere(center=vec3(0, room_height - 0.3, 0), radius=0.25))
    materials.append(material.DiffuseLight(warm_light * 25))

    # 镜子前的灯
    spheres.append(Sphere(center=vec3(0, 1.8, room_depth/2 - 1.8), radius=0.15))
    materials.append(material.DiffuseLight(warm_light * 20))

    # 相机参数
    cam_params = dict(
        lookfrom=vec3(0, room_height/2, -room_depth/2 + 0.5),
        lookat=vec3(0, room_height/2, room_depth/2),
        vup=vec3(0, 1, 0),
        vfov=60.0,
        defocus_angle=0.0,
        focus_dist=5.0,
        scene_mode='bathroom'
    )

    return spheres, materials, cam_params


def create_veach_mis_scene():
    """
    Veach MIS 场景 (Veach Multiple Importance Sampling)
    
    多强度光源（测试importance sampling）
    原创场景：测试复杂光照下的重要性采样
    """
    spheres = []
    materials = []

    # 材质
    white = vec3(0.9, 0.9, 0.9)
    gray = vec3(0.4, 0.4, 0.4)

    R = 1000.0

    # === 基础房间 ===
    # 地面
    spheres.append(Sphere(center=vec3(0, -R, 0), radius=R))
    materials.append(material.Lambert(gray))

    # 天花板
    spheres.append(Sphere(center=vec3(0, 5 + R, 0), radius=R))
    materials.append(material.Lambert(white))

    # 后墙
    spheres.append(Sphere(center=vec3(0, 2.5, 4 + R), radius=R))
    materials.append(material.Lambert(white))

    # 左墙
    spheres.append(Sphere(center=vec3(-5 - R, 2.5, 0), radius=R))
    materials.append(material.Lambert(white))

    # 右墙
    spheres.append(Sphere(center=vec3(5 + R, 2.5, 0), radius=R))
    materials.append(material.Lambert(white))

    # === 前墙（带不同强度光源） ===
    spheres.append(Sphere(center=vec3(0, 2.5, -4 - R), radius=R))
    materials.append(material.Lambert(white))

    # === 多个不同强度的光源 ===
    # 场景核心：同一位置、不同强度的光源
    light_x_positions = [-2, 0, 2]
    light_intensities = [1.0, 10.0, 100.0]  # 测试MIS效果

    for i, (x, intensity) in enumerate(zip(light_x_positions, light_intensities)):
        # 顶部小灯
        spheres.append(Sphere(center=vec3(x, 4.5, 0), radius=0.15))
        materials.append(material.DiffuseLight(vec3(intensity, intensity, intensity)))

        # 底部小灯
        spheres.append(Sphere(center=vec3(x, 0.5, 0), radius=0.15))
        materials.append(material.DiffuseLight(vec3(intensity, intensity, intensity)))

    # === 高反射表面（测试焦散） ===
    # 镜面地板区域
    mirror_floor = Sphere(center=vec3(-3.5, -R + 0.1, -2), radius=2.0)
    spheres.append(mirror_floor)
    materials.append(material.Metal(vec3(0.98, 0.98, 1.0), 0.0))

    # 玻璃球（测试折射和caustics）
    spheres.append(Sphere(center=vec3(-3.5, 1.0, -2), radius=0.8))
    materials.append(material.Dielectric(1.5))

    # === 低反射表面（对比） ===
    # 漫反射地板区域
    spheres.append(Sphere(center=vec3(3.5, -R + 0.1, -2), radius=2.0))
    materials.append(material.Lambert(vec3(0.2, 0.2, 0.2)))

    # 漫反射球
    spheres.append(Sphere(center=vec3(3.5, 1.0, -2), radius=0.8))
    materials.append(material.Lambert(vec3(0.6, 0.4, 0.3)))

    # === 中间区域（复杂材质球） ===
    for i in range(3):
        x = -1.5 + i * 1.5
        z = -2

        # 玻璃球
        spheres.append(Sphere(center=vec3(x, 0.6, z), radius=0.4))
        materials.append(material.Dielectric(1.5))

        # 金属球（不同粗糙度）
        fuzz = i * 0.1  # 0.0, 0.1, 0.2
        spheres.append(Sphere(center=vec3(x, 0.6, z + 2), radius=0.4))
        materials.append(material.Metal(vec3(0.9, 0.9, 0.95), fuzz))

    # === 观察球（用于评估光照分布） ===
    observation_spheres = [
        vec3(-3, 1.0, 2),
        vec3(0, 1.0, 2),
        vec3(3, 1.0, 2),
    ]
    for pos in observation_spheres:
        spheres.append(Sphere(center=pos, radius=0.3))
        materials.append(material.Lambert(vec3(0.8, 0.8, 0.8)))

    # === 辅助光源（确保场景明亮） ===
    spheres.append(Sphere(center=vec3(0, 2.5, 0), radius=0.3))
    materials.append(material.DiffuseLight(vec3(5, 5, 5)))

    # 相机参数
    cam_params = dict(
        lookfrom=vec3(0, 2.5, -3),
        lookat=vec3(0, 2.5, 0),
        vup=vec3(0, 1, 0),
        vfov=50.0,
        defocus_angle=0.0,
        focus_dist=3.0,
        scene_mode='veach_mis'
    )

    return spheres, materials, cam_params


# 场景字典
SCENES = {
    'cornell_box': create_cornell_box_scene,
    'two_room': create_two_room_scene,
    'night_scene': create_night_scene,
    'random': create_random_scene,
    'classroom': create_classroom_scene,      # 新增
    'bathroom': create_bathroom_scene,       # 新增
    'veach_mis': create_veach_mis_scene,       # 新增
}


def get_scene(scene_name='random'):
    """
    获取指定场景的配置
    
    Args:
        scene_name: 场景名称 ('cornell_box', 'two_room', 'night_scene', 'random',
                              'classroom', 'bathroom', 'veach_mis')
    
    Returns:
        (spheres, materials, cam_params)
    
    场景说明:
        - cornell_box: 标准 Cornell Box (5面墙 + 2球 + 面光源)
        - two_room: 两室一门 (测试窄缝漏光)
        - night_scene: 夜间场景 (多光源 + 高反射)
        - random: 随机小球场景 (默认)
        - classroom: 教室场景 (黑板 + 窗户光)
        - bathroom: 浴室场景 (白瓷砖 + 镜面 + caustics)
        - veach_mis: Veach MIS 场景 (多强度光源测试重要性采样)
    """
    if scene_name not in SCENES:
        raise ValueError(f"Unknown scene: {scene_name}. Available: {list(SCENES.keys())}")
    
    return SCENES[scene_name]()
