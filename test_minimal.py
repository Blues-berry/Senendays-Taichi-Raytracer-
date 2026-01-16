"""最小测试 - 只测试基本功能"""
print("开始...")

import taichi as ti
ti.init(arch=ti.gpu)
print("Taichi初始化成功")

import main
print("Main导入成功")

world, cam = main.setup_scene('cornell_box')
print(f"场景设置成功: {cam.scene_mode}")
print(f"分辨率: {cam.img_res}")

# 渲染1帧
print("渲染1帧...")
cam.render(world, 2)
ti.sync()
print("渲染成功")

print("\n所有测试通过！")
