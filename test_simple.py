"""更简单的测试脚本"""
import sys
import traceback

print("测试1: 导入所有模块...")
try:
    import taichi as ti
    ti.init(arch=ti.gpu)
    print("✓ taichi")
    
    import main
    print("✓ main")
    
    from camera_ms_aic import MultiScaleGrid
    print("✓ MultiScaleGrid")
    
    from camera_motion_comp import MotionCompensatedTemporalFilter
    print("✓ MotionCompensatedTemporalFilter")
    
except Exception as e:
    print(f"✗ 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n测试2: 设置场景...")
try:
    world, cam = main.setup_scene('cornell_box')
    print(f"✓ 场景: {cam.scene_mode}")
except Exception as e:
    print(f"✗ 场景设置失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n测试3: 创建MS Grid...")
try:
    vec3 = ti.types.vector(3, float)
    GRID_RESOLUTIONS = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    grid_origin = vec3(-8.0, -1.0, -8.0)
    grid_cell_size = 1.0
    ms_grid = MultiScaleGrid(GRID_RESOLUTIONS, grid_origin, grid_cell_size)
    print(f"✓ 创建成功，显存: {ms_grid.get_memory_usage_mb():.2f} MB")
except Exception as e:
    print(f"✗ MS Grid创建失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n测试4: 创建MCTF...")
try:
    mctf = MotionCompensatedTemporalFilter(cam.img_res)
    print(f"✓ 创建成功，分辨率: {cam.img_res}")
except Exception as e:
    print(f"✗ MCTF创建失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n测试5: 更新MS Grid (1次)...")
try:
    camera_pos = cam.camera_origin
    ms_grid.update_all_levels(world, 0.01, camera_pos)
    ti.sync()
    print("✓ 更新成功")
except Exception as e:
    print(f"✗ 更新失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n测试6: 渲染1帧...")
try:
    cam.render(world, 2)
    ti.sync()
    print("✓ 渲染成功")
except Exception as e:
    print(f"✗ 渲染失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n所有测试通过！")
