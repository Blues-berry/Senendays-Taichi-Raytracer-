"""测试导入和基本功能"""
import sys
import traceback

print("开始测试...")
print("=" * 60)

try:
    print("1. 导入 taichi...")
    import taichi as ti
    ti.init(arch=ti.gpu)
    print("✓ taichi 导入成功")
except Exception as e:
    print(f"✗ taichi 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. 导入 main...")
    import main
    print("✓ main 导入成功")
except Exception as e:
    print(f"✗ main 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. 导入 camera_ms_aic...")
    from camera_ms_aic import MultiScaleGrid
    print("✓ camera_ms_aic 导入成功")
except Exception as e:
    print(f"✗ camera_ms_aic 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n4. 导入 camera_motion_comp...")
    from camera_motion_comp import MotionCompensatedTemporalFilter
    print("✓ camera_motion_comp 导入成功")
except Exception as e:
    print(f"✗ camera_motion_comp 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n5. 设置场景...")
    world, cam = main.setup_scene('cornell_box')
    print(f"✓ 场景设置成功")
    print(f"  场景模式: {cam.scene_mode}")
    print(f"  图像分辨率: {cam.img_res}")
except Exception as e:
    print(f"✗ 场景设置失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n6. 创建 MultiScaleGrid...")
    GRID_RESOLUTIONS = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    grid_origin = ti.types.vector(3, float)(-8.0, -1.0, -8.0)
    grid_cell_size = 1.0
    ms_grid = MultiScaleGrid(GRID_RESOLUTIONS, grid_origin, grid_cell_size)
    print(f"✓ MultiScaleGrid 创建成功")
    print(f"  层数: {len(GRID_RESOLUTIONS)}")
    print(f"  显存: {ms_grid.get_memory_usage_mb():.2f} MB")
except Exception as e:
    print(f"✗ MultiScaleGrid 创建失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n7. 创建 MotionCompensatedTemporalFilter...")
    mctf = MotionCompensatedTemporalFilter(cam.img_res)
    print(f"✓ MotionCompensatedTemporalFilter 创建成功")
    print(f"  分辨率: {cam.img_res}")
except Exception as e:
    print(f"✗ MotionCompensatedTemporalFilter 创建失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)
