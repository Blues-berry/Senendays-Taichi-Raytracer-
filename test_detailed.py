"""详细诊断脚本 - 将错误写入日志文件"""
import sys
import traceback
from datetime import datetime

log_file = f"test_detailed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(message):
    """同时输出到控制台和日志文件"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def log_error(message):
    """记录错误信息"""
    log(f"\n{'='*60}")
    log(f"ERROR: {message}")
    log('='*60)
    with open(log_file, 'a', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    traceback.print_exc()

log("="*60)
log("详细诊断测试开始")
log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"日志文件: {log_file}")
log("="*60 + "\n")

# 测试1: 导入taichi
log("[测试1] 导入taichi...")
try:
    import taichi as ti
    ti.init(arch=ti.gpu, random_seed=42)
    log("✓ Taichi初始化成功")
    log(f"  版本: {ti.__version__}")
    log(f"  架构: {ti.arch_supports(ti.gpu)}")
except Exception as e:
    log_error(f"Taichi初始化失败: {e}")
    sys.exit(1)

# 测试2: 导入main
log("\n[测试2] 导入main...")
try:
    import main
    log("✓ main模块导入成功")
    log(f"  可用函数: {[x for x in dir(main) if not x.startswith('_')][:5]}...")
except Exception as e:
    log_error(f"main模块导入失败: {e}")
    sys.exit(1)

# 测试3: 设置场景
log("\n[测试3] 设置场景...")
try:
    world, cam = main.setup_scene('cornell_box')
    log("✓ 场景设置成功")
    log(f"  场景模式: {cam.scene_mode}")
    log(f"  图像分辨率: {cam.img_res}")
    log(f"  相机位置: {cam.camera_origin}")
except Exception as e:
    log_error(f"场景设置失败: {e}")
    sys.exit(1)

# 测试4: 导入camera_ms_aic
log("\n[测试4] 导入camera_ms_aic...")
try:
    from camera_ms_aic import MultiScaleGrid
    log("✓ camera_ms_aic模块导入成功")
except Exception as e:
    log_error(f"camera_ms_aic模块导入失败: {e}")
    sys.exit(1)

# 测试5: 创建MultiScaleGrid
log("\n[测试5] 创建MultiScaleGrid...")
try:
    vec3 = ti.types.vector(3, float)
    GRID_RESOLUTIONS = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    grid_origin = vec3(-8.0, -1.0, -8.0)
    grid_cell_size = 1.0
    ms_grid = MultiScaleGrid(GRID_RESOLUTIONS, grid_origin, grid_cell_size)
    log("✓ MultiScaleGrid创建成功")
    log(f"  层数: {len(GRID_RESOLUTIONS)}")
    log(f"  分辨率: {GRID_RESOLUTIONS}")
    log(f"  显存: {ms_grid.get_memory_usage_mb():.2f} MB")
except Exception as e:
    log_error(f"MultiScaleGrid创建失败: {e}")
    sys.exit(1)

# 测试6: 导入camera_motion_comp
log("\n[测试6] 导入camera_motion_comp...")
try:
    from camera_motion_comp import MotionCompensatedTemporalFilter
    log("✓ camera_motion_comp模块导入成功")
except Exception as e:
    log_error(f"camera_motion_comp模块导入失败: {e}")
    sys.exit(1)

# 测试7: 创建MotionCompensatedTemporalFilter
log("\n[测试7] 创建MotionCompensatedTemporalFilter...")
try:
    mctf = MotionCompensatedTemporalFilter(cam.img_res)
    log("✓ MotionCompensatedTemporalFilter创建成功")
    log(f"  分辨率: {cam.img_res}")
    log(f"  空间σ: {mctf.spatial_sigma}")
    log(f"  时间σ: {mctf.temporal_sigma}")
except Exception as e:
    log_error(f"MotionCompensatedTemporalFilter创建失败: {e}")
    sys.exit(1)

# 测试8: 更新MS Grid (1次)
log("\n[测试8] 更新MultiScaleGrid (1次)...")
try:
    import time
    camera_pos = cam.camera_origin
    start = time.time()
    ms_grid.update_all_levels(world, 0.01, camera_pos)
    ti.sync()
    elapsed = time.time() - start
    log("✓ MultiScaleGrid更新成功")
    log(f"  更新时间: {elapsed*1000:.1f} ms")
    log(f"  更新速率: {1/elapsed:.1f} updates/sec")
except Exception as e:
    log_error(f"MultiScaleGrid更新失败: {e}")
    sys.exit(1)

# 测试9: 渲染1帧
log("\n[测试9] 渲染1帧...")
try:
    start = time.time()
    cam.render(world, 2)
    ti.sync()
    elapsed = time.time() - start
    log("✓ 渲染成功")
    log(f"  渲染时间: {elapsed*1000:.1f} ms")
    log(f"  FPS: {1/elapsed:.1f}")
except Exception as e:
    log_error(f"渲染失败: {e}")
    sys.exit(1)

# 测试10: 渲染PT参考 (5 spp - 快速测试)
log("\n[测试10] 渲染PT参考 (5 spp, 快速测试)...")
try:
    start = time.time()
    cam.render_pt_reference(world, target_spp=5, chunk_spp=5, reset=True)
    ti.sync()
    elapsed = time.time() - start
    log("✓ PT参考渲染成功")
    log(f"  渲染时间: {elapsed:.2f} 秒")
except Exception as e:
    log_error(f"PT参考渲染失败: {e}")
    sys.exit(1)

# 测试11: 计算MSE
log("\n[测试11] 计算MSE...")
try:
    import numpy as np
    current_frame = cam.frame.to_numpy()
    pt_frame = cam.pt_frame.to_numpy()

    # 归一化到[0, 1]
    if current_frame.max() > 255.0:
        current_frame = current_frame / 255.0
    if pt_frame.max() > 255.0:
        pt_frame = pt_frame / 255.0

    # 计算MSE
    diff = current_frame - pt_frame
    mse = np.mean(diff ** 2)
    log("✓ MSE计算成功")
    log(f"  MSE: {mse:.6e}")
    if mse < 1e-2:
        log("  ✓ 质量检查通过 (MSE < 0.01)")
    else:
        log("  ⚠ 质量检查警告 (MSE >= 0.01)")
except Exception as e:
    log_error(f"MSE计算失败: {e}")
    sys.exit(1)

# 总结
log("\n" + "="*60)
log("所有测试通过！")
log("="*60)
log("\n测试总结:")
log(f"  ✓ Taichi环境正常")
log(f"  ✓ Main模块导入成功")
log(f"  ✓ 场景设置正常")
log(f"  ✓ MultiScaleGrid创建成功")
log(f"  ✓ MotionCompensatedTemporalFilter创建成功")
log(f"  ✓ MultiScaleGrid更新正常")
log(f"  ✓ 渲染功能正常")
log(f"  ✓ PT参考渲染正常")
log(f"  ✓ MSE计算正常")
log("\n代码已准备就绪，可以开始正式实验！")
log(f"\n详细日志已保存到: {log_file}")
