"""
快速测试新场景脚本

测试所有新场景的基本渲染功能
"""

import taichi as ti
from scenes.scene_configs import get_scene
from world import World
from camera import Camera
import time

# 初始化 Taichi
ti.init(arch=ti.gpu, random_seed=42)

# 测试场景列表
scenes = ['cornell_box', 'two_room', 'night_scene', 'random',
          'classroom', 'bathroom', 'veach_mis']

def test_scene(scene_name: str, frames: int = 10):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"Testing scene: {scene_name}")
    print(f"{'='*60}")

    # 1. 创建场景
    start_time = time.time()
    spheres, materials, cam_params = get_scene(scene_name)
    setup_time = time.time() - start_time

    print(f"  Setup time: {setup_time:.2f}s")
    print(f"  Spheres: {len(spheres)}")
    print(f"  Materials: {len(materials)}")
    print(f"  Camera: lookfrom={cam_params['lookfrom']}, vfov={cam_params['vfov']}")

    # 2. 初始化世界和相机
    world = World(spheres, materials)
    cam = Camera(world, **cam_params)

    # 3. 适应网格
    start_time = time.time()
    cam.adapt_grid_to_scene(spheres, verbose=False)
    adapt_time = time.time() - start_time
    print(f"  Grid adaptation time: {adapt_time:.2f}s")

    # 4. 设置光源列表
    cam.set_light_sources(spheres, materials)

    # 5. 渲染几帧测试
    mode_int = 2  # Adaptive mode
    print(f"\n  Rendering {frames} frames...")

    frame_times = []
    for i in range(frames):
        start = time.time()
        cam.update_grid(world, 0.01)
        cam.render(world, mode_int)
        cam.asvgf_filter()
        ti.sync()
        frame_time = time.time() - start
        frame_times.append(frame_time)

        if (i + 1) % 5 == 0:
            avg_fps = 1.0 / (sum(frame_times[-5:]) / 5)
            print(f"    Frame {i+1}/{frames}: {frame_time*1000:.1f}ms, FPS={avg_fps:.1f}")

    # 6. 统计
    avg_frame_time = sum(frame_times) / len(frame_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    nx, ny, nz = cam.grid_res
    grid_mem_mb = float(nx * ny * nz * 3 * 4) / (1024.0 * 1024.0)

    print(f"\n  === Statistics ===")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Average frame time: {avg_frame_time*1000:.1f}ms")
    print(f"  Grid resolution: {cam.grid_res}")
    print(f"  Grid memory: {grid_mem_mb:.2f} MB")

    return {
        'scene': scene_name,
        'spheres': len(spheres),
        'avg_fps': avg_fps,
        'grid_memory_mb': grid_mem_mb,
    }


def main():
    """测试所有场景"""
    print("\n" + "="*60)
    print("New Scenes Quick Test")
    print("="*60)

    results = []

    for scene_name in scenes:
        try:
            result = test_scene(scene_name, frames=20)
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ ERROR in {scene_name}: {e}")
            import traceback
            traceback.print_exc()

    # 输出汇总表格
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Scene':<15} | {'Spheres':<8} | {'FPS':<8} | {'Memory (MB)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['scene']:<15} | {r['spheres']:<8} | {r['avg_fps']:<8.1f} | {r['grid_memory_mb']:<12.2f}")

    print("\n✅ Test completed!")


if __name__ == '__main__':
    main()
