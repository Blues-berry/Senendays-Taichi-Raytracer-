"""
统一运行脚本 - 场景测试与数据分析
Run all scenes and generate plots
"""

import os
import sys
import time
from datetime import datetime

# 添加路径 - 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import taichi as ti
import numpy as np

# 导入main以确保taichi已初始化
import main

from scenes.scene_configs import get_scene
from camera import Camera
from world import World
import material

# 导入绘图工具
from plots.plotting_tools import (
    plot_convergence_scatter,
    plot_mse_curves,
    plot_mse_with_heatmap_markers,
    create_heatmap_collage,
    save_results_summary
)

vec3 = ti.types.vector(3, float)


def log_message(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def create_output_dir(name):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_scene_test(scene_name='cornell_box', frames=100):
    """
    运行单个场景测试
    
    Args:
        scene_name: 场景名称
        frames: 渲染帧数
    
    Returns:
        dict: 包含渲染时间、FPS等数据
    """
    log_message(f"Running scene: {scene_name}")
    
    # 获取场景配置
    spheres_list, materials_list, cam_params = get_scene(scene_name)
    
    # Taichi已经在main.py中初始化，这里不需要重复初始化
    # if ti.arch is None:
    #     ti.init(arch=ti.gpu, random_seed=42)
    
    # 创建世界和相机
    world = World(spheres_list, materials_list)
    cam = Camera(world, **cam_params)
    
    # 适应网格
    cam.adapt_grid_to_scene(spheres_list, verbose=False)
    
    # 渲染循环
    start_time = time.time()
    fps_history = []
    
    for frame in range(frames):
        frame_start = time.time()
        
        # 更新网格（1%概率）
        cam.update_grid(world, 0.01)
        
        # 渲染Hybrid模式
        cam.render(world, 2)

        # 应用A-SVGF
        cam.asvgf_filter()

        # 计算FPS (with filtering for extreme values)
        frame_time = time.time() - frame_start
        if frame_time > 1e-6:
            fps = 1.0 / frame_time
            # Filter out unreasonable FPS values
            if fps < 0.1 or fps > 10000:
                fps = 0.0
        else:
            fps = 0.0
        fps_history.append(fps)

        if (frame + 1) % 20 == 0:
            log_message(f"  Frame {frame+1}/{frames}: FPS={fps:.2f}")
    
    total_time = time.time() - start_time
    avg_fps = np.mean(fps_history)
    
    log_message(f"Scene {scene_name} completed: {total_time:.2f}s, Avg FPS={avg_fps:.2f}")
    
    return {
        'scene': scene_name,
        'frames': frames,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'fps_history': fps_history
    }


def run_all_scenes():
    """运行所有场景测试"""
    log_message("="*70)
    log_message("Running All Scenes Test")
    log_message("="*70)
    
    scenes_to_test = ['cornell_box', 'night_scene', 'random']  # 先测试这三个
    
    results = []
    output_dir = create_output_dir("all_scenes")
    
    for scene in scenes_to_test:
        try:
            result = run_scene_test(scene, frames=50)
            results.append(result)
        except Exception as e:
            log_message(f"Scene {scene} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    save_results_summary(results, os.path.join(output_dir, 'scene_results.csv'))
    
    # 绘制FPS对比
    fps_data = {r['scene']: r['fps_history'] for r in results}
    from plots.plotting_tools import plot_fps_comparison
    plot_fps_comparison(fps_data, os.path.join(output_dir, 'fps_comparison.png'))
    
    log_message(f"\nAll scenes completed. Results saved to: {output_dir}")


def run_two_room_stress_test():
    """
    运行两室一门场景的"压力测试"
    测试深度遮挡检测和窄缝漏光
    """
    log_message("="*70)
    log_message("Two-Room Stress Test (Pressure Test)")
    log_message("="*70)
    
    output_dir = create_output_dir("two_room_stress_test")
    
    # 获取场景配置
    spheres_list, materials_list, cam_params = get_scene('two_room')
    
    # Taichi已经在main.py中初始化，这里不需要重复初始化
    # if ti.arch is None:
    #     ti.init(arch=ti.gpu, random_seed=42)
    
    # 创建世界和相机
    world = World(spheres_list, materials_list)
    cam = Camera(world, **cam_params)
    
    log_message("Scene: Two-Room-One-Door")
    log_message("Testing: 1) Depth occlusion prevents light leak")
    log_message("         2) Indirect light through door slit")
    
    # 适应网格
    cam.adapt_grid_to_scene(spheres_list, verbose=False)
    
    # 渲染100帧
    log_message("\nRendering 100 frames...")
    start_time = time.time()
    
    for frame in range(100):
        cam.update_grid(world, 0.01)
        cam.render(world, 2)
        cam.asvgf_filter()
        
        if (frame + 1) % 20 == 0:
            log_message(f"  Frame {frame+1}/100")
    
    total_time = time.time() - start_time
    
    # 保存最终结果
    log_message("\nSaving results...")
    
    # 保存Hybrid渲染
    from ti.tools import imwrite
    hybrid_path = os.path.join(output_dir, 'hybrid_render.png')
    imwrite(cam.frame, hybrid_path)
    log_message(f"  Hybrid render: {hybrid_path}")
    
    # 构建PT参考（高精度）
    log_message("Building PT reference (256 spp)...")
    cam.clear_pt_reference()
    for _ in range(256):
        cam.render_pt(world)
        ti.sync()
    
    pt_path = os.path.join(output_dir, 'pt_reference.png')
    imwrite(cam.pt_frame, pt_path)
    log_message(f"  PT reference: {pt_path}")
    
    # 生成误差热力图
    cam.render_error_heatmap()
    heatmap_path = os.path.join(output_dir, 'error_heatmap.png')
    imwrite(cam.frame, heatmap_path)
    log_message(f"  Error heatmap: {heatmap_path}")
    
    # 创建三联对比图
    from plots.plotting_tools import create_comparison_triple
    comparison_path = os.path.join(output_dir, 'comparison_triple.png')
    create_comparison_triple(hybrid_path, pt_path, heatmap_path, comparison_path)
    log_message(f"  Comparison: {comparison_path}")
    
    log_message(f"\nStress test completed in {total_time:.2f}s")
    log_message(f"Results saved to: {output_dir}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Unified Raytracer Testing and Analysis")
    print("="*70)
    print("\nAvailable tests:")
    print("  1. Run all scenes (cornell_box, night_scene, random)")
    print("  2. Two-room stress test")
    print("  3. Convergence speed benchmark")
    print("  4. Error heatmap sequence")
    print("="*70 + "\n")
    
    # 默认运行所有场景测试
    try:
        print("Running: All scenes test...")
        run_all_scenes()
        
        print("\n" + "="*70)
        print("Next: Run two-room stress test...")
        print("="*70 + "\n")
        run_two_room_stress_test()
        
    except KeyboardInterrupt:
        log_message("\nTest interrupted by user")
    except Exception as e:
        log_message(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
