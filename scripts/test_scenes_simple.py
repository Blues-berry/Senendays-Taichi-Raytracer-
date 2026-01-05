"""
简单场景测试脚本 - 测试所有场景的基本功能
Simple Scene Test Script
"""

import os
import sys
import time
from datetime import datetime

# 添加路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入main以使用已初始化的taichi
import main

from scenes.scene_configs import get_scene


def log_message(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def test_scene(scene_name, frames=20):
    """测试单个场景"""
    log_message(f"\n{'='*60}")
    log_message(f"Testing scene: {scene_name}")
    log_message(f"{'='*60}")
    
    # 获取场景配置
    spheres_list, materials_list, cam_params = get_scene(scene_name)
    
    # 重新初始化世界和相机
    world = main.World(spheres_list, materials_list)
    cam = main.Camera(world, **cam_params)
    cam.scene_mode = scene_name
    
    log_message(f"  Spheres: {len(spheres_list)}, Materials: {len(materials_list)}")
    
    # 适应网格
    cam.adapt_grid_to_scene(spheres_list, verbose=False)
    
    # 渲染测试
    log_message(f"  Rendering {frames} frames...")
    start_time = time.time()
    
    fps_history = []
    for frame in range(frames):
        frame_start = time.time()
        
        cam.update_grid(world, 0.01)
        cam.render(world, 2)  # Hybrid mode
        cam.asvgf_filter()
        
        frame_time = time.time() - frame_start
        if frame_time > 1e-6:
            fps = 1.0 / frame_time
            # Filter out unreasonable FPS values
            if fps < 0.1 or fps > 10000:
                fps = 0.0
        else:
            fps = 0.0
        fps_history.append(fps)
        
        if (frame + 1) % 5 == 0:
            log_message(f"    Frame {frame+1}/{frames}: FPS={fps:.2f}")
    
    total_time = time.time() - start_time
    avg_fps = sum(fps_history) / len(fps_history)
    
    log_message(f"  Completed in {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")
    
    return {
        'scene': scene_name,
        'frames': frames,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'fps_history': fps_history
    }


def test_two_room_scene():
    """专门测试两室一门场景"""
    log_message(f"\n{'='*60}")
    log_message("Two-Room Stress Test (Pressure Test)")
    log_message(f"{'='*60}")
    
    # 获取场景配置
    spheres_list, materials_list, cam_params = get_scene('two_room')
    
    # 初始化
    world = main.World(spheres_list, materials_list)
    cam = main.Camera(world, **cam_params)
    cam.scene_mode = 'two_room'
    
    log_message("  Scene: Two-Room-One-Door")
    log_message("  Room A (z<0) has light source")
    log_message("  Room B (z>0) receives indirect light only")
    
    # 适应网格
    cam.adapt_grid_to_scene(spheres_list, verbose=False)
    
    # 渲染100帧
    log_message(f"  Rendering 100 frames...")
    start_time = time.time()
    
    for frame in range(100):
        cam.update_grid(world, 0.01)
        cam.render(world, 2)
        cam.asvgf_filter()
        
        if (frame + 1) % 20 == 0:
            log_message(f"    Frame {frame+1}/100")
    
    total_time = time.time() - start_time
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"two_room_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    from ti.tools import imwrite
    
    hybrid_path = os.path.join(output_dir, 'hybrid_render.png')
    imwrite(cam.frame, hybrid_path)
    log_message(f"\n  Saved hybrid render: {hybrid_path}")
    
    # 构建PT参考（128 spp）
    log_message("  Building PT reference (128 spp)...")
    cam.clear_pt_reference()
    for _ in range(128):
        cam.render_pt(world)
        main.ti.sync()
    
    pt_path = os.path.join(output_dir, 'pt_reference.png')
    imwrite(cam.pt_frame, pt_path)
    log_message(f"  Saved PT reference: {pt_path}")
    
    # 生成误差热力图
    cam.render_error_heatmap()
    heatmap_path = os.path.join(output_dir, 'error_heatmap.png')
    imwrite(cam.frame, heatmap_path)
    log_message(f"  Saved error heatmap: {heatmap_path}")
    
    log_message(f"\n  Stress test completed in {total_time:.2f}s")
    log_message(f"  All results saved to: {output_dir}")
    
    return output_dir


def main_test():
    """主测试函数"""
    print("\n" + "="*70)
    print("Simple Scene Testing")
    print("="*70)
    
    # 测试所有基本场景
    scenes = ['cornell_box', 'night_scene', 'random']
    results = []
    
    for scene in scenes:
        try:
            result = test_scene(scene, frames=20)
            results.append(result)
        except Exception as e:
            log_message(f"Scene {scene} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试两室场景
    try:
        two_room_dir = test_two_room_scene()
    except Exception as e:
        log_message(f"Two-room test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 打印摘要
    log_message("\n" + "="*70)
    log_message("Test Summary:")
    log_message("="*70)
    for r in results:
        log_message(f"  {r['scene']}: {r['frames']} frames, Avg FPS={r['avg_fps']:.2f}")


if __name__ == "__main__":
    try:
        main_test()
    except KeyboardInterrupt:
        log_message("\nTest interrupted by user")
    except Exception as e:
        log_message(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
