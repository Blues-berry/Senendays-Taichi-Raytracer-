"""
Test GPU Timing / GPU计时测试
Verify GPU timing in benchmark.py works correctly
验证benchmark.py中的GPU耗时统计是否正常工作
"""

import taichi as ti
import time
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

def test_gpu_timing():
    """测试GPU计时功能 / Test GPU timing functionality"""
    print("=== 测试GPU计时功能 / Testing GPU Timing ===")
    
    # 初始化Taichi
    ti.init(arch=ti.cuda)
    
    # 设置场景
    world, cam = main.setup_scene('cornell_box')
    cam.scene_mode = 'cornell_box'
    
    # 预热 / Warm-up
    print("预热中 / Warming up...")
    ti.sync()
    cam.update_grid(world, 0.01)
    cam.render(world, 2)  # Hybrid mode
    ti.sync()
    
    # 测试多次渲染的GPU耗时 / Test GPU timing for multiple renders
    print("开始GPU耗时测试 / Starting GPU timing test...")
    frame_times = []
    
    for i in range(10):
        # 精确计时 / Precise timing
        ti.sync()
        start_time = time.perf_counter()
        
        # 执行渲染 / Execute rendering
        cam.update_grid(world, 0.01)
        cam.render(world, 2)  # Hybrid mode
        cam.asvgf_filter()
        
        # 强制同步 / Force synchronization
        ti.sync()
        end_time = time.perf_counter()
        
        frame_time_ms = (end_time - start_time) * 1000.0
        frame_times.append(frame_time_ms)
        
        print(f"Frame {i+1}: {frame_time_ms:.2f} ms")
    
    # 统计结果 / Statistics
    avg_time = np.mean(frame_times)
    min_time = np.min(frame_times)
    max_time = np.max(frame_times)
    std_time = np.std(frame_times)
    
    print(f"\n=== GPU耗时统计 / GPU Timing Statistics ===")
    print(f"平均耗时 / Average: {avg_time:.2f} ms")
    print(f"最小耗时 / Minimum: {min_time:.2f} ms")
    print(f"最大耗时 / Maximum: {max_time:.2f} ms")
    print(f"标准差 / Std Dev: {std_time:.2f} ms")
    print(f"等效FPS / Equivalent FPS: {1000.0 / avg_time:.1f}")
    
    # 测试不同模式的耗时 / Test timing for different modes
    print(f"\n=== 不同渲染模式耗时对比 / Timing Comparison by Mode ===")
    modes = [
        (0, "Path Tracing"),
        (1, "Grid"), 
        (2, "Hybrid")
    ]
    
    for mode, mode_name in modes:
        # 预热 / Warm-up
        ti.sync()
        cam.render(world, mode)
        ti.sync()
        
        # 测试5帧 / Test 5 frames
        times = []
        for _ in range(5):
            ti.sync()
            start = time.perf_counter()
            
            if mode == 0:  # PT
                cam.render(world, mode)
            elif mode == 1:  # Grid
                cam.update_grid(world, 0.01)
                cam.render(world, mode)
            else:  # Hybrid
                cam.update_grid(world, 0.01)
                cam.render(world, mode)
                cam.asvgf_filter()
            
            ti.sync()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
        
        avg = np.mean(times)
        print(f"{mode_name}: {avg:.2f} ms (FPS: {1000.0 / avg:.1f})")

if __name__ == "__main__":
    try:
        test_gpu_timing()
        print("\nGPU计时测试完成! / GPU timing test completed!")
    except Exception as e:
        print(f"测试过程中出错 / Error during test: {e}")
        import traceback
        traceback.print_exc()
