"""
Grid Resolution Performance Analyzer
统计不同网格分辨率下的显存占用和计算开销
"""

import taichi as ti
import numpy as np
import time
import csv
import os
from datetime import datetime
import main
from main import spheres, cam, world
import utils

# 测试的网格分辨率列表
GRID_RESOLUTIONS = [
    (16, 16, 16),
    (32, 32, 32), 
    (64, 64, 64),
    (128, 128, 128)  # 可选的高分辨率测试
]

# 测试参数
TEST_FRAMES = 100  # 每个分辨率测试的帧数
SCENE_MODE = 'cornell_box'

def calculate_total_grid_memory(grid_resolution):
    """计算网格系统的总显存占用（MB）"""
    # 根据camera.py中定义的字段计算
    # irradiance_grid: 3 floats per cell
    # grid_update_weight: 1 float per cell  
    # irradiance_mean_lum: 1 float per cell
    # irradiance_variance: 1 float per cell
    # grid_mean_distance: 1 float per cell
    # grid_update_weight_tmp: 1 float per cell
    
    total_floats_per_cell = 3 + 1 + 1 + 1 + 1 + 1  # 8 floats per cell
    bytes_per_float = 4
    total_cells = np.prod(grid_resolution)
    
    total_bytes = total_cells * total_floats_per_cell * bytes_per_float
    total_mb = total_bytes / (1024 * 1024)
    
    return total_mb

def test_grid_performance(grid_resolution):
    """测试指定网格分辨率的性能"""
    print(f"\n=== 测试网格分辨率: {grid_resolution} ===")
    
    # 临时修改配置
    import experiment_config as cfg
    original_resolution = cfg.GRID_RESOLUTION
    cfg.GRID_RESOLUTION = grid_resolution
    
    try:
        # 重新初始化场景和相机
        world, cam = main.setup_scene(SCENE_MODE)
        cam.scene_mode = SCENE_MODE
        
        # 计算显存占用
        memory_usage = calculate_total_grid_memory(grid_resolution)
        print(f"网格显存占用: {memory_usage:.2f} MB")
        
        # 预热
        ti.sync()
        cam.adapt_grid_to_scene(spheres, verbose=False)
        cam.update_grid(world, 0.01)
        cam.render(world, 2)  # Hybrid mode
        ti.sync()
        
        # 性能测试
        grid_update_times = []
        render_times = []
        total_times = []
        
        for frame in range(TEST_FRAMES):
            # 测试网格更新耗时
            start_time = time.perf_counter()
            cam.update_grid(world, 0.01)
            ti.sync()
            grid_update_time = time.perf_counter() - start_time
            grid_update_times.append(grid_update_time * 1000)  # 转换为ms
            
            # 测试渲染耗时
            start_time = time.perf_counter()
            cam.render(world, 2)  # Hybrid mode
            ti.sync()
            render_time = time.perf_counter() - start_time
            render_times.append(render_time * 1000)  # 转换为ms
            
            total_times.append(grid_update_times[-1] + render_times[-1])
        
        # 统计结果
        avg_grid_update = np.mean(grid_update_times)
        avg_render = np.mean(render_times)
        avg_total = np.mean(total_times)
        
        print(f"平均网格更新时间: {avg_grid_update:.2f} ms")
        print(f"平均渲染时间: {avg_render:.2f} ms") 
        print(f"平均总时间: {avg_total:.2f} ms")
        print(f"等效FPS: {1000.0 / avg_total:.1f}")
        
        return {
            'grid_resolution': grid_resolution,
            'grid_cells': np.prod(grid_resolution),
            'memory_mb': memory_usage,
            'avg_grid_update_ms': avg_grid_update,
            'avg_render_ms': avg_render,
            'avg_total_ms': avg_total,
            'estimated_fps': 1000.0 / avg_total
        }
        
    finally:
        # 恢复原始配置
        cfg.GRID_RESOLUTION = original_resolution

def run_grid_resolution_analysis():
    """运行网格分辨率分析"""
    print("开始网格分辨率性能分析...")
    print(f"测试分辨率: {GRID_RESOLUTIONS}")
    print(f"每个分辨率测试帧数: {TEST_FRAMES}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"grid_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行测试
    results = []
    for resolution in GRID_RESOLUTIONS:
        try:
            result = test_grid_performance(resolution)
            results.append(result)
        except Exception as e:
            print(f"测试分辨率 {resolution} 时出错: {e}")
            continue
    
    # 保存结果到CSV
    csv_path = os.path.join(output_dir, "grid_resolution_performance.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'grid_resolution', 'grid_cells', 'memory_mb', 
            'avg_grid_update_ms', 'avg_render_ms', 'avg_total_ms', 'estimated_fps'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n结果已保存到: {csv_path}")
    
    # 打印对比表
    print("\n=== 网格分辨率性能对比 ===")
    print(f"{'分辨率':<12} {'网格单元':<10} {'显存(MB)':<10} {'网格更新(ms)':<14} {'总时间(ms)':<12} {'FPS':<8}")
    print("-" * 70)
    for r in results:
        resolution_str = f"{r['grid_resolution'][0]}^3"
        print(f"{resolution_str:<12} {r['grid_cells']:<10} {r['memory_mb']:<10.2f} "
              f"{r['avg_grid_update_ms']:<14.2f} {r['avg_total_ms']:<12.2f} {r['estimated_fps']:<8.1f}")
    
    return results, csv_path

def generate_tradeoff_data():
    """生成画质vs性能权衡曲线数据"""
    print("\n=== 生成画质vs性能权衡数据 ===")
    
    # 这里需要实际的渲染质量指标，暂时使用模拟数据
    # 在实际应用中，应该使用MSE或其他质量指标
    results, csv_path = run_grid_resolution_analysis()
    
    # 添加质量指标（这里用网格密度作为代理指标）
    # 实际应用中应该通过与ground truth的MSE来计算
    for result in results:
        # 模拟：网格越密集，质量越好，但收益递减
        grid_density = result['grid_cells'] ** (1/3) / 16.0  # 相对于16^3的密度
        result['quality_score'] = 1.0 - np.exp(-0.5 * (grid_density - 1.0))
        result['memory_efficiency'] = result['quality_score'] / result['memory_mb']
        result['performance_efficiency'] = result['quality_score'] / result['avg_total_ms']
    
    # 保存扩展结果
    output_dir = os.path.dirname(csv_path)
    tradeoff_csv_path = os.path.join(output_dir, "grid_tradeoff_analysis.csv")
    
    with open(tradeoff_csv_path, 'w', newline='') as f:
        fieldnames = [
            'grid_resolution', 'grid_cells', 'memory_mb', 
            'avg_total_ms', 'estimated_fps', 'quality_score',
            'memory_efficiency', 'performance_efficiency'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"权衡分析数据已保存到: {tradeoff_csv_path}")
    
    # 找出最优权衡点
    best_memory_efficiency = max(results, key=lambda x: x['memory_efficiency'])
    best_performance_efficiency = max(results, key=lambda x: x['performance_efficiency'])
    
    print(f"\n=== 最优权衡点分析 ===")
    print(f"最佳内存效率: {best_memory_efficiency['grid_resolution']} (效率: {best_memory_efficiency['memory_efficiency']:.6f})")
    print(f"最佳性能效率: {best_performance_efficiency['grid_resolution']} (效率: {best_performance_efficiency['performance_efficiency']:.6f})")
    
    return results, tradeoff_csv_path

if __name__ == "__main__":
    try:
        results, csv_path = run_grid_resolution_analysis()
        tradeoff_results, tradeoff_csv_path = generate_tradeoff_data()
        print(f"\n分析完成！结果文件:")
        print(f"- 性能数据: {csv_path}")
        print(f"- 权衡分析: {tradeoff_csv_path}")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()