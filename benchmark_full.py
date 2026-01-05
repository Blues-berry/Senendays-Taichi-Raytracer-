"""
多场景基准测试脚本
依次运行所有场景，每个场景用三种方法测试，完成后生成误差热力图

运行顺序：
1. Path Tracing (参考)
2. Pure Grid
3. Hybrid Adaptive
4. Error Heatmap (生成热力图)
"""

import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
import main
from main import spheres, materials
import utils

from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 初始化 Taichi
ti.init(arch=ti.gpu, random_seed=42)

# 渲染模式
RENDER_MODE_PT = 0      # Path Tracing (Ground Truth)
RENDER_MODE_GRID = 1    # Pure Grid
RENDER_MODE_HYBRID = 2  # Hybrid Adaptive
RENDER_MODE_ERROR = 3   # Error Heatmap

# 所有场景列表
ALL_SCENES = [
    'cornell_box',
    'two_room',
    'night_scene',
    'random',
    'classroom',
    'bathroom',
    'veach_mis',
]

# 测试配置
TEST_FRAMES_PER_MODE = 200    # 每种模式测试的帧数
PT_REFERENCE_FRAMES = 150       # PT 参考帧数
MOVEMENT_FRAME = 100           # 物体移动帧
ERROR_HEATMAP_SPP = 512       # 误差热力图 PT 参考 SPP

# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "results"
output_dir = os.path.join(results_dir, f"multi_scene_benchmark_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")


def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def save_screenshot(frame_buffer, filename):
    """Save a screenshot"""
    filepath = os.path.join(output_dir, filename)
    ti.tools.imwrite(frame_buffer, filepath)
    log_message(f"Saved screenshot: {filepath}")


def calculate_mse(current_linear, reference_linear):
    """Calculate MSE in linear space"""
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)

    # Normalize to [0, 1] range
    if curr_f.max() > 255.0:
        curr_f = curr_f / 255.0
    if ref_f.max() > 255.0:
        ref_f = ref_f / 255.0

    # Handle NaN and Inf values
    curr_f = np.nan_to_num(curr_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_f = np.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate MSE
    diff = curr_f - ref_f
    mse = np.mean(diff ** 2)
    return float(mse)


def run_scene_benchmark(scene_name: str):
    """运行单个场景的完整基准测试（三种模式 + Error 模式）"""
    log_message(f"\n{'='*60}")
    log_message(f"Testing scene: {scene_name}")
    log_message(f"{'='*60}\n")

    # 创建场景
    world, cam = main.setup_scene(scene_name)
    cam.scene_mode = scene_name

    # 初始化帧缓冲区
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)

    # CSV 数据
    csv_data = []

    # PT 参考（线性空间）
    pt_reference_linear = None

    # 运行顺序: PT -> Grid -> Hybrid -> Error
    modes = [
        (RENDER_MODE_PT, "Path_Tracing"),
        (RENDER_MODE_GRID, "Pure_Grid"),
        (RENDER_MODE_HYBRID, "Hybrid"),
        (RENDER_MODE_ERROR, "Error_Heatmap"),
    ]

    for mode_idx, (mode, mode_name) in enumerate(modes):
        log_message(f"\n--- Mode {mode_idx+1}/4: {mode_name} ---")

        # 重置帧缓冲区
        current_frame.fill(0.001)

        # 适应网格
        cam.adapt_grid_to_scene(main.spheres, verbose=False)

        # 预热
        ti.sync()
        if mode == RENDER_MODE_GRID or mode == RENDER_MODE_HYBRID:
            cam.update_grid(world, 0.01)
        cam.render(world, mode)
        if mode == RENDER_MODE_HYBRID:
            cam.asvgf_filter()
        ti.sync()

        # 模式特定的初始化
        if mode == RENDER_MODE_PT:
            # 累积 PT 参考帧
            log_message("Accumulating PT reference...")
            pt_accum = np.zeros((*cam.img_res, 3), dtype=np.float32)
            for _ in range(PT_REFERENCE_FRAMES):
                cam.render_pt(world)
                ti.sync()
                pt_accum += cam.pt_frame.to_numpy().astype(np.float32)
            pt_reference_linear = pt_accum / PT_REFERENCE_FRAMES
            log_message(f"PT reference accumulated ({PT_REFERENCE_FRAMES} frames)")

        elif mode == RENDER_MODE_ERROR:
            # 进入 Error 模式，需要 PT 参考
            if pt_reference_linear is None:
                log_message("ERROR: No PT reference available!")
                continue

            # 先渲染一帧 Hybrid 结果
            cam.update_grid(world, 0.01)
            cam.render(world, RENDER_MODE_HYBRID)
            cam.asvgf_filter()
            ti.sync()

            # 生成误差热力图
            log_message("Generating error heatmap...")
            cam.render_error_heatmap()

            # 保存误差热力图
            heatmap_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
            @ti.kernel
            def copy_heatmap(dst: ti.template(), src: ti.template()):
                for i, j in src:
                    dst[i, j] = src[i, j]
            copy_heatmap(heatmap_buffer, cam.frame)
            save_screenshot(heatmap_buffer, f"{scene_name}_error_heatmap.png")

            # Error 模式只运行一帧，跳过后续循环
            continue

        # 主测试循环
        for frame in range(TEST_FRAMES_PER_MODE):
            # 物体移动（仅在第 100 帧）
            if frame == MOVEMENT_FRAME and len(main.spheres) > 0:
                # 移动最后一个大球
                big_sphere_index = len(main.spheres) - 1
                old_x = main.spheres[big_sphere_index].center[0]
                main.spheres[big_sphere_index].center[0] = old_x + 1.0
                log_message(f"Object movement at frame {frame}: x {old_x:.2f} -> {main.spheres[big_sphere_index].center[0]:.2f}")

                # 重新适应网格
                if mode == RENDER_MODE_GRID or mode == RENDER_MODE_HYBRID:
                    cam.adapt_grid_to_scene(main.spheres, verbose=False)

                # 重置帧缓冲区（观察收敛）
                current_frame.fill(0.001)

            # 渲染帧
            ti.sync()
            start_time = time.perf_counter()

            if mode == RENDER_MODE_PT:
                cam.render(world, mode)
            elif mode == RENDER_MODE_GRID:
                cam.update_grid(world, 0.01)
                cam.render(world, mode)
            elif mode == RENDER_MODE_HYBRID:
                cam.update_grid(world, 0.01)
                cam.render(world, mode)
                cam.asvgf_filter()

            ti.sync()
            frame_time = time.perf_counter() - start_time

            # 计算 FPS
            fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0

            # 计算 MSE（相对于 PT 参考）
            mse = 0.0
            if pt_reference_linear is not None and mode != RENDER_MODE_PT:
                current_linear = cam.frame.to_numpy()
                mse = calculate_mse(current_linear, pt_reference_linear)

            # 记录 CSV 数据
            csv_data.append({
                'scene': scene_name,
                'mode': mode_name,
                'frame': frame + 1,
                'fps': fps,
                'mse': mse,
                'gpu_time_ms': frame_time * 1000.0,
                'timestamp': datetime.now().isoformat(),
                'grid_memory_mb': float(cam.grid_res[0] * cam.grid_res[1] * cam.grid_res[2] * 3 * 4 / (1024.0 * 1024.0)),
            })

            # 保存关键帧截图
            if frame in [5, 50, 100, 150, 199]:
                save_screenshot(current_frame, f"{scene_name}_{mode_name}_frame_{frame+1}.png")

            # 更新显示
            weight = 1.0 / (frame + 2)
            @ti.kernel
            def average_frames(dst: ti.template(), src: ti.template(), w: float):
                for i, j in src:
                    dst[i, j] = (1.0 - w) * dst[i, j] + w * src[i, j]
            average_frames(current_frame, cam.frame, weight)

            # 每 20 帧打印进度
            if (frame + 1) % 20 == 0:
                log_message(f"  Frame {frame+1}/{TEST_FRAMES_PER_MODE}: FPS={fps:.1f}, MSE={mse:.6e}")

        # 保存该模式的最终结果
        log_message(f"  {mode_name} completed")
        save_screenshot(current_frame, f"{scene_name}_{mode_name}_result.png")

    # 保存该场景的 CSV 数据
    csv_path = os.path.join(output_dir, f"{scene_name}_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['scene', 'mode', 'frame', 'fps', 'mse', 'gpu_time_ms', 'timestamp', 'grid_memory_mb']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    log_message(f"Saved CSV: {csv_path}")

    return csv_data


def run_all_scenes():
    """运行所有场景的基准测试"""
    log_message("="*60)
    log_message("Multi-Scene Benchmark Starting")
    log_message("="*60)

    all_data = {}

    # 依次运行每个场景
    for scene_idx, scene_name in enumerate(ALL_SCENES):
        log_message(f"\n\n>>> SCENE {scene_idx+1}/{len(ALL_SCENES)}: {scene_name}")

        try:
            scene_data = run_scene_benchmark(scene_name)
            all_data[scene_name] = scene_data
        except Exception as e:
            log_message(f"ERROR in scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()

    # 生成汇总 CSV
    log_message("\n\nGenerating summary CSV...")
    summary_csv_path = os.path.join(output_dir, "all_scenes_summary.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['scene', 'mode', 'avg_fps', 'avg_mse', 'grid_memory_mb']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scene_name, data in all_data.items():
            # 按模式分组统计
            modes = {}
            for row in data:
                mode = row['mode']
                if mode not in modes:
                    modes[mode] = {'fps': [], 'mse': [], 'memory': row['grid_memory_mb']}
                modes[mode]['fps'].append(row['fps'])
                modes[mode]['mse'].append(row['mse'])

            # 写入每种模式的统计
            for mode_name, mode_data in modes.items():
                avg_fps = np.mean(mode_data['fps'])
                avg_mse = np.mean(mode_data['mse'])
                writer.writerow({
                    'scene': scene_name,
                    'mode': mode_name,
                    'avg_fps': avg_fps,
                    'avg_mse': avg_mse,
                    'grid_memory_mb': mode_data['memory'],
                })

    log_message(f"Saved summary CSV: {summary_csv_path}")

    # 自动生成图表
    log_message("\n\nAuto-generating analysis plots...")
    auto_generate_plots(all_data)

    log_message("\n\n" + "="*60)
    log_message("Benchmark completed successfully!")
    log_message("="*60)


def auto_generate_plots(all_data: Dict[str, List[Dict]]):
    """自动生成所有分析图表"""
    try:
        # 1. MSE 对比图（所有场景）
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (scene_name, data) in enumerate(all_data.items()):
            ax = axes[idx]

            # 按模式分组
            modes = {}
            for row in data:
                mode = row['mode']
                if mode not in modes:
                    modes[mode] = {'frame': [], 'mse': []}
                modes[mode]['frame'].append(row['frame'])
                modes[mode]['mse'].append(row['mse'])

            # 绘制每种模式的 MSE 曲线
            for mode_name, mode_data in modes.items():
                if mode_name == 'Path_Tracing':
                    continue  # PT 参考不绘制
                ax.plot(mode_data['frame'], mode_data['mse'], label=mode_name, linewidth=1.5)

            ax.set_yscale('log')
            ax.set_xlabel('Frame')
            ax.set_ylabel('MSE (log scale)')
            ax.set_title(scene_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        mse_comparison_path = os.path.join(output_dir, 'all_scenes_mse_comparison.png')
        plt.savefig(mse_comparison_path, dpi=200)
        plt.close()
        log_message(f"Saved MSE comparison: {mse_comparison_path}")

        # 2. FPS 对比图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, (scene_name, data) in enumerate(all_data.items()):
            ax = axes[idx]

            # 按模式分组
            modes = {}
            for row in data:
                mode = row['mode']
                if mode not in modes:
                    modes[mode] = {'fps': []}
                modes[mode]['fps'].append(row['fps'])

            # 计算平均 FPS
            avg_fps = {mode: np.mean(data['fps']) for mode, data in modes.items()}

            # 柱状图
            x_pos = np.arange(len(avg_fps))
            ax.bar(x_pos, avg_fps.values(), width=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(avg_fps.keys(), rotation=45, ha='right')
            ax.set_ylabel('FPS')
            ax.set_title(scene_name)
            ax.grid(True, alpha=0.3, axis='y')

        # 删除最后一个子图（如果只有7个场景）
        if len(all_data) < 8:
            axes[7].remove()

        plt.tight_layout()
        fps_comparison_path = os.path.join(output_dir, 'all_scenes_fps_comparison.png')
        plt.savefig(fps_comparison_path, dpi=200)
        plt.close()
        log_message(f"Saved FPS comparison: {fps_comparison_path}")

        # 3. 性能提升比图
        performance_data = []
        for scene_name, data in all_data.items():
            # 按模式分组
            modes = {}
            for row in data:
                mode = row['mode']
                if mode not in modes:
                    modes[mode] = {'fps': []}
                modes[mode]['fps'].append(row['fps'])

            # 计算 Grid 和 Hybrid 相对于 PT 的性能提升
            pt_fps = np.mean(modes.get('Path_Tracing', {'fps': [1.0]})['fps'])
            grid_fps = np.mean(modes.get('Pure_Grid', {'fps': [1.0]})['fps'])
            hybrid_fps = np.mean(modes.get('Hybrid', {'fps': [1.0]})['fps'])

            performance_data.append({
                'scene': scene_name,
                'pt_fps': pt_fps,
                'grid_fps': grid_fps,
                'grid_speedup': grid_fps / pt_fps if pt_fps > 0 else 0,
                'hybrid_fps': hybrid_fps,
                'hybrid_speedup': hybrid_fps / pt_fps if pt_fps > 0 else 0,
            })

        # 绘制性能提升比
        fig, ax = plt.subplots(figsize=(14, 6))
        scenes = [d['scene'] for d in performance_data]
        grid_speedups = [d['grid_speedup'] for d in performance_data]
        hybrid_speedups = [d['hybrid_speedup'] for d in performance_data]

        x = np.arange(len(scenes))
        width = 0.35

        ax.bar(x - width/2, grid_speedups, width, label='Pure Grid (vs PT)', color='steelblue')
        ax.bar(x + width/2, hybrid_speedups, width, label='Hybrid (vs PT)', color='coral')

        ax.set_xlabel('Scene')
        ax.set_ylabel('Speedup (x)')
        ax.set_title('Performance Speedup vs Path Tracing')
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        speedup_path = os.path.join(output_dir, 'all_scenes_speedup_comparison.png')
        plt.savefig(speedup_path, dpi=200)
        plt.close()
        log_message(f"Saved speedup comparison: {speedup_path}")

        # 4. 汇总报告
        summary_path = os.path.join(output_dir, 'benchmark_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Multi-Scene Benchmark Summary\n")
            f.write("="*60 + "\n\n")

            for data in performance_data:
                f.write(f"\n{data['scene']}:\n")
                f.write(f"  Path Tracing FPS: {data['pt_fps']:.1f}\n")
                f.write(f"  Pure Grid FPS: {data['grid_fps']:.1f} ({data['grid_speedup']:.1f}x speedup)\n")
                f.write(f"  Hybrid FPS: {data['hybrid_fps']:.1f} ({data['hybrid_speedup']:.1f}x speedup)\n")

        log_message(f"Saved summary report: {summary_path}")

    except Exception as e:
        log_message(f"WARNING: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    log_message("Starting multi-scene benchmark...")
    try:
        run_all_scenes()
    except KeyboardInterrupt:
        log_message("Benchmark interrupted by user")
    except Exception as e:
        log_message(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
