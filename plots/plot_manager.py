"""
统一绘图管理器 (Unified Plot Manager)
整合所有绘图功能，提供统一的接口
"""

import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# 导入基础绘图工具
from .plotting_tools import (
    setup_mpl_style,
    plot_convergence_scatter,
    plot_mse_curves,
    plot_mse_with_heatmap_markers,
    plot_tradeoff_curves,
    create_heatmap_collage,
    create_comparison_triple,
    plot_fps_comparison,
    save_results_summary,
    load_csv_data
)

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_results_dir(results_root: str = "results", pattern: str = "benchmark_results_*"):
    """查找最新的结果目录"""
    pattern_path = os.path.join(results_root, pattern)
    dirs = glob.glob(pattern_path)
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def read_benchmark_csv(csv_path: str):
    """读取benchmark CSV文件"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    rows = []
    with open(csv_path, "r", newline="", encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "frame": int(r["frame"]),
                "mode": r["mode"],
                "fps": float(r["fps"]),
                "mse": float(r["mse"]),
                "timestamp": r.get("timestamp", ""),
                "gpu_time_ms": float(r.get("gpu_time_ms", 0.0)),
            })
    return rows


def read_benchmark_data(results_dir: str):
    """读取benchmark_results.csv并按mode分组"""
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    try:
        rows = read_benchmark_csv(csv_path)
    except FileNotFoundError:
        print(f"benchmark_results.csv not found: {csv_path}")
        return None

    data = {}
    for r in rows:
        mode = r["mode"]
        if mode not in data:
            data[mode] = {"frame": [], "fps": [], "mse": [], "timestamp": [], "gpu_time_ms": []}
        data[mode]["frame"].append(r["frame"])
        data[mode]["fps"].append(r["fps"])
        data[mode]["mse"].append(r["mse"])
        data[mode]["timestamp"].append(r.get("timestamp", ""))
        data[mode]["gpu_time_ms"].append(r.get("gpu_time_ms", 0.0))

    # 确保按frame排序
    for mode, values in data.items():
        if values["frame"]:
            order = np.argsort(np.array(values["frame"], dtype=np.int64))
            for k in ("frame", "fps", "mse", "timestamp", "gpu_time_ms"):
                values[k] = [values[k][i] for i in order]

    return data


def plot_mse_over_time(rows, out_dir: str, title: str = "MSE over time (log scale)"):
    """绘制MSE随时间变化的曲线"""
    modes = sorted({r["mode"] for r in rows})
    colors = {
        "Path Tracing": "#1f77b4",
        "Pure Grid": "#ff7f0e",
        "Hybrid": "#2ca02c",
    }

    plt.figure(figsize=(12, 6))

    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        frames = np.array([r["frame"] for r in mode_rows], dtype=np.int64)
        mse = np.array([r["mse"] for r in mode_rows], dtype=np.float64)

        mask = mse > 0
        if np.any(mask):
            plt.semilogy(
                frames[mask],
                mse[mask],
                label=mode,
                linewidth=2,
                alpha=0.85,
                color=colors.get(mode, None),
            )

    plt.xlabel("Frame")
    plt.ylabel("MSE (log scale)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "mse_over_time.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close()


def plot_detailed_mse_analysis(data, output_dir):
    """创建详细的MSE分析图（4个子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {
        'Path Tracing': '#1f77b4',
        'Pure Grid': '#ff7f0e', 
        'Hybrid': '#2ca02c'
    }
    
    # Plot 1: 移动前MSE
    ax = axes[0, 0]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            mse_values = np.array(values['mse'])
            
            pre_mask = frames < 150
            if np.any(pre_mask):
                mse_pre = mse_values[pre_mask]
                non_zero_mask = mse_pre > 0
                if np.any(non_zero_mask):
                    ax.semilogy(frames[pre_mask][non_zero_mask], mse_pre[non_zero_mask],
                              color=colors[mode], label=mode, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE Before Displacement (Frames 0-149)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 移动后MSE
    ax = axes[0, 1]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            mse_values = np.array(values['mse'])
            
            post_mask = frames >= 150
            if np.any(post_mask):
                mse_post = mse_values[post_mask]
                non_zero_mask = mse_post > 0
                if np.any(non_zero_mask):
                    ax.semilogy(frames[post_mask][non_zero_mask], mse_post[non_zero_mask],
                              color=colors[mode], label=mode, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE After Displacement (Frames 150+)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: 收敛趋势
    ax = axes[1, 0]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            mse_values = np.array(values['mse'])
            
            window_size = 20
            if len(mse_values) >= window_size:
                moving_avg = np.convolve(mse_values, np.ones(window_size)/window_size, mode='valid')
                moving_frames = frames[window_size-1:]
                
                non_zero_mask = moving_avg > 0
                if np.any(non_zero_mask):
                    ax.semilogy(moving_frames[non_zero_mask], moving_avg[non_zero_mask],
                              color=colors[mode], label=f'{mode} (moving avg)', linewidth=2)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE Convergence Trend (Moving Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: FPS对比
    ax = axes[1, 1]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            fps_values = np.array(values['fps'])
            
            non_zero_mask = fps_values > 0
            if np.any(non_zero_mask):
                ax.plot(frames[non_zero_mask], fps_values[non_zero_mask],
                       color=colors[mode], label=mode, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('FPS')
    ax.set_title('Performance Comparison (FPS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "detailed_mse_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed MSE analysis saved to: {output_path}")
    plt.close()


def generate_summary_report(data, output_dir):
    """生成文本摘要报告"""
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Raytracing Benchmark Summary Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for mode, values in data.items():
            if values['frame']:
                f.write(f"{mode} Mode:\n")
                f.write(f"  Total frames: {len(values['frame'])}\n")
                
                # MSE statistics
                mse_values = np.array(values['mse'])
                non_zero_mse = mse_values[mse_values > 0]
                if len(non_zero_mse) > 0:
                    f.write(f"  MSE - Min: {non_zero_mse.min():.6f}\n")
                    f.write(f"  MSE - Max: {non_zero_mse.max():.6f}\n")
                    f.write(f"  MSE - Mean: {non_zero_mse.mean():.6f}\n")
                    f.write(f"  MSE - Median: {np.median(non_zero_mse):.6f}\n")
                
                # FPS statistics
                fps_values = np.array(values['fps'])
                non_zero_fps = fps_values[fps_values > 0]
                if len(non_zero_fps) > 0:
                    f.write(f"  FPS - Min: {non_zero_fps.min():.1f}\n")
                    f.write(f"  FPS - Max: {non_zero_fps.max():.1f}\n")
                    f.write(f"  FPS - Mean: {non_zero_fps.mean():.1f}\n")
                    f.write(f"  FPS - Median: {np.median(non_zero_fps):.1f}\n")
                
                f.write("\n")
    
    print(f"Summary report saved to: {report_path}")


def plot_grid_performance_tradeoff(csv_path):
    """绘制网格分辨率性能权衡曲线"""
    # 读取CSV数据
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    output_dir = os.path.dirname(csv_path)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 准备数据
    grid_cells = [float(d['grid_cells']) for d in data]
    memory_mb = [float(d['memory_mb']) for d in data]
    avg_total_ms = [float(d['avg_total_ms']) for d in data]
    estimated_fps = [float(d['estimated_fps']) for d in data]
    
    # 子图1: 网格单元数 vs 内存占用
    ax1.scatter(grid_cells, memory_mb, s=100, alpha=0.7, c='blue')
    ax1.set_xlabel('网格单元总数')
    ax1.set_ylabel('显存占用 (MB)')
    ax1.set_title('网格规模 vs 显存占用')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    for d in data:
        resolution = d['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')
        ax1.annotate(resolution, (float(d['grid_cells']), float(d['memory_mb'])), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 子图2: 网格单元数 vs 渲染时间
    ax2.scatter(grid_cells, avg_total_ms, s=100, alpha=0.7, c='red')
    ax2.set_xlabel('网格单元总数')
    ax2.set_ylabel('平均渲染时间 (ms)')
    ax2.set_title('网格规模 vs 渲染性能')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    for d in data:
        resolution = d['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')
        ax2.annotate(resolution, (float(d['grid_cells']), float(d['avg_total_ms'])), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 子图3: 内存占用 vs FPS
    ax3.scatter(memory_mb, estimated_fps, s=100, alpha=0.7, c='green')
    ax3.set_xlabel('显存占用 (MB)')
    ax3.set_ylabel('估计 FPS')
    ax3.set_title('内存占用 vs 帧率')
    
    for d in data:
        resolution = d['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')
        ax3.annotate(resolution, (float(d['memory_mb']), float(d['estimated_fps'])), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    performance_plot_path = os.path.join(output_dir, 'grid_performance_tradeoff.png')
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grid performance tradeoff saved to: {performance_plot_path}")
    return performance_plot_path


def plot_benchmark_gpu_time(csv_path):
    """绘制基准测试中的GPU耗时数据"""
    try:
        # 读取CSV数据
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data or 'gpu_time_ms' not in data[0]:
            print("警告: CSV中未找到gpu_time_ms列")
            return None
            
        output_dir = os.path.dirname(csv_path)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 收集所有模式
        modes = list(set(d['mode'] for d in data))
        
        # 子图1: GPU耗时随时间变化
        for mode in modes:
            mode_data = [d for d in data if d['mode'] == mode]
            frames = [float(d['frame']) for d in mode_data]
            gpu_times = [float(d['gpu_time_ms']) for d in mode_data if d['gpu_time_ms']]
            
            if frames and gpu_times:
                ax1.plot(frames, gpu_times, 
                        label=mode, alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('帧数')
        ax1.set_ylabel('GPU耗时 (ms)')
        ax1.set_title('各渲染模式GPU耗时对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: GPU耗时分布（箱线图）
        gpu_data_by_mode = {}
        for mode in modes:
            gpu_times = [float(d['gpu_time_ms']) for d in data if d['mode'] == mode and d['gpu_time_ms']]
            gpu_data_by_mode[mode] = gpu_times
        
        ax2.boxplot(gpu_data_by_mode.values(), labels=gpu_data_by_mode.keys())
        ax2.set_ylabel('GPU耗时 (ms)')
        ax2.set_title('GPU耗时分布统计')
        
        plt.tight_layout()
        gpu_time_plot_path = os.path.join(output_dir, 'gpu_time_analysis.png')
        plt.savefig(gpu_time_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GPU time analysis saved to: {gpu_time_plot_path}")
        
        return gpu_time_plot_path
        
    except Exception as e:
        print(f"绘制GPU耗时图时出错: {e}")
        return None


def generate_all_plots(results_dir: str = None, plots_dir: str = None):
    """
    生成所有图表
    
    Args:
        results_dir: 结果目录路径（默认为最新）
        plots_dir: 图表输出目录（默认为results_dir/plots）
    """
    # 查找结果目录
    if results_dir is None:
        results_dir = find_latest_results_dir()
        if not results_dir:
            print("未找到结果目录")
            return
    
    # 设置输出目录
    if plots_dir is None:
        plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"使用结果目录: {results_dir}")
    print(f"输出图表到: {plots_dir}\n")
    
    # 1. 读取benchmark数据
    data = read_benchmark_data(results_dir)
    if data:
        print("生成MSE对比图...")
        plot_mse_comparison(data, plots_dir)
        
        print("生成详细MSE分析...")
        plot_detailed_mse_analysis(data, plots_dir)
        
        print("生成摘要报告...")
        generate_summary_report(data, plots_dir)
        
        print("生成GPU耗时分析...")
        if os.path.join(results_dir, "benchmark_results.csv"):
            plot_benchmark_gpu_time(os.path.join(results_dir, "benchmark_results.csv"))
    
    # 2. 检查网格分辨率数据
    grid_csv = os.path.join(results_dir, "grid_resolution_performance.csv")
    if os.path.exists(grid_csv):
        print("生成网格分辨率权衡曲线...")
        plot_grid_performance_tradeoff(grid_csv)
    
    # 3. 检查收敛速度数据
    convergence_csv = os.path.join(results_dir, "convergence_summary.csv")
    if os.path.exists(convergence_csv):
        print("生成收敛速度散点图...")
        # 使用plotting_tools中的函数
        pass
    
    print(f"\n所有图表已保存到: {plots_dir}")


def plot_mse_comparison(data, output_dir: str):
    """MSE对比图兼容包装函数"""
    rows = []
    for mode, values in data.items():
        for i, frame in enumerate(values.get("frame", [])):
            rows.append({
                "frame": int(frame),
                "mode": mode,
                "fps": float(values["fps"][i]) if i < len(values.get("fps", [])) else 0.0,
                "mse": float(values["mse"][i]) if i < len(values.get("mse", [])) else 0.0,
                "timestamp": values.get("timestamp", [""])[i] if i < len(values.get("timestamp", [])) else "",
            })

    plot_mse_over_time(rows, output_dir, title="MSE over time (log scale)")


if __name__ == "__main__":
    # 命令行调用
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None, help="结果目录路径")
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir)
