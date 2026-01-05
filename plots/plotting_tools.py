"""
统一绘图工具模块 (Unified Plotting Tools)
整合所有绘图和数据分析功能
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import os
from datetime import datetime


def setup_mpl_style():
    """设置Matplotlib样式，适合论文使用"""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
    })


def plot_convergence_scatter(results, output_path, title="Convergence Speed Comparison"):
    """
    绘制收敛速度散点图
    
    Args:
        results: list of dict, each containing:
            - strategy: 策略名称
            - converged_frame: 收敛帧数
            - total_gpu_time_ms: 总GPU时间(ms)
            - color: 颜色
            - description: 描述
        output_path: 输出路径
        title: 图表标题
    """
    setup_mpl_style()
    
    plt.figure(figsize=(10, 7))
    
    for result in results:
        plt.scatter(
            result['total_gpu_time_ms'],
            result['converged_frame'],
            s=300,
            c=result['color'],
            alpha=0.7,
            edgecolors='black',
            linewidths=2,
            label=result['strategy']
        )
        
        # 添加标签
        label_text = f"{result['converged_frame']} frames"
        plt.annotate(
            label_text,
            (result['total_gpu_time_ms'], result['converged_frame']),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9
        )
    
    plt.xlabel('Total GPU Time (ms)', fontweight='bold')
    plt.ylabel('Frames to Convergence', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mse_curves(results, output_path, title="MSE Convergence Curves"):
    """
    绘制MSE收敛曲线
    
    Args:
        results: list of dict, each containing:
            - strategy: 策略名称
            - mse_history: MSE历史列表
            - color: 颜色
        output_path: 输出路径
        title: 图表标题
    """
    setup_mpl_style()
    
    plt.figure(figsize=(10, 6))
    
    for result in results:
        if result['mse_history']:
            frames = range(1, len(result['mse_history']) + 1)
            plt.plot(
                frames,
                result['mse_history'],
                label=result['strategy'],
                color=result['color'],
                linewidth=2,
                marker='o',
                markersize=3
            )
    
    plt.xlabel('Frame after movement', fontweight='bold')
    plt.ylabel('MSE vs PT Reference', fontweight='bold')
    plt.yscale('log')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', which='both')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mse_with_heatmap_markers(mse_history, heatmap_frames, output_path, title="MSE Convergence"):
    """
    绘制MSE曲线，并标注热力图生成帧
    
    Args:
        mse_history: MSE历史列表
        heatmap_frames: 热力图生成帧列表
        output_path: 输出路径
        title: 图表标题
    """
    setup_mpl_style()
    
    plt.figure(figsize=(8, 5))
    frames = range(1, len(mse_history) + 1)
    
    plt.plot(frames, mse_history, 'r-', linewidth=2, marker='o', markersize=4)
    plt.yscale('log')
    plt.xlabel('Frame after movement', fontweight='bold')
    plt.ylabel('MSE vs PT Reference', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.grid(True, linestyle='--', which='both')
    
    # 标记热力图生成帧
    for frame in heatmap_frames:
        if frame - 1 < len(mse_history):
            plt.axvline(x=frame, color='blue', linestyle=':', alpha=0.7)
            plt.text(
                frame, 
                mse_history[frame-1], 
                f'  Frame {frame}',
                verticalalignment='center',
                fontsize=9,
                color='blue'
            )
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tradeoff_curves(data, output_path, title="Grid Resolution Trade-off"):
    """
    绘制网格分辨率权衡曲线
    
    Args:
        data: dict with keys:
            - resolutions: 分辨率列表 ['16^3', '32^3', '64^3']
            - memory_mb: 显存占用列表
            - gpu_time_ms: GPU时间列表
            - mse_values: MSE值列表
        output_path: 输出路径
        title: 图表标题
    """
    setup_mpl_style()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    resolutions = data['resolutions']
    x = np.arange(len(resolutions))
    
    # 1. 显存占用
    ax1.bar(x, data['memory_mb'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Grid Resolution')
    ax1.set_ylabel('GPU Memory (MB)')
    ax1.set_title('Memory Usage')
    ax1.set_xticks(x)
    ax1.set_xticklabels(resolutions)
    ax1.grid(True, alpha=0.3)
    
    # 2. GPU时间
    ax2.bar(x, data['gpu_time_ms'], color='coral', alpha=0.7)
    ax2.set_xlabel('Grid Resolution')
    ax2.set_ylabel('GPU Time per Frame (ms)')
    ax2.set_title('Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(resolutions)
    ax2.grid(True, alpha=0.3)
    
    # 3. MSE
    ax3.bar(x, data['mse_values'], color='lightgreen', alpha=0.7)
    ax3.set_xlabel('Grid Resolution')
    ax3.set_ylabel('MSE')
    ax3.set_title('Quality')
    ax3.set_xticks(x)
    ax3.set_xticklabels(resolutions)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.suptitle(title, fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_heatmap_collage(heatmap_paths, output_path, grid_size=(2, 2)):
    """
    创建热力图序列拼图
    
    Args:
        heatmap_paths: 热力图路径列表
        output_path: 输出路径
        grid_size: 网格尺寸 (rows, cols)
    """
    images = []
    for path in heatmap_paths:
        if os.path.exists(path):
            img = Image.open(path)
            images.append(img)
    
    if len(images) != grid_size[0] * grid_size[1]:
        print(f"Warning: Expected {grid_size[0] * grid_size[1]} images, got {len(images)}")
        return None
    
    w, h = images[0].size
    collage = Image.new('RGB', (w * grid_size[1], h * grid_size[0]))
    
    idx = 0
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            pos = (col * w, row * h)
            collage.paste(images[idx], pos)
            idx += 1
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    collage.save(output_path, 'PNG', dpi=(300, 300))
    
    return output_path


def create_comparison_triple(hybrid_path, pt_path, error_path, output_path):
    """
    创建三联对比图：Hybrid + PT + Error
    
    Args:
        hybrid_path: Hybrid渲染路径
        pt_path: PT参考路径
        error_path: 误差热力图路径
        output_path: 输出路径
    """
    hybrid = Image.open(hybrid_path).convert('RGB')
    pt = Image.open(pt_path).convert('RGB')
    error = Image.open(error_path).convert('RGB')
    
    w, h = hybrid.size
    comparison = Image.new('RGB', (w * 3, h))
    
    comparison.paste(hybrid, (0, 0))
    comparison.paste(pt, (w, 0))
    comparison.paste(error, (w * 2, 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison.save(output_path, 'PNG', dpi=(300, 300))
    
    return output_path


def plot_fps_comparison(fps_data, output_path, title="FPS Comparison"):
    """
    绘制FPS对比图
    
    Args:
        fps_data: dict with keys as mode names, values as FPS lists
        output_path: 输出路径
        title: 图表标题
    """
    setup_mpl_style()
    
    plt.figure(figsize=(10, 6))
    
    modes = list(fps_data.keys())
    x = np.arange(len(modes))
    
    for i, mode in enumerate(modes):
        fps_values = fps_data[mode]
        plt.plot(
            range(len(fps_values)),
            fps_values,
            label=mode,
            linewidth=2,
            marker='o',
            markersize=3
        )
    
    plt.xlabel('Frame', fontweight='bold')
    plt.ylabel('FPS', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results_summary(results, output_path):
    """
    保存结果摘要到CSV
    
    Args:
        results: list of dict
        output_path: 输出CSV路径
    """
    import csv
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        if results and len(results) > 0:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def load_csv_data(csv_path):
    """
    从CSV加载数据
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        list of dict
    """
    import csv
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)
