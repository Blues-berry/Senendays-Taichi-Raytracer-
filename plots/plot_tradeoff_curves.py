"""
Trade-off Curve Plotter
绘制画质 vs 性能的权衡曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def plot_grid_performance_tradeoff(csv_path):
    """绘制网格分辨率性能权衡曲线"""
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建输出目录
    output_dir = os.path.dirname(csv_path)
    
    # 1. 网格分辨率 vs 性能
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 子图1: 网格单元数 vs 内存占用
    ax1.scatter(df['grid_cells'], df['memory_mb'], s=100, alpha=0.7, c='blue')
    ax1.set_xlabel('网格单元总数')
    ax1.set_ylabel('显存占用 (MB)')
    ax1.set_title('网格规模 vs 显存占用')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 添加标签
    for i, row in df.iterrows():
        resolution = f"{row['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')}"
        ax1.annotate(resolution, (row['grid_cells'], row['memory_mb']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 子图2: 网格单元数 vs 渲染时间
    ax2.scatter(df['grid_cells'], df['avg_total_ms'], s=100, alpha=0.7, c='red')
    ax2.set_xlabel('网格单元总数')
    ax2.set_ylabel('平均渲染时间 (ms)')
    ax2.set_title('网格规模 vs 渲染性能')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # 添加标签
    for i, row in df.iterrows():
        resolution = f"{row['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')}"
        ax2.annotate(resolution, (row['grid_cells'], row['avg_total_ms']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 子图3: 内存占用 vs FPS
    ax3.scatter(df['memory_mb'], df['estimated_fps'], s=100, alpha=0.7, c='green')
    ax3.set_xlabel('显存占用 (MB)')
    ax3.set_ylabel('估计 FPS')
    ax3.set_title('内存占用 vs 帧率')
    
    # 添加标签
    for i, row in df.iterrows():
        resolution = f"{row['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')}"
        ax3.annotate(resolution, (row['memory_mb'], row['estimated_fps']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    performance_plot_path = os.path.join(output_dir, 'grid_performance_tradeoff.png')
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 权衡分析图 (如果有质量数据)
    if 'quality_score' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 内存效率权衡
        ax1.scatter(df['memory_mb'], df['quality_score'], s=100, alpha=0.7, c='blue')
        ax1.set_xlabel('显存占用 (MB)')
        ax1.set_ylabel('渲染质量得分')
        ax1.set_title('内存-质量权衡曲线')
        
        # 添加标签和效率值
        for i, row in df.iterrows():
            resolution = f"{row['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')}"
            efficiency = row['memory_efficiency']
            ax1.annotate(f"{resolution}\n({efficiency:.4f})", 
                        (row['memory_mb'], row['quality_score']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 性能效率权衡
        ax2.scatter(df['avg_total_ms'], df['quality_score'], s=100, alpha=0.7, c='red')
        ax2.set_xlabel('平均渲染时间 (ms)')
        ax2.set_ylabel('渲染质量得分')
        ax2.set_title('性能-质量权衡曲线')
        
        # 添加标签和效率值
        for i, row in df.iterrows():
            resolution = f"{row['grid_resolution'].replace('(', '').replace(')', '').replace(', ', 'x')}"
            efficiency = row['performance_efficiency']
            ax2.annotate(f"{resolution}\n({efficiency:.4f})", 
                        (row['avg_total_ms'], row['quality_score']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        quality_plot_path = os.path.join(output_dir, 'quality_performance_tradeoff.png')
        plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"质量权衡曲线已保存到: {quality_plot_path}")
    
    # 3. 综合性能雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 标准化数据到0-1范围用于雷达图
    metrics = ['memory_mb', 'avg_total_ms', 'estimated_fps']
    if 'quality_score' in df.columns:
        metrics.append('quality_score')
    
    # 为内存和时间使用倒数（越小越好）
    df_norm = df.copy()
    df_norm['memory_mb'] = 1 - (df['memory_mb'] - df['memory_mb'].min()) / (df['memory_mb'].max() - df['memory_mb'].min())
    df_norm['avg_total_ms'] = 1 - (df['avg_total_ms'] - df['avg_total_ms'].min()) / (df['avg_total_ms'].max() - df['avg_total_ms'].min())
    df_norm['estimated_fps'] = (df['estimated_fps'] - df['estimated_fps'].min()) / (df['estimated_fps'].max() - df['estimated_fps'].min())
    if 'quality_score' in df.columns:
        df_norm['quality_score'] = df['quality_score']  # 已经是0-1范围
    
    # 雷达图设置
    categories = []
    if 'memory_mb' in metrics:
        categories.append('内存效率')
    if 'avg_total_ms' in metrics:
        categories.append('速度效率')
    if 'estimated_fps' in metrics:
        categories.append('帧率性能')
    if 'quality_score' in metrics:
        categories.append('渲染质量')
    
    # 计算每个维度的角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 为每个分辨率绘制雷达图
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = []
        if 'memory_mb' in metrics:
            values.append(df_norm.iloc[i]['memory_mb'])
        if 'avg_total_ms' in metrics:
            values.append(df_norm.iloc[i]['avg_total_ms'])
        if 'estimated_fps' in metrics:
            values.append(df_norm.iloc[i]['estimated_fps'])
        if 'quality_score' in metrics:
            values.append(df_norm.iloc[i]['quality_score'])
        
        values += values[:1]  # 闭合图形
        
        resolution = str(row['grid_resolution']).replace('(', '').replace(')', '').replace(', ', 'x')
        ax.plot(angles, values, 'o-', linewidth=2, label=resolution, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('网格分辨率综合性能雷达图', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    radar_plot_path = os.path.join(output_dir, 'grid_performance_radar.png')
    plt.savefig(radar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"雷达图已保存到: {radar_plot_path}")
    print(f"性能权衡曲线已保存到: {performance_plot_path}")
    
    return performance_plot_path

def plot_benchmark_gpu_time(csv_path):
    """绘制基准测试中的GPU耗时数据"""
    try:
        df = pd.read_csv(csv_path)
        if 'gpu_time_ms' not in df.columns:
            print("警告: CSV中未找到gpu_time_ms列")
            return None
            
        # 创建输出目录
        output_dir = os.path.dirname(csv_path)
        
        # 按模式分组统计GPU耗时
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1: GPU耗时随时间变化
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode]
            ax1.plot(mode_data['frame'], mode_data['gpu_time_ms'], 
                    label=mode, alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('帧数')
        ax1.set_ylabel('GPU耗时 (ms)')
        ax1.set_title('各渲染模式GPU耗时对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: GPU耗时分布
        df.boxplot(column='gpu_time_ms', by='mode', ax=ax2)
        ax2.set_ylabel('GPU耗时 (ms)')
        ax2.set_title('GPU耗时分布统计')
        
        plt.tight_layout()
        gpu_time_plot_path = os.path.join(output_dir, 'gpu_time_analysis.png')
        plt.savefig(gpu_time_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GPU耗时分析图已保存到: {gpu_time_plot_path}")
        
        # 输出统计摘要
        print("\n=== GPU耗时统计摘要 ===")
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode]
            print(f"{mode}:")
            print(f"  平均耗时: {mode_data['gpu_time_ms'].mean():.2f} ms")
            print(f"  最小耗时: {mode_data['gpu_time_ms'].min():.2f} ms")
            print(f"  最大耗时: {mode_data['gpu_time_ms'].max():.2f} ms")
            print(f"  标准差: {mode_data['gpu_time_ms'].std():.2f} ms")
        
        return gpu_time_plot_path
        
    except Exception as e:
        print(f"绘制GPU耗时图时出错: {e}")
        return None

def main():
    """主函数"""
    print("开始绘制权衡曲线...")
    
    # 查找最新的结果目录
    import glob
    result_dirs = glob.glob("grid_analysis_results_*")
    benchmark_dirs = glob.glob("results/benchmark_results_*")
    
    if result_dirs:
        # 绘制网格分辨率权衡曲线
        latest_grid_dir = max(result_dirs, key=os.path.getctime)
        grid_csv = os.path.join(latest_grid_dir, "grid_resolution_performance.csv")
        if os.path.exists(grid_csv):
            plot_grid_performance_tradeoff(grid_csv)
        else:
            grid_csv = os.path.join(latest_grid_dir, "grid_tradeoff_analysis.csv")
            if os.path.exists(grid_csv):
                plot_grid_performance_tradeoff(grid_csv)
    
    if benchmark_dirs:
        # 绘制基准测试GPU耗时
        latest_benchmark_dir = max(benchmark_dirs, key=os.path.getctime)
        benchmark_csv = os.path.join(latest_benchmark_dir, "benchmark_results.csv")
        if os.path.exists(benchmark_csv):
            plot_benchmark_gpu_time(benchmark_csv)
    
    print("绘图完成!")

if __name__ == "__main__":
    main()