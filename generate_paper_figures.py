"""
Paper Figure Generation Script

自动生成论文所需的所有高质量图表。
运行此脚本后，所有图表将保存到 `paper_figures/` 目录。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path
import glob

# 配置
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 100

# 创建输出目录
output_dir = "paper_figures"
Path(output_dir).mkdir(exist_ok=True)

print("="*60)
print("Paper Figure Generator")
print("="*60)
print(f"Output directory: {output_dir}")
print()

# 查找最新实验结果
result_dirs = glob.glob("results/new_methods_benchmark_*")
if not result_dirs:
    print("ERROR: No experiment results found!")
    print("Please run: python experiment_new_methods.py")
    exit(1)

results_dir = sorted(result_dirs)[-1]
print(f"Using results from: {results_dir}")
print()

# 定义模式
modes = ['Grid', 'Hybrid', 'MS_AIC', 'MCTF', 'FULL']
mode_colors = {
    'Grid': '#1f77b4',      # blue
    'Hybrid': '#ff7f0e',    # orange
    'MS_AIC': '#2ca02c',     # green
    'MCTF': '#d62728',      # red
    'FULL': '#9467bd'        # purple
}
mode_labels = {
    'Grid': 'Pure Grid',
    'Hybrid': 'Hybrid (Baseline)',
    'MS_AIC': 'MS-AIC (Ours)',
    'MCTF': 'MCTF (Ours)',
    'FULL': 'Full Method (Ours)'
}

# ==================== 图1: MSE 收敛对比曲线 ====================
print("[1/6] Generating Figure 1: MSE Convergence Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

# 读取数据
for mode in modes:
    csv_path = os.path.join(results_dir, f'cornell_box_{mode}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 平滑曲线（移动平均）
        window = 5
        mse_smooth = df['mse'].rolling(window=window, center=True).mean()

        ax.plot(df['frame'], mse_smooth,
                label=mode_labels[mode],
                color=mode_colors[mode],
                linewidth=2,
                alpha=0.9)

ax.set_yscale('log')
ax.set_xlabel('Frame', fontweight='bold')
ax.set_ylabel('MSE (log scale)', fontweight='bold')
ax.set_title('MSE Convergence Comparison', fontweight='bold', fontsize=13)
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 标记物体移动
ax.axvline(x=200, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Object Movement')
ax.axvspan(200, 210, color='red', alpha=0.1, label='Convergence Phase')

plt.tight_layout()
fig1_path = os.path.join(output_dir, 'fig1_mse_convergence.pdf')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig1_path}")

# ==================== 图2: 性能对比柱状图 ====================
print("[2/6] Generating Figure 2: Performance Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 收集数据
avg_fps = {}
avg_mse = {}
final_mse = {}

for mode in modes:
    csv_path = os.path.join(results_dir, f'cornell_box_{mode}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        avg_fps[mode] = df['fps'].mean()
        avg_mse[mode] = df['mse'].mean()
        final_mse[mode] = df['mse'].iloc[-1]

# FPS 对比
x = np.arange(len(modes))
width = 0.6
fps_values = [avg_fps[m] for m in modes]
bars1 = ax1.bar(x, fps_values, width,
                  color=[mode_colors[m] for m in modes],
                  alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels([mode_labels[m] for m in modes],
                     rotation=45, ha='right')
ax1.set_ylabel('FPS', fontweight='bold')
ax1.set_title('Performance (FPS)', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_yscale('log')

# 添加数值标签
for bar, val in zip(bars1, fps_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.0f}',
             ha='center', va='bottom', fontsize=8)

# MSE 对比
mse_values = [avg_mse[m] for m in modes]
bars2 = ax2.bar(x, mse_values, width,
                  color=[mode_colors[m] for m in modes],
                  alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels([mode_labels[m] for m in modes],
                     rotation=45, ha='right')
ax2.set_ylabel('MSE', fontweight='bold')
ax2.set_title('Quality (MSE)', fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars2, mse_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2e}',
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig2_path = os.path.join(output_dir, 'fig2_performance_comparison.pdf')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}")

# ==================== 图3: 质量-性能权衡曲线 ====================
print("[3/6] Generating Figure 3: Quality-Performance Tradeoff...")

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制散点图
for mode in modes:
    csv_path = os.path.join(results_dir, f'cornell_box_{mode}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # 使用稳态数据（Frame 300-600）
        steady_state = df[df['frame'] >= 300]
        if len(steady_state) > 0:
            ax.scatter(steady_state['fps'].mean(),
                     steady_state['mse'].mean(),
                     label=mode_labels[mode],
                     color=mode_colors[mode],
                     s=200, alpha=0.7,
                     edgecolor='black', linewidth=1)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('FPS (log scale)', fontweight='bold')
ax.set_ylabel('MSE (log scale)', fontweight='bold')
ax.set_title('Quality-Performance Tradeoff', fontweight='bold', fontsize=13)
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 添加理想区域标注
ax.axvline(x=100, color='green', linestyle=':', alpha=0.5, label='Real-time threshold (100 FPS)')
ax.axhline(y=1e-3, color='orange', linestyle=':', alpha=0.5, label='High-quality threshold (1e-3)')

plt.tight_layout()
fig3_path = os.path.join(output_dir, 'fig3_quality_performance_tradeoff.pdf')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig3_path}")

# ==================== 图4: 收敛速度对比 ====================
print("[4/6] Generating Figure 4: Convergence Speed Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

# 计算收敛帧数（达到目标 MSE 的帧数）
target_mse_thresholds = [1e-2, 5e-3, 1e-3, 5e-4]

for threshold in target_mse_thresholds:
    convergence_frames = []
    for mode in modes:
        csv_path = os.path.join(results_dir, f'cornell_box_{mode}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # 找到第一个低于阈值的帧
            converged = df[df['mse'] <= threshold]
            if len(converged) > 0:
                frame = converged.iloc[0]['frame']
            else:
                frame = 600  # 未收敛

            convergence_frames.append(frame)
        else:
            convergence_frames.append(600)

    # 绘制柱状图
    x = np.arange(len(modes))
    width = 0.15
    offset = (target_mse_thresholds.index(threshold) - 1.5) * width

    ax.bar(x + offset, convergence_frames, width,
           label=f'MSE ≤ {threshold:.0e}',
           alpha=0.7, edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([mode_labels[m] for m in modes],
                     rotation=45, ha='right')
ax.set_ylabel('Convergence Frame', fontweight='bold')
ax.set_title('Convergence Speed Comparison', fontweight='bold', fontsize=13)
ax.legend(loc='best', framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
fig4_path = os.path.join(output_dir, 'fig4_convergence_speed.pdf')
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig4_path}")

# ==================== 图5: 误差热力图对比 ====================
print("[5/6] Generating Figure 5: Error Heatmap Comparison...")

# 检查是否有误差热力图
error_heatmaps = {}
for mode in ['Hybrid', 'MS_AIC', 'MCTF', 'FULL']:
    # 尝试查找误差热力图（这里假设已生成）
    heatmap_path = os.path.join(results_dir, f'cornell_box_{mode}_error.png')
    if os.path.exists(heatmap_path):
        from PIL import Image
        error_heatmaps[mode] = np.array(Image.open(heatmap_path))

if error_heatmaps:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (mode, heatmap) in enumerate(error_heatmaps.items()):
        axes[idx].imshow(heatmap)
        axes[idx].set_title(mode_labels[mode], fontweight='bold')
        axes[idx].axis('off')

        # 添加颜色条
        if idx == 0:
            cbar = plt.colorbar(axes[idx].images[0], ax=axes[idx],
                              orientation='horizontal', fraction=0.046, pad=0.04)
            cbar.set_label('Error Magnitude', fontweight='bold')

    plt.suptitle('Error Heatmaps (Low=Blue, High=Red)',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, 'fig5_error_heatmaps.pdf')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig5_path}")
else:
    print("  ⚠ No error heatmaps found, skipping Figure 5")

# ==================== 图6: 综合对比表 ====================
print("[6/6] Generating Figure 6: Summary Comparison Table...")

# 准备表格数据
table_data = []
for mode in modes:
    csv_path = os.path.join(results_dir, f'cornell_box_{mode}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # 计算指标
        avg_fps_val = df['fps'].mean()
        avg_mse_val = df['mse'].mean()
        final_mse_val = df['mse'].iloc[-1]

        # 收敛帧数（达到 1e-3 的帧数）
        converged = df[df['mse'] <= 1e-3]
        convergence_frame = converged.iloc[0]['frame'] if len(converged) > 0 else 600

        # 相对改进
        if mode == 'Hybrid':
            baseline_avg_mse = avg_mse_val
            baseline_fps = avg_fps_val
        elif mode != 'Hybrid' and 'baseline_avg_mse' in locals():
            mse_improvement = (baseline_avg_mse - avg_mse_val) / baseline_avg_mse * 100
            fps_change = (avg_fps_val - baseline_fps) / baseline_fps * 100
        else:
            mse_improvement = 0.0
            fps_change = 0.0

        table_data.append({
            'Method': mode_labels[mode],
            'Avg FPS': f'{avg_fps_val:.1f}',
            'Avg MSE': f'{avg_mse_val:.2e}',
            'Final MSE': f'{final_mse_val:.2e}',
            'Conv. Frame': f'{convergence_frame}',
            'MSE Improv.': f'{mse_improvement:.1f}%' if mse_improvement != 0 else '-',
            'FPS Change': f'{fps_change:.1f}%' if fps_change != 0 else '-'
        })

df_table = pd.DataFrame(table_data)

# 绘制表格
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_table.values,
              colLabels=df_table.columns,
              cellLoc='center',
              loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# 设置表头样式
for i in range(len(df_table.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置行样式
for i in range(1, len(df_table) + 1):
    for j in range(len(df_table.columns)):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')

plt.title('Comprehensive Performance Summary',
          fontweight='bold', fontsize=14, pad=20)

fig6_path = os.path.join(output_dir, 'fig6_summary_table.pdf')
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig6_path}")

# ==================== 完成 ====================
print()
print("="*60)
print("All figures generated successfully!")
print("="*60)
print()
print(f"Output directory: {output_dir}")
print()
print("Generated figures:")
print("  1. fig1_mse_convergence.pdf          - MSE 收敛对比曲线")
print("  2. fig2_performance_comparison.pdf      - 性能对比柱状图")
print("  3. fig3_quality_performance_tradeoff.pdf - 质量-性能权衡曲线")
print("  4. fig4_convergence_speed.pdf        - 收敛速度对比")
print("  5. fig5_error_heatmaps.pdf           - 误差热力图（如有）")
print("  6. fig6_summary_table.pdf            - 综合对比表")
print()
print("Next steps:")
print("  1. Review all figures")
print("  2. Adjust styling if needed (edit this script)")
print("  3. Insert figures into LaTeX document")
print()
