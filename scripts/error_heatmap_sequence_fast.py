"""
误差热力图序列生成 (指令4) - 快速版
针对物体移动后的第 1、5、20、100 帧，分别生成 Hybrid 模式与 PT 模式的差值热力图
"""

import taichi as ti
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import main

vec3 = ti.types.vector(3, float)

# ========== 配置参数 ==========
SCENE_MODE = 'cornell_box'
MOVEMENT_FRAME = 50  # 提前移动以便更快生成
HEATMAP_FRAMES = [1, 5, 20, 100]
PT_REFERENCE_SPP = 128  # 降低样本数以提高速度
REFERENCE_CHUNK_SPP = 8

# 输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"heatmap_sequence_{timestamp}")
os.makedirs(output_dir, exist_ok=True)


def log_message(message):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def build_pt_reference(cam, world, target_spp=128, chunk_spp=8):
    """构建高精度PT参考"""
    log_message(f"构建PT参考 ({target_spp} spp)...")
    cam.clear_pt_reference()
    cam.render_pt_reference(world, target_spp=target_spp, chunk_spp=chunk_spp, reset=True)
    log_message(f"PT参考构建完成，当前spp: {cam.pt_spp_count[None]}")


def save_heatmap(cam, frame_label, output_dir):
    """保存误差热力图"""
    log_message(f"生成热力图: {frame_label}")

    # 渲染热力图（覆盖 cam.frame）
    cam.render_error_heatmap()

    # 获取热力图数据
    heatmap_np = cam.frame.to_numpy()
    heatmap_np = np.clip(heatmap_np * 255, 0, 255).astype(np.uint8)

    # 保存
    filename = f"ERROR_{frame_label}.png"
    filepath = os.path.join(output_dir, filename)

    img = Image.fromarray(heatmap_np)
    img.save(filepath, 'PNG')

    log_message(f"  已保存: {filepath}")
    return filepath


def save_comparison_image(cam, frame_label, output_dir):
    """保存对比图：Hybrid + PT + Error"""
    log_message(f"生成对比图: {frame_label}")

    # 1. Hybrid结果（需要重新渲染，因为cam.frame可能被热力图覆盖）
    hybrid_np = cam.frame.to_numpy()
    hybrid_np = np.clip(hybrid_np * 255, 0, 255).astype(np.uint8)

    # 2. PT参考
    pt_np = cam.pt_frame.to_numpy()
    pt_np = np.clip(pt_np * 255, 0, 255).astype(np.uint8)

    # 3. 误差热力图
    cam.render_error_heatmap()
    heatmap_np = cam.frame.to_numpy()
    heatmap_np = np.clip(heatmap_np * 255, 0, 255).astype(np.uint8)

    # 4. 拼接
    width, height = hybrid_np.shape[1], hybrid_np.shape[0]
    comparison = np.zeros((height, width * 3, 3), dtype=np.uint8)
    comparison[:, :width, :] = hybrid_np
    comparison[:, width:2*width, :] = pt_np
    comparison[:, 2*width:3*width, :] = heatmap_np

    # 保存
    filename = f"COMPARISON_{frame_label}.png"
    filepath = os.path.join(output_dir, filename)

    img = Image.fromarray(comparison)
    img.save(filepath, 'PNG')

    log_message(f"  已保存对比图: {filepath}")
    return filepath


def save_hybrid_render(cam, frame_label, output_dir):
    """保存Hybrid渲染结果"""
    hybrid_np = cam.frame.to_numpy()
    hybrid_np = np.clip(hybrid_np * 255, 0, 255).astype(np.uint8)

    filename = f"RENDER_{frame_label}.png"
    filepath = os.path.join(output_dir, filename)

    img = Image.fromarray(hybrid_np)
    img.save(filepath, 'PNG')

    log_message(f"  已保存渲染结果: {filepath}")
    return filepath


def create_sequence_collage(heatmaps_info, output_dir):
    """创建热力图序列拼图"""
    log_message("\n创建热力图序列拼图...")

    images = []
    for info in heatmaps_info:
        img_path = info['heatmap_path']
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)

    if len(images) != len(HEATMAP_FRAMES):
        log_message(f"警告: 只找到 {len(images)}/{len(HEATMAP_FRAMES)} 张热力图")
        return None

    w, h = images[0].size

    # 2x2网格
    collage = Image.new('RGB', (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]

    for img, pos in zip(images, positions):
        collage.paste(img, pos)

    collage_path = os.path.join(output_dir, 'heatmap_sequence_collage.png')
    collage.save(collage_path, 'PNG', dpi=(300, 300))
    log_message(f"热力图序列拼图已保存: {collage_path}")

    return collage_path


def create_mse_plot(mse_history, output_dir):
    """创建MSE收敛曲线图"""
    log_message("\n创建MSE收敛曲线图...")

    frames = list(range(1, len(mse_history) + 1))
    mse_values = mse_history

    plt.figure(figsize=(8, 5))
    plt.plot(frames, mse_values, 'r-', linewidth=2, marker='o', markersize=4)
    plt.yscale('log')
    plt.xlabel('Frame after movement', fontsize=12, fontweight='bold')
    plt.ylabel('MSE vs PT Reference', fontsize=12, fontweight='bold')
    plt.title('MSE Convergence - Hybrid Mode', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')

    # 标记热力图生成的帧
    for frame in HEATMAP_FRAMES:
        if frame - 1 < len(mse_history):
            plt.axvline(x=frame, color='blue', linestyle=':', alpha=0.7)
            plt.text(frame, mse_history[frame-1], f'  Frame {frame}',
                    verticalalignment='center', fontsize=9, color='blue')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'mse_convergence.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_message(f"MSE收敛曲线已保存: {plot_path}")


def run_heatmap_sequence():
    """主函数"""
    log_message("="*70)
    log_message("Error Heatmap Sequence Generation (Task 4)")
    log_message("="*70)
    log_message(f"Scene: {SCENE_MODE}")
    log_message(f"Movement frame: {MOVEMENT_FRAME}")
    log_message(f"Heatmap frames: {HEATMAP_FRAMES}")
    log_message(f"PT reference spp: {PT_REFERENCE_SPP}")
    log_message(f"Output directory: {output_dir}")
    log_message("")

    # 初始化场景
    log_message("Initializing scene...")
    world, cam = main.setup_scene(SCENE_MODE)
    cam.scene_mode = SCENE_MODE

    import main as main_module
    scene_materials = main_module.materials if hasattr(main_module, 'materials') else []
    scene_spheres = main_module.spheres if hasattr(main_module, 'spheres') else []

    cam.set_light_sources(scene_spheres, scene_materials)
    cam.adapt_grid_to_scene(scene_spheres, verbose=False)

    # 热身
    log_message("Warmup rendering...")
    ti.sync()
    cam.update_grid(world, 0.01)
    cam.render(world, 2)
    cam.asvgf_filter()
    ti.sync()

    # 渲染到移动帧
    log_message(f"Rendering to movement frame ({MOVEMENT_FRAME})...")
    for frame in range(MOVEMENT_FRAME):
        cam.update_grid(world, 0.01)
        cam.render(world, 2)
        cam.asvgf_filter()

        import experiment_config as cfg
        cam.compute_adaptive_weights(
            cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
            cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
            cfg.ADAPTIVE_MAX_MULTIPLIER
        )

    # 移动物体
    log_message("\nMoving object...")
    if len(scene_spheres) > 0:
        light_index = len(scene_spheres) - 1
        old_pos = scene_spheres[light_index].center[0]
        scene_spheres[light_index].center[0] = old_pos + 1.0
        log_message(f"Object moved: X {old_pos:.3f} -> {scene_spheres[light_index].center[0]:.3f}")

    # 重新适应网格
    cam.adapt_grid_to_scene(scene_spheres, verbose=False)

    # 构建PT参考（移动后）
    build_pt_reference(cam, world, target_spp=PT_REFERENCE_SPP, chunk_spp=REFERENCE_CHUNK_SPP)

    # 记录MSE历史
    mse_history = []

    # 生成热力图序列
    log_message("\nGenerating heatmap sequence...")
    heatmaps_info = []
    current_frame = 0
    max_frame = max(HEATMAP_FRAMES)

    while current_frame < max_frame:
        current_frame += 1

        # 渲染当前帧
        cam.update_grid(world, 0.01)
        cam.render(world, 2)

        import experiment_config as cfg
        cam.compute_adaptive_weights(
            cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
            cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
            cfg.ADAPTIVE_MAX_MULTIPLIER
        )

        cam.asvgf_filter()

        # 计算MSE
        current_mse = cam.compute_mse()
        mse_history.append(current_mse)

        # 检查是否需要生成热力图
        if current_frame in HEATMAP_FRAMES:
            frame_label = f"frame_{current_frame:03d}"

            # 1. 保存热力图（会覆盖cam.frame）
            heatmap_path = save_heatmap(cam, frame_label, output_dir)

            # 2. 保存对比图（需要重新渲染）
            cam.update_grid(world, 0.01)
            cam.render(world, 2)
            cam.asvgf_filter()
            save_comparison_image(cam, frame_label, output_dir)

            # 3. 保存Hybrid渲染结果
            save_hybrid_render(cam, frame_label, output_dir)

            heatmaps_info.append({
                'frame': current_frame,
                'mse': current_mse,
                'heatmap_path': heatmap_path
            })

        log_message(f"Frame {current_frame:3d}/{max_frame}: MSE = {current_mse:.6e}")

    # 生成汇总图
    log_message("\n" + "="*70)
    log_message("Generating summary plots...")
    log_message("="*70)

    collage_path = create_sequence_collage(heatmaps_info, output_dir)
    mse_plot_path = create_mse_plot(mse_history, output_dir)

    # 保存数据到CSV
    csv_path = os.path.join(output_dir, 'heatmap_sequence_data.csv')
    with open(csv_path, 'w', newline='') as f:
        import csv as csv_module
        writer = csv_module.writer(f)
        writer.writerow(['frame', 'mse'])
        for i, mse in enumerate(mse_history, 1):
            writer.writerow([i, mse])
    log_message(f"MSE data saved: {csv_path}")

    # 打印摘要
    log_message("\n" + "="*70)
    log_message("Generation complete!")
    log_message("="*70)
    for info in heatmaps_info:
        log_message(f"Frame {info['frame']:3d}: MSE = {info['mse']:.6e}")

    log_message(f"\nAll outputs saved to: {output_dir}")
    log_message("  - Heatmap sequence collage: heatmap_sequence_collage.png")
    log_message(f"  - MSE convergence plot: mse_convergence.png")
    log_message(f"  - Data file: heatmap_sequence_data.csv")


if __name__ == "__main__":
    try:
        run_heatmap_sequence()
    except KeyboardInterrupt:
        log_message("\nGeneration interrupted by user")
    except Exception as e:
        log_message(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
