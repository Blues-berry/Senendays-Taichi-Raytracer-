"""
收敛速度统计 (指令2)
比较三种更新策略的收敛速度：
1. 固定低概率更新
2. 固定高概率更新  
3. 自适应更新逻辑

输出：横轴为算力消耗，纵轴为收敛时间的散点图
"""

import taichi as ti
import numpy as np
import time
import os
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import main
from main import spheres, cam, world

vec3 = ti.types.vector(3, float)

# ========== 配置参数 ==========
# 收敛阈值：MSE降至PT参考值的1.05倍
CONVERGENCE_THRESHOLD_RATIO = 1.05

# 测试配置
SCENE_MODE = 'cornell_box'
MOVEMENT_FRAME = 200
MAX_POST_MOVE_FRAMES = 300  # 移动后最多追踪300帧
PT_REFERENCE_SPP = 256  # PT参考样本数

# 三种更新策略配置
STRATEGIES = [
    {
        "name": "Fixed Low Probability",
        "update_prob": 0.01,  # 1%
        "adaptive_enabled": False,
        "description": "固定低概率更新 (1%)",
        "color": "blue"
    },
    {
        "name": "Fixed High Probability",
        "update_prob": 0.10,  # 10%
        "adaptive_enabled": False,
        "description": "固定高概率更新 (10%)",
        "color": "green"
    },
    {
        "name": "Adaptive Logic",
        "update_prob": 0.01,  # 基础1% + 自适应提升
        "adaptive_enabled": True,
        "description": "自适应更新 (基础1% + 运动区域提升)",
        "color": "red"
    }
]

# 输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"convergence_benchmark_{timestamp}")
os.makedirs(output_dir, exist_ok=True)


def log_message(message):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def calculate_mse(img1, img2):
    """计算MSE（线性空间）"""
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # 归一化到[0,1]范围
    if img1_f.max() > 1.1:
        img1_f = img1_f / 255.0
    if img2_f.max() > 1.1:
        img2_f = img2_f / 255.0

    mse = np.mean((img1_f - img2_f) ** 2)
    return float(mse)


def run_single_strategy(strategy_config, strategy_idx):
    """
    运行单个策略的收敛测试

    返回:
        dict: 包含收敛帧数、计算量等指标
    """
    log_message(f"\n{'='*60}")
    log_message(f"策略 {strategy_idx + 1}/3: {strategy_config['name']}")
    log_message(f"配置: {strategy_config['description']}")
    log_message(f"{'='*60}\n")

    # 重新初始化场景（确保公平比较）
    world, cam = main.setup_scene(SCENE_MODE)
    cam.scene_mode = SCENE_MODE

    # 直接获取场景的materials列表
    # setup_scene 会在main模块中设置 materials 和 spheres 全局变量
    import main as main_module
    scene_materials = main_module.materials if hasattr(main_module, 'materials') else []
    scene_spheres = main_module.spheres if hasattr(main_module, 'spheres') else []

    cam.set_light_sources(scene_spheres, scene_materials)
    cam.adapt_grid_to_scene(scene_spheres, verbose=False)

    # 应用策略配置
    update_prob = strategy_config['update_prob']
    adaptive_enabled = strategy_config['adaptive_enabled']

    # 热身
    log_message("热身渲染...")
    ti.sync()
    cam.update_grid(world, update_prob)
    cam.render(world, 2)  # Hybrid mode
    cam.asvgf_filter()
    ti.sync()

    # 构建PT参考（移动前的基准）
    log_message(f"构建PT参考（{PT_REFERENCE_SPP} spp）...")
    pt_ref_accum = np.zeros((*cam.img_res, 3), dtype=np.float32)
    for _ in range(PT_REFERENCE_SPP):
        cam.render_pt(world)
        ti.sync()
        pt_ref_accum += cam.pt_frame.to_numpy().astype(np.float32)
    pt_reference_linear = pt_ref_accum / float(PT_REFERENCE_SPP)
    baseline_mse = calculate_mse(pt_reference_linear, pt_reference_linear)

    # 收敛阈值
    convergence_threshold = baseline_mse * CONVERGENCE_THRESHOLD_RATIO
    log_message(f"基准MSE: {baseline_mse:.6e}")
    log_message(f"收敛阈值（{CONVERGENCE_THRESHOLD_RATIO}x）: {convergence_threshold:.6e}")

    # 主循环：达到移动帧 -> 移动物体 -> 追踪收敛
    frame_count = 0
    total_gpu_time_ms = 0.0
    converged_frame = None
    mse_history = []

    log_message(f"开始渲染，将在第 {MOVEMENT_FRAME} 帧移动物体...")

    while frame_count < MOVEMENT_FRAME:
        ti.sync()
        start_time = time.perf_counter()

        cam.update_grid(world, update_prob)
        cam.render(world, 2)  # Hybrid mode
        cam.asvgf_filter()

        if adaptive_enabled:
            import experiment_config as cfg
            cam.compute_adaptive_weights(
                cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
                cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
                cfg.ADAPTIVE_MAX_MULTIPLIER
            )

        ti.sync()
        frame_time = time.perf_counter() - start_time
        total_gpu_time_ms += frame_time * 1000.0

        frame_count += 1

    # 移动物体
    import main as main_module
    scene_spheres = main_module.spheres if hasattr(main_module, 'spheres') else []

    if len(scene_spheres) > 0:
        light_index = len(scene_spheres) - 1
        old_pos = scene_spheres[light_index].center[0]
        scene_spheres[light_index].center[0] = old_pos + 1.0
        log_message(f"物体移动: X {old_pos:.3f} -> {scene_spheres[light_index].center[0]:.3f}")

    # 重新适应网格
    cam.adapt_grid_to_scene(scene_spheres, verbose=False)

    # 追踪收敛
    log_message("开始追踪收敛...")
    post_move_frame = 0
    current_frame_buffer = np.zeros((*cam.img_res, 3), dtype=np.float32)
    frame_weight = 0.0

    while post_move_frame < MAX_POST_MOVE_FRAMES:
        ti.sync()
        start_time = time.perf_counter()

        cam.update_grid(world, update_prob)
        cam.render(world, 2)  # Hybrid mode

        if adaptive_enabled:
            import experiment_config as cfg
            cam.compute_adaptive_weights(
                cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
                cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
                cfg.ADAPTIVE_MAX_MULTIPLIER
            )

        cam.asvgf_filter()

        ti.sync()
        frame_time = time.perf_counter() - start_time
        total_gpu_time_ms += frame_time * 1000.0

        # 累积帧（模拟渐进式渲染）
        frame_weight = 1.0 / (post_move_frame + 1)
        current_linear = cam.frame.to_numpy()
        current_frame_buffer = current_frame_buffer * (1 - frame_weight) + current_linear * frame_weight

        # 计算MSE
        mse = calculate_mse(current_frame_buffer, pt_reference_linear)
        mse_history.append(mse)

        log_message(f"  Frame {post_move_frame + 1:3d}: MSE = {mse:.6e} (阈值: {convergence_threshold:.6e})")

        # 检查收敛
        if mse < convergence_threshold:
            converged_frame = post_move_frame + 1
            log_message(f"*** 收敛于第 {converged_frame} 帧！***")
            break

        post_move_frame += 1

    if converged_frame is None:
        log_message(f"警告: 未在 {MAX_POST_MOVE_FRAMES} 帧内收敛")
        converged_frame = MAX_POST_MOVE_FRAMES

    return {
        "strategy": strategy_config['name'],
        "converged_frame": converged_frame,
        "total_gpu_time_ms": total_gpu_time_ms,
        "final_mse": mse_history[-1] if mse_history else None,
        "mse_history": mse_history,
        "color": strategy_config['color'],
        "description": strategy_config['description']
    }


def plot_convergence_scatter(results):
    """
    绘制收敛速度散点图
    横轴：算力消耗（GPU时间 ms）
    纵轴：收敛时间（帧数）
    """
    plt.figure(figsize=(10, 7))

    for result in results:
        plt.scatter(
            result['total_gpu_time_ms'],
            result['converged_frame'],
            s=300,  # 点的大小
            c=result['color'],
            alpha=0.7,
            edgecolors='black',
            linewidths=2,
            label=result['strategy']
        )

        # 添加标签
        label_text = f"{result['converged_frame']}帧"
        plt.annotate(
            label_text,
            (result['total_gpu_time_ms'], result['converged_frame']),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9
        )

    plt.xlabel('总GPU计算时间 (ms)', fontsize=14, fontweight='bold')
    plt.ylabel('收敛所需帧数', fontsize=14, fontweight='bold')
    plt.title(f'收敛速度对比 (场景: {SCENE_MODE})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, 'convergence_speed_scatter.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"散点图已保存: {plot_path}")


def plot_mse_curves(results):
    """
    绘制MSE收敛曲线
    """
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

    plt.xlabel('移动后帧数', fontsize=14, fontweight='bold')
    plt.ylabel('MSE vs PT Reference', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.title('MSE收敛曲线对比', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'mse_convergence_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"MSE曲线已保存: {plot_path}")


def save_summary_csv(results):
    """保存结果到CSV"""
    csv_path = os.path.join(output_dir, 'convergence_summary.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '策略名称',
            '描述',
            '收敛帧数',
            '总GPU时间(ms)',
            '平均每帧时间(ms)',
            '最终MSE',
            '收敛效率(帧/秒)'
        ])

        for result in results:
            avg_frame_time = result['total_gpu_time_ms'] / (MOVEMENT_FRAME + result['converged_frame'])
            efficiency = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0

            writer.writerow([
                result['strategy'],
                result['description'],
                result['converged_frame'],
                f"{result['total_gpu_time_ms']:.2f}",
                f"{avg_frame_time:.2f}",
                f"{result['final_mse']:.6e}" if result['final_mse'] else "N/A",
                f"{efficiency:.2f}"
            ])

    log_message(f"CSV摘要已保存: {csv_path}")


def main_benchmark():
    """主函数：运行所有策略测试"""
    log_message("="*70)
    log_message("收敛速度基准测试 (指令2)")
    log_message("="*70)
    log_message(f"场景模式: {SCENE_MODE}")
    log_message(f"移动帧: {MOVEMENT_FRAME}")
    log_message(f"收敛阈值: {CONVERGENCE_THRESHOLD_RATIO}x PT参考")
    log_message(f"PT参考样本数: {PT_REFERENCE_SPP} spp")
    log_message(f"输出目录: {output_dir}")
    log_message("")

    results = []

    for i, strategy in enumerate(STRATEGIES):
        try:
            result = run_single_strategy(strategy, i)
            results.append(result)
        except Exception as e:
            log_message(f"策略 {strategy['name']} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    if results:
        log_message("\n" + "="*70)
        log_message("测试完成！生成结果...")
        log_message("="*70)

        save_summary_csv(results)
        plot_convergence_scatter(results)
        plot_mse_curves(results)

        # 打印摘要
        log_message("\n" + "="*70)
        log_message("结果摘要:")
        log_message("="*70)
        for result in results:
            log_message(f"\n{result['strategy']}:")
            log_message(f"  - 收敛帧数: {result['converged_frame']}")
            log_message(f"  - 总GPU时间: {result['total_gpu_time_ms']:.2f} ms")
            log_message(f"  - 最终MSE: {result['final_mse']:.6e}")

        log_message(f"\n所有结果已保存至: {output_dir}")
    else:
        log_message("警告: 没有成功完成任何测试")


if __name__ == "__main__":
    try:
        main_benchmark()
    except KeyboardInterrupt:
        log_message("\n测试被用户中断")
    except Exception as e:
        log_message(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
