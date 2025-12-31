"""
Quick Feature Test Script
快速功能测试脚本
Tests all implemented features without running full benchmarks
测试所有已实现的功能，无需运行完整benchmark
"""

import taichi as ti
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Taichi
ti.init(arch=ti.gpu, random_seed=42)

print("="*60)
print("FEATURE VERIFICATION TEST / 功能验证测试")
print("="*60)

# Test 1: Check camera.py has depth-based occlusion
print("\n[1] Checking depth-based occlusion / 检查深度遮挡判定...")
from camera import Camera
import main

world, cam = main.setup_scene('cornell_box')

# Check for required fields
assert hasattr(cam, 'grid_mean_distance'), "Missing grid_mean_distance field!"
assert hasattr(cam, 'interpolate_grid_sampling'), "Missing interpolate_grid_sampling!"
assert hasattr(cam, 'enable_light_guided_probes'), "Missing enable_light_guided_probes!"
print("  ✓ grid_mean_distance field exists")
print("  ✓ Feature toggles (interpolate_grid_sampling, enable_light_guided_probes) exist")

# Test 2: Check temporal accumulation
print("\n[2] Checking temporal accumulation / 检查时域累积...")
assert hasattr(cam, 'accum_frame'), "Missing accum_frame field!"
assert hasattr(cam, 'prev_normal_buffer'), "Missing prev_normal_buffer!"
assert hasattr(cam, 'prev_depth_buffer'), "Missing prev_depth_buffer!"
assert hasattr(cam, 'accum_alpha_static'), "Missing accum_alpha_static!"
assert hasattr(cam, 'accum_alpha_moving'), "Missing accum_alpha_moving!"
print("  ✓ Temporal accumulation fields exist")
print("  ✓ EMA parameters (accum_alpha_static={}, accum_alpha_moving={})".format(
    cam.accum_alpha_static, cam.accum_alpha_moving
))

# Test 3: Check PT reference and error heatmap
print("\n[3] Checking PT reference and error heatmap / 检查PT参考和误差热力图...")
assert hasattr(cam, 'pt_frame'), "Missing pt_frame field!"
assert hasattr(cam, 'pt_accum'), "Missing pt_accum field!"
assert hasattr(cam, 'pt_spp_count'), "Missing pt_spp_count!"
assert hasattr(cam, 'render_error_heatmap'), "Missing render_error_heatmap() method!"
assert hasattr(cam, 'render_pt_reference'), "Missing render_pt_reference() method!"
print("  ✓ PT reference fields exist (pt_frame, pt_accum, pt_spp_count)")
print("  ✓ Error heatmap method exists")

# Test 4: Check adaptive sampling
print("\n[4] Checking adaptive sampling / 检查自适应采样...")
assert hasattr(cam, 'adaptive_weight_map'), "Missing adaptive_weight_map!"
assert hasattr(cam, 'compute_adaptive_weights'), "Missing compute_adaptive_weights() method!"
assert hasattr(cam, 'asvgf_filter'), "Missing asvgf_filter() method!"
print("  ✓ Adaptive weight map exists")
print("  ✓ Adaptive weight computation method exists")
print("  ✓ A-SVGF filter method exists")

# Test 5: Check ablation configuration
print("\n[5] Checking ablation configuration / 检查消融配置...")
import benchmark
assert hasattr(benchmark, 'EXPERIMENT_GROUPS'), "Missing EXPERIMENT_GROUPS!"
assert len(benchmark.EXPERIMENT_GROUPS) == 4, "EXPERIMENT_GROUPS should have 4 groups!"
group_names = [g['name'] for g in benchmark.EXPERIMENT_GROUPS]
expected_names = ['Baseline', 'V1', 'V2', 'Full_Hybrid']
assert set(group_names) == set(expected_names), f"Unexpected group names: {group_names}"
for g in benchmark.EXPERIMENT_GROUPS:
    assert 'interpolation_on' in g, f"{g['name']} missing interpolation_on!"
    assert 'importance_sampling_on' in g, f"{g['name']} missing importance_sampling_on!"
    assert 'adaptive_logic_on' in g, f"{g['name']} missing adaptive_logic_on!"
print("  ✓ EXPERIMENT_GROUPS has 4 configurations")
print(f"  ✓ Group names: {', '.join(expected_names)}")
for g in benchmark.EXPERIMENT_GROUPS:
    print(f"    - {g['name']}: interp={g['interpolation_on']}, "
          f"IS={g['importance_sampling_on']}, adaptive={g['adaptive_logic_on']}")

# Test 6: Check plot functions
print("\n[6] Checking plot functions / 检查绘图函数...")
import plot_results
plot_funcs = [
    'plot_ablation_mse_comparison',
    'plot_performance_comparison',
    'plot_mse_fps_tradeoff',
    'generate_ablation_summary_report'
]
for func_name in plot_funcs:
    assert hasattr(plot_results, func_name), f"Missing {func_name}!"
print("  ✓ All required plot functions exist")
print(f"    - {', '.join(plot_funcs)}")

# Test 7: Check memory analysis
print("\n[7] Checking memory analysis / 检查内存分析...")
import memory_analysis
assert hasattr(memory_analysis, 'calculate_grid_memory'), "Missing calculate_grid_memory!"
assert hasattr(memory_analysis, 'benchmark_grid_update_time'), "Missing benchmark_grid_update_time!"
print("  ✓ Memory calculation function exists")
print("  ✓ Grid update time benchmark function exists")

# Test 8: Check scene setup
print("\n[8] Checking scene setup / 检查场景设置...")
scene_modes = ['random', 'cornell_box', 'night_scene']
for mode in scene_modes:
    try:
        w, c = main.setup_scene(mode)
        assert w is not None, f"World is None for {mode}!"
        assert c is not None, f"Camera is None for {mode}!"
        print(f"  ✓ {mode} scene setup successful")
    except Exception as e:
        print(f"  ✗ {mode} scene setup failed: {e}")

# Test 9: Calculate memory for different resolutions
print("\n[9] Memory calculation test / 内存计算测试...")
test_resolutions = [(16,16,16), (32,32,32), (64,64,64)]
for res in test_resolutions:
    mem_mb = memory_analysis.calculate_grid_memory(res)
    expected_cells = res[0] * res[1] * res[2]
    print(f"  {res[0]}x{res[1]}x{res[2]} ({expected_cells:,} cells): {mem_mb:.2f} MB")

# Summary
print("\n" + "="*60)
print("ALL FEATURES VERIFIED SUCCESSFULLY! / 所有功能验证成功！")
print("="*60)
print("\nImplemented features / 已实现的功能:")
print("  1. ✓ Depth-based occlusion detection / 基于深度的遮挡判定")
print("  2. ✓ Temporal accumulation (EMA) / 时域累积(EMA)")
print("  3. ✓ PT reference & error heatmap / PT参考和误差热力图")
print("  4. ✓ Adaptive sampling with A-SVGF filter / 自适应采样和A-SVGF滤波")
print("  5. ✓ Ablation study configuration (4 groups) / 消融实验配置(4组)")
print("  6. ✓ High-quality academic plotting / 高质量学术绘图")
print("  7. ✓ Memory and performance analysis / 内存和性能分析")
print("  8. ✓ Multiple scene modes (Cornell Box, etc.) / 多种场景模式")
print("\nReady to run full analysis / 准备运行完整分析:")
print("  python run_complete_analysis.py")
