# 基准测试完整指南 (Benchmark Guide)

本文档详细说明基准测试系统、运行流程和输出分析。

---

## 概述

基准测试系统自动化对比三种渲染模式的性能和质量：
- **Path Tracing (PT)**: 物理精确的参考标准
- **Pure Grid**: 纯网格快速渲染
- **Hybrid Adaptive**: 混合自适应算法

---

## 快速开始

### 运行完整基准测试

```bash
python benchmark.py
```

测试完成后会自动生成分析图表到 `results/benchmark_results_YYYYMMDD_HHMMSS/plots/`

### 使用指定场景运行

```python
# 修改 benchmark.py 中的场景参数
run_group_experiments('classroom')  # 或任意场景名称
```

---

## 测试流程

### 1. 初始化阶段

```python
# 创建时间戳目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(results_dir, f"benchmark_results_{timestamp}")

# 初始化场景
spheres, materials, cam_params = get_scene('random')  # 默认场景
world = World(spheres, materials)
cam = Camera(world, **cam_params)
cam.adapt_grid_to_scene(spheres)

# 设置光源
cam.set_light_sources(spheres, materials)
```

### 2. 测试阶段

基准测试按顺序运行三种模式：

#### 模式1: Path Tracing (参考)
```python
# 渲染 150 帧
for frame in range(150):
    start_time = time.perf_counter()
    cam.render_pt(world)
    ti.sync()
    frame_time = time.perf_counter() - start_time
    # 记录 FPS
```

**输出**:
- `result_path_tracing.png` - 最终渲染结果
- PT 参考缓存（用于后续 MSE 计算）

#### 模式2: Pure Grid
```python
# 渲染 600 帧
for frame in range(600):
    start_time = time.perf_counter()
    cam.update_grid(world, 0.01)
    cam.render(world, RENDER_MODE_GRID)
    ti.sync()
    frame_time = time.perf_counter() - start_time
    # 计算 MSE vs PT 参考
    mse = cam.compute_mse()
    # 保存到 CSV
```

**输出**:
- `result_pure_grid.png` - 最终渲染结果
- 每 100 帧的截图（`pure_grid_frame_*.png`）

#### 模式3: Hybrid Adaptive
```python
# 渲染 600 帧（前 300 帧静态，后 300 帧动态）
for frame in range(600):
    # 第 200 帧移动大球（测试动态响应）
    if frame == 200:
        big_sphere.center[0] += 0.5
        cam.adapt_grid_to_scene(spheres)

    start_time = time.perf_counter()
    cam.update_grid(world, 0.01)
    cam.render(world, RENDER_MODE_HYBRID)
    cam.asvgf_filter()
    cam.compute_adaptive_weights(...)
    ti.sync()
    frame_time = time.perf_counter() - start_time

    # 收敛检测（移动后）
    if frame > 200:
        # 检测 5 帧连续 MSE 变化 < 0.1%
        if consecutive_small_mse_changes >= 5:
            print(f"Recovered in {frame - 200} frames")

    # 关键帧保存截图
    if frame in [5, 50, 100, 150]:
        save_screenshot(...)
```

**输出**:
- `result_hybrid.png` - 最终渲染结果
- 移动后第 5、50 帧截图
- 误差热力图（`ERROR_move_*.png`）

### 3. 自动生成图表

```python
from plots.plot_manager import generate_all_plots
generate_all_plots(output_dir)
```

生成的图表：
- `mse_over_time.png` - MSE 随时间变化
- `detailed_mse_analysis.png` - 4 子图详细分析
- `summary_report.txt` - 文本摘要
- `BENCHMARK_ANALYSIS_REPORT.md` - 完整分析报告

---

## 输出文件结构

```
results/
└── benchmark_results_20251226_172131/
    ├── benchmark_results.csv          # 完整测试数据
    ├── result_path_tracing.png        # PT 结果
    ├── result_pure_grid.png          # Grid 结果
    ├── result_hybrid.png             # Hybrid 结果
    ├── path_tracing_frame_5.png     # PT 第5帧
    ├── path_tracing_frame_50.png    # PT 第50帧
    ├── path_tracing_frame_150.png   # PT 第150帧
    ├── pure_grid_frame_100.png      # Grid 第100帧
    ├── pure_grid_frame_200.png      # Grid 第200帧
    ├── pure_grid_frame_300.png      # Grid 第300帧
    ├── hybrid_move_1_frame_5.png     # Hybrid 移动后第5帧
    ├── hybrid_move_1_frame_50.png    # Hybrid 移动后第50帧
    ├── ERROR_move_1_frame_5.png     # 误差热力图（移动后第5帧）
    ├── ERROR_move_1_frame_50.png    # 误差热力图（移动后第50帧）
    └── plots/
        ├── mse_over_time.png
        ├── detailed_mse_analysis.png
        ├── summary_report.txt
        └── BENCHMARK_ANALYSIS_REPORT.md
```

---

## CSV 数据格式

### benchmark_results.csv

```csv
frame,mode,fps,mse,gpu_time_ms,grid_cells,memory_mb,timestamp
0,Path Tracing,45.2,0.000000e+00,22.12,262144,3.00,2025-12-26T17:21:33.123456
1,Path Tracing,43.8,0.000000e+00,22.83,262144,3.00,2025-12-26T17:21:33.567890
...
150,Pure Grid,567.3,4.450000e-06,1.76,262144,3.00,2025-12-26T17:23:45.987654
151,Pure Grid,580.1,4.420000e-06,1.72,262144,3.00,2025-12-26T17:23:45.990123
...
200,Hybrid,71.9,4.450000e-06,13.91,262144,3.00,2025-12-26T17:24:01.234567
201,Hybrid,68.3,2.340000e-05,14.64,262144,3.00,2025-12-26T17:24:01.249876
...
```

**字段说明**:
- `frame`: 帧序号
- `mode`: 渲染模式（Path Tracing, Pure Grid, Hybrid）
- `fps`: 帧率
- `mse`: 均方误差（相对于 PT 参考）
- `gpu_time_ms`: GPU 执行时间（毫秒）
- `grid_cells`: 网格单元总数
- `memory_mb`: 网格显存占用（MB）
- `timestamp`: 时间戳

---

## 关键功能

### 1. 动态场景测试

每 200 帧自动移动大球，测试算法的动态响应：
```python
if frame % 200 == 0 and frame > 0:
    big_sphere.center[0] += 0.5
    cam.adapt_grid_to_scene(spheres)
```

### 2. 收敛检测

移动后监控 MSE 变化，检测收敛：
```python
if consecutive_small_mse_changes >= 5:
    recovery_frames = frame - move_frame
    print(f"Recovered in {recovery_frames} frames")
```

### 3. 自动截图

在关键帧自动保存截图：
- 移动后第 5、50 帧
- 每种模式的第 100、200、300 帧

### 4. 误差热力图

实时生成 Hybrid vs PT 的误差热力图：
```python
cam.render_error_heatmap()
```

**颜色映射**:
- 蓝色/青色: 低误差 (< 1e-5)
- 绿色: 中等误差 (1e-5 - 1e-4)
- 黄色: 较高误差 (1e-4 - 1e-3)
- 红色: 高误差 (> 1e-3)

---

## 性能分析

### 预期性能范围

| 模式 | 预期 FPS | GPU 时间 | MSE 范围 |
|------|----------|----------|----------|
| Path Tracing | 1-5 | 200-1000ms | 0.0 (参考) |
| Pure Grid | 80-200 | 5-12ms | 0.01-0.1 |
| Hybrid | 40-100 | 10-25ms | 0.001-0.05 |

*注：实际性能因场景复杂度、硬件配置而异*

### 性能提升比

基于 `BENCHMARK_ANALYSIS_REPORT.md` 的结果：

| 模式 | 平均 FPS | 相对 PT 倍数 | MSE |
|------|---------|-------------|-----|
| Path Tracing | 1.1 | 1x (基准) | 0.0 |
| Pure Grid | 108.3 | **98.5x** | 4.45e-06 |
| Hybrid | 71.9 | **65.4x** | 4.45e-06 |

---

## 自动分析功能

### plot_manager.py 功能

`generate_all_plots()` 函数自动生成：

#### 1. MSE 随时间变化
```python
plot_mse_over_time(csv_path)
```
- 横轴: 帧数
- 纵轴: MSE (log scale)
- 三条曲线: PT, Pure Grid, Hybrid

#### 2. 详细 MSE 分析
```python
plot_detailed_mse_analysis(csv_path)
```
4 子图：
1. MSE 随时间（全部帧）
2. 移动前/后对比
3. 收敛速度
4. 稳态误差

#### 3. FPS 对比
```python
plot_fps_comparison(csv_path)
```
- 柱状图对比三种模式的 FPS
- 误差棒显示标准差

#### 4. 摘要报告
```python
save_results_summary(csv_path)
```
文本报告包含：
- 统计数据（均值、标准差、最大/最小值）
- 性能提升比
- 质量评估

---

## 自定义测试

### 修改测试帧数

```python
# benchmark.py
TEST_FRAMES = {
    'pt': 150,           # PT 帧数
    'grid': 600,         # Grid 帧数
    'hybrid': 600,       # Hybrid 帧数
}
```

### 修改移动间隔

```python
# benchmark.py
MOVE_INTERVAL = 200  # 每 N 帧移动一次
```

### 修改截图帧

```python
# benchmark.py
SCREENSHOT_FRAMES = [5, 50, 100, 150, 200, 300]
```

### 禁用自动分析

```python
# benchmark.py
AUTO_GENERATE_PLOTS = False  # 禁用自动生成图表
```

---

## 手动分析

### 使用已有数据生成图表

```bash
python -c "from plots.plot_manager import generate_all_plots; generate_all_plots('results/benchmark_results_20251226_172131')"
```

### 单独调用绘图函数

```python
from plots.plotting_tools import (
    plot_mse_curves,
    plot_fps_comparison,
    create_heatmap_collage
)

# 绘制 MSE 曲线
plot_mse_curves(csv_path, 'output.png', title='MSE Convergence')

# 绘制 FPS 对比
plot_fps_comparison(csv_data, 'fps_comparison.png')

# 创建热力图拼图
create_heatmap_collage(heatmap_files, 'collage.png')
```

---

## 常见问题

### Q: FPS 显示异常高（> 10000）
A: GPU 同步问题。确保使用 `ti.sync()`：
```python
ti.sync()
frame_time = time.perf_counter() - start_time
```

### Q: MSE 始终为 0
A: PT 参考缓存未正确设置。检查：
```python
if mode == 'Path Tracing':
    # 确保 PT 缓存在此阶段正确保存
    cam.save_pt_reference()
```

### Q: 测试运行很慢
A: 可能原因：
1. 场景复杂度（球体数过多）
2. 网格分辨率过高
3. 采样率过高

解决方法：
```python
# experiment_config.py
GRID_RESOLUTION = (24, 24, 24)  # 降低网格分辨率
MAX_PROBE_SAMPLES = 8          # 降低采样数
```

### Q: 内存不足
A: 降低网格分辨率：
```python
GRID_RESOLUTION = (16, 16, 16)  # 最小配置
```

---

## 扩展：消融实验

基准测试支持配置不同的算法变体进行消融实验：

```python
# 消融实验配置
ABLATION_CONFIGS = {
    'Baseline': {
        'use_trilinear': False,
        'use_adaptive': False,
        'use_nee': False,
    },
    'V1_TriLinear': {
        'use_trilinear': True,
        'use_adaptive': False,
        'use_nee': False,
    },
    'V2_Adaptive': {
        'use_trilinear': True,
        'use_adaptive': True,
        'use_nee': False,
    },
    'Full_Hybrid': {
        'use_trilinear': True,
        'use_adaptive': True,
        'use_nee': True,
    },
}
```

输出每个配置的独立 CSV 文件：`ablation_*.csv`

---

## 相关文档

- **PARAMETERS_GUIDE.md**: 所有参数详细说明
- **SCENES_GUIDE.md**: 场景配置指南
- **PAPER_IMPLEMENTATION_GUIDE.md**: 论文实现指南
- **README_INTEGRATED.md**: 项目整体文档
