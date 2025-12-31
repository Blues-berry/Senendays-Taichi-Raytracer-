# 论文实现指南 (Paper Implementation Guide)

本文档说明了为支持学术论文而实现的所有功能和使用方法。

## ✅ 已实现功能清单 (Implemented Features)

### 第一阶段：算法深度优化 (Algorithm Deep Optimization)

#### ✅ 指令1：基于深度的遮挡判定 (Depth-based Occlusion Detection)
**状态：已实现**
- **文件位置**：`camera.py`
- **实现细节**：
  - `grid_mean_distance` 字段：存储每个网格点的平均击中距离
  - `_probe_contrib()` 函数：检查实际距离与平均距离的偏差
  - 20% 相对阈值：超出则认为被遮挡，权重置为 0
  - 应用位置：`sample_irradiance_grid()`, `get_ray_color_grid()`, `get_ray_color_hybrid()`
- **效果**：显著减少漏光（Light Leaking）现象

#### ✅ 指令2：时域滤波与降噪 (Temporal Filtering & Denoising)
**状态：已实现**
- **文件位置**：`camera.py` (行 108-236)
- **实现细节**：
  - `accum_frame`：时域累积缓存（EMA）
  - `prev_normal_buffer`, `prev_depth_buffer`：上一帧的 G-buffer
  - 运动检测：深度相对变化 > 2% 或法线点积 < 0.98
  - EMA 权重：静态区域 alpha=0.10，运动区域 alpha=0.80
- **效果**：在极低采样率下获得平滑画质

### 第二阶段：自动化消融实验 (Automated Ablation Experiments)

#### ✅ 指令3：消融实验配置 (Ablation Study Configuration)
**状态：已实现**
- **文件位置**：`benchmark.py` (行 27-52, 229-381)
- **实验组**：
  - **Baseline**：全关（无插值、无重要性采样、无自适应）
  - **V1**：仅三线性插值
  - **V2**：插值 + 自适应权重更新
  - **Full_Hybrid**：全开（所有功能）
- **输出**：每个实验组一个独立 CSV 文件（`ablation_*.csv`）

#### ✅ 指令4：高质量学术对比图 (High-Quality Academic Plots)
**状态：已实现**
- **文件位置**：`plot_results.py`
- **生成的图表**：
  1. `ablation_mse_comparison.pdf` - MSE 随帧数对比曲线（对数坐标）
  2. `performance_comparison.pdf` - FPS 和 GPU 时间对比
  3. `quality_performance_tradeoff.pdf` - 质量-性能权衡曲线
  4. `detailed_mse_analysis.pdf` - 详细 MSE 分析
  5. `ablation_summary_report.txt` - 消融实验摘要报告
- **特点**：
  - 对数纵轴（便于观察收敛过程）
  - Frame 200 处标记物体位移
  - 300 DPI 输出（适合插入 LaTeX）

### 第三阶段：多场景压力测试 (Multi-scene Stress Testing)

#### ✅ 指令5：Cornell Box 场景
**状态：已实现**
- **文件位置**：`main.py` (行 91-138)
- **场景构成**：
  - 五面墙体：左墙红色、右墙绿色（验证颜色溢出）
  - 顶部强发光面光源（半径 0.85，强度 25）
  - 高反射金属球（fuzz=0.01）
  - 折射玻璃球（折射率 1.5）
- **网格适配**：自动计算 AABB 并调整网格范围

#### ✅ 指令6：误差热力图 (Error Heatmap)
**状态：已实现**
- **文件位置**：`camera.py` (行 303-348)
- **实现**：`render_error_heatmap()` 内核
- **伪彩色映射**：
  - 蓝色 → 青色 → 绿色 → 黄色 → 红色
  - 冷色表示低误差，红色表示高误差
- **输出**：`ERROR_*.png` 文件

### 第四阶段：量化性能分析 (Quantitative Performance Analysis)

#### ✅ 指令7：显存与计算开销统计 (Memory & Performance Analysis)
**状态：已实现**
- **文件位置**：`memory_analysis.py`
- **测试分辨率**：16³, 32³, 48³, 64³, 80³
- **输出**：
  1. `memory_performance_analysis.csv` - 详细数据
  2. `tradeoff_curves.pdf` - 内存 vs 更新时间权衡曲线
  3. `comprehensive_analysis.pdf` - 综合分析图表
  4. `memory_analysis_report.txt` - 文本报告

---

## 📋 使用方法 (Usage)

### 快速开始：一键运行完整分析

```bash
python run_complete_analysis.py
```

这将执行以下步骤：
1. 运行消融实验（4 个配置组）
2. 执行显存和性能分析
3. 生成所有论文图表

### 分步执行

#### 1. 仅运行消融实验

```bash
python benchmark.py
```

输出文件位于 `results/benchmark_results_YYYYMMDD_HHMMSS/`：
- `ablation_Baseline.csv`
- `ablation_V1.csv`
- `ablation_V2.csv`
- `ablation_Full_Hybrid.csv`

#### 2. 仅生成图表（使用已有数据）

```bash
python plot_results.py --results_dir results/benchmark_results_YYYYMMDD_HHMMSS
```

或使用最新结果目录：

```bash
python plot_results.py
```

#### 3. 仅运行显存分析

```bash
python memory_analysis.py
```

输出文件位于 `results/memory_analysis_YYYYMMDD_HHMMSS/`。

#### 4. 选择不同场景

```bash
python benchmark.py  # 默认使用 cornell_box
```

修改 `benchmark.py` 行 676 的参数：

```python
run_group_experiments('cornell_box')  # 或 'random', 'night_scene'
```

---

## 📊 输出文件说明 (Output Files)

### 消融实验输出

| 文件名 | 描述 |
|--------|------|
| `ablation_Baseline.csv` | 基线配置的 MSE/FPS/GPU 时间数据 |
| `ablation_V1.csv` | 仅插值配置的数据 |
| `ablation_V2.csv` | 插值+自适应配置的数据 |
| `ablation_Full_Hybrid.csv` | 完整混合配置的数据 |
| `result_*.png` | 每组实验的结果截图 |
| `ERROR_*.png` | 误差热力图 |

### 图表输出

| 文件名 | 描述 |
|--------|------|
| `ablation_mse_comparison.pdf` | 四组配置的 MSE 对比曲线（论文主图） |
| `performance_comparison.pdf` | FPS 和 GPU 时间对比 |
| `quality_performance_tradeoff.pdf` | 质量-性能权衡曲线 |
| `detailed_mse_analysis.pdf` | 详细 MSE 分析（4 子图） |

### 显存分析输出

| 文件名 | 描述 |
|--------|------|
| `memory_performance_analysis.csv` | 不同分辨率的显存和时间数据 |
| `tradeoff_curves.pdf` | 权衡曲线 |
| `comprehensive_analysis.pdf` | 综合分析（4 子图） |
| `memory_analysis_report.txt` | 文本摘要报告 |

---

## 🔧 配置参数 (Configuration)

### 实验配置 (`experiment_config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GRID_RESOLUTION` | (32, 32, 32) | 网格分辨率 |
| `ADAPTIVE_BRIGHTNESS_THRESHOLD` | 0.05 | 亮度对比阈值 |
| `ADAPTIVE_SAMPLING_MULTIPLIER` | 1.0 | 自适应采样倍数 |
| `ADAPTIVE_MAX_MULTIPLIER` | 2.0 | 最大采样倍数 |
| `LIGHT_IMPORTANCE_SCALE` | 2.0 | 光源重要性采样缩放 |
| `VARIANCE_SAMPLING_SCALE` | 2.0 | 方差采样缩放 |
| `MAX_PROBE_SAMPLES` | 16 | 最大探针采样数 |

### Benchmark 参数 (`benchmark.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `test_frames` | 450 | 每组测试帧数 |
| `movement_frame` | 200 | 物体位移触发帧 |
| `pt_ref_spp_frames` | 150 | PT 参考样本帧数 |
| `reference_spp` | 512 | 热力图 PT 参考 SPP |

---

## 📈 论文图表建议 (Paper Figure Suggestions)

### 图 1：Teaser Image
使用 `result_Full_Hybrid.png` 展示 Cornell Box 的渲染结果。

### 图 2：消融研究 MSE 对比
使用 `ablation_mse_comparison.pdf`：
- 横轴：Frame
- 纵轴：MSE (log scale)
- 四条曲线：Baseline, V1, V2, Full_Hybrid
- 垂直虚线标记物体位移

### 图 3：误差热力图
使用 `ERROR_Full_Hybrid_move_5.png` 和 `ERROR_Full_Hybrid_move_50.png`：
- 展示阴影边缘的收敛过程
- 蓝色 = 低误差，红色 = 高误差

### 图 4：性能对比
使用 `performance_comparison.pdf`：
- 左子图：FPS 对比
- 右子图：GPU 时间对比

### 图 5：质量-性能权衡
使用 `quality_performance_tradeoff.pdf`：
- 横轴：FPS
- 纵轴：MSE (log scale)
- 每个配置的散点图

### 图 6：显存与分辨率权衡
使用 `tradeoff_curves.pdf`：
- 展示不同网格分辨率下的显存占用和更新时间

---

## 📝 论文写作要点 (Writing Points)

### 1. 漏光问题
- 提及基于深度的遮挡判定（20% 相对阈值）
- 展示误差热力图证明有效性

### 2. 噪声抑制
- 描述时域 EMA 累积
- 静态区域强累积 (alpha=0.10)，运动区域快速更新 (alpha=0.80)

### 3. 消融实验
- 清晰对比四个配置的差异
- Full_Hybrid 在保持实时性能的同时显著降低 MSE

### 4. 泛化性
- Cornell Box 验证颜色溢出和间接光照
- 不同网格分辨率下的性能表现

---

## 🐛 故障排除 (Troubleshooting)

### 显存不足
降低 `experiment_config.py` 中的 `GRID_RESOLUTION`：
```python
GRID_RESOLUTION = (24, 24, 24)  # 从 32³ 降低
```

### FPS 显示为 0
检查 `benchmark.py` 行 490-500 的 FPS 上限设置。

### 误差热力图全红
增加 PT 参考样本数：
```python
cam.render_pt_reference(world, target_spp=1024, chunk_spp=16, reset=True)
```

---

## 📧 联系与支持 (Contact & Support)

如有问题或需要进一步定制，请检查：
1. `benchmark.py` - 实验配置
2. `experiment_config.py` - 算法参数
3. `plot_results.py` - 绘图逻辑
4. `memory_analysis.py` - 性能分析
