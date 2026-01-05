# 绘图文件整合与 Benchmark 分析报告

## 整合完成

### plots/ 目录结构

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块初始化 |
| `plotting_tools.py` | 基础绘图工具函数（散点图、MSE曲线、热力图等） |
| `plot_manager.py` | 统一绘图管理器，整合所有绘图功能 |
| `plot_results_legacy.py` | 旧版绘图脚本（从scripts/迁移） |
| `plot_tradeoff_curves.py` | 权衡曲线绘制（从scripts/迁移） |

### scripts/ 目录（清理后）

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块初始化 |
| `benchmark_utils.py` | 基准测试工具函数 |
| `convergence_speed_benchmark.py` | 收敛速度统计（指令2） |
| `error_heatmap_sequence_fast.py` | 误差热力图序列（指令4） |
| `generate_heatmap.py` | 生成单帧热力图 |
| `grid_resolution_analyzer.py` | 网格分辨率分析（指令5） |
| `memory_analysis.py` | 内存分析 |
| `run_all_tests.py` | 统一测试运行脚本 |
| `test_scenes_simple.py` | 简单场景测试脚本 |

---

## Benchmark 分析结果

### 测试配置
- 结果目录: `results/benchmark_results_20251226_172131/`
- 测试帧数: 600 帧
- 移动事件: 第150帧发生物体位移
- 测试模式: Path Tracing, Pure Grid, Hybrid

### 性能数据

| 模式 | 平均FPS | 相对PT倍数 | MSE | 质量 |
|------|---------|-------------|-----|------|
| Path Tracing | 1.1 | 1x (基准) | 0.0 | 参考标准 |
| Pure Grid | 108.3 | **98.5x** | 4.45e-06 | 几乎无差异 |
| Hybrid | 71.9 | **65.4x** | 4.45e-06 | 几乎无差异 |

### 详细分析

#### Path Tracing 模式
- FPS范围: 1.0 - 2.0
- 平均FPS: 1.1
- 作为参考标准，MSE = 0.0

#### Pure Grid 模式
- FPS范围: 1.3 - 925.3
- 平均FPS: 108.3
- 初始帧有性能波动（网格初始化）
- 稳定后维持在 90-110 FPS
- MSE: 4.45e-06 (极小)

#### Hybrid 模式
- FPS范围: 1.4 - 107.5
- 平均FPS: 71.9
- 稳定后维持在 70-75 FPS
- MSE: 4.45e-06 (与 Pure Grid 相同)
- 性能略低于 Pure Grid（自适应开销）

---

## 生成的图表

### 1. `mse_over_time.png`
三种模式的 MSE 随时间变化曲线（对数坐标）

### 2. `detailed_mse_analysis.png`
包含 4 个子图:
- **MSE Before Displacement (Frames 0-149)**: 移动前的MSE曲线
- **MSE After Displacement (Frames 150+)**: 移动后的MSE曲线
- **MSE Convergence Trend**: 移动平均收敛趋势
- **Performance Comparison (FPS)**: FPS 性能对比

### 3. `summary_report.txt`
文本摘要报告，包含:
- 各模式的帧数统计
- MSE 统计（最小、最大、均值、中位数）
- FPS 统计（最小、最大、均值、中位数）

---

## 关键发现

### 优势
1. **Pure Grid**: 极高帧率（~100x PT），适合静态场景
2. **Hybrid**: 平衡性能与质量（~65x PT），适合动态场景
3. **质量一致**: 两种加速方法的 MSE 极低，与 PT 几乎无差异

### 劣势
1. **Pure Grid**: 动态场景需要完全重建网格，存在瞬时开销
2. **Hybrid**: 自适应逻辑增加约 30% 的计算成本

### 应用建议
- **静态场景**: 使用 Pure Grid 获得最高帧率
- **动态场景**: 使用 Hybrid 获得平滑的收敛体验
- **离线渲染**: 使用 Path Tracing 获得最高质量

---

## 收敛性分析

### 移动前 (Frames 0-149)
- Pure Grid 和 Hybrid 快速收敛
- MSE 稳定在 4.45e-06

### 移动后 (Frames 150+)
- Pure Grid: 立即重建网格，短暂波动后快速恢复
- Hybrid: 自适应逻辑逐步更新网格，收敛更平滑
- 两者在约 50-100 帧后恢复稳定

---

## 使用绘图管理器

```python
from plots.plot_manager import generate_all_plots

# 生成所有图表
generate_all_plots('results/benchmark_results_20251226_172131')
```

生成的图表将保存到 `results/benchmark_results_20251226_172131/plots/` 目录。
