# 网格分辨率性能分析工具

## 概述

本套工具用于分析不同网格分辨率下的显存占用和计算开销，帮助绘制"画质 vs 性能"的权衡曲线（Trade-off Curve）。

## 文件说明

### 核心工具

1. **`grid_resolution_analyzer.py`** - 网格分辨率性能分析器
   - 测试不同网格分辨率(16³, 32³, 64³, 128³)的性能
   - 计算显存占用、网格更新时间、渲染时间
   - 生成性能数据和质量权衡分析

2. **`plot_tradeoff_curves.py`** - 权衡曲线绘制工具
   - 绘制网格规模 vs 显存占用曲线
   - 绘制网格规模 vs 渲染性能曲线  
   - 绘制内存-质量、性能-质量权衡曲线
   - 生成综合性能雷达图

3. **`test_gpu_timing.py`** - GPU计时功能测试
   - 验证GPU耗时统计功能
   - 测试不同渲染模式的性能差异

### 修改的文件

1. **`benchmark.py`** - 增强的基准测试
   - 新增 `gpu_time_ms` 列记录每帧GPU耗时
   - 支持更详细的性能分析
   - CSV格式: `frame,mode,fps,mse,gpu_time_ms,timestamp`

## 使用方法

### 1. 运行网格分辨率分析

```bash
python grid_resolution_analyzer.py
```

输出：
- `grid_analysis_results_[timestamp]/grid_resolution_performance.csv` - 基础性能数据
- `grid_analysis_results_[timestamp]/grid_tradeoff_analysis.csv` - 权衡分析数据

### 2. 绘制权衡曲线

```bash
python plot_tradeoff_curves.py
```

输出图表：
- `grid_performance_tradeoff.png` - 网格规模 vs 性能关系图
- `quality_performance_tradeoff.png` - 质量-性能权衡曲线
- `grid_performance_radar.png` - 综合性能雷达图
- `gpu_time_analysis.png` - GPU耗时分析图（如果有基准测试数据）

### 3. 运行基准测试（包含GPU耗时）

```bash
python benchmark.py
```

增强的基准测试现在会记录每帧的GPU耗时，便于后续分析。

### 4. 测试GPU计时功能

```bash
python test_gpu_timing.py
```

## 数据格式

### 网格分辨率性能数据格式

```csv
grid_resolution,grid_cells,memory_mb,avg_grid_update_ms,avg_render_ms,avg_total_ms,estimated_fps
"(16, 16, 16)",4096,0.125,0.45,2.34,2.79,358.4
"(32, 32, 32)",32768,1.000,3.67,8.92,12.59,79.4
"(64, 64, 64)",262144,8.000,29.36,71.36,100.72,9.9
```

### 权衡分析数据格式

```csv
grid_resolution,grid_cells,memory_mb,avg_total_ms,estimated_fps,quality_score,memory_efficiency,performance_efficiency
"(16, 16, 16)",4096,0.125,2.79,358.4,0.393,3.144,0.141
"(32, 32, 32)",32768,1.000,12.59,79.4,0.632,0.632,0.050
"(64, 64, 64)",262144,8.000,100.72,9.9,0.865,0.108,0.009
```

### 基准测试数据格式（增强版）

```csv
frame,mode,fps,mse,gpu_time_ms,timestamp
0,Path Tracing,12.5,0.000000,80.00,2024-01-01T10:00:00
1,Path Tracing,13.2,0.000000,75.76,2024-01-01T10:00:01
...
450,Hybrid,45.8,0.001234,21.83,2024-01-01T10:07:30
```

## 分析指标说明

### 性能指标
- **memory_mb**: 网格系统显存占用（MB）
- **avg_grid_update_ms**: 平均网格更新时间（毫秒）
- **avg_render_ms**: 平均渲染时间（毫秒）
- **avg_total_ms**: 平均总时间（毫秒）
- **estimated_fps**: 估计帧率

### 质量指标
- **quality_score**: 渲染质量得分（0-1，越高越好）
- **memory_efficiency**: 内存效率 = quality_score / memory_mb
- **performance_efficiency**: 性能效率 = quality_score / avg_total_ms

### 网格分辨率对应关系
- **16³**: 低分辨率，高性能，低质量
- **32³**: 中等分辨率，平衡性能与质量
- **64³**: 高分辨率，高质量，低性能
- **128³**: 极高分辨率，最高质量，最低性能

## 权衡曲线解读

### 1. 网格规模 vs 显存占用
- **关系**: 显存占用随网格单元数线性增长
- **理想**: 选择满足质量需求的最小网格

### 2. 网格规模 vs 渲染性能  
- **关系**: 渲染时间随网格规模近似立方增长
- **理想**: 在可接受的帧率范围内选择最大网格

### 3. 质量-性能权衡
- **效率值**: 越高表示性价比越好
- **甜蜜点**: 效率值最大的配置

## 实际应用建议

### 实时渲染应用
- 推荐：**32³** 网格分辨率
- 原因：性能与质量平衡较好，内存占用适中

### 高质量离线渲染
- 推荐：**64³** 或 **128³** 网格分辨率  
- 原因：质量优先，可接受较低帧率

### 移动设备/低功耗场景
- 推荐：**16³** 网格分辨率
- 原因：最低内存占用和计算开销

## 故障排除

### 常见问题

1. **GPU耗时异常高**
   - 检查是否使用GPU后端（ti.cuda）
   - 确认taichi版本兼容性

2. **显存计算不准确**
   - 检查camera.py中网格字段定义
   - 确认float大小（通常4字节）

3. **绘图无中文显示**
   - 安装中文字体：`pip install matplotlib seaborn`
   - 修改字体配置

### 调试技巧

1. **单独测试网格分辨率**:
   ```bash
   python test_gpu_timing.py
   ```

2. **检查数据完整性**:
   ```python
   import pandas as pd
   df = pd.read_csv('grid_resolution_performance.csv')
   print(df.describe())
   ```

3. **验证GPU同步**:
   - 确保所有ti.sync()调用正确
   - 检查异步操作是否完成

## 扩展功能

### 添加自定义网格分辨率
在 `grid_resolution_analyzer.py` 中修改：
```python
GRID_RESOLUTIONS = [
    (24, 24, 24),  # 自定义分辨率
    (48, 48, 48),
    # ... 其他分辨率
]
```

### 添加新的性能指标
可以扩展 `test_grid_performance()` 函数，添加：
- 功耗测量
- 温度监控
- 内存带宽利用率

### 集成到CI/CD
可以将这些工具集成到自动化测试流程中，持续监控性能变化。