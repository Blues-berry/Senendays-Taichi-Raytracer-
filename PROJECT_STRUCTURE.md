# 项目结构整理 (Project Structure)

文件已按功能分类整理到不同目录。

## 目录结构

```
Senendays-Taichi-Raytracer-
├── main.py                 # 主程序（入口）
├── camera.py                # 相机与渲染核心
├── world.py                 # 场景世界
├── hittable.py              # 可击中物体基类
├── material.py              # 材质定义
├── ray.py                  # 光线类
├── utils.py                 # 工具函数
│
├── scenes/                  # 【新增】场景配置
│   ├── __init__.py
│   └── scene_configs.py      # 所有场景定义（cornell_box, two_room, night_scene, random）
│
├── scripts/                 # 【新增】测试与分析脚本
│   ├── __init__.py
│   ├── convergence_speed_benchmark.py     # 指令2: 收敛速度统计
│   ├── error_heatmap_sequence_fast.py     # 指令4: 误差热力图序列
│   ├── test_scenes_simple.py             # 简单场景测试
│   ├── run_all_tests.py                # 统一运行脚本
│   ├── benchmark_utils.py              # 基准测试工具
│   ├── grid_resolution_analyzer.py      # 网格分辨率分析
│   ├── memory_analysis.py              # 显存分析
│   ├── plot_results.py                # 结果绘图
│   ├── plot_tradeoff_curves.py        # 权衡曲线
│   └── generate_heatmap.py            # 热力图生成
│
├── plots/                   # 【新增】绘图工具
│   ├── __init__.py
│   └── plotting_tools.py              # 统一绘图工具模块
│
├── results/                 # 测试结果输出目录
│
├── test/                    # 测试文件
│
└── *.md                     # 文档文件
```

## 使用说明

### 运行场景测试

```bash
# 测试所有基本场景
python scripts/test_scenes_simple.py
```

### 运行收敛速度基准测试（指令2）

```bash
python scripts/convergence_speed_benchmark.py
```

**输出**:
- `convergence_speed_scatter.png` - 散点图（算力消耗 vs 收敛时间）
- `mse_convergence_curves.png` - MSE收敛曲线
- `convergence_summary.csv` - 数据摘要

### 运行误差热力图序列生成（指令4）

```bash
python scripts/error_heatmap_sequence_fast.py
```

**输出**:
- `ERROR_frame_*.png` - 各帧误差热力图
- `COMPARISON_frame_*.png` - 三联对比图（Hybrid + PT + Error）
- `RENDER_frame_*.png` - Hybrid渲染结果
- `heatmap_sequence_collage.png` - 热力图序列拼图
- `mse_convergence.png` - MSE收敛曲线（带帧标记）
- `heatmap_sequence_data.csv` - 数据文件

### 场景配置

所有场景定义在 `scenes/scene_configs.py` 中：

| 场景名称 | 描述 | 用途 |
|-----------|------|------|
| `cornell_box` | 标准Cornell Box | 基准测试 |
| `two_room` | 两室一门 | 压力测试（窄缝漏光） |
| `night_scene` | 夜间场景 | 测试暗光环境 |
| `random` | 随机小球 | 默认场景 |

在代码中使用：

```python
from scenes.scene_configs import get_scene

spheres, materials, cam_params = get_scene('cornell_box')
world = World(spheres, materials)
cam = Camera(world, **cam_params)
```

### 绘图工具

所有绘图函数在 `plots/plotting_tools.py` 中：

```python
from plots.plotting_tools import (
    plot_convergence_scatter,
    plot_mse_curves,
    plot_mse_with_heatmap_markers,
    create_heatmap_collage,
    create_comparison_triple,
    save_results_summary
)
```

## 指令对应关系

| 指令 | 脚本文件 | 状态 |
|--------|-----------|------|
| 指令1: 运动感知采样 | camera.py（需增强） | 待实现 |
| 指令2: 收敛速度统计 | scripts/convergence_speed_benchmark.py | ✅ 已完成 |
| 指令3: 两室一门场景 | scenes/scene_configs.py (create_two_room_scene) | ✅ 已完成 |
| 指令4: 误差热力图序列 | scripts/error_heatmap_sequence_fast.py | ✅ 已完成 |
| 指令5: 网格分辨率权衡 | scripts/grid_resolution_analyzer.py | 需运行 |

## 下一步工作

1. **运行场景测试**:
   ```bash
   python scripts/test_scenes_simple.py
   ```

2. **运行两室一门压力测试**（验证深度遮挡）:
   - 检查是否阻止光线穿墙
   - 检查是否正确捕捉窄缝间接光

3. **指令5：网格分辨率权衡分析**:
   - 统计 16³, 32³, 64³ 的显存占用
   - 统计每帧更新耗时
   - 计算单位计算量下的误差下降率

4. **指令1：运动感知采样**:
   - 在 camera.py 中实现 velocity_field
   - 根据运动速度提升采样率
   - 实现预采样（Pre-sampling）

## 论文数据支持

已生成的数据可用于论文：

1. **收敛速度散点图** - 证明自适应算法的效率提升
2. **热力图序列** - 直观展示快速"抹平"高误差区域
3. **两室一门场景** - 证明复杂遮挡下的表现

需要我提供 Methodology（方法论）部分的数学公式推导建议吗？
