# Taichi 光线追踪渲染器

基于 Python 和 Taichi 库构建的高性能自适应光线追踪渲染器，支持多种渲染模式、完整性能基准测试和学术级分析。

![Raytracing in One Weekend Image](images/raytracing_in_one_weekend.png)
### [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) 最终渲染效果

![Cornell Box Demo](https://github.com/user-attachments/assets/a9d083fd-600e-46ca-bb94-6e5f22c839dd)
### Cornell Box 场景渲染

---

## 📚 文档导航

- **[场景配置指南 (SCENES_GUIDE.md)](SCENES_GUIDE.md)** - 所有场景详细说明和使用方法
- **[基准测试指南 (BENCHMARK_GUIDE.md)](BENCHMARK_GUIDE.md)** - 基准测试系统、运行流程和输出分析
- **[参数配置指南 (PARAMETERS_GUIDE.md)](PARAMETERS_GUIDE.md)** - 所有可配置参数的详细说明
- **[论文实现指南 (PAPER_IMPLEMENTATION_GUIDE.md)](PAPER_IMPLEMENTATION_GUIDE.md)** - 学术功能实现和论文支持

---

## 🚀 为什么选择 Taichi？

虽然 C++ 通常比 Python 快几个数量级，但 Taichi 库能将 Python 代码编译成原生的 CPU 或 GPU 代码，让开发者避免了编写高效 C++/CUDA 代码所需的数千行代码和令人沮丧的调试时间，同时不牺牲性能。

---

## 📦 安装依赖

```bash
pip install taichi numpy matplotlib
```

---

## 🎮 渲染模式

本渲染器支持三种渲染模式：

### 1. Path Tracing (PT) - 路径追踪
- **模式**: `0`
- **特点**: 物理精确的全局光照渲染
- **速度**: 慢速 (1-5 FPS)
- **质量**: 最高，作为参考标准
- **用途**: 基准参考、最终渲染

### 2. Pure Grid - 纯网格渲染
- **模式**: `1`
- **特点**: 基于空间网格的快速渲染
- **速度**: 快速 (80-200 FPS)
- **质量**: 中等，适合实时预览
- **用途**: 快速预览、静态场景

### 3. Hybrid Adaptive - 混合自适应
- **模式**: `2`
- **特点**: 结合网格和路径追踪的自适应算法
- **速度**: 中等 (40-100 FPS)
- **质量**: 优于网格，接近路径追踪
- **用途**: 实时应用、动态场景

---

## 🏗️ 项目结构

```
Senendays-Taichi-Raytracer/
├── 📄 核心文件
│   ├── main.py                 # 主程序入口
│   ├── ablation.py            # 单场景消融实验（原 benchmark.py，含自动分析）
│   ├── camera.py              # 相机与渲染核心
│   ├── material.py            # 材质系统
│   ├── world.py               # 场景管理
│   ├── ray.py                # 光线定义
│   ├── hittable.py           # 碰撞检测
│   ├── utils.py              # 工具函数
│   └── experiment_config.py   # 实验参数配置
│
├── 🎬 场景配置
│   ├── __init__.py
│   └── scene_configs.py      # 场景定义（7个场景）
│
├── 🧪 测试与分析脚本
│   ├── __init__.py
│   ├── benchmark_utils.py              # 基准测试工具
│   ├── convergence_speed_benchmark.py  # 收敛速度分析
│   ├── error_heatmap_sequence_fast.py  # 误差热力图序列
│   ├── grid_resolution_analyzer.py      # 网格分辨率分析
│   ├── memory_analysis.py              # 显存分析
│   ├── generate_heatmap.py            # 热力图生成
│   ├── test_scenes_simple.py         # 简单场景测试
│   ├── run_all_tests.py             # 统一测试脚本
│   └── test_new_scenes.py          # 新场景快速测试
│
├── 📊 绘图工具
│   ├── __init__.py
│   ├── plotting_tools.py       # 基础绘图函数
│   ├── plot_manager.py         # 统一绘图管理器（推荐使用）
│   ├── plot_results_legacy.py  # 旧版绘图兼容
│   └── plot_tradeoff_curves.py # 权衡曲线
│
├── 📁 结果输出
│   └── results/              # 所有数据统一保存目录
│       ├── benchmark_results_YYYYMMDD_HHMMSS/
│       ├── convergence_benchmark_YYYYMMDD_HHMMSS/
│       └── heatmap_sequence_YYYYMMDD_HHMMSS/
│
├── 🖼️ 示例图像
│   └── images/               # 渲染效果展示
│
├── 🧪 测试文件
│   └── test/                # 单元测试和验证脚本
│
└── 📚 文档文件
    ├── README.md             # 本文档
    ├── SCENES_GUIDE.md      # 场景配置指南
    ├── BENCHMARK_GUIDE.md    # 基准测试指南
    ├── PARAMETERS_GUIDE.md   # 参数配置指南
    └── PAPER_IMPLEMENTATION_GUIDE.md  # 论文实现指南
```

---

## 📖 快速开始

### 1. 运行主程序（交互式渲染）

```bash
python main.py
```

### 2. 切换场景

编辑 `main.py` 顶部：
```python
CURRENT_SCENE = 'classroom'  # 或 cornell_box, two_room, night_scene, random, bathroom, veach_mis
```

### 3. 运行单场景消融实验

```bash
python ablation.py
```

测试默认场景（cornell_box），完成后自动生成分析图表。

### 4. 运行多场景全面测试（推荐）

```bash
python benchmark_full.py
```

依次测试所有 7 个场景，每个场景完整运行三种方法 + Error 热力图。

### 5. 快速测试所有场景

```bash
python scripts/test_new_scenes.py
```

快速验证所有场景的基本渲染功能。

---

## 🎨 场景类型

| 场景 | 描述 | 主要测试 | 推荐用途 |
|------|------|---------|---------|
| `random` | 随机小球场景 | 基础渲染 | 默认场景 |
| `cornell_box` | 标准 Cornell Box | 颜色溢出、间接光 | 经典测试 |
| `two_room` | 两室一门 | 窄缝漏光 | 复杂遮挡 |
| `night_scene` | 夜间场景 | 多光源、高反射 | 光照复杂度 |
| `classroom` | 教室场景 | 环境光、间接光 | 室内场景 |
| `bathroom` | 浴室场景 | 镜面、caustics | 高反射 |
| `veach_mis` | Veach MIS | 重要性采样 | 采样效率 |

详细说明见：[SCENES_GUIDE.md](SCENES_GUIDE.md)

---

## 📊 性能数据

### 基准测试结果

| 模式 | 平均 FPS | 相对 PT 倍数 | MSE | 适用场景 |
|------|---------|-------------|-----|---------|
| Path Tracing | 1.1 | 1x (基准) | 0.0 | 参考标准 |
| Pure Grid | 108.3 | **98.5x** | 4.45e-06 | 静态场景 |
| Hybrid | 71.9 | **65.4x** | 4.45e-06 | 动态场景 |

### 自动生成的图表

基准测试完成后自动生成：
- `mse_over_time.png` - MSE 随时间变化
- `detailed_mse_analysis.png` - 4 子图详细分析
- `summary_report.txt` - 文本摘要报告
- `BENCHMARK_ANALYSIS_REPORT.md` - 完整分析报告

---

## 🔧 技术特性

### 核心算法

1. **自适应网格更新**
   - 基于亮度对比的动态采样
   - 运动区域加速更新
   - 方差引导采样

2. **深度防漏光**
   - 距离不匹配检测
   - 20% 相对阈值判定遮挡

3. **时域滤波 (A-SVGF)**
   - 运动检测（深度和法线变化）
   - EMA 累积（静态 vs 动态区域）
   - 边界保持

4. **重要性采样 (NEE)**
   - 光源直接采样
   - 多重要性采样 (MIS)

### 精确的性能测量
- **GPU同步**: 使用 `ti.sync()` 确保测量实际执行时间
- **高精度计时**: `time.perf_counter()` 提供微秒级精度
- **异常值过滤**: 根据模式特点设置合理 FPS 上限

---

## 🧪 实验功能

### 消融实验

支持配置不同的算法变体：
- Baseline: 全关
- V1: 仅三线性插值
- V2: 插值 + 自适应
- Full: 全开

### 动态场景测试
- 每 200 帧自动移动大球
- 实时监控收敛过程
- 自动保存关键帧截图

### 收敛检测
- 检测 5 帧连续 MSE 变化 < 0.1% 为收敛
- 自动记录恢复帧数

---

## 🔮 未来改进

1. **几何体扩展**: 添加 Plane, Box 等基本几何体
2. **更复杂的材质**: 支持纹理、BSDF
3. **运动模糊**: 时间抗锯齿
4. **分布式渲染**: 多 GPU 支持

---

## 🐛 故障排除

### 常见问题

#### 1. FPS 显示异常高（> 10000）
- **原因**: GPU 异步执行导致计时不准确
- **解决**: 已通过 `ti.sync()` 修复

#### 2. MSE 始终为 0
- **原因**: PT 参考帧未正确设置
- **解决**: 检查参考帧存储时机

#### 3. 显存不足
- **解决**: 降低网格分辨率：
```python
# experiment_config.py
GRID_RESOLUTION = (16, 16, 16)  # 从 32³ 降低
```

#### 4. 测试运行很慢
- **解决**:
  1. 降低网格分辨率
  2. 降低采样率
  3. 使用更简单的场景

---

## 📖 详细文档

### 场景配置
[SCENES_GUIDE.md](SCENES_GUIDE.md)
- 所有场景的详细说明
- 场景元素和材质组合
- 自定义场景方法

### 基准测试
[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)
- 测试流程详解
- 输出文件说明
- 自动分析功能
- 消融实验配置

### 参数配置
[PARAMETERS_GUIDE.md](PARAMETERS_GUIDE.md)
- 所有可配置参数
- 参数调优指南
- 计算公式
- 常见问题

### 论文实现
[PAPER_IMPLEMENTATION_GUIDE.md](PAPER_IMPLEMENTATION_GUIDE.md)
- 已实现功能清单
- 消融实验
- 论文图表建议
- 写作要点

---

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---

## 🙏 致谢

- [Taichi Graphics](https://taichi-lang.org/) - 高性能计算框架
- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) - 光线追踪基础理论
- [Cornell Box](https://en.wikipedia.org/wiki/Cornell_box) - 标准测试场景

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

**Happy Ray Tracing! 🚀**
