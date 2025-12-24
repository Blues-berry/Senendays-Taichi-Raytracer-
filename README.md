# Taichi 光线追踪渲染器

基于 Python 和 Taichi 库构建的高性能光线追踪渲染器，支持多种渲染模式和完整的性能基准测试。

![Raytracing in One Weekend Image](images/raytracing_in_one_weekend.png)
### [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) 最终渲染效果

![Cornell Box Demo](https://github.com/user-attachments/assets/a9d083fd-600e-46ca-bb94-6e5f22c839dd)
### Cornell Box 场景渲染

## 🚀 为什么选择 Taichi？

虽然 C++ 通常比 Python 快几个数量级，但 Taichi 库能将 Python 代码编译成原生的 CPU 或 GPU 代码，让开发者避免了编写高效 C++/CUDA 代码所需的数千行代码和令人沮丧的调试时间，同时不牺牲性能。

## 📦 安装依赖

```bash
pip install taichi numpy
```

## 🎮 渲染模式

本渲染器支持三种渲染模式：

### 1. **Path Tracing (PT)** - 路径追踪
- **模式**: `0`
- **特点**: 物理精确的全局光照渲染
- **速度**: 慢速 (10-50 FPS)
- **质量**: 最高，作为参考标准

### 2. **Pure Grid** - 纯网格渲染
- **模式**: `1` 
- **特点**: 基于空间网格的快速渲染
- **速度**: 快速 (200-800 FPS)
- **质量**: 中等，适合实时预览

### 3. **Hybrid Adaptive** - 混合自适应
- **模式**: `2`
- **特点**: 结合网格和路径追踪的自适应算法
- **速度**: 中等 (40-120 FPS)
- **质量**: 优于网格，接近路径追踪

## 🛠️ 使用方法

### 基础渲染
```bash
# 运行主程序（交互式渲染）
python main.py
```

### 性能基准测试
```bash
# 运行完整基准测试（各模式300帧）
python benchmark.py
```

## 📁 项目结构

```
Senendays-Taichi-Raytracer/
├── 📄 核心文件
│   ├── main.py                 # 主程序入口
│   ├── benchmark.py           # 性能基准测试
│   ├── camera.py              # 摄像机和渲染逻辑
│   ├── material.py            # 材质系统
│   ├── world.py               # 场景管理
│   ├── ray.py                 # 光线定义
│   ├── hittable.py            # 碰撞检测
│   └── utils.py               # 工具函数
│
├── 📊 实验配置
│   ├── experiment_config.py   # 实验参数配置
│   └── experiment_automation.py # 实验自动化
│
├── 📁 结果输出
│   └── results/              # 所有数据统一保存目录
│       ├── benchmark_results_YYYYMMDD_HHMMSS/
│       │   ├── benchmark_results.csv    # 基准测试数据
│       │   ├── result_path_tracing.png   # PT结果图
│       │   ├── result_pure_grid.png      # Grid结果图
│       │   ├── result_hybrid.png         # Hybrid结果图
│       │   ├── path_tracing_frame_5.png  # PT第5帧
│       │   ├── path_tracing_frame_50.png # PT第50帧
│       │   ├── ... (各模式指定帧截图)
│       │   └── path_tracing_frame_150.png # PT第150帧
│       └── experiment_YYYYMMDD_HHMMSS/
│           ├── experiment_log.csv        # 实验日志
│           ├── output_*.png              # 最终输出
│           └── *.png                     # 实验截图
│
├── 🖼️ 示例图像
│   └── images/               # 渲染效果展示
│
└── 📚 文档文件
    ├── README.md             # 本文档
    └── *.md                  # 各种修复和更新文档
```

## 📊 性能数据与分析

### 基准测试输出

运行基准测试后会生成详细的性能数据：

#### CSV 数据格式
```csv
frame,mode,fps,mse,timestamp
0,Path Tracing,45.2,0.000000,2025-12-24T11:30:00.123456
1,Path Tracing,43.8,0.000000,2025-12-24T11:30:00.567890
...
300,Pure Grid,567.3,0.023456,2025-12-24T11:35:15.987654
```

#### 预期性能范围
| 模式 | 预期 FPS | 理论范围 | MSE 范围 |
|------|----------|----------|----------|
| Path Tracing | 20-60 | 10-100 | 0.0 (参考) |
| Pure Grid | 200-800 | 100-2000 | 0.02-0.1 |
| Hybrid | 40-120 | 20-200 | 0.01-0.05 |

### 自动截图功能

基准测试会在以下帧数自动保存截图：

- **固定帧数**: 5, 50, 100, 150帧
- **模式结束**: 每种模式的最后帧

文件命名规则：`{mode}_frame_{frame_number}.png`

## 🔧 技术特性

### 精确的性能测量
- **GPU同步**: 使用 `ti.sync()` 确保测量实际执行时间
- **高精度计时**: `time.perf_counter()` 提供微秒级精度
- **异常值过滤**: 根据模式特点设置合理FPS上限
- **实时数据**: 每10帧自动保存，防止数据丢失

### 智能FPS过滤
```python
max_fps = {
    RENDER_MODE_PT: 200,     # Path Tracing 慢速
    RENDER_MODE_GRID: 2000,  # Grid 方法快速  
    RENDER_MODE_HYBRID: 500   # Hybrid 中等速度
}
```

### MSE质量评估
- 以Path Tracing结果为参考基准
- 实时计算其他模式相对于PT的质量差异
- 支持收敛分析和质量对比

## 🧪 实验功能

### 动态场景测试
- 每200帧自动移动主要球体
- 实时监控渲染收敛过程
- 自动在关键帧保存截图用于分析

### 收敛检测
- 检测5帧连续MSE变化小于0.1%为收敛
- 自动记录恢复帧数
- 支持动态场景性能评估

## 🐛 故障排除

### 常见问题

#### 1. FPS显示异常高
- **现象**: FPS显示2000+
- **原因**: GPU异步执行导致计时不准确
- **解决**: 已通过GPU同步修复

#### 2. MSE始终为0
- **现象**: MSE列全为0
- **原因**: PT参考帧未正确设置
- **解决**: 已修复参考帧存储时机

#### 3. 数据丢失
- **现象**: 程序异常退出后CSV数据不完整
- **解决**: 每10帧自动保存，异常退出时保存剩余数据

### 测试验证
```bash
# 运行测试脚本验证修复
python test_fps_fix.py
python test_data_saving.py
python test_sync_timing.py
```

## 📈 性能优化建议

### 1. 硬件优化
- **GPU**: 推荐NVIDIA RTX系列，支持CUDA
- **内存**: 建议8GB以上，特别是大分辨率场景
- **存储**: SSD可提升场景加载速度

### 2. 参数调优
- **分辨率**: 降低分辨率可显著提升FPS
- **网格分辨率**: 调整`grid_res`平衡精度和速度
- **采样率**: 增加采样率提升质量但降低FPS

### 3. 模式选择策略
- **开发阶段**: 使用Pure Grid快速预览
- **测试验证**: 使用Hybrid获得平衡的性能和质量
- **最终渲染**: 使用Path Tracing获得最高质量

## 📚 API 参考

### 主要类和方法

#### Camera类
```python
# 渲染场景
cam.render(world, mode)  # mode: 0=PT, 1=Grid, 2=Hybrid

# 获取MSE
mse = cam.compute_mse()

# 自适应网格调整
cam.adapt_grid_to_scene(spheres, verbose=True)
```

#### 材质系统
```python
# 创建不同材质
material_lambert = Lambertian(color)
material_metal = Metal(color, fuzz)
material_dielectric = Dielectric(refractive_index)
```

## 🤝 贡献指南

### 代码风格
- 遵循PEP 8规范
- 使用有意义的变量名和函数名
- 添加适当的注释和文档字符串

### 提交规范
- 修复bug: 使用`fix:`前缀
- 新功能: 使用`feat:`前缀
- 性能优化: 使用`perf:`前缀

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- [Taichi Graphics](https://taichi-lang.org/) - 高性能计算框架
- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) - 光线追踪基础理论
- [Cornell Box](https://en.wikipedia.org/wiki/Cornell_box) - 标准测试场景

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

**Happy Ray Tracing! 🚀**