# Test Suite / 测试套件

此目录包含所有项目的测试脚本，用于验证核心功能和性能。

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `run_all_tests.py` | 一键运行所有测试脚本 |
| `quick_benchmark.py` | 精简版消融实验 (2组配置, 100帧) |
| `test_update_grid.py` | 快速测试网格更新功能 |
| `test_data_saving.py` | 测试数据保存功能 |
| `test_fps_fix.py` | 测试FPS计算修复 |
| `test_gpu_timing.py` | 测试GPU计时功能 |
| `test_sync_timing.py` | 测试GPU同步计时 |
| `test_features.py` | 验证所有已实现的功能 |

## 使用方法

### 运行所有测试
```bash
cd test
python run_all_tests.py
```

### 单独运行测试
```bash
cd test

# 快速benchmark (推荐)
python quick_benchmark.py

# 功能验证
python test_features.py

# GPU计时测试
python test_gpu_timing.py

# 数据保存测试
python test_data_saving.py

# FPS计算测试
python test_fps_fix.py

# 同步计时测试
python test_sync_timing.py

# 网格更新测试
python test_update_grid.py
```

## 测试说明

### 1. run_all_tests.py
按顺序运行所有测试脚本，输出测试总结报告。

### 2. quick_benchmark.py
运行精简版的消融实验：
- 配置: Baseline 和 Full_Hybrid (2组)
- 测试帧数: 100帧
- 场景: Cornell Box
- 输出: CSV数据 + 渲染结果图

### 3. test_features.py
验证所有已实现的功能：
- 基于深度的遮挡判定
- 时域累积(EMA)
- PT参考和误差热力图
- 自适应采样和A-SVGF滤波
- 消融实验配置(4组)
- 高质量学术绘图
- 内存和性能分析

### 4. test_gpu_timing.py
测试GPU计时准确性：
- 测量不同渲染模式的耗时
- 验证同步机制的正确性

### 5. test_data_saving.py
测试数据保存功能：
- CSV文件写入
- 时间戳目录创建
- 截图保存路径

### 6. test_fps_fix.py
测试FPS计算修复：
- 正常情况 (60 FPS)
- 极小帧时间过滤
- 高FPS情况 (1000 FPS)

### 7. test_sync_timing.py
测试GPU同步计时：
- 无同步 vs 有同步
- 验证异步执行检测

### 8. test_update_grid.py
快速测试网格更新功能是否正常。

## 输出目录

测试结果将保存在 `test/test_results_YYYYMMDD_HHMMSS/` 目录下。

## 快速开始

```bash
# 从项目根目录运行
cd Senendays-Taichi-Raytracer-

# 运行所有测试
python test/run_all_tests.py

# 或运行快速benchmark
python test/quick_benchmark.py
```

## 注意事项

- 所有测试脚本会自动添加父目录到Python路径
- 需要确保Taichi已正确安装 (建议使用GPU后端)
- `quick_benchmark.py` 是最推荐的测试方式，运行时间约2分钟
