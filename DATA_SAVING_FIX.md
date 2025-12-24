# 数据保存修复说明

## 修复的问题

### 1. 原始问题
- `experiment_log.csv` 文件为空（0字节）
- 数据没有正确保存
- 每次运行会覆盖之前的数据

### 2. 修复内容

#### Benchmark.py 修复：
1. **自动创建时间戳目录**：每次运行都会创建格式为 `benchmark_results_YYYYMMDD_HHMMSS` 的目录
2. **实时数据保存**：每50帧自动刷新数据到CSV文件，防止数据丢失
3. **错误处理**：添加了异常捕获和优雅退出处理
4. **进度显示**：在GUI中显示已保存的记录数

#### Main.py 修复：
1. **实验目录管理**：创建格式为 `experiment_YYYYMMDD_HHMMSS` 的时间戳目录
2. **日志文件修复**：确保实验日志正确写入指定目录
3. **截图路径修复**：所有截图都保存到对应的时间戳目录
4. **最终输出修复**：最终渲染图片保存到实验目录

## 使用方法

### 运行基准测试：
```bash
python benchmark.py
```

### 运行主程序（带实验自动化）：
```bash
python main.py
```

## 输出结构

### Benchmark 运行后：
```
benchmark_results_20251224_111402/
├── benchmark_results.csv    # 性能数据
├── result_path_tracing.png  # PT模式结果图
├── result_pure_grid.png     # 网格模式结果图
└── result_hybrid.png        # 混合模式结果图
```

### Main 程序运行后：
```
experiment_20251224_111402/
├── experiment_log.csv       # 实验日志
├── Adaptive_move_1_frame_5.png    # 移动后第5帧截图
├── Adaptive_move_1_frame_50.png   # 移动后第50帧截图
└── output_Adaptive_20251224_111402.png  # 最终输出图
```

## 数据格式

### Benchmark CSV 格式：
| frame | mode | fps | mse | timestamp |
|-------|------|-----|-----|-----------|
| 1     | Path Tracing | 30.5 | 0.0 | 2025-12-24T11:14:02.308595 |

### Experiment Log CSV 格式：
| frame_index | mode | FPS | MSE |
|-------------|------|-----|-----|
| 100         | Adaptive | 35.7 | 0.000567 |

## 新增功能

1. **数据安全**：
   - 每50帧自动保存，防止程序崩溃导致数据丢失
   - 异常处理确保优雅退出

2. **版本管理**：
   - 每次运行创建独立目录，避免数据覆盖
   - 时间戳确保可追溯性

3. **可视化改进**：
   - 实时显示保存的记录数
   - 清晰的日志输出

## 验证修复

运行测试脚本验证修复效果：
```bash
python test_data_saving.py
```

## 故障排除

如果仍有保存问题：
1. 检查目录权限
2. 确保磁盘空间充足
3. 检查文件是否被其他程序占用
4. 运行测试脚本验证基本功能