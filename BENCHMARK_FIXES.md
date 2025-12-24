# Benchmark 数据异常修复报告

## 🚨 发现的问题

### 1. FPS 数据异常
- **症状**: 大量 0 FPS 值和不合理的高值（如 2987 FPS）
- **原因**: 
  - 使用 `time.time()` 精度不够
  - 没有过滤异常值
  - 除零保护不够严格

### 2. MSE 计算问题
- **症状**: PT 模式 MSE 始终为 0，但没有正确的参考帧
- **原因**: PT 参考帧存储时机不正确，每帧都在更新

### 3. 数据保存问题
- **症状**: 数据可能丢失，保存时机不当
- **原因**: 每50帧保存一次，间隔太长

## ✅ 修复方案

### 1. FPS 计算改进
```python
# 修复前
frame_time = time.time() - start_time
fps = 1.0 / frame_time if frame_time > 0 else 0

# 修复后
frame_time = time.perf_counter() - start_time
fps = 1.0 / frame_time if frame_time > 0.0001 else 0
if fps > 10000:  # 过滤异常值
    fps = 0.0
    log_message(f"Warning: Unusually high FPS detected and filtered")
```

### 2. MSE 计算修复
```python
# 修复前：每帧都更新 PT 参考
if render_mode == RENDER_MODE_PT:
    pt_reference = current_frame.to_numpy()

# 修复后：只在 PT 模式结束时存储参考帧
if render_mode == RENDER_MODE_PT and current_mode_frames == mode_frames - 1:
    pt_reference = current_frame.to_numpy()
    log_message("PT reference frame stored for MSE comparison")
```

### 3. 数据保存优化
```python
# 修复前：每50帧保存
if len(benchmark_data) >= 50:
    flush_benchmark_data()

# 修复后：每10帧保存，更及时
if len(benchmark_data) >= 10:
    flush_benchmark_data()
```

### 4. CSV 写入改进
```python
# 新增：智能头部写入
file_exists = os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["frame", "mode", "fps", "mse", "timestamp"])
    writer.writerows(benchmark_data)
```

## 📊 预期效果

### 正常 FPS 范围
- **Path Tracing**: 10-50 FPS（计算密集）
- **Pure Grid**: 100-1000 FPS（快速）
- **Hybrid**: 30-100 FPS（中等）

### 正常 MSE 范围
- **Pure Grid vs PT**: 0.02-0.1
- **Hybrid vs PT**: 0.01-0.05

### 数据完整性
- 每10帧自动保存，减少数据丢失风险
- 异常退出时自动保存剩余数据
- 实时显示已保存的记录数

## 🔧 使用建议

### 1. 运行基准测试
```bash
python benchmark.py
```

### 2. 查看实时数据
- GUI 左下角显示 "Data: X records"
- 控制台显示保存日志
- 数据实时写入时间戳目录

### 3. 结果分析
每个运行会创建独立目录：
```
benchmark_results_20251224_111838/
├── benchmark_results.csv  # 详细性能数据
├── result_path_tracing.png
├── result_pure_grid.png
└── result_hybrid.png
```

## 🧪 验证方法

运行测试脚本验证修复：
```bash
python test_fps_fix.py
python test_data_saving.py
```

## 📈 性能监控

### 关键指标
1. **FPS 稳定性**: 避免异常 0 或超高值
2. **MSE 收敛**: 随帧数增加而降低
3. **数据完整性**: 所有帧数据都被记录

### 异常处理
- FPS > 10000: 自动过滤并警告
- frame_time < 0.0001s: 设置为 0 FPS
- 程序中断: 自动保存已收集数据

## 🎯 下一步改进

1. **统计分析**: 添加平均 FPS、标准差等统计指标
2. **可视化**: 生成实时性能图表
3. **对比模式**: 支持多次运行结果对比
4. **配置化**: 可调节帧数、保存间隔等参数