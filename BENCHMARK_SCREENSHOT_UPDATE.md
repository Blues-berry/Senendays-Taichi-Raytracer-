# Benchmark 截图功能更新

## 📸 **新增截图时机**

现在 `benchmark.py` 会在以下帧数自动保存截图：

### 🔸 **固定帧数截图**
- **第 5 帧**: `{mode}_frame_5.png`
- **第 50 帧**: `{mode}_frame_50.png` 
- **第 100 帧**: `{mode}_frame_100.png`
- **第 150 帧**: `{mode}_frame_150.png`

### 🔸 **模式结束截图**
- **最后帧**: `result_{mode}.png`

## 📁 **输出文件命名规则**

```
results/benchmark_results_YYYYMMDD_HHMMSS/
├── path_tracing_frame_5.png      # PT模式第5帧
├── path_tracing_frame_50.png     # PT模式第50帧
├── path_tracing_frame_100.png    # PT模式第100帧
├── path_tracing_frame_150.png    # PT模式第150帧
├── result_path_tracing.png       # PT模式最终结果
├── pure_grid_frame_5.png          # Grid模式第5帧
├── pure_grid_frame_50.png         # Grid模式第50帧
├── pure_grid_frame_100.png        # Grid模式第100帧
├── pure_grid_frame_150.png        # Grid模式第150帧
├── result_pure_grid.png           # Grid模式最终结果
├── hybrid_frame_5.png             # Hybrid模式第5帧
├── hybrid_frame_50.png            # Hybrid模式第50帧
├── hybrid_frame_100.png           # Hybrid模式第100帧
├── hybrid_frame_150.png           # Hybrid模式第150帧
├── result_hybrid.png               # Hybrid模式最终结果
└── benchmark_results.csv          # 基准测试数据
```

## 🎯 **使用场景**

1. **渐进式分析**: 可以观察到每种渲染模式在不同帧数的渐进效果
2. **性能对比**: 对比不同模式在相同帧数下的渲染质量
3. **收敛分析**: 观察各种模式何时达到可接受的视觉效果
4. **实验记录**: 为研究报告提供详细的中间过程图像

## 🔧 **修改内容**

在 `benchmark.py` 中添加了固定帧数的截图逻辑：

```python
# Save screenshot at specified frames: 5, 50, 100, 150
if current_mode_frames + 1 in [5, 50, 100, 150]:
    mode_name = get_mode_name(render_mode).lower().replace(" ", "_")
    filename = f"{mode_name}_frame_{current_mode_frames + 1}.png"
    save_screenshot(gui, filename)
```

现在运行基准测试时，会自动在指定帧数保存截图，方便进行详细的渲染效果分析！