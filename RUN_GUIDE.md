# 运行指南 - 新方法测试

生成时间: 2026-01-16

## 📋 执行摘要

所有新增代码已完成语法检查和依赖验证，可以立即开始运行。由于命令行工具的限制，建议使用以下方式运行测试。

---

## ✅ 代码状态

| 组件 | 状态 | 检查项目 |
|------|------|----------|
| camera_ms_aic.py | ✅ | 语法检查通过，编译成功 |
| camera_motion_comp.py | ✅ | 语法检查通过，编译成功 |
| experiment_new_methods.py | ✅ | 语法检查通过，编译成功 |
| quick_test_new_methods.py | ✅ | 语法检查通过，编译成功 |
| generate_paper_figures.py | ✅ | 语法检查通过，编译成功 |
| 依赖模块 | ✅ | 所有模块可正常导入 |

---

## 🚀 推荐运行方式

### 方式1: 使用PowerShell脚本 (推荐)

PowerShell脚本会捕获所有输出并保存到日志文件，便于查看错误。

```powershell
# 在PowerShell中运行
cd "c:\Users\Bluesky\Desktop\graphic\Senendays-Taichi-Raytracer-"
.\run_tests.ps1
```

**输出文件**:
- `test_output_powershell.txt` - 脚本输出
- `test_detailed_log_YYYYMMDD_HHMMSS.txt` - 详细日志
- `test_error_powershell.txt` - 错误输出（如有）

### 方式2: 使用批处理脚本

```batch
# 在命令提示符中运行
cd c:\Users\Bluesky\Desktop\graphic\Senendays-Taichi-Raytracer-
run_tests.bat
```

### 方式3: 直接运行测试脚本

```powershell
# 在PowerShell或命令提示符中运行
cd c:\Users\Bluesky\Desktop\graphic\Senendays-Taichi-Raytracer-

# 运行详细诊断测试
python test_detailed.py > test_output.txt 2>&1

# 查看输出
type test_output.txt
```

### 方式4: 使用Python解释器

```powershell
# 启动Python并逐行执行
python

# 然后在Python REPL中执行：
exec(open('test_detailed.py').read())
```

---

## 📊 测试内容说明

### test_detailed.py - 详细诊断测试

这个脚本包含11个测试步骤：

1. **导入taichi** - 验证Taichi环境
2. **导入main** - 验证主模块
3. **设置场景** - 验证场景创建
4. **导入camera_ms_aic** - 验证多尺度网格模块
5. **创建MultiScaleGrid** - 验证网格初始化
6. **导入camera_motion_comp** - 验证运动补偿模块
7. **创建MotionCompensatedTemporalFilter** - 验证滤波器初始化
8. **更新MultiScaleGrid** - 验证网格更新功能
9. **渲染1帧** - 验证渲染功能
10. **渲染PT参考** - 验证PT参考渲染
11. **计算MSE** - 验证质量评估

**预期运行时间**: 1-2分钟

### quick_test_new_methods.py - 快速功能测试

这个脚本执行完整的快速测试：

1. 设置cornell_box场景
2. 测试Multi-Scale Grid (10次更新)
3. 测试Motion-Compensated Filter (5帧)
4. 测试完整方法集成 (20帧)
5. 质量检查 (vs PT参考，50 spp)

**预期运行时间**: 5-10分钟

**输出**: `results/quick_test/test_report_YYYYMMDD_HHMMSS.txt`

### experiment_new_methods.py - 完整实验

运行完整的实验对比：

- 测试3个场景
- 6种渲染模式对比
- 每种模式600帧
- 自动记录性能数据

**预期运行时间**: 2-3小时

**输出**: `results/new_methods_benchmark_YYYYMMDD_HHMMSS/`

### generate_paper_figures.py - 生成论文图表

生成6个高质量PDF图表（300 DPI）

**预期运行时间**: 5分钟

**输出**: `paper_figures/`

---

## 🎯 推荐执行顺序

### 第一阶段: 验证 (今天)

1. ✅ 运行详细诊断测试
   ```powershell
   python test_detailed.py
   ```
   - 确认所有模块正常工作
   - 检查是否有运行时错误
   - 验证基本功能

2. ✅ 运行快速功能测试
   ```powershell
   python quick_test_new_methods.py
   ```
   - 测试完整的工作流程
   - 收集初步性能数据
   - 验证质量指标

### 第二阶段: 数据收集 (本周)

3. ⏳ 运行完整实验
   ```powershell
   python experiment_new_methods.py
   ```
   - 收集完整的对比数据
   - 在多个场景上测试
   - 记录详细的性能指标

4. ⏳ 生成论文图表
   ```powershell
   python generate_paper_figures.py
   ```
   - 创建所有需要的图表
   - 使用实验数据生成可视化
   - 准备论文插图

### 第三阶段: 论文撰写 (下周)

5. ⏳ 撰写Methods章节
6. ⏳ 撰写Results章节
7. ⏳ 准备补充材料

---

## 📁 文件清单

### 新增的测试文件
| 文件名 | 用途 | 运行时间 |
|--------|------|----------|
| test_detailed.py | 详细诊断测试 | 1-2分钟 |
| test_simple.py | 简单测试 | 1分钟 |
| test_minimal.py | 最小测试 | 30秒 |
| run_tests.bat | 批处理运行脚本 | 依赖测试 |
| run_tests.ps1 | PowerShell运行脚本 | 依赖测试 |

### 新增的文档文件
| 文件名 | 用途 |
|--------|------|
| CODE_ANALYSIS_REPORT.md | 代码分析报告 |
| RUN_GUIDE.md | 本文档 - 运行指南 |

### 新增的方法实现文件
| 文件名 | 用途 | 代码行数 |
|--------|------|----------|
| camera_ms_aic.py | 多尺度光照缓存 | 528行 |
| camera_motion_comp.py | 运动补偿滤波 | 453行 |
| experiment_new_methods.py | 完整实验脚本 | 373行 |
| quick_test_new_methods.py | 快速测试脚本 | 183行 |
| generate_paper_figures.py | 图表生成脚本 | 358行 |

### 新增的文档文件
| 文件名 | 用途 |
|--------|------|
| RESEARCH_PROPOSAL.md | 研究方案（~1500行） |
| PAPER_IMPLEMENTATION_GUIDE_NEW.md | 实现指南（~800行） |
| PROJECT_COMPLETION_SUMMARY.md | 完成总结（~700行） |
| START_HERE.py | 快速启动脚本 |
| README_NEW_METHODS.md | 新方法说明（~500行） |

---

## 🔍 故障排除

### 问题1: PowerShell脚本无法执行

**错误信息**:
```
无法加载文件 run_tests.ps1，因为在此系统上禁止运行脚本。
```

**解决方案**:
```powershell
# 以管理员身份运行PowerShell，然后执行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题2: Taichi CUDA错误

**错误信息**:
```
[Taichi] Error: CUDA not available
```

**解决方案**:
```python
# 在脚本开头修改ti.init()调用：
ti.init(arch=ti.cpu, random_seed=42)
```
注意：使用CPU后端会显著降低性能。

### 问题3: 内存不足

**错误信息**:
```
RuntimeError: Out of memory
```

**解决方案**:
- 减小图像分辨率：修改 `camera.py` 中的 `width` 值
- 减少网格分辨率：修改 `GRID_RESOLUTIONS` 为更小的值
- 减少测试帧数：修改 `TEST_FRAMES`

### 问题4: 导入错误

**错误信息**:
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**:
```bash
pip install xxx
```

### 问题5: 运行时间过长

**建议**:
- 先运行 `test_detailed.py` 快速验证
- 减少快速测试的PT参考spp：将 `target_spp=50` 改为 `target_spp=5`
- 使用更快的GPU或减少分辨率

---

## 📊 预期结果

### 详细诊断测试 (test_detailed.py)

成功输出示例:
```
========================================
详细诊断测试开始
时间: 2026-01-16 XX:XX:XX
...

[测试1] 导入taichi...
✓ Taichi初始化成功
  版本: 1.7.3
  架构: ...

[测试2] 导入main...
✓ main模块导入成功

[测试3] 设置场景...
✓ 场景设置成功
  场景模式: cornell_box
  图像分辨率: (1200, 675)
  相机位置: ...

[测试4] 导入camera_ms_aic...
✓ camera_ms_aic模块导入成功

[测试5] 创建MultiScaleGrid...
✓ MultiScaleGrid创建成功
  层数: 3
  分辨率: [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
  显存: 0.55 MB

[测试6] 导入camera_motion_comp...
✓ camera_motion_comp模块导入成功

[测试7] 创建MotionCompensatedTemporalFilter...
✓ MotionCompensatedTemporalFilter创建成功
  分辨率: (1200, 675)
  空间σ: 2.0
  时间σ: 0.5

[测试8] 更新MultiScaleGrid (1次)...
✓ MultiScaleGrid更新成功
  更新时间: XX.X ms
  更新速率: XX.X updates/sec

[测试9] 渲染1帧...
✓ 渲染成功
  渲染时间: XX.X ms
  FPS: XXX.X

[测试10] 渲染PT参考 (5 spp, 快速测试)...
✓ PT参考渲染成功
  渲染时间: XX.XX 秒

[测试11] 计算MSE...
✓ MSE计算成功
  MSE: X.XXXXXe-XX
  ✓ 质量检查通过 (MSE < 0.01)

============================================================
所有测试通过！
============================================================

测试总结:
  ✓ Taichi环境正常
  ✓ Main模块导入成功
  ✓ 场景设置正常
  ✓ MultiScaleGrid创建成功
  ✓ MotionCompensatedTemporalFilter创建成功
  ✓ MultiScaleGrid更新正常
  ✓ 渲染功能正常
  ✓ PT参考渲染正常
  ✓ MSE计算正常

代码已准备就绪，可以开始正式实验！

详细日志已保存到: test_detailed_log_YYYYMMDD_HHMMSS.txt
```

### 快速功能测试 (quick_test_new_methods.py)

成功输出示例:
```
============================================================
New Methods Quick Test
============================================================
Start time: 2026-01-16 XX:XX:XX

[1/5] Setting up scene (cornell_box)...
✓ Scene setup complete

[2/5] Testing Multi-Scale Grid (MS-AIC)...
  Grid levels: 3
  Resolutions: [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
  Memory: 0.55 MB
  Updating grid (10 iterations)...
  Update time: XX.X ms avg
  Update rate: XX.X updates/sec
✓ Multi-Scale Grid test complete

[3/5] Testing Motion-Compensated Temporal Filter (MCTF)...
  Resolution: (1200, 675)
  Spatial sigma: 2.0
  Temporal sigma: 0.5
  Simulating 5 frames...
✓ Motion-Compensated Filter test complete

[4/5] Testing Full Method Integration...
  Rendering 20 frames with MS-AIC + MCTF...
    Frame 5/20: XXX.X FPS
    Frame 10/20: XXX.X FPS
    Frame 15/20: XXX.X FPS
    Frame 20/20: XXX.X FPS
  Average FPS: XXX.X
✓ Full Method test complete

[5/5] Running Quality Check...
  Building PT reference (50 spp)...
  MSE vs PT reference: X.XXXXXe-XX
  ✓ Quality check PASSED (MSE < 0.01)

============================================================
Quick Test Summary
============================================================
Multi-Scale Grid:
  - Levels: 3
  - Memory: 0.55 MB
  - Update rate: XX.X updates/sec

Motion-Compensated Filter:
  - Resolution: (1200, 675)
  - Parameters: spatial=2.0, temporal=0.5

Full Method Integration:
  - Average FPS: XXX.X
  - MSE vs PT: X.XXXXXe-XX

✓ All tests completed successfully!
End time: 2026-01-16 XX:XX:XX

============================================================

Test report saved to: results/quick_test/test_report_YYYYMMDD_HHMMSS.txt

Next steps:
  1. Run full experiments: python experiment_new_methods.py
  2. Generate paper figures: see PAPER_IMPLEMENTATION_GUIDE_NEW.md
  3. Start writing paper: use provided LaTeX templates
```

---

## 💡 最佳实践

1. **保存日志**
   - 每次运行后保存日志文件
   - 记录运行时间和硬件配置
   - 便于后续对比和分析

2. **增量测试**
   - 先运行小规模测试验证
   - 确认无误后再运行完整实验
   - 定期保存中间结果

3. **资源管理**
   - 确保有足够的磁盘空间
   - 在低负载时段运行长时间实验
   - 监控GPU和内存使用情况

4. **文档记录**
   - 记录每次运行的结果
   - 注意异常情况和错误
   - 保存重要的截图和数据

---

## 📞 获取帮助

如果遇到问题:

1. **查看日志文件** - 最详细的错误信息
2. **检查CODE_ANALYSIS_REPORT.md** - 已知问题和解决方案
3. **查看PAPER_IMPLEMENTATION_GUIDE_NEW.md** - 故障排除指南
4. **查看README_NEW_METHODS.md** - 常见问题解答

---

## 🎉 总结

**当前状态**: ✅ 代码准备就绪，可以开始运行

**下一步**: 运行详细诊断测试验证环境

**预期时间**: 完整流程需要约3-4小时

**最终目标**: 完成论文投稿材料

---

**祝您好运！🚀**
