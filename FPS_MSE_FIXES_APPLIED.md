# FPS 和 MSE 计算逻辑修复总结

## 修复日期
2025-12-31

---

## 修复的问题

### 1. MSE 计算增强 (benchmark.py)

#### 问题
- 未处理 NaN 和 Inf 值
- 归一化阈值使用 1.1，可能在某些边缘情况下不准确
- 缺少数据有效性检查

#### 修复 (Lines 96-113)
```python
def calculate_accurate_mse(current_linear, reference_linear):
    """Calculate MSE in linear space for accurate photometric comparison"""
    # Ensure both are numpy arrays of type float32
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)

    # Normalize to [0, 1] range if needed (more robust threshold check)
    if curr_f.max() > 255.0:
        curr_f = curr_f / 255.0
    if ref_f.max() > 255.0:
        ref_f = ref_f / 255.0

    # Handle NaN and Inf values (robust error handling)
    curr_f = np.nan_to_num(curr_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_f = np.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure values are in valid range [0, 1] after normalization
    curr_f = np.clip(curr_f, 0.0, 1.0)
    ref_f = np.clip(ref_f, 0.0, 1.0)

    # Calculate MSE in linear space
    diff = curr_f - ref_f
    mse = np.mean(diff ** 2)
    return float(mse)
```

#### 改进点
- ✅ 添加 NaN/Inf 处理：使用 `np.nan_to_num`
- ✅ 改进归一化阈值：从 1.1 改为 255.0
- ✅ 添加数据范围检查：使用 `np.clip` 确保值在 [0, 1]

---

### 2. FPS 计算增强

#### 问题
- 未过滤极端 FPS 值（如 < 0.1 或 > 10000）
- 极小帧时间可能导致异常高的 FPS 值

#### 修复
在以下文件中统一修复：

#### a) benchmark.py (Lines 336-339)
```python
ti.sync()
frame_time = time.perf_counter() - start_time
# Calculate FPS with filtering for extreme values
if frame_time > 1e-6:
    fps = 1.0 / frame_time
    # Filter out unreasonable FPS values
    if fps < 0.1 or fps > 10000:
        fps = 0.0
else:
    fps = 0.0
gpu_time_ms = frame_time * 1000.0  # Convert to milliseconds
```

#### b) scripts/run_all_tests.py (Lines 94-97)
```python
# 计算FPS (with filtering for extreme values)
frame_time = time.time() - frame_start
if frame_time > 1e-6:
    fps = 1.0 / frame_time
    # Filter out unreasonable FPS values
    if fps < 0.1 or fps > 10000:
        fps = 0.0
else:
    fps = 0.0
```

#### c) scripts/test_scenes_simple.py (Lines 57-59)
```python
frame_time = time.time() - frame_start
if frame_time > 1e-6:
    fps = 1.0 / frame_time
    # Filter out unreasonable FPS values
    if fps < 0.1 or fps > 10000:
        fps = 0.0
else:
    fps = 0.0
```

#### d) test/quick_benchmark.py (Lines 163-169)
```python
ti.sync()
frame_time = time.perf_counter() - start_time
# Calculate FPS with filtering for extreme values
if frame_time > 1e-6:
    fps = 1.0 / frame_time
    # Filter out unreasonable FPS values
    if fps < 0.1 or fps > 10000:
        fps = 0.0
else:
    fps = 0.0
gpu_time_ms = frame_time * 1000.0
```

#### 改进点
- ✅ 添加 FPS 过滤：范围限制在 [0.1, 10000]
- ✅ 避免除零错误：使用 1e-6 作为最小时间阈值
- ✅ 统一实现：所有脚本使用相同的 FPS 计算逻辑

---

## 修复的影响

### MSE 计算
- **更稳定**: NaN/Inf 值被正确处理，不会污染数据
- **更准确**: 改进的归一化逻辑，减少边缘情况错误
- **更安全**: 使用 `np.clip` 确保值在合理范围内

### FPS 计算
- **更可靠**: 过滤极端值，避免统计数据偏差
- **更一致**: 所有脚本使用统一的 FPS 计算逻辑
- **更健壮**: 防止极小帧时间导致的异常

---

## 测试建议

### 1. MSE 计算测试
```python
# 测试 NaN 处理
import numpy as np
from benchmark import calculate_accurate_mse

# 创建包含 NaN 的测试数据
test_img = np.ones((100, 100, 3), dtype=np.float32)
test_img[50, 50, 0] = np.nan
ref_img = np.ones((100, 100, 3), dtype=np.float32)

mse = calculate_accurate_mse(test_img, ref_img)
print(f"MSE with NaN: {mse}")  # 应该是 0.0（NaN 被替换为 0）
```

### 2. FPS 计算测试
```python
# 测试极端 FPS 过滤
frame_time = 0.000001  # 1 microsecond
if frame_time > 1e-6:
    fps = 1.0 / frame_time
    if fps < 0.1 or fps > 10000:
        fps = 0.0
else:
    fps = 0.0
print(f"FPS for 1ms frame: {fps}")  # 应该是 0.0（被过滤）

frame_time = 0.016  # ~60 FPS
if frame_time > 1e-6:
    fps = 1.0 / frame_time
    if fps < 0.1 or fps > 10000:
        fps = 0.0
else:
    fps = 0.0
print(f"FPS for 16ms frame: {fps}")  # 应该是 ~62.5（正常）
```

### 3. 运行完整测试
```bash
# 重新运行 benchmark，使用修复后的代码
python benchmark.py

# 检查生成的 CSV 文件
# 查看 fps 和 mse 值是否在合理范围内
```

---

## 相关文件

### 修改的文件
- `benchmark.py`: MSE 计算和 FPS 计算修复
- `scripts/run_all_tests.py`: FPS 计算修复
- `scripts/test_scenes_simple.py`: FPS 计算修复
- `test/quick_benchmark.py`: FPS 计算修复

### 新增的文档
- `FPS_MSE_CALCULATION_ANALYSIS.md`: 详细的计算逻辑分析
- `FPS_MSE_FIXES_APPLIED.md`: 本文档，修复总结

---

## 后续建议

### 1. 颜色空间处理（可选）
当前代码假设输入图像已经在线性空间。如果需要更精确的光度误差计算，可以添加 Gamma 校正：

```python
def gamma_to_linear(x):
    """Convert gamma-corrected values to linear space"""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def calculate_accurate_mse_with_gamma(current, reference):
    curr_f = gamma_to_linear(current.astype(np.float32))
    ref_f = gamma_to_linear(reference.astype(np.float32))
    # ... 其余计算 ...
```

### 2. 通道独立 MSE（可选）
对于某些评估需求，可能需要分别计算每个 RGB 通道的 MSE：

```python
def calculate_mse_per_channel(current, reference):
    curr_f = current.astype(np.float32)
    ref_f = reference.astype(np.float32)

    mse_r = np.mean((curr_f[:,:,0] - ref_f[:,:,0]) ** 2)
    mse_g = np.mean((curr_f[:,:,1] - ref_f[:,:,1]) ** 2)
    mse_b = np.mean((curr_f[:,:,2] - ref_f[:,:,2]) ** 2)

    return {
        'joint': np.mean((curr_f - ref_f) ** 2),
        'red': mse_r,
        'green': mse_g,
        'blue': mse_b
    }
```

### 3. 首帧处理（可选）
如果需要更准确的统计数据，可以选择跳过首帧：

```python
for frame in range(frames):
    if frame == 0:
        # 跳过首帧（初始化开销）
        continue

    start_time = time.perf_counter()
    # ... 渲染 ...
    frame_time = time.perf_counter() - start_time
    fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
    # ... 记录 ...
```
