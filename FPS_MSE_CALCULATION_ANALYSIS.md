# FPS 和 MSE 计算逻辑分析文档

## 一、FPS (Frames Per Second) 计算逻辑

### 1. 基本计算公式

```python
frame_time = time.perf_counter() - start_time
fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
```

### 2. 实现位置

| 文件 | 行号 | 说明 |
|------|------|------|
| `benchmark.py` | 328-330 | 主基准测试中的 FPS 计算 |
| `scripts/test_scenes_simple.py` | 51-59 | 简单场景测试 |
| `scripts/run_all_tests.py` | 83-97 | 统一测试脚本 |
| `scripts/grid_resolution_analyzer.py` | 79-96 | 网格分辨率分析 |
| `test/quick_benchmark.py` | 147-169 | 快速基准测试 |

### 3. 计算流程

#### benchmark.py 中的实现
```python
# Line 301-329
ti.sync()
start_time = time.perf_counter()  # 开始计时

# 渲染过程
if COMPARE_RENDER_MODE in (RENDER_MODE_GRID, RENDER_MODE_HYBRID):
    cam.update_grid(world, 0.01)
cam.render(world, COMPARE_RENDER_MODE)
if COMPARE_RENDER_MODE == RENDER_MODE_HYBRID:
    cam.asvgf_filter()

if bool(g.get("adaptive_logic_on", False)):
    import experiment_config as cfg
    cam.compute_adaptive_weights(
        cfg.ADAPTIVE_BRIGHTNESS_THRESHOLD,
        cfg.ADAPTIVE_SAMPLING_MULTIPLIER,
        cfg.ADAPTIVE_MAX_MULTIPLIER,
    )

ti.sync()
frame_time = time.perf_counter() - start_time  # 结束计时

# FPS 计算
fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
gpu_time_ms = frame_time * 1000.0  # 转换为毫秒
```

### 4. FPS 计算逻辑分析

#### ✅ 正确点
1. **使用 `time.perf_counter()`**: 提供微秒级精度，适合性能测试
2. **包含完整渲染流程**: 包括网格更新、渲染、滤波、自适应权重计算
3. **ti.sync() 同步**: 确保所有 GPU 操作完成后才计时
4. **阈值保护**: 避免除以零或极小值导致异常

#### ⚠️ 潜在问题

##### 问题 1: 时间单位转换
```python
fps = 1.0 / frame_time  # frame_time 单位是秒
```
- **正确性**: ✅ 正确，perf_counter 返回的是秒
- **GPU 时间计算**: `frame_time * 1000.0` 正确转换为毫秒

##### 问题 2: 首帧异常
```python
fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
```
- **分析**: 首帧通常需要初始化时间，可能比后续帧慢很多
- **影响**: 首帧 FPS 可能异常低（0.1-0.5 FPS）
- **是否需要修复**: 取决于是否需要包含首帧数据

##### 问题 3: 高 FPS 过滤
```python
fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
```
- 当 `frame_time < 0.000001` 秒时，`fps` 会被设为 0.0
- 这对应 FPS > 1,000,000 的极端情况
- **当前处理**: 直接设为 0，可能在统计时造成误解

---

## 二、MSE (Mean Squared Error) 计算逻辑

### 1. 基本计算公式

```python
def calculate_accurate_mse(current_linear, reference_linear):
    # 转换为 float32
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)
    
    # 归一化到 [0, 1] 范围
    if curr_f.max() > 1.1:
        curr_f = curr_f / 255.0
    if ref_f.max() > 1.1:
        ref_f = ref_f / 255.0
    
    # 线性空间计算 MSE
    mse = np.mean((curr_f - ref_f) ** 2)
    return float(mse)
```

### 2. 实现位置

| 文件 | 行号 | 说明 |
|------|------|------|
| `benchmark.py` | 96-114 | MSE 计算函数 |
| `test/quick_benchmark.py` | 61-67 | 快速测试中的 MSE 计算 |

### 3. 计算流程

#### benchmark.py 中的实现

```python
# Line 96-110: MSE 计算函数
def calculate_accurate_mse(current_linear, reference_linear):
    """Calculate MSE in linear space for accurate photometric comparison"""
    # Ensure both are numpy arrays of type float32
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)

    # Normalize to [0, 1] range if needed
    if curr_f.max() > 1.1:
        curr_f = curr_f / 255.0
    if ref_f.max() > 1.1:
        ref_f = ref_f / 255.0

    # Calculate MSE in linear space
    mse = np.mean((curr_f - ref_f) ** 2)
    return float(mse)

# Line 280-285: 生成 PT 参考图
pt_accum = np.zeros((*cam.img_res, 3), dtype=np.float32)
for _ in range(pt_ref_spp_frames):
    cam.render_pt(world)
    ti.sync()
    pt_accum += cam.pt_frame.to_numpy().astype(np.float32)
pt_reference_linear = pt_accum / float(pt_ref_spp_frames)

# Line 332-333: 计算当前帧的 MSE
current_linear = cam.frame.to_numpy()
mse = calculate_accurate_mse(current_linear, pt_reference_linear)
```

### 4. MSE 计算逻辑分析

#### ✅ 正确点

1. **线性空间计算**: 在线性空间计算 MSE，更准确反映物理亮度差异
2. **参考图平均**: 使用多帧 PT 渲染结果作为参考，降低噪声
3. **自动归一化**: 检测像素值范围并自动归一化

#### ⚠️ 潜在问题

##### 问题 1: 颜色空间处理
```python
# 当前代码使用线性空间直接计算
curr_f = current_linear.astype(np.float32)
ref_f = reference_linear.astype(np.float32)
mse = np.mean((curr_f - ref_f) ** 2)
```

**分析**:
- 代码注释说"线性空间"，但未明确输入是否已经在线性空间
- 如果 `cam.frame` 是 Gamma 校正后的值，应该先进行 Gamma 校正

**建议**:
```python
# 如果输入是 sRGB (gamma 2.2)，需要转换到线性空间
def gamma_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

curr_linear = gamma_to_linear(curr_f)
ref_linear = gamma_to_linear(ref_f)
mse = np.mean((curr_linear - ref_linear) ** 2)
```

##### 问题 2: 归一化阈值 1.1
```python
if curr_f.max() > 1.1:
    curr_f = curr_f / 255.0
```

**分析**:
- 使用 1.1 作为阈值（略大于 1.0）
- 如果图片 HDR 或部分区域值 > 1.0，可能会被错误归一化
- **当前实现**: 可能存在边缘情况处理不当

**建议**:
```python
# 更明确的判断逻辑
if curr_f.max() > 255.0:
    curr_f = curr_f / 255.0
elif curr_f.max() > 1.0:
    # 可能是 HDR 或已归一化的数据
    pass
```

##### 问题 3: 未处理 NaN/Inf
```python
mse = np.mean((curr_f - ref_f) ** 2)
```

**分析**:
- 如果参考图或当前帧包含 NaN 或 Inf，计算结果也会是 NaN
- 需要添加数据有效性检查

**建议**:
```python
diff = curr_f - ref_f
diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
mse = np.mean(diff ** 2)
```

##### 问题 4: 单通道 vs 多通道
```python
mse = np.mean((curr_f - ref_f) ** 2)
```

**分析**:
- 当前计算 RGB 三个通道的联合 MSE
- 对于某些评估需求，可能需要分别计算每个通道的 MSE

**可选改进**:
```python
# 联合 MSE (当前实现)
mse_joint = np.mean((curr_f - ref_f) ** 2)

# 通道独立 MSE
mse_r = np.mean((curr_f[:,:,0] - ref_f[:,:,0]) ** 2)
mse_g = np.mean((curr_f[:,:,1] - ref_f[:,:,1]) ** 2)
mse_b = np.mean((curr_f[:,:,2] - ref_f[:,:,2]) ** 2)
mse_per_channel = (mse_r + mse_g + mse_b) / 3.0
```

---

## 三、数据保存逻辑

### CSV 字段定义

```python
fieldnames=[
    "frame",              # 帧索引
    "mse",               # MSE 值
    "fps",               # FPS 值
    "gpu_time_ms",       # GPU 时间（毫秒）
    "timestamp",         # 时间戳
    "interpolation_on",   # 是否启用插值
    "importance_sampling_on",  # 是否启用重要性采样
    "adaptive_logic_on",  # 是否启用自适应逻辑
    "movement_applied",  # 是否应用了物体移动
    "grid_memory_mb",    # 网格内存占用（MB）
]
```

### 数据记录流程

```python
# Line 336-348: 记录每帧数据
group_rows.append({
    "frame": int(f),
    "mse": float(mse),
    "fps": float(fps),
    "gpu_time_ms": float(gpu_time_ms),
    "timestamp": datetime.now().isoformat(),
    "interpolation_on": bool(g.get("interpolation_on", False)),
    "importance_sampling_on": bool(g.get("importance_sampling_on", False)),
    "adaptive_logic_on": bool(g.get("adaptive_logic_on", False)),
    "movement_applied": bool(moved_this_frame),
    "grid_memory_mb": float(
        cam.grid_res[0] * cam.grid_res[1] * cam.grid_res[2] * 3 * 4 / (1024.0 * 1024.0)
    ),
})
```

### 内存计算分析

```python
grid_memory_mb = (
    cam.grid_res[0] * cam.grid_res[1] * cam.grid_res[2] * 3 * 4 / (1024.0 * 1024.0)
)
```

**公式解析**:
- `cam.grid_res[0] * cam.grid_res[1] * cam.grid_res[2]`: 网格总单元数
- `3`: RGB 三个通道
- `4`: float32 每个元素 4 字节
- `/ (1024.0 * 1024.0)`: 转换为 MB

**验证** (以 64^3 网格为例):
```
64 * 64 * 64 = 262,144 单元
262,144 * 3 * 4 = 3,145,728 字节
3,145,728 / (1024 * 1024) = 3.0 MB
```

**正确性**: ✅ 计算正确

---

## 四、总结与建议

### ✅ 当前实现优点

1. **FPS 计算**: 使用高精度计时器，包含完整渲染流程
2. **MSE 计算**: 在线性空间计算，自动归一化
3. **数据记录**: 完整记录每帧的元数据
4. **内存估算**: 准确计算网格内存占用

### ⚠️ 需要修复的问题

#### 优先级 1: 颜色空间处理
- **问题**: 未明确输入图像的颜色空间
- **修复**: 确保所有输入都在线性空间后再计算 MSE
- **影响**: 可能导致 MSE 值不准确

#### 优先级 2: NaN/Inf 处理
- **问题**: 未处理异常值
- **修复**: 添加 `np.nan_to_num` 处理
- **影响**: 极端情况下可能导致数据损坏

#### 优先级 3: 首帧过滤
- **问题**: 首帧 FPS 异常低
- **修复**: 可选择跳过首帧或单独统计
- **影响**: 影响统计数据

### 建议的修复代码

```python
def calculate_accurate_mse(current_linear, reference_linear):
    """Calculate MSE in linear space with robust error handling"""
    # 转换为 float32
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)
    
    # 归一化到 [0, 1] 范围
    if curr_f.max() > 255.0:
        curr_f = curr_f / 255.0
    if ref_f.max() > 255.0:
        ref_f = ref_f / 255.0
    
    # 处理 NaN 和 Inf
    curr_f = np.nan_to_num(curr_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_f = np.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 确保值在合理范围内
    curr_f = np.clip(curr_f, 0.0, 1.0)
    ref_f = np.clip(ref_f, 0.0, 1.0)
    
    # 计算线性空间 MSE
    diff = curr_f - ref_f
    mse = np.mean(diff ** 2)
    
    return float(mse)

def calculate_fps_with_filter(frame_time, min_fps_threshold=0.1, max_fps_threshold=1000):
    """Calculate FPS with filtering for extreme values"""
    if frame_time <= 1e-6:
        return 0.0  # 极小时间，避免除零
    
    fps = 1.0 / frame_time
    
    # 过滤异常值
    if fps < min_fps_threshold or fps > max_fps_threshold:
        return 0.0
    
    return fps
```

---

## 五、附录：相关代码位置索引

### FPS 计算
- `benchmark.py`: 328-330, 481-488
- `scripts/test_scenes_simple.py`: 51-59
- `scripts/run_all_tests.py`: 83-97
- `test/quick_benchmark.py`: 147-169

### MSE 计算
- `benchmark.py`: 96-114
- `test/quick_benchmark.py`: 61-67

### 数据保存
- `benchmark.py`: 336-348, 179-199
- `test/quick_benchmark.py`: 87-89
