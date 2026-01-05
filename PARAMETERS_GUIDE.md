# 参数配置完整指南 (Parameters Configuration Guide)

本文档详细说明项目中的所有可配置参数。

---

## 概述

项目参数分为三类：
1. **实验配置** (`experiment_config.py`): 全局实验参数
2. **渲染参数** (`camera.py`, `benchmark.py`): 渲染和测试参数
3. **场景参数** (`scenes/scene_configs.py`): 场景定义参数

---

## 1. 实验配置 (experiment_config.py)

### 实验设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EXPERIMENT_DURATION_FRAMES` | 1000 | 总运行帧数 |
| `SPHERE_MOVE_INTERVAL` | 200 | 大球移动间隔（帧） |
| `BASE_UPDATE_PROBABILITY` | 0.05 | 网格基础更新概率 |

### 性能监控

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FPS_LOG_INTERVAL` | 100 | FPS 日志间隔（帧） |
| `MEMORY_LOG_INTERVAL` | 100 | 内存日志间隔（帧） |

### 收敛检测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MIN_FRAMES_BEFORE_CONVERGENCE_CHECK` | 5 | 移动后开始收敛检查的最小帧数 |
| `CONVERGENCE_CHECK_INTERVAL` | 10 | 收敛检查间隔（帧） |

### 截图设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SAVE_SCREENSHOT_AFTER_MOVE_FRAME` | 5 | 移动后保存截图的帧数 |
| `SAVE_SCREENSHOT_CONVERGED_FRAME` | 50 | 收敛时保存截图的帧数 |

### 自适应逻辑设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ADAPTIVE_BOOST_MULTIPLIER` | 1.0 | 自适应区域的权重倍数 |
| `ADAPTIVE_INFLUENCE_RADIUS` | 3.0 | 影响半径（球半径的倍数） |
| `GAUSSIAN_BLUR_ENABLED` | False | 是否启用高斯模糊平滑权重 |

### 网格设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GRID_RESOLUTION` | (32, 32, 32) | 网格分辨率 (nx, ny, nz) |
| `GRID_PADDING` | 0.5 | 场景 AABB 的扩展距离 |

**网格显存计算**:
```python
grid_memory_mb = (
    GRID_RESOLUTION[0] * GRID_RESOLUTION[1] * GRID_RESOLUTION[2] *
    3 * 4 /  # RGB 3 通道 * float32 4 字节
    (1024.0 * 1024.0)  # 转换为 MB
)
```

| 分辨率 | 单元总数 | 显存 (MB) |
|--------|---------|-----------|
| (16, 16, 16) | 4,096 | 0.05 |
| (32, 32, 32) | 32,768 | 0.37 |
| (64, 64, 64) | 262,144 | 3.00 |
| (128, 128, 128) | 2,097,152 | 24.00 |

### 自适应采样设置（基于亮度）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ADAPTIVE_BRIGHTNESS_THRESHOLD` | 0.05 | 触发额外采样的亮度对比阈值 |
| `ADAPTIVE_SAMPLING_MULTIPLIER` | 1.0 | 阈值以上时的额外采样倍数 |
| `ADAPTIVE_MAX_MULTIPLIER` | 2.0 | 最大采样倍数上限 |

**工作原理**:
```python
# 计算亮度对比
if brightness > threshold:
    weight = 1.0 + min(sampling_multiplier, max_multiplier)
else:
    weight = 1.0
```

### 重要性采样 / NEE (Next Event Estimation)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LIGHT_IMPORTANCE_SCALE` | 2.0 | 光源重要性采样缩放因子 |

**作用**: 在网格探测时，发光材质的击中获得更高权重。

### 方差引导采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VARIANCE_SAMPLING_SCALE` | 2.0 | 方差转换为额外探针采样数的缩放因子 |
| `MAX_PROBE_SAMPLES` | 16 | 每个网格单元的最大探针采样数 |

**工作原理**:
```python
extra_samples = variance * VARIANCE_SAMPLING_SCALE
total_samples = min(base_samples + extra_samples, MAX_PROBE_SAMPLES)
```

### 距离防漏光

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DISTANCE_MISMATCH_THRESHOLD` | 1.0 | 检测不匹配的阈值（网格单元尺寸的倍数） |

**作用**: 如果实际击中距离与网格平均距离偏差超过此阈值，认为被遮挡，权重置为 0。

### 输出设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_DIRECTORY` | "experiment_results" | 输出目录 |
| `SAVE_FPS_DATA` | True | 是否保存 FPS 数据 |
| `SAVE_CONVERGENCE_DATA` | True | 是否保存收敛数据 |
| `SAVE_SCREENSHOTS` | True | 是否保存截图 |
| `SAVE_SUMMARY` | True | 是否保存摘要 |

### 对比模式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RUN_COMPARISON` | True | 是否运行对比实验 |
| `COMPARISON_ADAPTIVE_FIRST` | True | 是否先运行自适应实验 |

### 日志设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VERBOSE_LOGGING` | True | 启用详细日志 |
| `TIMESTAMP_LOGGING` | True | 为所有日志添加时间戳 |

---

## 2. 渲染参数

### benchmark.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `test_frames` | 600 | 每组测试的帧数 |
| `movement_frame` | 200 | 物体位移触发帧 |
| `pt_ref_spp_frames` | 150 | PT 参考样本累积帧数 |
| `reference_spp` | 512 | 热力图 PT 参考 SPP |

### 渲染模式

```python
mode_map = {
    'PT': 0,      # Path Tracing
    'Grid': 1,    # Pure Grid
    'Adaptive': 2, # Hybrid Adaptive
    'ERROR': 3     # Error Heatmap
}
```

### FPS 上限过滤

```python
max_fps = {
    RENDER_MODE_PT: 200,      # Path Tracing 慢速
    RENDER_MODE_GRID: 2000,    # Grid 方法快速
    RENDER_MODE_HYBRID: 500    # Hybrid 中等速度
}

if fps > max_fps[mode]:
    fps = 0.0  # 标记为异常值
```

---

## 3. 场景参数 (scenes/scene_configs.py)

### 场景通用参数

#### 相机参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lookfrom` | vec3(...) | 相机位置 |
| `lookat` | vec3(...) | 观察目标 |
| `vup` | vec3(0, 1, 0) | 上方向向量 |
| `vfov` | 40-60 | 垂直视场角（度） |
| `defocus_angle` | 0.0-0.6 | 景深角度（0=无景深） |
| `focus_dist` | 8.0-12.0 | 聚焦距离 |
| `scene_mode` | str | 场景模式标识 |

#### 大球近似平面

```python
R = 1000.0  # 大球半径，越大越接近平面
```

墙壁、地板、天花板使用大球近似：
```python
# 地面
Sphere(center=vec3(0, -(R + half), 0), radius=R)

# 左墙
Sphere(center=vec3(-(R + half), 0, 0), radius=R)
```

---

## 4. 材质参数

### Lambert（漫反射）

```python
material.Lambert(color)
```

**参数**:
- `color`: vec3(R, G, B), 范围 [0, 1]

**示例**:
```python
white = vec3(0.73, 0.73, 0.73)
material.Lambert(white)
```

### Metal（金属）

```python
material.Metal(color, fuzz)
```

**参数**:
- `color`: vec3(R, G, B), 范围 [0, 1]
- `fuzz`: float, 粗糙度 [0.0, 1.0]

**fuzz 值说明**:
- `0.0`: 完美镜面反射
- `0.1`: 轻微模糊
- `0.5`: 中等模糊
- `1.0`: 高度模糊

**示例**:
```python
# 镜面
mirror = material.Metal(vec3(0.98, 0.98, 1.0), 0.0)

# 粗糙金属
rough_metal = material.Metal(vec3(0.8, 0.7, 0.6), 0.5)
```

### Dielectric（电介质/玻璃）

```python
material.Dielectric(refractive_index)
```

**参数**:
- `refractive_index`: float, 折射率

**常见材质折射率**:
- 空气: 1.0
- 水: 1.33
- 玻璃: 1.5
- 钻石: 2.42

**示例**:
```python
# 水
water = material.Dielectric(1.33)

# 玻璃
glass = material.Dielectric(1.5)

# 钻石
diamond = material.Dielectric(2.42)
```

### DiffuseLight（发光）

```python
material.DiffuseLight(color)
```

**参数**:
- `color`: vec3(R, G, B), 发光强度

**强度参考**:
- 弱光: vec3(1, 1, 1) - vec3(5, 5, 5)
- 中等: vec3(10, 10, 10) - vec3(25, 25, 25)
- 强光: vec3(50, 50, 50) - vec3(100, 100, 100)

**示例**:
```python
# 弱光
weak_light = material.DiffuseLight(vec3(5, 5, 5))

# 面光源（Cornell Box）
area_light = material.DiffuseLight(vec3(25, 25, 25))

# 彩色光
colored_light = material.DiffuseLight(vec3(20, 10, 5))  # 暖橙色
```

---

## 5. 参数调优指南

### 性能优先

**目标**: 最大化 FPS，可牺牲部分质量

```python
# experiment_config.py
GRID_RESOLUTION = (16, 16, 16)          # 最低网格分辨率
MAX_PROBE_SAMPLES = 4                     # 降低采样数
ADAPTIVE_SAMPLING_MULTIPLIER = 0.5         # 减少自适应采样
ADAPTIVE_BRIGHTNESS_THRESHOLD = 0.1        # 提高阈值，减少触发
```

### 质量优先

**目标**: 最大化渲染质量，可接受较低 FPS

```python
# experiment_config.py
GRID_RESOLUTION = (64, 64, 64)           # 高网格分辨率
MAX_PROBE_SAMPLES = 32                    # 增加采样数
ADAPTIVE_SAMPLING_MULTIPLIER = 2.0         # 增加自适应采样
ADAPTIVE_BRIGHTNESS_THRESHOLD = 0.02       # 降低阈值，更灵敏
VARIANCE_SAMPLING_SCALE = 3.0              # 增加方差引导
```

### 平衡配置（推荐）

**目标**: 在性能和质量之间取得平衡

```python
# experiment_config.py
GRID_RESOLUTION = (32, 32, 32)           # 中等网格分辨率
MAX_PROBE_SAMPLES = 16                    # 默认采样数
ADAPTIVE_SAMPLING_MULTIPLIER = 1.0         # 默认倍数
ADAPTIVE_BRIGHTNESS_THRESHOLD = 0.05       # 默认阈值
```

### 显存受限

**目标**: 降低显存占用

```python
# experiment_config.py
GRID_RESOLUTION = (16, 16, 16)  # 0.05 MB
# 或 (24, 24, 24)  # 0.16 MB
```

---

## 6. 参数计算公式

### FPS 计算

```python
frame_time = time.perf_counter() - start_time
fps = 1.0 / frame_time if frame_time > 1e-6 else 0.0
```

### MSE 计算

```python
def calculate_accurate_mse(current_linear, reference_linear):
    # 归一化到 [0, 1]
    curr_f = current_linear.astype(np.float32)
    ref_f = reference_linear.astype(np.float32)

    if curr_f.max() > 255.0:
        curr_f = curr_f / 255.0
    if ref_f.max() > 255.0:
        ref_f = ref_f / 255.0

    # 处理异常值
    curr_f = np.nan_to_num(curr_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_f = np.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算线性空间 MSE
    diff = curr_f - ref_f
    mse = np.mean(diff ** 2)
    return float(mse)
```

### 网格显存计算

```python
grid_memory_mb = (
    nx * ny * nz *          # 单元总数
    3 * 4 /               # RGB 3 通道 * float32 4 字节
    (1024.0 * 1024.0)    # 转换为 MB
)
```

### 收敛检测

```python
rel_change = abs(current_mse - prev_mse) / (abs(prev_mse) + 1e-12)

if rel_change < 0.001:  # 相对变化 < 0.1%
    consecutive_small += 1
else:
    consecutive_small = 0

if consecutive_small >= 5:  # 连续 5 帧小变化
    print("Converged!")
```

---

## 7. 常见问题

### Q: 如何调整场景亮度？
A: 修改发光材质的强度：
```python
material.DiffuseLight(vec3(20, 20, 20))  # 提高亮度
```

### Q: 如何减少噪点？
A: 增加采样数：
```python
MAX_PROBE_SAMPLES = 32  # 从 16 增加
```

### Q: 如何提高 FPS？
A: 降低网格分辨率：
```python
GRID_RESOLUTION = (24, 24, 24)  # 从 32³ 降低
```

### Q: 显存不足怎么办？
A: 进一步降低网格分辨率：
```python
GRID_RESOLUTION = (16, 16, 16)  # 最低配置
```

---

## 相关文档

- **SCENES_GUIDE.md**: 场景配置指南
- **BENCHMARK_GUIDE.md**: 基准测试指南
- **README_INTEGRATED.md**: 项目整体文档
