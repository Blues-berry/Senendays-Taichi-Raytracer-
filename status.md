请按照以下优先级和具体要求，逐步修改现有代码，重点解决渲染质量（尤其是漏光）和逻辑一致性问题。目标是让渲染结果在低样本数下接近参考路径追踪，同时为 ablation 实验提供可信的结果。
优先级 1：立即修复防漏光机制（最关键）
当前防漏光开关（NORMAL_WEIGHTING_ENABLED、DISTANCE_WEIGHTING_ENABLED、NEIGHBOR_CLAMPING_ENABLED）在 experiment_config.py 中已定义，但 camera.py 中并未实际使用，导致漏光问题严重。
任务：

在 Camera 类的 __init__ 中添加缺失的 field：Pythonself.normal_grid = ti.Vector.field(3, ti.f32, shape=self.grid_res)
修改或替换当前的 irradiance 采样函数（sample_irradiance_grid 或类似函数），实现带权重的插值：
支持以下三种权重（根据配置开关启用）：
法线权重：pow(max(dot(cell_normal, query_normal), 0.0), cfg.NORMAL_POWER)
距离权重：1.0 / (dist * dist + 1e-4)，若 dist > cfg.DISTANCE_CUTOFF_MULTIPLIER * self.grid_cell_size 则权重为 0
邻域钳制（Neighbor Clamping）：最终 irradiance 值不得超过周围 26 个邻域 cell 的最大值

权重归一化后计算加权平均
如果所有权重都为 0，则返回零或背景色

在 experiment_config.py 中添加对应的 Camera 属性（或通过参数传递）：Pythonself.normal_weighting_enabled = cfg.NORMAL_WEIGHTING_ENABLED
self.distance_weighting_enabled = cfg.DISTANCE_WEIGHTING_ENABLED
self.neighbor_clamping_enabled = cfg.NEIGHBOR_CLAMPING_ENABLED

优先级 2：物体移动后必须清除 grid 数据
当前 adapt_grid_to_scene 只在初始化时调用，物体移动后旧缓存数据残留，导致鬼影和持续漏光。
任务：

在 main.py 的交互循环中，物体移动（move_big_spheres 或类似逻辑）后，立即调用：Pythoncam.clear_grid_data()
# 可选：cam.adapt_grid_to_scene(spheres, verbose=False)  # 如果需要重新计算 AABB
确保 clear_grid_data 方法清除所有相关 field，包括：Pythonself.irradiance_grid.fill(0.0)
self.normal_grid.fill(0.0)
self.grid_update_weight.fill(1.0)
# 根据实际使用的其他统计 field 也一并清零

优先级 3：将 ablation 开关正确传递给 Camera
当前 benchmark.py 定义了 ablation 组（包括 normal_weighting_on），但 Camera 没有接收这些开关，导致 ablation 实验无法区分不同配置的效果。
任务：

修改 Camera 的构造函数，增加参数：Pythondef __init__(self, world, ..., interpolation_on=True, importance_sampling_on=False,
             adaptive_logic_on=False, normal_weighting_on=False, ...):
    self.interpolation_on = interpolation_on
    self.importance_sampling_on = importance_sampling_on
    self.adaptive_logic_on = adaptive_logic_on
    self.normal_weighting_on = normal_weighting_on
    # ... 其他开关
在渲染相关函数中，根据这些开关控制行为，例如：
是否使用三线性插值
是否使用光源引导探针
是否启用自适应权重更新
是否启用 normal weighting 等防漏光机制

在 main.py 或 benchmark 脚本中创建 Camera 时，传入对应组的开关值。

优先级 4：其他小问题修复

非均匀 grid_res 的 cell size 计算
当前 adapt_grid_to_scene 使用单一 cell = max_dim / max(nx,ny,nz)，可能导致某些方向分辨率不足。
建议：改为分别计算每个方向的 cell size，或强制使用立方体 cell（取最大边长）。
缺少 max_light_sources 定义
在 set_light_sources 中使用 maxl = int(self.max_light_sources)，但未定义。
修复：在 __init__ 中添加：Pythonself.max_light_sources = 64
场景配置导入问题main.py 中有 from scenes.scene_configs import get_scene，但未提供该模块。
建议：暂时注释掉，或补全该模块；若不使用，可直接在 setup_scene 中保留手动场景构建逻辑。

优先级 5：验证与输出要求
完成以上修改后，请执行以下步骤并报告结果：

在 cornell_box 场景下运行基准测试（benchmark.py），比较开启/关闭 normal weighting 的视觉差异。
保存修复前后在物体移动后第 5 帧和第 50 帧的截图 + 误差热图。
检查 MSE 曲线是否显著改善（尤其 Full_Hybrid 组）。
确认性能是否仍在实时范围内（目标 ≥ 30 FPS）。

目标：修复后渲染画面应明显减少漏光、颜色出血和鬼影，阴影边缘更清晰，整体接近参考路径追踪效果。
请按以上顺序逐项完成，每完成一项可简要报告进展或遇到的问题。完成后可继续优化性能或增加更多场景支持。