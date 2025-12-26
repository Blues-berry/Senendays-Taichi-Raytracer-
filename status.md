运行结果: 基准运行成功并完成（见终端日志）。输出目录： benchmark_results_20251226_172131，包含 benchmark_results.csv 和若干截图（例如 result_hybrid.png、after_displacement_*）。

已实现:

G-Buffer：在 camera.py 增加 normal_buffer 与 depth_buffer 并在命中时写入。
A-SVGF：实现轻量级 asvgf_filter()（3x3 法线/深度/亮度引导滤波）并在 Hybrid 后调用。
自适应采样：加入 adaptive_weight_map 与内核 compute_adaptive_weights(...)；在 experiment_config.py 添加参数 ADAPTIVE_BRIGHTNESS_THRESHOLD、ADAPTIVE_SAMPLING_MULTIPLIER、ADAPTIVE_MAX_MULTIPLIER。
调整了网格更新（更多探针、指数平滑）与将模糊开关驱动自 experiment_config.py。
关键文件: camera.py, experiment_config.py, main.py, benchmark.py

下一步建议（选一）:

我现在打开并预览 benchmark_results.csv；或
我列出并展示几张 hybrid / after_displacement 的截图以比较滤波前后差异；或
我帮你微调参数（例如调整 ADAPTIVE_BRIGHTNESS_THRESHOLD、ADAPTIVE_SAMPLING_MULTIPLIER、grid_samples_per_update、grid_probe_depth、grid_update_alpha）并再跑一次短时测试。