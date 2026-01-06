你的目标：在当前 Taichi hybrid irradiance grid path tracer 的基础上，通过针对性优化大幅减少漏光（light leaking），使渲染质量在低样本数（≤8 spp）下接近参考路径追踪，同时保持实时性能（≥30 FPS @ 1200x675），为发表论文准备高质量结果。

优先级顺序（必须严格遵守）：

1. **最高优先：彻底解决漏光问题（视觉质量决定论文成败）**
   - 立即实现以下三项防漏光机制（任选其二即可见效，全做最佳）：
     a. **法线加权插值（Normal-weighted interpolation）**：新增一个 normal_grid (vec3, same shape as irradiance_grid)，在探针更新时累加击中表面的法线。在三线性采样 irradiance 时，对每个邻域 cell 计算 normal.dot(query_normal)^8 作为权重，仅信任法线一致的 cell。
     b. **距离加权 + 阈值截断**：采样时计算 query_pos 到 cell_center 的距离，使用 1/(dist² + ε) 权重，并如果距离 > 1.5 × cell_size 则权重设为 0。
     c. **Neighbor Clamping（可选进阶）**：采样得到的 irradiance 值不得超过周围 26 个邻域 cell 的最大值（经典 radiance caching 防漏光技巧）。
   - 物体移动后必须立即 cam.irradiance_grid.fill(0.0) 和 normal_grid.fill(0.0)，防止旧缓存残留导致鬼影/漏光。

2. **次高优先：提升有效分辨率但控制内存与性能**
   - 将 grid_res 从 (32,32,32) 提升到 (64,48,64) 或 (64,64,64)，同时保持 GRID_PADDING ≥ 1.0。
   - 若显存/性能紧张，可改为非均匀 grid（如 xz 方向更高分辨率）。

3. **创新点强化（论文差异化关键）**
   - 保留并突出当前三项可开关组件：
     • Tri-linear interpolation (interpolation_on)
     • Light-guided probes / importance sampling (importance_sampling_on)
     • Adaptive weight update (adaptive_logic_on)
   - 在 ablation 中新增一组：只开 normal-weighted interpolation 的变体，证明其对漏光抑制的独立贡献。
   - 强调“零预计算、自动 grid 适应、动态场景下 <50 帧完全恢复”的轻量优势，与需要重预计算的 neural caching、wavelet caching 方法形成鲜明对比。

4. **性能约束**
   - 所有优化后，Cornell Box 场景在单张 RTX 3060/4070 上必须保持 ≥30 FPS（含 GUI）。
   - 禁止引入神经网络、额外 BVH、复杂数据结构（如 octree、wavelets），保持纯 Taichi field + kernel 的轻量实现。

5. **输出要求**
   - 修复后立即运行 benchmark.py（scene='cornell_box'），生成：
     • 修复前 vs 修复后对比截图（frame 50、move+5、move+50）
     • MSE 曲线（log scale，四组 ablation）
     • 误差热图（尤其突出阴影边缘和颜色出血区域的改善）
   - 如果漏光基本消失、MSE 显著下降（Full_Hybrid < 1e-3）、视觉接近 256 spp PT 参考，即可停止优化，开始准备论文图表。

请按以上顺序逐步实现，每完成一项立即测试视觉效果并报告。目标是让审稿人看到截图时认为“这是干净的实时路径追踪结果”，而不是“明显的 grid caching 伪影”。