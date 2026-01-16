# 论文研究方案：实时光照缓存与自适应渲染

## 📊 项目现状分析

### 已实现的核心技术
1. **深度防漏光机制** (Depth-based Occlusion Detection)
2. **时域 EMA 滤波** (Temporal Accumulation)
3. **法线加权插值** (Normal-weighted Interpolation)
4. **距离加权插值** (Distance-weighted Interpolation)
5. **邻域限制** (26-neighbor Clamping)
6. **方差引导采样** (Variance-guided Sampling)
7. **重要性采样** (Light Importance Sampling)

### 当前性能指标
- **Grid 模式**: ~2000 FPS, 显存 0.37MB (32³)
- **Hybrid 模式**: ~500 FPS, 显存 0.37MB
- **Path Tracing**: ~200 FPS

### 已有实验数据
- 5组消融实验配置
- 7个不同场景测试
- 误差热力图分析
- MSE 收敛曲线
- 性能对比

---

## 🎯 论文创新点建议

### 方案A：多尺度自适应光照缓存 (MS-AIC)
**创新点**：
1. **多分辨率网格金字塔**：使用 16³, 32³, 64³ 三层网格
2. **自适应层级选择**：根据距离相机、动态区域等因素自动选择合适层级
3. **层级间信息传递**：低分辨率结果上采样指导高分辨率更新

**优势**：
- 显存占用降低 40-60%
- 远处物体使用低精度网格，近距离物体使用高精度
- 动态区域优先更新高精度层

**实现复杂度**：中等（2-3周）

---

### 方案B：基于学习的网格更新预测 (Learned Grid Update Prediction)
**创新点**：
1. **轻量级 CNN**：预测每个网格单元的更新概率
2. **特征输入**：法线变化、深度变化、方差、历史更新频率
3. **端到端训练**：使用少量预渲染数据训练网络

**优势**：
- 智能化更新策略，减少不必要的计算
- 自动适应不同场景特性
- 可解释性强（网络输出更新概率图）

**实现复杂度**：高（3-4周，需要 PyTorch/TensorFlow）

---

### 方案C：时变场景的时空一致性滤波 (Spatiotemporal Consistency Filtering)
**创新点**：
1. **运动补偿时域累积**：基于 G-buffer 的运动向量预测下一帧位置
2. **双边滤波器**：结合空间和时域相似性
3. **自适应历史长度**：根据运动速度调整累积历史长度

**优势**：
- 显著减少运动拖尾
- 更好的时域一致性
- 适用于快速运动场景

**实现复杂度**：中等（2-3周）

---

### 方案D：混合方差引导采样 (Hybrid Variance-Guided Sampling)
**创新点**：
1. **双重采样策略**：结合重要性采样（NEE）和方差引导
2. **动态权重平衡**：根据场景光照复杂度自动调整
3. **采样效率估计**：在线评估采样效率并自适应调整

**优势**：
- 更充分利用光源重要性
- 方差大的区域获得更多采样
- 平衡质量和性能

**实现复杂度**：低（1-2周）

---

## 📝 推荐方案：方案A + 方案C 组合

### 理由
1. **技术互补**：方案A解决空间精度和显存问题，方案C解决时域一致性问题
2. **实现可行**：两个方案复杂度适中，可在合理时间内完成
3. **创新性充足**：多尺度缓存 + 运动补偿滤波是较新的研究方向
4. **实验设计清晰**：可分别验证每个组件的贡献

### 方法名称
**"Multi-Scale Adaptive Irradiance Caching with Motion-Compensated Temporal Filtering"**

### 核心贡献
1. 提出多分辨率光照缓存金字塔结构
2. 设计基于距离和动态性的自适应层级选择算法
3. 实现运动补偿的时域累积滤波器
4. 在多个复杂场景上验证方法的效率和鲁棒性

---

## 🔬 对比实验设计

### Baseline 方法
1. **Path Tracing (PT)**：传统路径追踪（参考真值）
2. **Pure Grid**：单层光照缓存（当前实现）
3. **Hybrid**：当前完整混合方法
4. **Adaptive Grid** [Zhou et al. 2020]：引用相关工作
5. **IR Caching** [Clarberg et al. 2005]：传统辐照度缓存

### 需要实现的新方法
1. **MS-AIC (Multi-Scale)**：仅多尺度部分
2. **Motion-Compensated**：仅运动补偿部分
3. **Full Method**：完整方法（多尺度 + 运动补偿）

### 对比维度
1. **质量指标**：
   - MSE (Mean Squared Error)
   - SSIM (Structural Similarity Index)
   - PSNR (Peak Signal-to-Noise Ratio)
   - LPIPS (Learned Perceptual Image Patch Similarity)

2. **性能指标**：
   - FPS (Frames Per Second)
   - GPU 时间
   - 显存占用
   - 网格更新开销

3. **收敛速度**：
   - MSE 随帧数下降曲线
   - 收敛到目标 MSE 所需帧数
   - 稳态 MSE

4. **视觉效果**：
   - 误差热力图
   - 放大细节对比
   - 运动序列对比

---

## 📊 实验场景设计

### 标准场景（7个）
1. **cornell_box**：颜色溢出测试
2. **two_room**：窄缝漏光测试
3. **night_scene**：多光源测试
4. **classroom**：环境光测试
5. **bathroom**：镜面反射测试
6. **veach_mis**：重要性采样测试
7. **random**：复杂材质测试

### 压力测试场景（新增）
1. **fast_motion**：快速运动物体（测试时域一致性）
2. **dynamic_lighting**：动态光源变化（测试网格更新）
3. **complex_geometry**：复杂遮挡关系（测试防漏光）
4. **low_memory**：显存受限场景（测试多尺度效率）

### 测试协议
- 每个场景运行 600 帧
- 在 Frame 200 处触发物体移动
- 记录每帧的 MSE、FPS、GPU 时间
- 保存关键帧截图（Frame 5, 50, 100, 150, 200, 205, 210, 250, 300, 400, 500, 600）
- 生成误差热力图

---

## 📈 论文结构建议

### 1. Abstract (300-400字)
- 问题陈述：实时渲染中的质量-性能权衡
- 方法概述：多尺度缓存 + 运动补偿滤波
- 主要贡献：4点核心贡献
- 结果摘要：相比 Baseline 提升 X% 质量，显存降低 Y%

### 2. Introduction (1.5-2页)
- 背景：实时渲染的重要性
- 问题：
  - 单一分辨率缓存无法兼顾精度和效率
  - 时域累积在运动场景下产生拖尾
- 挑战：
  - 如何设计高效的多尺度缓存结构
  - 如何实现鲁棒的运动补偿
- 贡献：明确列出 3-4 点贡献

### 3. Related Work (2页)
- 光照缓存技术
  - 传统的辐照度缓存
  - 光照传播网格
- 时域降噪技术
  - TAA/TAAU
  - SVGF
  - Adaptive Temporal Anti-Aliasing
- 多尺度渲染
  - 多分辨率光照
  - 级联阴影贴图
- 自适应采样
  - 重要性采样
  - 方差引导采样

### 4. Overview (1页)
- 系统架构图
- 关键组件：
  - 多尺度光照缓存金字塔
  - 自适应层级选择
  - 运动补偿时域滤波
  - 混合采样策略

### 5. Multi-Scale Irradiance Caching (2-3页)
- 网格金字塔设计
  - 三层结构（16³, 32³, 64³）
  - 数据结构设计
  - 显存占用分析
- 自适应层级选择算法
  - 距离相机远近判断
  - 动态区域检测
  - 层级切换策略
- 层级间信息传递
  - 低分辨率 → 高分辨率上采样
  - 双线性插值优化
- 实现细节（伪代码）

### 6. Motion-Compensated Temporal Filtering (2-3页)
- 运动向量估计
  - 基于 G-buffer（法线、深度）
  - 前向/后向光流
- 时域累积滤波器
  - 双边权重计算
  - 空间相似性
  - 时域相似性（运动补偿）
  - 自适应历史长度
- 理论分析
  - 收敛性证明
  - 复杂度分析

### 7. Hybrid Sampling Strategy (1-1.5页)
- 重要性采样（NEE）
- 方差引导采样
- 动态权重平衡
- 采样效率评估

### 8. Implementation (1页)
- 基于 Taichi 的 GPU 实现
- 场景设置
- 参数调优

### 9. Results (3-4页)
- 对比方法设置
- 定量结果表格
  - 多场景 MSE/SSIM/PSNR 对比
  - FPS 和显存对比
  - 收敛速度对比
- 可视化结果
  - 对比截图
  - 误差热力图
  - 放大细节
- 消融实验
  - 多尺度 vs 单尺度
  - 运动补偿 vs 无补偿
  - 各组件贡献分析
- 极限测试
  - 快速运动场景
  - 动态光源场景
  - 低显存场景

### 10. Discussion (1页)
- 方法的优点和局限性
- 失败案例分析
- 与现有方法的对比
- 未来工作方向

### 11. Conclusion (0.5页)
- 总结贡献
- 主要发现
- 潜在应用

---

## 🔧 实现计划

### 阶段1：核心算法实现（3周）
- Week 1: 多尺度网格金字塔
  - 数据结构设计
  - 三层网格初始化
  - 显存优化
- Week 2: 自适应层级选择
  - 距离判断算法
  - 动态区域检测
  - 层级切换逻辑
- Week 3: 运动补偿滤波
  - G-buffer 运动向量估计
  - 双边滤波器实现
  - 自适应历史长度

### 阶段2：集成与优化（2周）
- Week 4: 系统集成
  - 与现有代码整合
  - 参数调优
  - Bug 修复
- Week 5: 性能优化
  - GPU 内核优化
  - 减少数据传输
  - 并行化改进

### 阶段3：实验与数据收集（2周）
- Week 6: 基准实验
  - 运行所有场景
  - 收集定量数据
  - 生成截图和热力图
- Week 7: 对比实验
  - 实现对比方法
  - 完整消融实验
  - 压力测试场景

### 阶段4：论文撰写（2-3周）
- Week 8-9: 初稿撰写
- Week 10: 修改和润色

---

## 📦 交付物清单

### 代码
1. 完整实现代码（带注释）
2. 实验脚本和配置
3. 数据处理脚本

### 实验数据
1. 所有场景的 CSV 数据文件
2. 关键帧截图
3. 误差热力图
4. 对比图表

### 论文
1. 论文初稿（LaTeX）
2. 所有图表（高质量，300 DPI）
3. 补充材料（视频、代码链接）

---

## 🎯 成功指标

### 定量指标
- 相比 Hybrid 方法，MSE 降低 20-30%
- 显存占用降低 40-50%
- FPS 维持在 400-600 范围
- 收敛速度提升 30-40%

### 定性指标
- 视觉质量明显优于 Baseline
- 运动拖尾显著减少
- 光照效果更平滑自然
- 通过多场景压力测试

### 论文投稿目标
- 会议：SIGGRAPH, Eurographics, EGSR, I3D
- 期刊：TVCG, CGF

---

## 💡 备选创新方向

如果时间或资源有限，可以考虑以下简化方案：

### 简化方案1：仅多尺度缓存
- 实现方案A
- 对比单层网格
- 重点展示显存效率和精度权衡

### 简化方案2：仅运动补偿滤波
- 实现方案C
- 对比现有时域累积
- 重点展示运动一致性改进

### 简化方案3：参数自适应优化
- 基于现有框架
- 自动调优参数（更新概率、采样数等）
- 使用强化学习或贝叶斯优化

---

## 📚 参考文献建议

### 光照缓存
1. Clarberg, P., et al. "Instant Radiosity." EGSR 2005.
2. Zhou, K., et al. "Real-time Global Illumination with Adaptive Grid." SIGGRAPH 2020.
3. Kontkanen, J., et al. "Irradiance Caching for Global Illumination." SIGGRAPH 2005.

### 时域滤波
4. Schied, C., et al. "Spatiotemporal Variance-Guided Filtering." SIGGRAPH 2018.
5. Salvi, M., et al. "Adaptive Temporal Anti-Aliasing." HPG 2018.
6. Kuwahara, M., et al. "Temporal Upsampling." SIGGRAPH 2017.

### 多尺度渲染
7. Gollinet, V., et al. "Multi-view Global Illumination." EG 2019.
8. Hua, B.-S., et al. "Multi-resolution Illumination." I3D 2018.

### 自适应采样
9. Hachisuka, T., et al. "Adaptive Sampling for Monte Carlo Rendering." SIGGRAPH 2008.
10. Veach, E., and Guibas, L. J. "Metropolis Light Transport." SIGGRAPH 1997.

---

## 📞 下一步行动

1. **选择方案**：确定实现哪个创新方向（推荐方案A+C）
2. **详细设计**：撰写算法伪代码和数据结构设计
3. **开始实现**：按照实现计划逐步开发
4. **持续测试**：每完成一个组件立即测试
5. **记录实验**：详细记录所有实验结果

---

**预计时间线**：9-10周完成全部实现和实验

**预计论文质量**：可投稿 SIGGRAPH/Eurographics 等顶级会议
