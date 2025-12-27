指令 3：重构 benchmark.py 以支持自动化消融实验

“请修改 benchmark.py，允许通过配置参数开关以下功能：

interpolation_on (三线性插值)

importance_sampling_on (光源重要性采样)

adaptive_logic_on (自适应权重更新) 编写一个主循环，依次运行：Baseline（全关）、V1（仅插值）、V2（插值+自适应）、Full_Hybrid（全开）。将四组数据分别存入不同的 CSV，确保每组实验都包含相同的球体位移触发。”

指令 4：生成高质量学术对比图（Python 绘图指令）

“请修改 plot_results.py。读取消融实验产出的四份 CSV。在同一张图中绘制 MSE 随帧数变化的对比曲线。要求：使用 logarithmic（对数）纵轴，添加清晰的图例，并在位移发生帧（Frame 200）处画一条灰色虚线并标注 'Object Movement'。确保图片以 300DPI 的 PDF 格式输出，以便插入 LaTeX 论文。”