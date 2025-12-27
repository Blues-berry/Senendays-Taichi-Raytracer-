建议让 Cursor 修改 benchmark.py 来支持消融实验：

“请在 benchmark.py 中增加实验分组逻辑，对比以下三种配置：

Baseline: 关闭插值，关闭光源采样。

Optimized_V1: 开启插值，关闭光源采样。

Full_Hybrid: 两者全部开启。 要求自动生成三条 MSE 曲线在同一张图表上。