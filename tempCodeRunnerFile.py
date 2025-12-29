"""Quick test script to generate a heat-map without手动缩进错误。
运行：python tempCodeRunnerFile.py
"""
import taichi as ti
import main

# 1. 初始化场景（可改为 'random' / 'cornell_box' 等）
world, cam = main.setup_scene('cornell_box')

# 2. 渲染一帧 Hybrid 模式，作为待评估图像
RENDER_MODE_HYBRID = 2
cam.update_grid(world, 0.01)
cam.render(world, RENDER_MODE_HYBRID)

# 3. 生成高质量 PT 参考（512 spp）
cam.render_pt_reference(world, target_spp=512, chunk_spp=16, reset=True)

# 4. 计算误差热力图
cam.render_error_heatmap()

# 5. 保存结果
output = 'temp_heatmap.png'
ti.tools.imwrite(cam.frame, output)
print(f'Saved {output}')

