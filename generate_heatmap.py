import taichi as ti
import main

# 1. 初始化 Taichi 和场景
ti.init(arch=ti.gpu)
world, cam = main.setup_scene('cornell_box')

print("Step 1: Rendering a single frame of the Hybrid mode...")
# 2. 渲染一帧你想评估的模式（这里用 Hybrid 模式）到 cam.frame
RENDER_MODE_HYBRID = 2
cam.update_grid(world, 0.01)  # Hybrid/Grid 模式需要更新网格
cam.render(world, RENDER_MODE_HYBRID)
ti.sync()
print("Hybrid frame rendered.")

print("\nStep 2: Generating high-quality Path Tracing reference (512 spp)... This may take a moment.")
# 3. 生成高质量的路径追踪参考图到 cam.pt_frame
cam.render_pt_reference(world, target_spp=512, chunk_spp=16, reset=True)
ti.sync()
print("PT reference generated.")

print("\nStep 3: Calculating and rendering the error heatmap...")
# 4. 正确调用 render_error_heatmap 方法
cam.render_error_heatmap()
ti.sync()
print("Heatmap calculated.")

# 5. 保存最终的热力图
output_path = "heatmap_output.png"
ti.tools.imwrite(cam.frame, output_path)
print(f"\nDone! Heatmap saved to {output_path}")

