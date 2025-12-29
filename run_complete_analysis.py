"""
完整分析流程运行脚本
一键运行网格分辨率分析和基准测试，并生成所有图表
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"开始执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        end_time = time.time()
        
        print(f"✅ {description} 执行成功!")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        if result.stdout:
            print("输出:")
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 执行失败!")
        print(f"错误码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 执行过程中出现异常: {e}")
        return False

def main():
    """主函数 - 运行完整分析流程"""
    print("🚀 开始完整网格分辨率性能分析")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 分析步骤
    steps = [
        ("python test_gpu_timing.py", "GPU计时功能测试"),
        ("python grid_resolution_analyzer.py", "网格分辨率性能分析"),
        ("python plot_tradeoff_curves.py", "绘制权衡曲线"),
    ]
    
    # 可选步骤（基准测试）
    optional_steps = [
        ("python benchmark.py", "增强版基准测试（包含GPU耗时）"),
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    # 执行必要步骤
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"\n📊 步骤 {i}/{total_steps}: {desc}")
        
        if run_command(cmd, desc):
            success_count += 1
        else:
            print(f"⚠️  步骤 {i} 失败，但继续执行后续步骤...")
    
    # 询问是否运行基准测试
    print(f"\n📋 必要步骤完成: {success_count}/{total_steps}")
    print("是否运行增强版基准测试？这可能需要较长时间...")
    
    try:
        response = input("输入 'y' 继续，其他键跳过: ").lower().strip()
        if response == 'y':
            print("\n🔥 开始运行基准测试...")
            if run_command(optional_steps[0][0], optional_steps[0][1]):
                success_count += 1
                total_steps += 1
                print("🎉 基准测试完成!")
    except KeyboardInterrupt:
        print("\n⚠️  用户中断基准测试")
    except Exception as e:
        print(f"\n⚠️  基准测试过程出错: {e}")
    
    # 最终汇总
    print(f"\n{'='*60}")
    print("📊 分析完成汇总")
    print(f"{'='*60}")
    print(f"✅ 成功步骤: {success_count}/{total_steps}")
    print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 列出生成的文件
    print("\n📁 生成的文件:")
    
    # 查找网格分析结果
    import glob
    grid_dirs = glob.glob("grid_analysis_results_*")
    if grid_dirs:
        latest_grid = max(grid_dirs, key=os.path.getctime)
        print(f"📂 网格分析结果目录: {latest_grid}")
        for file in os.listdir(latest_grid):
            print(f"   📄 {file}")
    
    # 查找基准测试结果  
    benchmark_dirs = glob.glob("results/benchmark_results_*")
    if benchmark_dirs:
        latest_benchmark = max(benchmark_dirs, key=os.path.getctime)
        print(f"📂 基准测试结果目录: {latest_benchmark}")
        for file in os.listdir(latest_benchmark):
            if file.endswith('.csv') or file.endswith('.png'):
                print(f"   📄 {file}")
    
    # 查找生成的图表
    charts = glob.glob("*.png")
    if charts:
        print(f"📈 生成的图表:")
        for chart in charts:
            print(f"   📊 {chart}")
    
    print(f"\n🎯 下一步建议:")
    print("1. 查看生成的PNG图表了解性能权衡")
    print("2. 分析CSV数据选择最适合的网格分辨率")
    print("3. 根据实际需求调整配置参数")
    
    if success_count == total_steps:
        print("\n🎉 所有分析步骤都成功完成!")
    else:
        print(f"\n⚠️  有 {total_steps - success_count} 个步骤失败，请检查错误信息")

if __name__ == "__main__":
    main()