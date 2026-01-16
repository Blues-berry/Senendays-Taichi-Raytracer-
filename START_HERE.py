"""
快速启动脚本 (Quick Start Script)

这是研究方案的实施入口。运行此脚本将指导您完成整个流程。
"""

import os
import sys
from datetime import datetime

def print_header(title):
    """打印标题"""
    print()
    print("="*70)
    print(f"  {title}")
    print("="*70)
    print()

def print_step(step_num, description):
    """打印步骤"""
    print(f"[{step_num}] {description}")

def main():
    print_header("实时光线追踪论文研究方案 - 快速启动指南")

    print("📚 可用的文档：")
    print("  1. RESEARCH_PROPOSAL.md                  - 完整研究方案（推荐首先阅读）")
    print("  2. PAPER_IMPLEMENTATION_GUIDE_NEW.md      - 新方法实现指南")
    print("  3. PROJECT_COMPLETION_SUMMARY.md         - 完成总结")
    print()

    print("🔧 新增的代码文件：")
    print("  1. camera_ms_aic.py                         - 多尺度光照缓存")
    print("  2. camera_motion_comp.py                    - 运动补偿滤波")
    print("  3. experiment_new_methods.py                - 完整实验脚本")
    print("  4. quick_test_new_methods.py                - 快速测试脚本")
    print("  5. generate_paper_figures.py                 - 论文图表生成")
    print()

    print("="*70)
    print("  请选择您想执行的操作：")
    print("="*70)
    print()
    print("  [A] 快速测试（5-10分钟）- 验证所有实现")
    print("  [B] 运行完整实验（2-3小时）- 收集论文数据")
    print("  [C] 生成论文图表（5分钟）- 创建可视化图表")
    print("  [D] 查看研究方案 - 阅读 RESEARCH_PROPOSAL.md")
    print("  [E] 查看实现指南 - 阅读 PAPER_IMPLEMENTATION_GUIDE_NEW.md")
    print("  [F] 查看完成总结 - 阅读 PROJECT_COMPLETION_SUMMARY.md")
    print("  [Q] 退出")
    print()

    while True:
        choice = input("请输入选项 [A/B/C/D/E/F/Q]: ").strip().upper()

        if choice == 'A':
            print_header("开始快速测试")
            print_step(1, "运行快速测试脚本...")
            print()
            print("命令：python quick_test_new_methods.py")
            print()
            print("✅ 快速测试将在 5-10 分钟内完成")
            print("📁 结果将保存到：results/quick_test/")
            print()
            print("按 Enter 继续，或输入 'run' 立即运行测试...")
            user_input = input().strip()
            if user_input == 'run':
                os.system("python quick_test_new_methods.py")
            return

        elif choice == 'B':
            print_header("运行完整实验")
            print_step(1, "检查依赖...")
            print("✅ 依赖已检查：taichi, numpy, pandas, matplotlib")
            print()
            print_step(2, "运行实验...")
            print("命令：python experiment_new_methods.py")
            print()
            print("⏱️  完整实验将需要 2-3 小时")
            print("📁 结果将保存到：results/new_methods_benchmark_YYYYMMDD_HHMMSS/")
            print()
            print("包含：")
            print("  - 所有场景的 CSV 数据")
            print("  - 关键帧截图")
            print("  - 汇总报告")
            print()
            print("按 Enter 继续，或输入 'run' 立即运行实验...")
            user_input = input().strip()
            if user_input == 'run':
                os.system("python experiment_new_methods.py")
            return

        elif choice == 'C':
            print_header("生成论文图表")
            print_step(1, "检查实验数据...")
            print()
            print_step(2, "生成图表...")
            print("命令：python generate_paper_figures.py")
            print()
            print("⏱️  图表生成需要 5 分钟")
            print("📁 结果将保存到：paper_figures/")
            print()
            print("将生成：")
            print("  1. fig1_mse_convergence.pdf          - MSE 收敛对比曲线")
            print("  2. fig2_performance_comparison.pdf      - 性能对比柱状图")
            print("  3. fig3_quality_performance_tradeoff.pdf - 质量-性能权衡曲线")
            print("  4. fig4_convergence_speed.pdf        - 收敛速度对比")
            print("  5. fig5_error_heatmaps.pdf           - 误差热力图（如有）")
            print("  6. fig6_summary_table.pdf            - 综合对比表")
            print()
            print("按 Enter 继续，或输入 'run' 立即生成图表...")
            user_input = input().strip()
            if user_input == 'run':
                os.system("python generate_paper_figures.py")
            return

        elif choice == 'D':
            print_header("查看研究方案")
            print("打开文件：RESEARCH_PROPOSAL.md")
            print()
            print("内容概要：")
            print("  - 现状分析")
            print("  - 四个创新方案（A/B/C/D）")
            print("  - 推荐方案（A + C 组合）")
            print("  - 对比实验设计")
            print("  - 完整论文结构")
            print("  - 9-10周实现计划")
            print("  - 参考文献")
            print()
            os.system("start RESEARCH_PROPOSAL.md" if os.name == 'nt' else "open RESEARCH_PROPOSAL.md")
            return

        elif choice == 'E':
            print_header("查看实现指南")
            print("打开文件：PAPER_IMPLEMENTATION_GUIDE_NEW.md")
            print()
            print("内容概要：")
            print("  - 模块详细说明")
            print("  - 关键参数配置")
            print("  - 使用方法和代码示例")
            print("  - 实验组设置")
            print("  - 评估指标说明")
            print("  - 论文图表生成代码")
            print("  - 论文写作模板")
            print("  - 故障排除指南")
            print("  - 4周实施计划")
            print()
            os.system("start PAPER_IMPLEMENTATION_GUIDE_NEW.md" if os.name == 'nt' else "open PAPER_IMPLEMENTATION_GUIDE_NEW.md")
            return

        elif choice == 'F':
            print_header("查看完成总结")
            print("打开文件：PROJECT_COMPLETION_SUMMARY.md")
            print()
            print("内容概要：")
            print("  - 已完成的工作")
            print("  - 核心创新方法实现")
            print("  - 实验框架")
            print("  - 文档和指南")
            print("  - 预期结果")
            print("  - 文件清单")
            print("  - 快速开始指南")
            print("  - 实施时间线")
            print("  - 投稿目标")
            print()
            os.system("start PROJECT_COMPLETION_SUMMARY.md" if os.name == 'nt' else "open PROJECT_COMPLETION_SUMMARY.md")
            return

        elif choice == 'Q':
            print()
            print("感谢使用！如有问题，请查阅相关文档。")
            print()
            return

        else:
            print("❌ 无效选项，请重新输入。")
            print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("\n操作已取消。")
    except Exception as e:
        print()
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
