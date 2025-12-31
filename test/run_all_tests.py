#!/usr/bin/env python3
"""
Run all test scripts
运行所有测试脚本
"""

import subprocess
import sys
import os

# Test scripts in order
TEST_SCRIPTS = [
    ("test_update_grid.py", "Grid Update Test / 网格更新测试"),
    ("test_data_saving.py", "Data Saving Test / 数据保存测试"),
    ("test_fps_fix.py", "FPS Calculation Test / FPS计算测试"),
    ("test_sync_timing.py", "GPU Sync Timing Test / GPU同步计时测试"),
    ("test_features.py", "Feature Verification Test / 功能验证测试"),
]

def run_test(script_path, description):
    """Run a single test script / 运行单个测试脚本"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print(f"运行: {description}")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """Main entry point / 主入口"""
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*70)
    print("TEST SUITE / 测试套件")
    print("="*70)
    print(f"Test directory / 测试目录: {test_dir}")
    
    results = {}
    for script, desc in TEST_SCRIPTS:
        script_path = os.path.join(test_dir, script)
        if os.path.exists(script_path):
            success = run_test(script_path, desc)
            results[script] = success
        else:
            print(f"⚠ Script not found: {script_path}")
            results[script] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY / 测试总结")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for script, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {script}")
    
    print(f"\nTotal / 总计: {passed}/{total} tests passed / 通过")
    
    if passed == total:
        print("\n🎉 All tests passed! / 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. / {total - passed} 个测试失败。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
