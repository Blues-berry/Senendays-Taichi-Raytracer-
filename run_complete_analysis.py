"""
Complete Pipeline for Paper-Ready Analysis
Executes all benchmark, analysis, and plotting steps in sequence
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with return code {result.returncode}")
        return False
    
    print(f"\n[SUCCESS] {description} completed")
    return True


def find_latest_results_dir():
    """Find the most recent benchmark results directory"""
    import glob
    pattern = os.path.join("results", "benchmark_results_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run complete analysis pipeline for paper-ready results"
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip running ablation benchmark (use existing results)"
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip memory and performance analysis"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Specify results directory directly (overrides auto-detection)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="cornell_box",
        choices=["cornell_box", "random", "night_scene"],
        help="Scene mode for benchmark"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COMPLETE ANALYSIS PIPELINE FOR PAPER-READY RESULTS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Run ablation benchmark (unless skipped)
    results_dir = args.results_dir or find_latest_results_dir()
    
    if not args.skip_benchmark:
        if not run_command(
            [sys.executable, "benchmark.py"],
            "Ablation Study Benchmark (4 groups: Baseline, V1, V2, Full_Hybrid)"
        ):
            return 1
        
        # Find the new results directory
        results_dir = find_latest_results_dir()
        if not results_dir:
            print("[ERROR] No benchmark results directory found!")
            return 1
    else:
        if not results_dir:
            print("[ERROR] --skip-benchmark specified but no results directory found!")
            print("Please specify --results-dir or run the benchmark first.")
            return 1
        print(f"\n[INFO] Using existing results from: {results_dir}")
    
    # Step 2: Memory and performance analysis (unless skipped)
    if not args.skip_memory:
        if not run_command(
            [sys.executable, "memory_analysis.py"],
            "Memory and Grid Performance Analysis"
        ):
            print("[WARNING] Memory analysis failed, continuing with plotting...")
    
    # Step 3: Generate all plots
    if not run_command(
        [sys.executable, "plot_results.py", "--results_dir", results_dir],
        "Generate All Plots (MSE comparison, Performance, Trade-off curves)"
    ):
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults directory: {results_dir}")
    print(f"Plots directory: {os.path.join(results_dir, 'plots')}")
    
    # List generated files
    print("\nGenerated files:")
    if os.path.exists(results_dir):
        plots_dir = os.path.join(results_dir, "plots")
        if os.path.exists(plots_dir):
            for f in sorted(os.listdir(plots_dir)):
                print(f"  - {f}")
    
        # List CSV files
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        if csv_files:
            print("\nGenerated CSV files:")
            for f in sorted(csv_files):
                print(f"  - {f}")
    
    # List memory analysis results
    mem_dirs = [d for d in os.listdir("results") if d.startswith("memory_analysis_")]
    if mem_dirs:
        mem_dir = os.path.join("results", sorted(mem_dirs)[-1])
        print(f"\nMemory analysis directory: {mem_dir}")
        if os.path.exists(mem_dir):
            for f in sorted(os.listdir(mem_dir)):
                print(f"  - {f}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll files ready for paper submission!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
