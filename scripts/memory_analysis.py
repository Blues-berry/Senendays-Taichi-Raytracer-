"""
Memory and Performance Analysis Script
Analyzes VRAM usage and grid update timing across different grid resolutions
"""

import taichi as ti
import numpy as np
import time
import csv
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import setup_scene

# Grid resolutions to test
GRID_RESOLUTIONS = [
    (16, 16, 16),
    (32, 32, 32),
    (48, 48, 48),
    (64, 64, 64),
    (80, 80, 80),
]

def calculate_grid_memory(grid_res, verbose=False):
    """Calculate memory usage for irradiance grid in MB"""
    nx, ny, nz = grid_res
    
    # irradiance_grid: vec3 (3 floats) per cell
    grid_mem = nx * ny * nz * 3 * 4  # 3 floats * 4 bytes
    
    # grid_update_weight: 1 float per cell
    weight_mem = nx * ny * nz * 4
    
    # irradiance_mean_lum: 1 float per cell
    mean_lum_mem = nx * ny * nz * 4
    
    # irradiance_variance: 1 float per cell
    variance_mem = nx * ny * nz * 4
    
    # grid_mean_distance: 1 float per cell
    distance_mem = nx * ny * nz * 4
    
    # grid_update_weight_tmp: 1 float per cell (temp buffer for blur)
    weight_tmp_mem = nx * ny * nz * 4
    
    total_grid_mem = grid_mem + weight_mem + mean_lum_mem + variance_mem + distance_mem + weight_tmp_mem
    total_mb = total_grid_mem / (1024.0 * 1024.0)
    
    if verbose:
        print(f"  irradiance_grid: {grid_mem/1024/1024:.2f} MB")
        print(f"  grid_update_weight: {weight_mem/1024/1024:.2f} MB")
        print(f"  irradiance_mean_lum: {mean_lum_mem/1024/1024:.2f} MB")
        print(f"  irradiance_variance: {variance_mem/1024/1024:.2f} MB")
        print(f"  grid_mean_distance: {distance_mem/1024/1024:.2f} MB")
        print(f"  grid_update_weight_tmp: {weight_tmp_mem/1024/1024:.2f} MB")
        print(f"  Total Grid: {total_mb:.2f} MB")
    
    return total_mb


def benchmark_grid_update_time(world, cam, grid_res, num_iterations=10, verbose=False):
    """Benchmark grid update time in milliseconds"""
    update_times = []
    
    # Warm-up
    for _ in range(3):
        cam.update_grid(world, 0.05)
        ti.sync()
    
    # Benchmark
    for _ in range(num_iterations):
        ti.sync()
        start = time.perf_counter()
        cam.update_grid(world, 0.05)
        ti.sync()
        elapsed = time.perf_counter() - start
        update_times.append(elapsed * 1000.0)  # Convert to ms
    
    mean_time = np.mean(update_times)
    std_time = np.std(update_times)
    
    if verbose:
        print(f"  Mean update time: {mean_time:.2f} ms (+/- {std_time:.2f} ms)")
    
    return mean_time, std_time


def run_full_analysis(scene_mode='cornell_box'):
    """Run complete memory and performance analysis"""
    results = []
    
    print("=" * 60)
    print("GRID MEMORY AND PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Scene mode: {scene_mode}")
    print(f"Grid resolutions to test: {GRID_RESOLUTIONS}")
    print()
    
    for grid_res in GRID_RESOLUTIONS:
        print(f"\n--- Testing grid resolution {grid_res} ---")
        
        # Setup scene with new grid resolution
        world, cam = setup_scene(scene_mode)
        cam.grid_res = grid_res
        
        # Recreate grid fields with new resolution
        cam.irradiance_grid = ti.Vector.field(n=3, dtype=ti.f32, shape=grid_res)
        cam.grid_update_weight = ti.field(dtype=ti.f32, shape=grid_res)
        cam.irradiance_mean_lum = ti.field(dtype=ti.f32, shape=grid_res)
        cam.irradiance_variance = ti.field(dtype=ti.f32, shape=grid_res)
        cam.grid_mean_distance = ti.field(dtype=ti.f32, shape=grid_res)
        cam.grid_update_weight_tmp = ti.field(dtype=ti.f32, shape=grid_res)
        
        # Initialize
        cam.grid_update_weight.fill(1.0)
        cam.irradiance_mean_lum.fill(0.0)
        cam.irradiance_variance.fill(0.0)
        cam.grid_mean_distance.fill(1e9)
        cam.grid_update_weight_tmp.fill(1.0)
        
        # Adapt grid to scene
        cam.adapt_grid_to_scene([], verbose=False)
        
        # Calculate memory
        memory_mb = calculate_grid_memory(grid_res, verbose=True)
        
        # Benchmark update time
        mean_time, std_time = benchmark_grid_update_time(world, cam, grid_res, num_iterations=10, verbose=True)
        
        # Calculate theoretical cells
        num_cells = grid_res[0] * grid_res[1] * grid_res[2]
        
        results.append({
            'grid_res': f"{grid_res[0]}x{grid_res[1]}x{grid_res[2]}",
            'num_cells': num_cells,
            'memory_mb': memory_mb,
            'update_time_ms': mean_time,
            'update_time_std_ms': std_time,
        })
    
    return results


def save_results_to_csv(results, output_path):
    """Save analysis results to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'grid_res', 'num_cells', 'memory_mb', 
            'update_time_ms', 'update_time_std_ms'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


def plot_tradeoff_curves(results, output_path):
    """Plot memory vs performance trade-off curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    grid_res_labels = [r['grid_res'] for r in results]
    num_cells = [r['num_cells'] for r in results]
    memory_mb = [r['memory_mb'] for r in results]
    update_time_ms = [r['update_time_ms'] for r in results]
    
    # Plot 1: Memory vs Grid Resolution
    ax1.plot(num_cells, memory_mb, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_xlabel('Number of Grid Cells', fontsize=12)
    ax1.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax1.set_title('Grid Memory vs Resolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Annotate points
    for i, (label, x, y) in enumerate(zip(grid_res_labels, num_cells, memory_mb)):
        ax1.annotate(label, (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Update Time vs Memory (Trade-off Curve)
    ax2.plot(memory_mb, update_time_ms, 's-', linewidth=2, markersize=8, color='#ff7f0e')
    ax2.set_xlabel('Memory Usage (MB)', fontsize=12)
    ax2.set_ylabel('Grid Update Time (ms)', fontsize=12)
    ax2.set_title('Performance Trade-off: Memory vs Update Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Annotate points
    for i, (label, x, y) in enumerate(zip(grid_res_labels, memory_mb, update_time_ms)):
        ax2.annotate(label, (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trade-off plot saved to: {output_path}")
    plt.close()


def plot_comprehensive_analysis(results, output_path):
    """Plot comprehensive analysis with multiple metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    grid_res_labels = [r['grid_res'] for r in results]
    num_cells = [r['num_cells'] for r in results]
    memory_mb = [r['memory_mb'] for r in results]
    update_time_ms = [r['update_time_ms'] for r in results]
    update_time_std = [r['update_time_std_ms'] for r in results]
    
    # Plot 1: Memory Scaling
    ax = axes[0, 0]
    ax.plot(num_cells, memory_mb, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax.set_xlabel('Number of Grid Cells')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Scaling')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Update Time Scaling
    ax = axes[0, 1]
    ax.plot(num_cells, update_time_ms, 's-', linewidth=2, markersize=8, color='#ff7f0e')
    ax.fill_between(num_cells, 
                   np.array(update_time_ms) - np.array(update_time_std),
                   np.array(update_time_ms) + np.array(update_time_std),
                   alpha=0.2, color='#ff7f0e')
    ax.set_xlabel('Number of Grid Cells')
    ax.set_ylabel('Update Time (ms)')
    ax.set_title('Grid Update Time Scaling')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Memory per Cell
    ax = axes[1, 0]
    bytes_per_cell = np.array(memory_mb) * 1024 * 1024 / np.array(num_cells)
    ax.plot(num_cells, bytes_per_cell, '^-', linewidth=2, markersize=8, color='#2ca02c')
    ax.set_xlabel('Number of Grid Cells')
    ax.set_ylabel('Bytes per Cell')
    ax.set_title('Memory Efficiency (Bytes per Cell)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Trade-off Curve (Quality vs Performance proxy)
    ax = axes[1, 1]
    # Use memory as a proxy for "potential quality" (higher res = more detail)
    ax.plot(memory_mb, update_time_ms, 'd-', linewidth=2, markersize=8, color='#d62728')
    ax.set_xlabel('Memory Usage (MB) -> Quality')
    ax.set_ylabel('Update Time (ms) -> Performance')
    ax.set_title('Quality-Performance Trade-off Curve')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis plot saved to: {output_path}")
    plt.close()


def generate_summary_report(results, output_path):
    """Generate text summary report"""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GRID MEMORY AND PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        total_memory = sum(r['memory_mb'] for r in results)
        avg_memory = total_memory / len(results)
        avg_update_time = np.mean([r['update_time_ms'] for r in results])
        
        f.write(f"Total grid resolutions tested: {len(results)}\n")
        f.write(f"Average memory usage: {avg_memory:.2f} MB\n")
        f.write(f"Average update time: {avg_update_time:.2f} ms\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        for r in results:
            f.write(f"\nGrid Resolution: {r['grid_res']}\n")
            f.write(f"  Number of cells: {r['num_cells']:,}\n")
            f.write(f"  Memory usage: {r['memory_mb']:.2f} MB\n")
            f.write(f"  Update time: {r['update_time_ms']:.2f} ms (+/- {r['update_time_std_ms']:.2f})\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Summary report saved to: {output_path}")


def main():
    """Main execution function"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"memory_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Run analysis
    results = run_full_analysis(scene_mode='cornell_box')
    
    # Save results
    csv_path = os.path.join(output_dir, "memory_performance_analysis.csv")
    save_results_to_csv(results, csv_path)
    
    # Generate plots
    tradeoff_path = os.path.join(output_dir, "tradeoff_curves.pdf")
    plot_tradeoff_curves(results, tradeoff_path)
    
    comprehensive_path = os.path.join(output_dir, "comprehensive_analysis.pdf")
    plot_comprehensive_analysis(results, comprehensive_path)
    
    # Generate summary report
    report_path = os.path.join(output_dir, "memory_analysis_report.txt")
    generate_summary_report(results, report_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
