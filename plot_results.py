import csv
import os
import glob
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def find_latest_results_dir(results_root: str = "results"):
    pattern = os.path.join(results_root, "benchmark_results_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def read_benchmark_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "frame": int(r["frame"]),
                    "mode": r["mode"],
                    "fps": float(r["fps"]),
                    "mse": float(r["mse"]),
                    "timestamp": r.get("timestamp", ""),
                }
            )
    return rows


def plot_mse_over_time(rows, out_dir: str, title: str = "MSE over time (log scale)"):
    # Group by mode
    modes = sorted({r["mode"] for r in rows})
    colors = {
        "Path Tracing": "#1f77b4",
        "Pure Grid": "#ff7f0e",
        "Hybrid": "#2ca02c",
    }

    plt.figure(figsize=(12, 6))

    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        frames = np.array([r["frame"] for r in mode_rows], dtype=np.int64)
        mse = np.array([r["mse"] for r in mode_rows], dtype=np.float64)

        # Log scale can't show <=0; mask them out
        mask = mse > 0
        if np.any(mask):
            plt.semilogy(
                frames[mask],
                mse[mask],
                label=mode,
                linewidth=2,
                alpha=0.85,
                color=colors.get(mode, None),
            )

    plt.xlabel("Frame")
    plt.ylabel("MSE (log scale)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "mse_over_time.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")


def read_benchmark_data(results_dir: str):
    """Read benchmark_results.csv from a results directory and group by mode.

    Returns a dict:
        {mode: {'frame': [...], 'fps': [...], 'mse': [...], 'timestamp': [...]}}
    """
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    try:
        rows = read_benchmark_csv(csv_path)
    except FileNotFoundError:
        print(f"benchmark_results.csv not found: {csv_path}")
        return None

    data = {}
    for r in rows:
        mode = r["mode"]
        if mode not in data:
            data[mode] = {"frame": [], "fps": [], "mse": [], "timestamp": []}
        data[mode]["frame"].append(r["frame"])
        data[mode]["fps"].append(r["fps"])
        data[mode]["mse"].append(r["mse"])
        data[mode]["timestamp"].append(r.get("timestamp", ""))

    # Ensure sorted by frame within each mode
    for mode, values in data.items():
        if values["frame"]:
            order = np.argsort(np.array(values["frame"], dtype=np.int64))
            for k in ("frame", "fps", "mse", "timestamp"):
                values[k] = [values[k][i] for i in order]

    return data


# NOTE: main() is defined later in this file (below) as the richer CLI/plotter.

def plot_detailed_mse_analysis(data, output_dir):
    """Create detailed MSE analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {
        'Path Tracing': '#1f77b4',
        'Pure Grid': '#ff7f0e', 
        'Hybrid': '#2ca02c'
    }
    
    # Plot 1: MSE before displacement (frames 0-149)
    ax = axes[0, 0]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            mse_values = np.array(values['mse'])
            
            # Filter for pre-displacement frames
            pre_mask = frames < 150
            if np.any(pre_mask):
                mse_pre = mse_values[pre_mask]
                non_zero_mask = mse_pre > 0
                if np.any(non_zero_mask):
                    ax.semilogy(frames[pre_mask][non_zero_mask], mse_pre[non_zero_mask],
                              color=colors[mode], label=mode, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE Before Displacement (Frames 0-149)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MSE after displacement (frames 150+)
    ax = axes[0, 1]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            mse_values = np.array(values['mse'])
            
            # Filter for post-displacement frames
            post_mask = frames >= 150
            if np.any(post_mask):
                mse_post = mse_values[post_mask]
                non_zero_mask = mse_post > 0
                if np.any(non_zero_mask):
                    ax.semilogy(frames[post_mask][non_zero_mask], mse_post[non_zero_mask],
                              color=colors[mode], label=mode, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE After Displacement (Frames 150+)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Convergence analysis (MSE trend)
    ax = axes[1, 0]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            mse_values = np.array(values['mse'])
            
            # Calculate moving average for trend
            window_size = 20
            if len(mse_values) >= window_size:
                moving_avg = np.convolve(mse_values, np.ones(window_size)/window_size, mode='valid')
                moving_frames = frames[window_size-1:]
                
                non_zero_mask = moving_avg > 0
                if np.any(non_zero_mask):
                    ax.semilogy(moving_frames[non_zero_mask], moving_avg[non_zero_mask],
                              color=colors[mode], label=f'{mode} (moving avg)', linewidth=2)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE Convergence Trend (Moving Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison (FPS)
    ax = axes[1, 1]
    for mode, values in data.items():
        if values['frame']:
            frames = np.array(values['frame'])
            fps_values = np.array(values['fps'])
            
            non_zero_mask = fps_values > 0
            if np.any(non_zero_mask):
                ax.plot(frames[non_zero_mask], fps_values[non_zero_mask],
                       color=colors[mode], label=mode, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('FPS')
    ax.set_title('Performance Comparison (FPS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed analysis
    output_path = os.path.join(output_dir, "detailed_mse_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed MSE analysis saved to: {output_path}")
    
    plt.show()

def generate_summary_report(data, output_dir):
    """Generate a text summary report"""
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== Raytracing Benchmark Summary Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for mode, values in data.items():
            if values['frame']:
                f.write(f"{mode} Mode:\n")
                f.write(f"  Total frames: {len(values['frame'])}\n")
                
                # MSE statistics
                mse_values = np.array(values['mse'])
                non_zero_mse = mse_values[mse_values > 0]
                if len(non_zero_mse) > 0:
                    f.write(f"  MSE - Min: {non_zero_mse.min():.6f}\n")
                    f.write(f"  MSE - Max: {non_zero_mse.max():.6f}\n")
                    f.write(f"  MSE - Mean: {non_zero_mse.mean():.6f}\n")
                    f.write(f"  MSE - Median: {np.median(non_zero_mse):.6f}\n")
                
                # FPS statistics
                fps_values = np.array(values['fps'])
                non_zero_fps = fps_values[fps_values > 0]
                if len(non_zero_fps) > 0:
                    f.write(f"  FPS - Min: {non_zero_fps.min():.1f}\n")
                    f.write(f"  FPS - Max: {non_zero_fps.max():.1f}\n")
                    f.write(f"  FPS - Mean: {non_zero_fps.mean():.1f}\n")
                    f.write(f"  FPS - Median: {np.median(non_zero_fps):.1f}\n")
                
                f.write("\n")
    
    print(f"Summary report saved to: {report_path}")

def plot_mse_comparison(data, output_dir: str):
    """Compatibility wrapper around plot_mse_over_time."""
    rows = []
    for mode, values in data.items():
        for i, frame in enumerate(values.get("frame", [])):
            rows.append(
                {
                    "frame": int(frame),
                    "mode": mode,
                    "fps": float(values["fps"][i]) if i < len(values.get("fps", [])) else 0.0,
                    "mse": float(values["mse"][i]) if i < len(values.get("mse", [])) else 0.0,
                    "timestamp": values.get("timestamp", [""])[i]
                    if i < len(values.get("timestamp", []))
                    else "",
                }
            )

    plot_mse_over_time(rows, output_dir, title="MSE over time (log scale)")


def main():
    """Main function to plot benchmark results"""
    print("=== Raytracing Benchmark Results Plotter ===")

    # Find the latest results directory
    results_dir = find_latest_results_dir()
    if not results_dir:
        return

    # Read benchmark data
    data = read_benchmark_data(results_dir)
    if not data:
        return

    # Create plots directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating MSE comparison plot...")
    plot_mse_comparison(data, plots_dir)

    print("\nGenerating detailed MSE analysis...")
    plot_detailed_mse_analysis(data, plots_dir)

    print("\nGenerating summary report...")
    generate_summary_report(data, plots_dir)

    print(f"\nAll plots and reports saved to: {plots_dir}")
    print("Plot generation completed successfully!")

if __name__ == "__main__":
    main()
