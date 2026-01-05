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


def _read_ablation_csv(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "frame": int(r["frame"]),
                "mse": float(r["mse"]),
                "fps": float(r.get("fps", 0.0)),
                "gpu_time_ms": float(r.get("gpu_time_ms", 0.0)),
                "grid_memory_mb": float(r.get("grid_memory_mb", 0.0))
            })
    return rows


def plot_ablation_mse_comparison(
    results_dir: str,
    out_dir: str,
    groups=("Baseline", "V1", "V2", "Full_Hybrid"),
    movement_frame: int = 200,
):
    """High-quality academic comparison plot for the ablation study."""
    os.makedirs(out_dir, exist_ok=True)

    # Publication-ish defaults (still lightweight)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.0,
        }
    )

    plt.figure(figsize=(6.5, 3.8))

    palette = {
        "Baseline": "#1f77b4",
        "V1": "#ff7f0e",
        "V2": "#2ca02c",
        "Full_Hybrid": "#d62728",
    }

    for g in groups:
        csv_path = os.path.join(results_dir, f"ablation_{g}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing ablation CSV: {csv_path}")

        rows = _read_ablation_csv(csv_path)
        frames = np.array([r["frame"] for r in rows], dtype=np.int64)
        mse = np.array([r["mse"] for r in rows], dtype=np.float64)

        # log-scale can't show <= 0
        mask = mse > 0
        if np.any(mask):
            plt.semilogy(
                frames[mask],
                mse[mask],
                label=g,
                color=palette.get(g, None),
                alpha=0.95,
            )

    # Movement marker
    plt.axvline(movement_frame, color="#808080", linestyle="--", linewidth=1.5, alpha=0.9)
    plt.text(
        movement_frame + 3,
        plt.ylim()[1] / 3.0,
        "Object Movement",
        color="#606060",
        rotation=90,
        va="top",
        ha="left",
    )

    plt.xlabel("Frame")
    plt.ylabel("MSE (log scale)")
    plt.title("Ablation Study: MSE vs Frame")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.legend(loc="best", frameon=True)

    out_pdf = os.path.join(out_dir, "ablation_mse_comparison.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


def plot_performance_comparison(results_dir: str, out_dir: str, 
                               groups=("Baseline", "V1", "V2", "Full_Hybrid"),
                               movement_frame: int = 200):
    """Plot FPS and GPU time comparison across ablation groups."""
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    palette = {
        "Baseline": "#1f77b4",
        "V1": "#ff7f0e",
        "V2": "#2ca02c",
        "Full_Hybrid": "#d62728",
    }

    # FPS comparison
    for g in groups:
        csv_path = os.path.join(results_dir, f"ablation_{g}.csv")
        if not os.path.exists(csv_path):
            continue

        rows = _read_ablation_csv(csv_path)
        frames = np.array([r["frame"] for r in rows], dtype=np.int64)
        fps = np.array([r["fps"] for r in rows], dtype=np.float64)

        # Filter out zero FPS values
        mask = fps > 0
        if np.any(mask):
            ax1.plot(frames[mask], fps[mask], label=g, color=palette.get(g, None), alpha=0.9)

    ax1.set_xlabel("Frame")
    ax1.set_ylabel("FPS")
    ax1.set_title("Performance Comparison (FPS)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Movement marker on FPS plot
    ax1.axvline(movement_frame, color="#808080", linestyle="--", linewidth=1.5, alpha=0.7)

    # GPU time comparison
    for g in groups:
        csv_path = os.path.join(results_dir, f"ablation_{g}.csv")
        if not os.path.exists(csv_path):
            continue

        rows = _read_ablation_csv(csv_path)
        frames = np.array([r["frame"] for r in rows], dtype=np.int64)
        gpu_time = np.array([r["gpu_time_ms"] for r in rows], dtype=np.float64)

        # Filter out zero GPU time values
        mask = gpu_time > 0
        if np.any(mask):
            ax2.plot(frames[mask], gpu_time[mask], label=g, color=palette.get(g, None), alpha=0.9)

    ax2.set_xlabel("Frame")
    ax2.set_ylabel("GPU Time (ms)")
    ax2.set_title("GPU Time per Frame")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Movement marker on GPU time plot
    ax2.axvline(movement_frame, color="#808080", linestyle="--", linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(out_dir, "performance_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_mse_fps_tradeoff(results_dir: str, out_dir: str,
                          groups=("Baseline", "V1", "V2", "Full_Hybrid")):
    """Plot MSE vs FPS trade-off for quality-performance analysis."""
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    })

    plt.figure(figsize=(7, 5))

    palette = {
        "Baseline": "#1f77b4",
        "V1": "#ff7f0e",
        "V2": "#2ca02c",
        "Full_Hybrid": "#d62728",
    }

    for g in groups:
        csv_path = os.path.join(results_dir, f"ablation_{g}.csv")
        if not os.path.exists(csv_path):
            continue

        rows = _read_ablation_csv(csv_path)
        mse = np.array([r["mse"] for r in rows], dtype=np.float64)
        fps = np.array([r["fps"] for r in rows], dtype=np.float64)

        # Filter valid data
        mask = (mse > 0) & (fps > 0)
        if np.any(mask):
            # Use log scale for MSE
            plt.semilogy(fps[mask], mse[mask], 'o', label=g, 
                        color=palette.get(g, None), alpha=0.6, markersize=3)
            # Add mean marker
            mean_mse = np.mean(np.log10(mse[mask]))
            mean_fps = np.mean(fps[mask])
            plt.plot(mean_fps, 10**mean_mse, '*', markersize=12, 
                    color=palette.get(g, None), markeredgecolor='black', markeredgewidth=1)

    plt.xlabel("FPS (Performance)")
    plt.ylabel("MSE (Error, log scale)")
    plt.title("Quality-Performance Trade-off")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(out_dir, "quality_performance_tradeoff.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def generate_ablation_summary_report(results_dir: str, out_dir: str,
                                    groups=("Baseline", "V1", "V2", "Full_Hybrid")):
    """Generate comprehensive ablation study summary report."""
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, "ablation_summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDY SUMMARY REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for g in groups:
            csv_path = os.path.join(results_dir, f"ablation_{g}.csv")
            if not os.path.exists(csv_path):
                f.write(f"\n[{g}]\n")
                f.write("  CSV file not found\n")
                continue
            
            rows = _read_ablation_csv(csv_path)
            
            # Extract metrics
            mse_values = np.array([r["mse"] for r in rows])
            fps_values = np.array([r["fps"] for r in rows])
            gpu_time_values = np.array([r["gpu_time_ms"] for r in rows])
            
            # Filter valid data
            valid_mse = mse_values[mse_values > 0]
            valid_fps = fps_values[fps_values > 0]
            valid_gpu = gpu_time_values[gpu_time_values > 0]
            
            f.write(f"\n[{g}] Configuration:\n")
            if rows:
                # Read configuration from first row
                first_row = rows[0]
                f.write(f"  Interpolation: {'ON' if first_row.get('interpolation_on', False) else 'OFF'}\n")
                f.write(f"  Importance Sampling: {'ON' if first_row.get('importance_sampling_on', False) else 'OFF'}\n")
                f.write(f"  Adaptive Logic: {'ON' if first_row.get('adaptive_logic_on', False) else 'OFF'}\n")
            
            f.write(f"\n  Statistics:\n")
            f.write(f"    Total frames: {len(rows)}\n")
            
            if len(valid_mse) > 0:
                f.write(f"    MSE - Min: {valid_mse.min():.6e}\n")
                f.write(f"    MSE - Max: {valid_mse.max():.6e}\n")
                f.write(f"    MSE - Mean: {valid_mse.mean():.6e}\n")
                f.write(f"    MSE - Median: {np.median(valid_mse):.6e}\n")
                f.write(f"    MSE - Std: {valid_mse.std():.6e}\n")
            
            if len(valid_fps) > 0:
                f.write(f"    FPS - Min: {valid_fps.min():.2f}\n")
                f.write(f"    FPS - Max: {valid_fps.max():.2f}\n")
                f.write(f"    FPS - Mean: {valid_fps.mean():.2f}\n")
                f.write(f"    FPS - Median: {np.median(valid_fps):.2f}\n")
            
            if len(valid_gpu) > 0:
                f.write(f"    GPU Time - Min: {valid_gpu.min():.2f} ms\n")
                f.write(f"    GPU Time - Max: {valid_gpu.max():.2f} ms\n")
                f.write(f"    GPU Time - Mean: {valid_gpu.mean():.2f} ms\n")
                f.write(f"    GPU Time - Median: {np.median(valid_gpu):.2f} ms\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Ablation summary report saved to: {report_path}")


def main():
    """Main function to plot benchmark / ablation results"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory like results/benchmark_results_YYYYMMDD_HHMMSS (default: latest)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or find_latest_results_dir()
    if not results_dir:
        print("No results directory found under ./results")
        return

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Using results dir: {results_dir}")

    # 1) New: ablation comparison (4 CSVs)
    try:
        print("\nGenerating ablation MSE comparison...")
        plot_ablation_mse_comparison(results_dir, plots_dir)
    except FileNotFoundError as e:
        print(f"Ablation plot skipped: {e}")

    # 2) Performance comparison
    try:
        print("\nGenerating performance comparison...")
        plot_performance_comparison(results_dir, plots_dir)
    except Exception as e:
        print(f"Performance plot skipped: {e}")

    # 3) Quality-Performance trade-off
    try:
        print("\nGenerating quality-performance trade-off...")
        plot_mse_fps_tradeoff(results_dir, plots_dir)
    except Exception as e:
        print(f"Trade-off plot skipped: {e}")

    # 4) Ablation summary report
    try:
        print("\nGenerating ablation summary report...")
        generate_ablation_summary_report(results_dir, plots_dir)
    except Exception as e:
        print(f"Summary report skipped: {e}")

    # 5) Backwards-compatible: legacy benchmark_results.csv plots if present
    data = read_benchmark_data(results_dir)
    if data:
        print("\nGenerating legacy MSE comparison plot...")
        plot_mse_comparison(data, plots_dir)

        print("\nGenerating detailed MSE analysis...")
        plot_detailed_mse_analysis(data, plots_dir)

        print("\nGenerating summary report...")
        generate_summary_report(data, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")

if __name__ == "__main__":
    main()
