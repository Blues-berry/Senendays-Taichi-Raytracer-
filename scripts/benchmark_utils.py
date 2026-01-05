import os
import csv
import numpy as np
from datetime import datetime
import taichi as ti
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global variable to hold the output directory. This should be set by the main runner.
output_dir = "."

def set_output_dir(path):
    """Sets the global output directory for all utility functions."""
    global output_dir
    output_dir = path
    os.makedirs(output_dir, exist_ok=True)

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_screenshot(current_frame, filename):
    """Save a screenshot with the given filename into the output directory."""
    filepath = os.path.join(output_dir, filename)
    ti.tools.imwrite(current_frame, filepath)
    log_message(f"Saved screenshot: {filepath}")

