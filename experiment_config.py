"""
Experiment Configuration File
Controls the parameters for the automated raytracing experiments
"""

# Experiment Settings
EXPERIMENT_DURATION_FRAMES = 1000  # Total frames to run
SPHERE_MOVE_INTERVAL = 200        # Move big sphere every N frames
BASE_UPDATE_PROBABILITY = 0.05     # Base probability for grid updates (5%)

# Performance Monitoring
FPS_LOG_INTERVAL = 100             # Log FPS every N frames
MEMORY_LOG_INTERVAL = 100          # Log memory usage every N frames

# Convergence Detection
MIN_FRAMES_BEFORE_CONVERGENCE_CHECK = 5  # Minimum frames to wait after movement
CONVERGENCE_CHECK_INTERVAL = 10          # Check convergence every N frames during convergence

# Screenshot Settings
SAVE_SCREENSHOT_AFTER_MOVE_FRAME = 5     # Save screenshot N frames after sphere movement
SAVE_SCREENSHOT_CONVERGED_FRAME = 50      # Save screenshot at frame N during convergence

# Adaptive Logic Settings
ADAPTIVE_BOOST_MULTIPLIER = 3.0          # Weight multiplier for adaptive regions
ADAPTIVE_INFLUENCE_RADIUS = 3.0          # Influence radius as multiple of sphere radius
GAUSSIAN_BLUR_ENABLED = True             # Enable Gaussian blur for weight smoothing

# Grid Settings
GRID_RESOLUTION = (16, 16, 16)            # Grid resolution
GRID_PADDING = 0.5                        # Padding around scene AABB

# Output Settings
OUTPUT_DIRECTORY = "experiment_results"
SAVE_FPS_DATA = True
SAVE_CONVERGENCE_DATA = True
SAVE_SCREENSHOTS = True
SAVE_SUMMARY = True

# Comparison Mode
RUN_COMPARISON = True                     # Run both adaptive and non-adaptive experiments
COMPARISON_ADAPTIVE_FIRST = True         # Run adaptive experiment first

# Logging
VERBOSE_LOGGING = True                    # Enable detailed logging
TIMESTAMP_LOGGING = True                  # Add timestamps to all logs
