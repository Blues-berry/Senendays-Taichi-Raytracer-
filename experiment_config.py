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
ADAPTIVE_BOOST_MULTIPLIER = 1.0          # Weight multiplier for adaptive regions (reduced for testing)
ADAPTIVE_INFLUENCE_RADIUS = 3.0          # Influence radius as multiple of sphere radius
GAUSSIAN_BLUR_ENABLED = False            # Enable Gaussian blur for weight smoothing

# Grid Settings
GRID_RESOLUTION = (64, 64, 64)            # Grid resolution (upgraded from 32x32x32)
GRID_PADDING = 1.0                        # Padding around scene AABB

# Adaptive sampling settings (brightness-based)
ADAPTIVE_BRIGHTNESS_THRESHOLD = 0.05      # Luminance contrast threshold to trigger extra samples
ADAPTIVE_SAMPLING_MULTIPLIER = 1.0         # Additional multiplier when above threshold (adds to 1.0)
ADAPTIVE_MAX_MULTIPLIER = 2.0             # Cap multiplier (so weight <= 1 + this)

# Importance sampling / NEE
LIGHT_IMPORTANCE_SCALE = 2.0              # Scale applied to emissive hits during grid probing
# Variance-guided sampling
VARIANCE_SAMPLING_SCALE = 2.0             # Multiplier for converting variance -> extra probe samples
MAX_PROBE_SAMPLES = 16                    # Upper bound for per-cell probe samples

# Distance-based leak prevention
DISTANCE_MISMATCH_THRESHOLD = 1.0         # Threshold (in multiples of cell size) to detect mismatch

# Anti-Leak Mechanisms
NORMAL_WEIGHTING_ENABLED = True            # Enable normal-weighted interpolation (prevents cross-surface bleeding)
NORMAL_POWER = 8.0                        # Power for normal alignment (higher = stricter)
DISTANCE_WEIGHTING_ENABLED = True          # Enable distance-based weighting
DISTANCE_CUTOFF_MULTIPLIER = 1.5         # Zero weight if distance > CUTOFF * cell_size
NEIGHBOR_CLAMPING_ENABLED = True          # Enable neighbor clamping (clamp to max of 26 neighbors)

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
