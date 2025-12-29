# --- Experiment groups for ablation study ---
# The ablation switches are:
#   interpolation_on         (tri-linear interpolation)
#   importance_sampling_on   (light importance sampling / light-guided probes)
#   adaptive_logic_on        (adaptive weight update)
#
# Required runs (in order):
#   Baseline      : all OFF
#   V1            : interpolation only
#   V2            : interpolation + adaptive
#   Full_Hybrid   : all ON
EXPERIMENT_GROUPS = [
    {
        "name": "Baseline",
        "interpolation_on": False,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
    },
    {
        "name": "V1",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
    },
    {
        "name": "V2",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": True,
    },
    {
        "name": "Full_Hybrid",
        "interpolation_on": True,
        "importance_sampling_on": True,
        "adaptive_logic_on": True,
    },
]

