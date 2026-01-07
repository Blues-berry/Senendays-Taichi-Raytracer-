import ablation

# Quick sanity check: compare Normal weighting OFF vs ON (other anti-leak off)
ablation.EXPERIMENT_GROUPS = [
    {
        "name": "Normal_OFF",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
        "normal_weighting_on": False,
        "distance_weighting_on": False,
        "neighbor_clamping_on": False,
    },
    {
        "name": "Normal_ON",
        "interpolation_on": True,
        "importance_sampling_on": False,
        "adaptive_logic_on": False,
        "normal_weighting_on": True,
        "distance_weighting_on": False,
        "neighbor_clamping_on": False,
    },
]

ablation.run_group_experiments("cornell_box")
print("DONE quick 2 groups")

