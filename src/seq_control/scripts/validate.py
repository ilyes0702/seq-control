"""Simulation script validation of inverse control models.

This script initializes the plant, loads a trained controller and scalers,
generates reference trajectories (dynamic and constant), and runs the
simulation routine.
"""

import math

# Import utility functions
from seq_control.utils.validation_utils import *

# Import dictionaries
from seq_control.dicts.plant_dict import plant_dict 

def main():

    modes = ["open_loop", "closed_loop"]
    WINDOW_LEN_FRAC = 100
    

    
    # Validation on trajectories generated in the same batch as the training data
    for mode in modes:
        for pl, pl_dict in plant_dict.items():
            validate_multiple_controllers(
                models_dict=pl_dict["models_dict"],
                plant=pl_dict["plant"],
                dataset_io=pl_dict["val_data_io"],
                hyperparam_config=pl_dict["plant"].hyperparam_config,
                dirname=f"results/multi_model_validation_{pl}_{mode}",
                start_idx=2,
                window_len=WINDOW_LEN_FRAC,
                mode=mode,
                show_plots=True
            )

    # Validation on non-training-data-related reference trajectories        
    for pl, pl_dict in plant_dict.items():
        validate_controller_ext_ref_multi(
            models_dict=pl_dict["models_dict"],
            plant=pl_dict["plant"],
            y_ref=pl_dict["y_ref_const"],
            hyperparam_config=pl_dict["plant"].hyperparam_config,
            dirname=f"./plots_ref_multi_{pl}_closed_loop",
            start_idx=2,
            window_len=WINDOW_LEN_FRAC,
            mode="closed_loop",
            show_plots=True
        )

if __name__ == "__main__":
    main()   