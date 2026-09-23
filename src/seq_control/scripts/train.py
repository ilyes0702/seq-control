"""
Script to train inverse models.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).
"""

# Import utilities
from seq_control.config import *
from seq_control.utils.training_utils import * 

# Import dictionaries
from seq_control.dicts.plant_dict import plant_dict

if __name__ == "__main__":
    
    # Iterate through benchmark plants
    for plant_name, plant_data in plant_dict.items():
        plant = plant_data["plant"]
        hyperparam_config = plant.hyperparam_config
        train_data = plant_data["train_data_sw_ic"]
        dirname= f"{plant_name}_ic"
        model_dict = plant_data["models_dict"]

        # Iterate through the sequence models
        for co_name, co in model_dict.items():            
            model = co
            run_optuna_study(
                model_class=co.__class__,
                dataset=train_data,
                base_config=hyperparam_config,
                plant=plant,
                dirname=f"{dirname}/{co_name}",
                param_space=hyperparam_config[f"{co_name}_param_space"],                        
                n_trials=10
            )