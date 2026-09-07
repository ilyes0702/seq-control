"""
Script to train inverse models.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).
"""

from seq_control.config import *

# Import plants
from seq_control.classes.plants.IndForProteinProductionPlant import *
from seq_control.classes.plants.ChemostatPlant import *
from seq_control.classes.plants.TrophophasePlant import *
from seq_control.classes.plants.IdiophasePlant import *
from seq_control.classes.plants.CocultivationPlant import *

# Import inverse controller architectures
from seq_control.classes.controllers.MambaInverseController import *
from seq_control.classes.controllers.ESNInverseController import *
from seq_control.classes.controllers.LSTMInverseController import *
from seq_control.classes.controllers.TransformerInverseController import *

# Import utilities
from seq_control.utils.training_utils import * 
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.plotting_utils import *

if __name__ == "__main__":
    # Instantiate plant using hyperparameters.
    hyperparam_config = hyperparam_config_TrophophasePlant
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)

    # Directory name to store training artifacts (models, plots, logs).
    dirname= f"{plant.__class__.__name__}_training"
    # Load the dataset from disk. 
    train_data_sw_ic_path = (
       "src/seq_control/results/2026-09-07/2026-09-07_10-11-58/TrophophasePlant/sw_ic/dataset/2026-09-07_10-11-58_sw_ic_training_data.pt"
    )
    train_data_sw_ic = torch.load(train_data_sw_ic_path, weights_only=True)

    train_data_sw_sysid_path = (
           "src/seq_control/results/2026-09-07/2026-09-07_10-11-58/TrophophasePlant/sw_ic/dataset/2026-09-07_10-11-58_sw_ic_validation_data.pt"
        )
    train_data_sw_sysid = torch.load(train_data_sw_sysid_path, weights_only=True)
 
    # Initialize the inverse controller.
    controller = MambaInverseController(hyperparam_config=hyperparam_config)

    train_inverse_controller(
        model=controller,
        plant=plant,
        sw_ic_dataset=train_data_sw_ic,
        hyperparam_config=hyperparam_config,
        dirname=dirname,
        show_plots=True
    )  

    # run_optuna_study(
    #     n_trials=10,
    #     model_class=MambaInverseController,
    #     dataset=train_data_sw_ic,
    #     hyperparam_config=hyperparam_config,
    #     plant=plant
    # )