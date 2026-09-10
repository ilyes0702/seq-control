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
from seq_control.classes.sequence_models.MambaInverseController import *
from seq_control.classes.sequence_models.ESNInverseController import *
from seq_control.classes.sequence_models.LSTMInverseController import *
from seq_control.classes.sequence_models.TransformerInverseController import *

# Import utilities
from seq_control.utils.training_utils import * 
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.plotting_utils import *

if __name__ == "__main__":
    # Instantiate plant using hyperparameters.
    hyperparam_config = hyperparam_config_TrophophasePlant
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)

    # Directory name to store training artifacts (models, plots, logs).
    
    # Load the dataset from disk. 
    train_data_sw_ic_path = (
       "src/seq_control/results/2026-09-07/2026-09-07_10-11-58/TrophophasePlant/sw_ic/dataset/2026-09-07_10-11-58_sw_ic_training_data.pt"
    )
    train_data_sw_ic = torch.load(train_data_sw_ic_path, weights_only=True)

    train_data_sw_sysid_path = (
           "src/seq_control/results/2026-09-07/2026-09-07_10-11-58/TrophophasePlant/sw_sysid/dataset/2026-09-07_10-11-58_sw_sysid_training_data.pt"
        )
    train_data_sw_sysid = torch.load(train_data_sw_sysid_path, weights_only=True)

    plant_dict =    {
        # "ChemostatPlant": {
        #     "plant" : ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
        #     "train_data": torch.load("src/seq_control/results/2026-09-10/2026-09-10_17-44-16/ChemostatPlant/sw_ic/dataset/2026-09-10_17-44-16_sw_ic_training_data.pt", 
        #     weights_only=True)
        # },
        # "TrophophasePlant": {
        #             "plant" : TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
        #             "train_data": torch.load("src/seq_control/results/2026-09-10/2026-09-10_17-44-16/TrophophasePlant/sw_ic/dataset/2026-09-10_17-44-16_sw_ic_training_data.pt", 
        #             weights_only=True)
        #         },
        "IdiophasePlant": {
                    "plant" : IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
                    "train_data": torch.load("src/seq_control/results/2026-09-10/2026-09-10_17-44-16/IdiophasePlant/sw_ic/dataset/2026-09-10_17-44-16_sw_ic_training_data.pt", weights_only=True)
                },
        # "CoCultivationPlant": {
        #             "plant" : CoCultivationPlant(hyperparam_config=hyperparam_config_CoCultivationPlant),
        #             "train_data": torch.load("src/seq_control/results/2026-09-10/2026-09-10_17-44-16/CoCultivationPlant/sw_ic/dataset/2026-09-10_17-44-16_sw_ic_training_data.pt", weights_only=True)
        #         },
        # "IndForProteinProductionPlant" : {
        #     "plant" : IndForProteinProductionPlant(hyperparam_config=hyperparam_config_IndForProteinProductionPlant),
        #     "train_data": torch.load("src/seq_control/results/2026-09-10/2026-09-10_17-44-16/IndForProteinProductionPlant/sw_ic/dataset/2026-09-10_17-44-16_sw_ic_training_data.pt", weights_only=True)

        # }
    }
    # Initialize the inverse controller.
    controller_list = [
        MambaInverseController,
        #LSTMInverseController(hyperparam_config=hyperparam_config),
        #TransformerInverseController(hyperparam_config=hyperparam_config)
    ]
    controller = MambaInverseController(hyperparam_config=hyperparam_config)

    for co in controller_list:
        for plant_name, plant_data in plant_dict.items():

            print(plant_name)

            dirname_ic= f"{plant_name}_ic"

            plant = plant_data["plant"]
            hyperparam_config = plant.hyperparam_config
            model = co(hyperparam_config)
            
            train_data_sw_ic = plant_data["train_data"]

            train_sequence_model(
                model=model,
                plant=plant,
                sw_dataset=train_data_sw_ic,
                hyperparam_config=hyperparam_config,
                dirname=f"dirname_ic/{co.__class__.__name__}",
                show_plots=True
            )  

    # train_controller_esn(
    #     model=ESNInverseController(hyperparam_config=hyperparam_config),
    #     X_raw=train_data_sw_ic["X_raw"],
    #     Y_raw=train_data_sw_ic["Y_raw"],
    #     hyperparam_config=hyperparam_config,
    #     dirname=f"dirname_ic/esn"

    # )

    # dirname_sysid =  f"{plant.__class__.__name__}_sysid"

    # surr = MambaInverseController(hyperparam_config=hyperparam_config)
    # # Surrogate model
    # train_sequence_model(
    #         model=surr,
    #         plant=plant,
    #         sw_ic_dataset=train_data_sw_sysid,
    #         hyperparam_config=hyperparam_config,
    #         dirname=dirname_sysid,
    #         show_plots=True
    #     )  

    # run_optuna_study(
    #     n_trials=10,
    #     model_class=MambaInverseController,
    #     dataset=train_data_sw_ic,
    #     hyperparam_config=hyperparam_config,
    #     plant=plant
    # )