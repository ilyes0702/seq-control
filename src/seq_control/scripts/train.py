"""
Script to train inverse models.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).
"""

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
from seq_control.config import *
from seq_control.utils.training_utils import * 
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.plotting_utils import *

if __name__ == "__main__":

    # Define the plants to investigate along with the training data
    plant_dict =    {
        "ChemostatPlant": {
            "plant" : ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
            "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/ChemostatPlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_data.pt", 
            weights_only=True),
            "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/ChemostatPlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_training_data.pt", weights_only=True)
        },
        "TrophophasePlant": {
            "plant" : TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
            "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/TrophophasePlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_training_data.pt", 
            weights_only=True),
            "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/TrophophasePlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_training_data.pt", weights_only=True)
        },
        "IdiophasePlant": {
            "plant" : IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
            "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IdiophasePlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_training_data.pt", weights_only=True),
            "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IdiophasePlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_training_data.pt", weights_only=True)
        },
        "CoCultivationPlant": {
            "plant" : CoCultivationPlant(hyperparam_config=hyperparam_config_CoCultivationPlant),
            "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/CoCultivationPlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_training_data.pt", weights_only=True),
            "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/CoCultivationPlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_training_data.pt", weights_only=True)
        },
        "IndForProteinProductionPlant" : {
            "plant" : IndForProteinProductionPlant(hyperparam_config=hyperparam_config_IndForProteinProductionPlant),
            "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IndForProteinProductionPlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_training_data.pt", weights_only=True),
            "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IndForProteinProductionPlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_training_data.pt", weights_only=True)
        }
    }

    # Define inverse controller architectures to test
    controller_list = [
        MambaInverseController,
        LSTMInverseController,
        TransformerInverseController,
        ESNInverseController
    ]

    surrmod_list = [
        MambaSurrogateModel
    ]
    
    # Iterate through control architectures and benchmark plants
    for plant_name, plant_data in plant_dict.items():
    
        plant = plant_data["plant"]
        hyperparam_config = plant.hyperparam_config
        print("----------------------------",plant.__class__.__name__)
        train_data = plant_data["train_data_sw_ic"]
        dirname= f"{plant_name}_ic"
        #Training of the inverse controllers
        for co in controller_list:
            
            model = co(hyperparam_config)
            print("X_raw shape:", train_data["X_raw"].shape)  
            print("Y_raw shape:", train_data["Y_raw"].shape) 
            if co == ESNInverseController:
                train_controller_esn(
                        model=ESNInverseController(hyperparam_config=hyperparam_config),
                        X_raw=train_data["X_raw"],
                        Y_raw=train_data["Y_raw"],
                        hyperparam_config=hyperparam_config,
                        dirname=f"{dirname}/ESNInverseController"
                    )
            else:
                for plant_name, plant_data in plant_dict.items():
                    
                    train_sequence_model(
                        model=model,
                        plant=plant,
                        sw_dataset=train_data,
                        hyperparam_config=hyperparam_config,
                        dirname=f"{dirname}/{co.__name__}",
                        show_plots= False
                    )  

        # Training of the surrogate models
        # for surr in surrmod_list:
        #     train_data = plant_data["train_data_sysid"]
        #     dirname= f"{plant_name}_sysid"
        #     model = surr(hyperparam_config)
        #     # if surr == ESNInverseSurrogateModel:
        #     #     train_controller_esn(
        #     #             model=ESNInverseController(hyperparam_config=hyperparam_config),
        #     #             X_raw=train_data_sw_ic["X_raw"],
        #     #             Y_raw=train_data_sw_ic["Y_raw"],
        #     #             hyperparam_config=hyperparam_config,
        #     #             dirname=f"f{dirname_ic}/ESNInverseController"
        #     #         )
        #     # else:
        #     for plant_name, plant_data in plant_dict.items():
        #         train_sequence_model(
        #             model=model,
        #             plant=plant,
        #             sw_dataset=train_data,
        #             hyperparam_config=hyperparam_config,
        #             dirname=f"f{dirname}/{surr.__class__.__name__}",
        #             show_plots=True
        #         ) 

    
    # run_optuna_study(
    #     n_trials=10,
    #     model_class=MambaInverseController,
    #     dataset=train_data_sw_ic,
    #     hyperparam_config=hyperparam_config,
    #     plant=plant
    # )