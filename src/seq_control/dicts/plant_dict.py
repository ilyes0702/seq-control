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
from seq_control.utils.validation_utils import *

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#   CHANGE THESE DATES WHEN UPDATING THE DATASET OR TRAINED MODELS   #
DATA_REFERENCE_DATE = "2026-09-21"                                   #                           
DATA_REFERENCE_DATE_TIME = "2026-09-21_13-34-44"                     #
                                                                     #
MODEL_REFERENCE_DATE = "2026-09-21"                                  #
MODEL_REFERENCE_DATE_TIME = "2026-09-21_21-04-17"                    #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

plant_dict =    {
    "ChemostatPlant": {
        "plant" : ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
        "train_data_sw_ic": torch.load(f"src/seq_control/results/{DATA_REFERENCE_DATE}/{DATA_REFERENCE_DATE_TIME}/ChemostatPlant/sw_ic/dataset/{DATA_REFERENCE_DATE_TIME}_sw_ic_training_data.pt",
        weights_only=True),
        "val_data_io": torch.load(f"src/seq_control/results/{DATA_REFERENCE_DATE}/{DATA_REFERENCE_DATE_TIME}/ChemostatPlant/io/dataset/{DATA_REFERENCE_DATE_TIME}_val_io_data.pt"),
        "y_ref_const": [generate_reference_trajectory(
                            steps=hyperparam_config_ChemostatPlant["validation_trajectories"]["seq_len"],
                            dt=hyperparam_config_ChemostatPlant["training_data_cfg"]["dt"],
                            constant_val=hyperparam_config_ChemostatPlant["validation_trajectories"]["constant_value"][0],
                            )],
        "y_ref_exp_decay": [generate_exponential_decay_trajectory(
                                    steps=hyperparam_config_ChemostatPlant["validation_trajectories"]["seq_len"],
                                    dt=hyperparam_config_ChemostatPlant["training_data_cfg"]["dt"],
                                    y_start=hyperparam_config_ChemostatPlant["validation_trajectories"]["y_start"][0],
                                    y_target=hyperparam_config_ChemostatPlant["validation_trajectories"]["y_target"][0],
                                    tau=hyperparam_config_ChemostatPlant["validation_trajectories"]["tau"][0],
                                    )],
        "models_dict": {
            "MambaInverseController": {
                "model": load_model(MambaInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/MambaInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/MambaInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/MambaInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "LSTMInverseController": {
                "model": load_model(LSTMInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/LSTMInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/LSTMInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/LSTMInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "TransformerInverseController": {
                "model": load_model(TransformerInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/TransformerInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/TransformerInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/TransformerInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "ESNInverseController": {
                "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/ESNInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_production_model.pkl"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/ESNInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/ChemostatPlant_ic/ESNInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_scaler_y.pkl")
            }
        }                    
        
    },
    "TrophophasePlant": {
        "plant" : TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
        "train_data_sw_ic": torch.load(f"src/seq_control/results/{DATA_REFERENCE_DATE}/{DATA_REFERENCE_DATE_TIME}/TrophophasePlant/sw_ic/dataset/{DATA_REFERENCE_DATE_TIME}_sw_ic_training_data.pt", 
        weights_only=True),
        "val_data_io": torch.load(f"src/seq_control/results/{DATA_REFERENCE_DATE}/{DATA_REFERENCE_DATE_TIME}/TrophophasePlant/io/dataset/{DATA_REFERENCE_DATE_TIME}_val_io_data.pt"),
        "y_ref_const": [generate_reference_trajectory(
            steps=hyperparam_config_TrophophasePlant["validation_trajectories"]["seq_len"],
            dt=hyperparam_config_TrophophasePlant["training_data_cfg"]["dt"],
            constant_val=hyperparam_config_TrophophasePlant["validation_trajectories"]["constant_value"][0],
            )],
        "y_ref_exp_decay": [generate_exponential_decay_trajectory(
            steps=hyperparam_config_TrophophasePlant["validation_trajectories"]["seq_len"],
            dt=hyperparam_config_TrophophasePlant["training_data_cfg"]["dt"],
            y_start=hyperparam_config_TrophophasePlant["validation_trajectories"]["y_start"][0],
            y_target=hyperparam_config_TrophophasePlant["validation_trajectories"]["y_target"][0],
            tau=hyperparam_config_TrophophasePlant["validation_trajectories"]["tau"][0],
                                            )],
        "models_dict": {
            "MambaInverseController": {
                "model": load_model(MambaInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/MambaInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/MambaInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/MambaInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "LSTMInverseController": {
                "model": load_model(LSTMInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/LSTMInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/LSTMInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/LSTMInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "TransformerInverseController": {
                "model": load_model(TransformerInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/TransformerInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/TransformerInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/TransformerInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "ESNInverseController": {
                "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/ESNInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_production_model.pkl"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/ESNInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/TrophophasePlant_ic/ESNInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_scaler_y.pkl")
            }
        }      
    },
    "IdiophasePlant": {
        "plant" : IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
        "train_data_sw_ic": torch.load(f"src/seq_control/results/{DATA_REFERENCE_DATE}/{DATA_REFERENCE_DATE_TIME}/IdiophasePlant/sw_ic/dataset/{DATA_REFERENCE_DATE_TIME}_sw_ic_training_data.pt", weights_only=True),
        "val_data_io": torch.load(f"src/seq_control/results/{DATA_REFERENCE_DATE}/{DATA_REFERENCE_DATE_TIME}/IdiophasePlant/io/dataset/{DATA_REFERENCE_DATE_TIME}_val_io_data.pt"),
        "y_ref_const": [generate_reference_trajectory(
                            steps=hyperparam_config_IdiophasePlant["validation_trajectories"]["seq_len"],
                            dt=hyperparam_config_IdiophasePlant["training_data_cfg"]["dt"],
                            constant_val=hyperparam_config_IdiophasePlant["validation_trajectories"]["constant_value"][0],
                            ),
                generate_reference_trajectory(
                            steps=hyperparam_config_IdiophasePlant["validation_trajectories"]["seq_len"],
                            dt=hyperparam_config_IdiophasePlant["training_data_cfg"]["dt"],
                            constant_val=hyperparam_config_IdiophasePlant["validation_trajectories"]["constant_value"][1],
                            )
                ],
        "models_dict": {
            "MambaInverseController": {
                "model": load_model(MambaInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/MambaInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/MambaInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/MambaInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "LSTMInverseController": {
                "model": load_model(LSTMInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/LSTMInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/LSTMInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/LSTMInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "TransformerInverseController": {
                "model": load_model(TransformerInverseController,f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/TransformerInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_retrained_model.pt"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/TransformerInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/TransformerInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_final_scaler_y.pkl")
            },
            "ESNInverseController": {
                "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/ESNInverseController/final_best_model/{MODEL_REFERENCE_DATE_TIME}_final_production_model.pkl"),
                "x_scaler" : load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/ESNInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_scaler_x.pkl"),
                "y_scaler": load_scaler(f"src/seq_control/results/{MODEL_REFERENCE_DATE}/{MODEL_REFERENCE_DATE_TIME}/IdiophasePlant_ic/ESNInverseController/final_best_model/scalers/{MODEL_REFERENCE_DATE_TIME}_scaler_y.pkl")
            }
        }  
    }
}