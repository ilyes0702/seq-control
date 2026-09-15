"""Simulation script validation of inverse control models.

This script initializes the plant, loads a trained controller and scalers,
generates reference trajectories (dynamic and constant), and runs the
simulation routine.
"""

# Import utility functions
from seq_control.utils.validation_utils import *
from seq_control.utils.general_utils import *

# Import plants
from seq_control.classes.plants.IdiophasePlant import *
from seq_control.classes.plants.TrophophasePlant import *
from seq_control.classes.plants.ChemostatPlant import *
from seq_control.classes.plants.IndForProteinProductionPlant import *
from seq_control.classes.plants.CocultivationPlant import *

# Import controller models
from seq_control.classes.sequence_models.MambaInverseController import *
from seq_control.classes.sequence_models.ESNInverseController import *
from seq_control.classes.sequence_models.LSTMInverseController import *
from seq_control.classes.sequence_models.TransformerInverseController import *

def main():
    # Used data
    REF_data_date = "2026-09-15"
    REF_data_date_and_time = "2026-09-15_16-50-06"
    # Used models
    REF_mod_date = "2026-09-15"
    REF_mod_date_and_time = "2026-09-15_23-55-00"

    # Initialize plant instances and load training data    
    plant_dict =    {
        "ChemostatPlant": {
            "plant" : ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
            "val_data_io": torch.load(f"src/seq_control/results/{REF_data_date}/{REF_data_date_and_time}/ChemostatPlant/io/dataset/{REF_data_date_and_time}_val_io_data.pt"),
            "y_ref": [generate_exponential_decay_trajectory(
                    hyperparam_config_ChemostatPlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_ChemostatPlant["training_data_cfg"]["dt"],
                    hyperparam_config_ChemostatPlant["validation_trajectories"]["y_start"][0],
                    hyperparam_config_ChemostatPlant["validation_trajectories"]["y_target"][0],
                    hyperparam_config_ChemostatPlant["validation_trajectories"]["tau"][0]
                    )],
            "models_dict": {
                "MambaInverseController": {
                    "model": load_model(MambaInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/MambaInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "LSTMInverseController": {
                    "model": load_model(LSTMInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/LSTMInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "TransformerInverseController": {
                    "model": load_model(TransformerInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/TransformerInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "ESNInverseController": {
                    "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/ESNInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pkl"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/ChemostatPlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                }
            }
        },
        "TrophophasePlant": {
            "plant" : TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
            "val_data_io": torch.load(f"src/seq_control/results/{REF_data_date}/{REF_data_date_and_time}/TrophophasePlant/io/dataset/{REF_data_date_and_time}_val_io_data.pt"),
            "y_ref": [generate_exponential_decay_trajectory(
                    hyperparam_config_TrophophasePlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_TrophophasePlant["training_data_cfg"]["dt"],
                    hyperparam_config_TrophophasePlant["validation_trajectories"]["y_start"][0],
                    hyperparam_config_TrophophasePlant["validation_trajectories"]["y_target"][0],
                    hyperparam_config_TrophophasePlant["validation_trajectories"]["tau"][0]
                    )],
            "models_dict": {
                "MambaInverseController": {
                    "model": load_model(MambaInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/MambaInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "LSTMInverseController": {
                    "model": load_model(LSTMInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/LSTMInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")    
                },
                "TransformerInverseController": {
                    "model": load_model(TransformerInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/TransformerInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "ESNInverseController": {
                    "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/ESNInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pkl"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/TrophophasePlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                }
            }
        },
        "IdiophasePlant": {
            "plant" : IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
            "val_data_io": torch.load(f"src/seq_control/results/{REF_data_date}/{REF_data_date_and_time}/IdiophasePlant/io/dataset/{REF_data_date_and_time}_val_io_data.pt"),
            "y_ref": [
                generate_exponential_decay_trajectory(
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_IdiophasePlant["training_data_cfg"]["dt"],
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["y_start"][0],
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["y_target"][0],
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["tau"][0]
                    ),
                generate_exponential_decay_trajectory(
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_IdiophasePlant["training_data_cfg"]["dt"],
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["y_start"][1],
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["y_target"][1],
                    hyperparam_config_IdiophasePlant["validation_trajectories"]["tau"][1]
                    )
                ],
            "models_dict": {
                "MambaInverseController": {
                    "model": load_model(MambaInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/MambaInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "LSTMInverseController": {
                    "model": load_model(LSTMInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/LSTMInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")    
                },
                "TransformerInverseController": {
                    "model": load_model(TransformerInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/TransformerInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "ESNInverseController": {
                    "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/ESNInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pkl"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IdiophasePlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                }
            }
        },
        "CoCultivationPlant": {
            "plant" : CoCultivationPlant(hyperparam_config=hyperparam_config_CoCultivationPlant),
            "val_data_io": torch.load(f"src/seq_control/results/{REF_data_date}/{REF_data_date_and_time}/CoCultivationPlant/io/dataset/{REF_data_date_and_time}_val_io_data.pt"),
            "y_ref": [generate_exponential_decay_trajectory(
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_CoCultivationPlant["training_data_cfg"]["dt"],
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["y_start"][0],
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["y_target"][0],
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["tau"][0]
                    ),
                generate_exponential_decay_trajectory(
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_CoCultivationPlant["training_data_cfg"]["dt"],
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["y_start"][1],
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["y_target"][1],
                    hyperparam_config_CoCultivationPlant["validation_trajectories"]["tau"][1]
                    )
                ],
            "models_dict": {
                "MambaInverseController": {
                    "model": load_model(MambaInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/MambaInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "LSTMInverseController": {
                    "model": load_model(LSTMInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/LSTMInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")    
                },
                "TransformerInverseController": {
                    "model": load_model(TransformerInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/TransformerInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "ESNInverseController": {
                    "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/ESNInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pkl"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/CoCultivationPlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                }
            }
        },
        "IndForProteinProductionPlant" : {
            "plant" : IndForProteinProductionPlant(hyperparam_config=hyperparam_config_IndForProteinProductionPlant),
            "val_data_io": torch.load(f"src/seq_control/results/{REF_data_date}/{REF_data_date_and_time}/IndForProteinProductionPlant/io/dataset/{REF_data_date_and_time}_val_io_data.pt"),
            "y_ref": [generate_exponential_decay_trajectory(
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_IndForProteinProductionPlant["training_data_cfg"]["dt"],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["y_start"][0],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["y_target"][0],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["tau"][0]
                    ),
                generate_exponential_decay_trajectory(
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_IndForProteinProductionPlant["training_data_cfg"]["dt"],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["y_start"][1],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["y_target"][1],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["tau"][1]
                    ),
                generate_exponential_decay_trajectory(
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["seq_len"],
                    hyperparam_config_IndForProteinProductionPlant["training_data_cfg"]["dt"],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["y_start"][2],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["y_target"][2],
                    hyperparam_config_IndForProteinProductionPlant["validation_trajectories"]["tau"][2]
                    )
                ],
            "models_dict": {
                "MambaInverseController": {
                    "model": load_model(MambaInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/MambaInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/MambaInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "LSTMInverseController": {
                    "model": load_model(LSTMInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/LSTMInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/LSTMInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")    
                },
                "TransformerInverseController": {
                    "model": load_model(TransformerInverseController,f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/TransformerInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pt"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/TransformerInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                },
                "ESNInverseController": {
                    "model": load_model_esn(ESNInverseController, f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/ESNInverseController/fold_1/{REF_mod_date_and_time}_best_fold_model.pkl"),
                    "x_scaler" : load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_x.pkl"),
                    "y_scaler": load_scaler(f"src/seq_control/results/{REF_mod_date}/{REF_mod_date_and_time}/IndForProteinProductionPlant_ic/ESNInverseController/fold_1/scalers/{REF_mod_date_and_time}_scaler_y.pkl")
                }
            },
        }
    }

    for pl in plant_dict.keys():
        print(f"Validating plant: {pl}")
        validate_multiple_controllers(
            models_dict=plant_dict[pl]["models_dict"],
            plant=plant_dict[pl]["plant"],
            dataset_io=plant_dict[pl]["val_data_io"],
            hyperparam_config=plant_dict[pl]["plant"].hyperparam_config,
            dirname=f"results/multi_model_validation_{pl}",
            start_idx=2,
            window_len=100,
            mode="closed_loop",
            show_plots=True
        )

        
    # validate_multiple_controllers(
    #     models_dict=plant_dict["models_dict"],
    #     plant=plant_dict["TrophophasePlant"]["plant"],
    #     dataset_io=plant_dict["TrophophasePlant"]["val_data_io"],
    #     hyperparam_config=plant_dict["TrophophasePlant"]["plant"].hyperparam_config,
    #     dirname="results/multi_model_validation",
    #     start_idx=20,
    #     mode="open_loop",
    #     show_plots=True
    # )


    # validate_controller_ext_ref(
    #         model_dict["LSTMInverseController"]["model"],
    #         plant_dict["TrophophasePlant"]["plant"],
    #         r_smooth_decay,
    #         scaler_x=model_dict["MambaInverseController"]["x_scaler"],
    #         scaler_y=model_dict["MambaInverseController"]["y_scaler"],
    #         hyperparam_config=plant_dict["TrophophasePlant"]["plant"].hyperparam_config,
    #         dirname="./plots_ref",
    #         start_idx=2,
    #         mode="open_loop"
    #     )

    # for pl in plant_dict.keys():
    #     print(f"Validating plant: {pl}")
    #     validate_controller_ext_ref_multi(
    #         models_dict=plant_dict[pl]["models_dict"],
    #         plant=plant_dict[pl]["plant"],
    #         y_ref=plant_dict[pl]["y_ref"],
    #         hyperparam_config=plant_dict[pl]["plant"].hyperparam_config,
    #         dirname=f"./plots_ref_multi_{pl}",
    #         start_idx=20,
    #         window_len=10,
    #         mode="closed_loop",
    #         show_plots=False
    #     )


    # validate_controller(
    #     model = model_dict["MambaInverseController"]["model"],
    #     plant = plant_dict["ChemostatPlant"]["plant"],
    #     dataset_io= plant_dict["ChemostatPlant"]["val_data_io"],
    #     scaler_x = model_dict["MambaInverseController"]["x_scaler"],
    #     scaler_y = model_dict["MambaInverseController"]["y_scaler"],
    #     hyperparam_config=plant_dict["ChemostatPlant"]["plant"].hyperparam_config,
    #     dirname = "./plots_ref",
    #     start_idx=2,
    #     mode="closed_loop",
    #     show_plots = True
    # )

    

if __name__ == "__main__":
    main()   