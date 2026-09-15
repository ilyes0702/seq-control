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

    # Initialize plant instances and load training data    
    plant_dict =    {
            "ChemostatPlant": {
                "plant" : ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
                "val_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/ChemostatPlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_validation_data.pt", 
                weights_only=True),
                "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/ChemostatPlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_validation_data.pt", weights_only=True),
                "val_data_io": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/ChemostatPlant/io/dataset/2026-09-13_11-19-22_val_io_data.pt")
            },
            "TrophophasePlant": {
                "plant" : TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
                "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/TrophophasePlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_validation_data.pt", 
                weights_only=True),
                "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/TrophophasePlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_validation_data.pt", weights_only=True),
                "val_data_io": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/TrophophasePlant/io/dataset/2026-09-13_11-19-22_val_io_data.pt")
            },
            "IdiophasePlant": {
                "plant" : IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
                "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IdiophasePlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_validation_data.pt", weights_only=True),
                "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IdiophasePlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_validation_data.pt", weights_only=True),
                "val_data_io": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IdiophasePlant/io/dataset/2026-09-13_11-19-22_val_io_data.pt")
            },
            "CoCultivationPlant": {
                "plant" : CoCultivationPlant(hyperparam_config=hyperparam_config_CoCultivationPlant),
                "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/CoCultivationPlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_validation_data.pt", weights_only=True),
                "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/CoCultivationPlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_validation_data.pt", weights_only=True),
                "val_data_io": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/CoCultivationPlant/io/dataset/2026-09-13_11-19-22_val_io_data.pt")
            },
            "IndForProteinProductionPlant" : {
                "plant" : IndForProteinProductionPlant(hyperparam_config=hyperparam_config_IndForProteinProductionPlant),
                "train_data_sw_ic": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IndForProteinProductionPlant/sw_ic/dataset/2026-09-13_11-19-22_sw_ic_validation_data.pt", weights_only=True),
                "train_data_sysid": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IndForProteinProductionPlant/sw_sysid/dataset/2026-09-13_11-19-22_sw_sysid_validation_data.pt", weights_only=True),
                "val_data_io": torch.load("src/seq_control/results/2026-09-13/2026-09-13_11-19-22/IndForProteinProductionPlant/io/dataset/2026-09-13_11-19-22_val_io_data.pt")
            }
        }

    plant = plant_dict["ChemostatPlant"]["plant"]

    # Chemostat plant
    r_static_y_1 = generate_reference_trajectory(
        steps = plant.hyperparam_config["validation_trajectories"]["seq_len"],
        dt = plant.hyperparam_config["training_data_cfg"]["dt"],
        device = plant.hyperparam_config["train"]["device"],
        constant_val = plant.hyperparam_config["validation_trajectories"]["set_point"],
        amplitude = plant.hyperparam_config["validation_trajectories"]["amplitude"],
        period = plant.hyperparam_config["validation_trajectories"]["period"],
        mode = "dynamic"
        )

    r_smooth_decay = generate_exponential_decay_trajectory(
        steps = plant.hyperparam_config["validation_trajectories"]["seq_len"],
        dt = plant.hyperparam_config["training_data_cfg"]["dt"],
        y_start = plant.hyperparam_config["validation_trajectories"]["y_start"],       
        y_target = plant.hyperparam_config["validation_trajectories"]["y_target"],    
        tau = plant.hyperparam_config["validation_trajectories"]["tau"],           
    )

    # Load
    model_dict = {
        "MambaInverseController": {
            "model": load_model(MambaInverseController,"src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/MambaInverseController/fold_1/2026-09-14_13-58-01_best_fold_model.pt"),
            "x_scaler" : load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/MambaInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_x.pkl"),
            "y_scaler": load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/MambaInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_y.pkl")
        },
        "LSTMInverseController": {
            "model": load_model(LSTMInverseController,"src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/LSTMInverseController/fold_1/2026-09-14_13-58-01_best_fold_model.pt"),
            "x_scaler" : load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/LSTMInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_x.pkl"),
            "y_scaler": load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/LSTMInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_y.pkl")
        },
        "TransformerInverseController": {
            "model": load_model(TransformerInverseController,"src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/TransformerInverseController/fold_1/2026-09-14_13-58-01_best_fold_model.pt"),
            "x_scaler" : load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/TransformerInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_x.pkl"),
            "y_scaler": load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/TransformerInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_y.pkl")
        },
        "ESNInverseController": {
            "model": load_model_esn(ESNInverseController, "src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/ESNInverseController/fold_1/2026-09-14_13-58-01_best_fold_model.pkl"),
            "x_scaler" : load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/ESNInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_x.pkl"),
            "y_scaler": load_scaler("src/seq_control/results/2026-09-14/2026-09-14_13-58-01/ChemostatPlant_ic/ESNInverseController/fold_1/scalers/2026-09-14_13-58-01_scaler_y.pkl")
        }
    }

    # validate_multiple_controllers(
    #     models_dict=model_dict,
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

    validate_controller_ext_ref_multi(
        models_dict=model_dict,
        plant=plant,
        y_ref=r_static_y_1,
        hyperparam_config=plant.hyperparam_config,
        dirname="./plots_ref_multi",
        start_idx=2,
        u_ref=None,
        mode="closed_loop",
        show_plots=False
    )

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