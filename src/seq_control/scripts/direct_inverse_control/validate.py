"""Simulation script for running a trained Mamba inverse controller on the
Chemostat plant model.

This script initializes the plant, loads a trained controller and scalers,
generates reference trajectories (dynamic and constant), and runs the
simulation routine.
"""


# Import utility functions
from seq_control.utils.validation_utils import *
from seq_control.utils.general_utils import *
from seq_control.hyperparam_config import *

# Import plants
from seq_control.classes.plants.IdiophasePlant import *
from seq_control.classes.plants.TrophophasePlant import *
from seq_control.classes.plants.ChemostatPlant import *

# Import controller models
from seq_control.classes.controllers.MambaInverseController import *
from seq_control.classes.controllers.ESNInverseController import *
from seq_control.classes.controllers.LSTMInverseController import *
from seq_control.classes.controllers.TransformerInverseController import *

def main():
    # Initialize the plant model
    dirname = "results/run_1"
    hyperparam_config = hyperparam_config_TrophophasePlant   
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)


    val_data_path = (
        "src/seq_control/results/2026-09-07/2026-09-07_10-11-58/TrophophasePlant/io/dataset/2026-09-07_10-11-58_val_io_data.pt"
    )
    val_data = torch.load(val_data_path, weights_only=True)

    r_static_y_1 = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=0.015,
    )

    r_smooth_decay = generate_exponential_decay_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        y_start=0.12,       # Starts here
        y_target=0.02,      # Smoothly descends and turns to this constant value
        tau=0.1,            # Governs speed (lower = faster drop)
        device="cuda"
    )


    
    model = load_model(MambaInverseController, "src/seq_control/results/2026-09-07/2026-09-07_14-35-34/TrophophasePlant_training/fold_1/2026-09-07_14-35-34_best_fold_model.pt")

    x_scaler = load_scaler("src/seq_control/results/2026-09-07/2026-09-07_14-35-34/TrophophasePlant_training/fold_1/scalers/2026-09-07_14-35-34_scaler_x.pkl")

    y_scaler = load_scaler("src/seq_control/results/2026-09-07/2026-09-07_14-35-34/TrophophasePlant_training/fold_1/scalers/2026-09-07_14-35-34_scaler_y.pkl")

    print(f"x_scaler features: {getattr(x_scaler, 'n_features_in_', None)}")
    print(f"y_scaler features: {getattr(y_scaler, 'n_features_in_', None)}")

    validate_controller(
        model = model,
        plant = plant,
        dataset_io= val_data,
        scaler_x = x_scaler,
        scaler_y = y_scaler,
        hyperparam_config=hyperparam_config,
        dirname = dirname,
        start_idx=2,
        mode="open_loop",
        show_plots = True
    )

    # simulate_tracking_stateful(
    #     model=model,
    #     plant=plant,
    #     val_data=val_data,
    #     hyperparam_config=hyperparam_config,
    #     x_scaler=x_scaler,
    #     y_scaler=y_scaler,
    #     dirname=dirname
    # )

    # simulate_tracking_stateful_external_ref_trajectory(
    #     model,
    #     plant,
    #     [r_static_y_1.squeeze()],  # List of reference trajectories, one for each output dimension. Shape: [steps] for each trajectory.
    #     hyperparam_config,
    #     x_scaler,
    #     y_scaler,
    #     dirname,
    #     plot_individual_plots=False
    # )

    # comparison_summary = simulate_tracking_stateful_multi_model(
    #     models_dict=models_dict_1,
    #     plant=ChemostatPlant,
    #     r_trajectories=[r_static_y_1.squeeze()],
    #     hyperparam_config=hyperparam_config,
    #     dirname="results/model_benchmark_comparison"
    # )

if __name__ == "__main__":
    main()   