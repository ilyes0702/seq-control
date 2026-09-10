"""Simulation script validation of inverse control models.

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
from seq_control.classes.sequence_models.MambaInverseController import *
from seq_control.classes.sequence_models.ESNInverseController import *
from seq_control.classes.sequence_models.LSTMInverseController import *
from seq_control.classes.sequence_models.TransformerInverseController import *



def main():
    # Initialize the plant model
    import torch
    
    hyperparam_config = hyperparam_config_TrophophasePlant   
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)
    dirname = plant.__class__.__name__

    # Load validation data
    val_data_path = (
        "src/seq_control/results/2026-09-07/2026-09-07_10-11-58/TrophophasePlant/io/dataset/2026-09-07_10-11-58_val_io_data.pt"
    )
    val_data = torch.load(val_data_path, weights_only=True)
    print("Keys of val_data: ", val_data.keys)

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
        y_target=0.015,      # Smoothly descends and turns to this constant value
        tau=0.1,            # Governs speed (lower = faster drop)
        device="cuda"
    )

    # Load trained model and scalers

    model_list = [
        MambaInverseController,
        LSTMInverseController,
        TransformerInverseController
    ]

    
    model = load_model(MambaInverseController, "src/seq_control/results/2026-09-07/2026-09-07_14-35-34/TrophophasePlant_training/fold_1/2026-09-07_14-35-34_best_fold_model.pt")

    model = load_model(LSTMInverseController, "src/seq_control/results/2026-09-10/2026-09-10_16-49-12/dirname_ic/LSTMInverseController/fold_1/2026-09-10_16-49-12_best_fold_model.pt")

    model = load_model(TransformerInverseController, "src/seq_control/results/2026-09-10/2026-09-10_16-49-12/dirname_ic/TransformerInverseController/fold_1/2026-09-10_16-49-12_best_fold_model.pt")

    

    exit()
    x_scaler = load_scaler("src/seq_control/results/2026-09-07/2026-09-07_14-35-34/TrophophasePlant_training/fold_1/scalers/2026-09-07_14-35-34_scaler_x.pkl")

    y_scaler = load_scaler("src/seq_control/results/2026-09-07/2026-09-07_14-35-34/TrophophasePlant_training/fold_1/scalers/2026-09-07_14-35-34_scaler_y.pkl")

    validate_controller(
        model = model,
        plant = plant,
        dataset_io= val_data,
        scaler_x = x_scaler,
        scaler_y = y_scaler,
        hyperparam_config=hyperparam_config,
        dirname = dirname,
        start_idx=2,
        mode="closed_loop",
        show_plots = True
    )

    validate_controller_ext_ref(
        model,
        plant,
        r_smooth_decay,
        x_scaler,
        y_scaler,
        hyperparam_config,
        dirname,
        start_idx=2,
        mode="closed_loop"
    )

    import torch

    # 1. Setup device and extract lookback settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_y = hyperparam_config["training_data_cfg"]["n_y"]
    n_u = hyperparam_config["training_data_cfg"]["n_u"]

    # 2. Ensure your surrogate model is in eval mode on the correct device
    model = load_model(MambaSurrogateModel, "src/seq_control/results/2026-09-07/2026-09-07_20-54-33/TrophophasePlant_sysid/fold_1/2026-09-07_20-54-33_best_fold_model.pt")
    x_scaler = load_scaler("src/seq_control/results/2026-09-07/2026-09-07_20-54-33/TrophophasePlant_sysid/fold_1/scalers/2026-09-07_20-54-33_scaler_x.pkl")
    y_scaler = load_scaler("src/seq_control/results/2026-09-07/2026-09-07_20-54-33/TrophophasePlant_sysid/fold_1/scalers/2026-09-07_20-54-33_scaler_y.pkl")
    # model.load_state_dict(torch.load("mamba_surrogate.pt")) # Load trained weights if applicable
    model.eval()

    # 3. Instantiate MambaGradientMPC
    mpc = MambaGradientMPC(
        surrogate_model=model,
        scaler_x=x_scaler,       # StandardScaler fitted on input features (X_raw / v_k)
        scaler_y=y_scaler,       # StandardScaler fitted on targets (Y_raw / y_k)
        n_y=n_y,                 # Past output lookback count
        n_u=n_u,                 # Past control action lookback count
        horizon=15,              # Lookahead prediction steps (H)
        num_iters=20,            # Gradient optimization steps per timestep
        lr=0.03,                 # Adam learning rate for control trajectory
        u_min=-1.0,              # Actuator lower bound (or None if unconstrained)
        u_max=1.0,               # Actuator upper bound (or None if unconstrained)
        weight_y=15.0,           # Tracking error loss weight
        weight_u=0.01,           # Control action magnitude penalty
        weight_du=0.2,           # Slew rate / change in control action penalty
        device=device
    )

    # Execute evaluation and render stacked plots
    # img, metrics = evaluate_and_plot_mpc(
    #     mpc_controller=mpc,
    #     plant=plant,
    #     dataset_dict=val_data,
    #     Y_trajectories=val_data["y"],
    #     U_trajectories=val_data["u"],
    #     dt=0.01,
    #     trace_idx=0,           # Evaluate on first trace
    #     dirname="mpc_results"  # Output directory for saved plots
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