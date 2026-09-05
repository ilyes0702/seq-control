"""
Training Utility Functions
==========================

This module contains utility functions for the training of inverse controllers. 
"""

# Import standard libraries
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from seq_control.config import *
from seq_control.decorators.general_decorators import *
from seq_control.utils.loss_utils import *
from seq_control.utils.plotting_utils import *
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.general_utils import *

plt.style.use("src/seq_control/sample/style.mplstyle")

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#=== FUNCTION TO TRAIN A CONTROLLER IN OPEN LOOP MODE ===#
@track_resources
def train_controller_sanem(
    model,
    plant,
    Y_trajectories,
    U_trajectories,
    hyperparam_config,
    dirname,
    show_plots=False
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    
    device = train_cfg["device"]
    
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]


    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]

    # --- 1. GENERATE SLIDING WINDOW DATASET ---
    print(f"🔄 Slicing trajectories with sliding windows (n_y={n_y}, n_u={n_u})...")
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u
    )
    print("X_raw", X_raw.shape)
    print("Y_raw", Y_raw.shape)

    total_sequences = X_raw.shape[0]
    sliding_seq_len = X_raw.shape[1]
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # --- K-FOLD CROSS VALIDATION LOOP ---
    # Loop across folds. In case of 5-fold cross validation, the code in this for loop will run 5 times for different partitions of the data into training and validation data.
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction='None')
        elif loss_name == "MSELoss":
            criterion = torch.nn.MSELoss(reduction='none')

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")

        N_train, seq_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        train_x = torch.tensor(scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x), dtype=torch.float32)
        train_y = torch.tensor(scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y), dtype=torch.float32)

        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)

        val_x = torch.tensor(scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x), dtype=torch.float32)
        val_y = torch.tensor(scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y), dtype=torch.float32)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        if show_plots:
            curves_dir = f"{fold_dir}/transformed_data_curves"
            sample_x = train_x[0].numpy()
            sample_y = train_y[0].numpy()
            t_axis = np.arange(seq_len) * dt
            for out_idx in range(output_dim):
                plot_signals(t=t_axis, signals=[sample_y[:, out_idx]], labels=[f"Scaled u_{out_idx+1}"],
                             xlabel="Time (s)", ylabel="Standardized Units",
                             title=f"Fold {fold+1} | Transformed Input u_{out_idx+1}",
                             dirname=curves_dir, filename=f"scaled_u{out_idx+1}_curve")

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_train_channel_batch_loss = {ch: [] for ch in range(output_dim)}
        fold_train_channel_epoch_history = {ch: [] for ch in range(output_dim)}
        fold_val_channel_epoch_history = {ch: [] for ch in range(output_dim)}
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            epoch_train_channel_accum = {ch: 0.0 for ch in range(output_dim)}

            print(f"\n🎬 Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")
            shuffled_train_indices = torch.randperm(train_size)

            for i in range(0, train_size, mini_batch_size):
                batch_indices = shuffled_train_indices[i : i + mini_batch_size]
                current_batch_size = len(batch_indices)

                batch_x = train_x[batch_indices].to(device)
                batch_y = train_y[batch_indices].to(device)

                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                optimizer.zero_grad()
                u_pred_batch = model(batch_x)
                raw_loss = criterion(u_pred_batch, batch_y)
                # Reduce the loss using sum or mean
                loss = raw_loss.mean()

                loss.backward()
                optimizer.step()

                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size
                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)

                for ch in range(input_dim):
                    ch_loss_val = raw_loss[:, :, ch].mean().item()
                    epoch_train_channel_accum[ch] += ch_loss_val * current_batch_size
                    fold_train_channel_batch_loss[ch].append(ch_loss_val)

                global_batch_counter += 1

            scheduler.step()

            # --- 3. VALIDATION PASS ---
            model.eval()
            epoch_val_loss_accum = 0.0
            epoch_val_channel_accum = {ch: 0.0 for ch in range(output_dim)}

            all_val_preds = []
            all_val_trues = []

            with torch.no_grad():
                for i in range(0, val_size, mini_batch_size):
                    batch_val_x = val_x[i : i + mini_batch_size].to(device)
                    batch_val_y = val_y[i : i + mini_batch_size].to(device)
                    current_val_batch_size = len(batch_val_x)

                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    u_val_pred = model(batch_val_x)
                    raw_val_loss = criterion(u_val_pred, batch_val_y)

                    val_loss = raw_val_loss.mean()
                    epoch_val_loss_accum += val_loss.item() * current_val_batch_size

                    for ch in range(input_dim):
                        ch_val_loss_val = raw_val_loss[:, :, ch].mean().item()
                        epoch_val_channel_accum[ch] += ch_val_loss_val * current_val_batch_size

                    all_val_preds.append(u_val_pred.cpu().numpy())
                    all_val_trues.append(batch_val_y.cpu().numpy())

            val_all_preds_arr = np.concatenate(all_val_preds, axis=0)
            val_all_trues_arr = np.concatenate(all_val_trues, axis=0)

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0

            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {
                    "train_loss": [], "val_loss": [], "val_epochs": [],
                    **{f"train_loss_ch{ch+1}": [] for ch in range(output_dim)},
                    **{f"val_loss_ch{ch+1}": [] for ch in range(output_dim)}
                }

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            for ch in range(output_dim):
                mean_train_ch = epoch_train_channel_accum[ch] / train_size
                mean_val_ch = (epoch_val_channel_accum[ch] / val_size) if val_size > 0 else 0.0

                fold_train_channel_epoch_history[ch].append(mean_train_ch)
                fold_val_channel_epoch_history[ch].append(mean_val_ch)

                fold_histories[fold][f"train_loss_ch{ch+1}"].append(mean_train_ch)
                fold_histories[fold][f"val_loss_ch{ch+1}"].append(mean_val_ch)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Total Train Loss: {mean_train_loss:.6f} | Total Val Loss: {mean_val_loss:.6f}")

            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0
                save_model(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    early_stopped = True
                    break
        
        # --- 4. PLOT EXTENDED LOSS CURVES ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"
        epoch_axis = np.array(fold_histories[fold]["val_epochs"])

        plot_signals(t=np.array(fold_train_batch_indices),
                     signals=[np.array(fold_train_batch_loss)],
                     labels=[f"Fold {fold+1} Total Loss"],
                     xlabel="Optimization Steps",
                     ylabel="Loss",
                     dirname=fold_dir,
                     filename="granular_training_loss",
                     asp=0.3)

        plot_signals(t=epoch_axis,
                     signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
                     labels=["Avg Train Loss", "Avg Val Loss"],
                     xlabel="Epochs",
                     ylabel="Loss",
                     dirname=fold_dir,
                     filename="epoch_validation_loss",
                     asp=0.3)

        # --- PLOT ALL VALIDATION PREDICTIONS FOR THIS FOLD ---
        print(f"📈 Plotting all validation predictions for Fold {fold + 1}...")
        pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
        t_axis_val = np.arange(seq_len) * dt

        # --- EXTRACT METADATA FROM PLANT ---
        plot_cfg = plant.get_plot_config()

        # Search for control ('u') and time ('t') configurations
        u_cfg = next((cfg for cfg in plot_cfg if "u" in cfg["cols"]), None)
        t_cfg = next((cfg for cfg in plot_cfg if "t" in cfg["cols"]), None)

        # Dynamic x-axis label (fallback to default string if not found)
        xlabel_str = t_cfg["xlabel"] if t_cfg else "Time [h]"
        if isinstance(xlabel_str, list):
            xlabel_str = xlabel_str[0]

        # --- PLOT LOOP ---
        for seq_idx in range(len(val_all_preds_arr)):
            seq_pred_scaled = val_all_preds_arr[seq_idx]
            seq_true_scaled = val_all_trues_arr[seq_idx]

            seq_pred_unscaled = scaler_y.inverse_transform(seq_pred_scaled)
            seq_true_unscaled = scaler_y.inverse_transform(seq_true_scaled)

            for ch in range(input_dim):
                # Extract y-axis and curve labels dynamically per channel
                if u_cfg:
                    ylabel_str = u_cfg["ylabel"][ch] if isinstance(u_cfg["ylabel"], list) else u_cfg["ylabel"]
                    base_label = u_cfg["labels"][ch] if ch < len(u_cfg["labels"]) else f"u_{ch+1}"
                else:
                    ylabel_str = f"Control Unit u_{ch+1}"
                    base_label = f"u_{ch+1}"

                plot_signals(
                    t=t_axis_val,
                    signals=[seq_true_unscaled[:, ch], seq_pred_unscaled[:, ch]],
                    labels=[f"True {base_label}", f"Predicted {base_label}"],
                    xlabel=xlabel_str,
                    ylabel=ylabel_str,
                    dirname=pred_curves_dir,
                    filename=f"val_prediction_seq{seq_idx+1}_u{ch+1}",
                    asp=0.3
                )

    # --- METRIC AGGREGATION ACROSS FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))
    mean_cv_loss = avg_best_val_loss  # Open-loop mean CV loss

    # --- DATAFRAME CREATION & PRINTING ---
    summary_metrics = {
        "Metric": [
            "Avg Best Open-Loop Train Loss",
            "Avg Best Open-Loop Val Loss (Mean CV)",
        ],
        "Value": [
            avg_best_train_loss,
            mean_cv_loss,
        ],
    }

    summary_results_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 K-FOLD CROSS-VALIDATION SUMMARY")
    print("==========================================")
    print(summary_results_df.to_string(index=False))
    print("==========================================\n")

    return (
        fold_histories,
        {
            "train_loss": avg_best_train_loss,
            "val_loss": avg_best_val_loss,
        },
        mean_cv_loss,
        summary_results_df,
    )




#=== FUNCTION FOR TRAINING INVERSE CONTROLLER IN OPEN LOOP MODE ===#
@track_resources
def train_controller_open_loop(
    model,
    train_data,
    test_data,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation=True,
    run_sim_with_plots=False,
):
    """Train an inverse neural network controller in open-loop mode.

    This function prepares trajectory data using sliding windows, scales the inputs
    and targets using standard scalers, and trains the controller model. It handles
    the training loop, tracks evaluation metrics on test data, implements early
    stopping based on test loss, saves the best model/scalers, and generates
    training loss plots.

    :param model: The PyTorch neural network model representing the inverse controller.
    :type model: torch.nn.Module
    :param train_data: A tuple/list containing raw training trajectory arrays ``(Y_train, U_train)``.
    :type train_data: tuple[np.ndarray, np.ndarray]
    :param test_data: A tuple/list containing raw testing trajectory arrays ``(Y_test, U_test)``.
    :type test_data: tuple[np.ndarray, np.ndarray]
    :param hyperparam_config: Configuration dictionary containing training, data, and plant parameters.
        Must include keys ``'train'``, ``'training_data_cfg'``, and ``'plant'``.
    :type hyperparam_config: dict
    :param plant: The physical plant or dynamic system object being controlled.
    :type plant: object
    :param dirname: Path to the directory where models, scalers, configs, and plots will be saved.
    :type dirname: str
    :param show_plots: Whether to interactively display generated loss plots, defaults to False.
    :type show_plots: bool, optional
    :param run_simulation: Flag indicating whether open-loop simulation should run post-training, defaults to True.
    :type run_simulation: bool, optional
    :param run_sim_with_plots: Flag indicating whether simulation results should be plotted, defaults to False.
    :type run_sim_with_plots: bool, optional

    :returns: A tuple containing:
        - **model** (*torch.nn.Module*): The trained model (loaded with state of the best epoch).
        - **history** (*dict*): Dictionary with key-value pairs for ``'train_loss'`` and ``'test_loss'`` lists across epochs.
        - **summary_df** (*pd.DataFrame*): Summary DataFrame containing the best training and testing loss values.
    :rtype: tuple[torch.nn.Module, dict, pandas.DataFrame]

    .. note::
        The function automatically fits ``StandardScaler`` instances on the training split only and transforms 
        both training and test data accordingly to prevent data leakage.
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    patience = train_cfg.get("val_patience_epochs", 10)
    min_delta = train_cfg.get("val_min_delta", 1e-4)

    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]

    # Helper function to extract or slice dataset arrays
    def _prepare_split(split_data, split_name):
        arg1, arg2 = split_data
        
        print(
            f"🔄 Slicing {split_name} trajectories with sliding windows (n_y={n_y}, n_u={n_u})..."
            )
        return create_inverse_controller_dataset(
            Y_trajectories=arg1, U_trajectories=arg2, n_y=n_y, n_u=n_u
        )
            
    # --- 1. PREPARE SPLITS ---
    train_x_raw, train_y_raw = _prepare_split(train_data, "train")
    test_x_raw, test_y_raw = _prepare_split(test_data, "test")

    print(
        f"📊 Dataset shapes -> Train: {train_x_raw.shape} | Test: {test_x_raw.shape}"
    )

    # --- 2. FIT SCALERS ON TRAIN DATA ONLY ---
    print("⚖️ Fitting StandardScalers on Training Data...")
    N_train, seq_len_train, dim_x = train_x_raw.shape
    dim_y = train_y_raw.shape[-1]

    train_x_flat = train_x_raw.reshape(-1, dim_x)
    train_y_flat = train_y_raw.reshape(-1, dim_y)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_x_flat)
    scaler_y.fit(train_y_flat)

    # Transform Train
    train_x = torch.tensor(
        scaler_x.transform(train_x_flat).reshape(N_train, seq_len_train, dim_x),
        dtype=torch.float32,
    )
    train_y = torch.tensor(
        scaler_y.transform(train_y_flat).reshape(N_train, seq_len_train, dim_y),
        dtype=torch.float32,
    )

    # Transform Test
    N_test, seq_len_test, _ = test_x_raw.shape
    test_x_flat = test_x_raw.reshape(-1, dim_x)
    test_y_flat = test_y_raw.reshape(-1, dim_y)
    test_x = torch.tensor(
        scaler_x.transform(test_x_flat).reshape(N_test, seq_len_test, dim_x),
        dtype=torch.float32,
    )
    test_y = torch.tensor(
        scaler_y.transform(test_y_flat).reshape(N_test, seq_len_test, dim_y),
        dtype=torch.float32,
    )

    # Save Config and Scalers
    save_to_json(hyperparam_config, dirname, "hyperparam_config")
    save_scaler_object(scaler_x, dirname=dirname, filename="scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="scaler_y")

    # --- SETUP MODEL, OPTIMIZER, & LOSS ---
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=train_cfg["lr_decay_rate"]
    )

    loss_name = train_cfg["loss_function"].replace("()", "")
    if loss_name == "NormalizedRMSELoss":
        criterion = NormalizedRMSELoss(reduction="none")
    elif loss_name == "MSELoss":
        criterion = torch.nn.MSELoss(reduction="none")

    train_epoch_history = []
    test_epoch_history = []
    train_batch_loss_history = []

    best_test_loss = float("inf")
    patience_counter = 0

    # --- 3. TRAINING LOOP (TESTING PERFORMANCE EVERY EPOCH) ---
    for epoch in range(epochs):
        model.train()
        epoch_train_loss_accum = 0.0

        print(f"\n🎬 Starting Epoch {epoch+1}/{epochs}")
        shuffled_train_indices = torch.randperm(N_train)

        # Batch Training Pass
        for i in range(0, N_train, mini_batch_size):
            batch_indices = shuffled_train_indices[i : i + mini_batch_size]
            current_batch_size = len(batch_indices)

            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)

            if hasattr(model, "reset_memory"):
                model.reset_memory(
                    batch_size=current_batch_size, device=device
                )

            optimizer.zero_grad()
            u_pred_batch = model(batch_x)
            raw_loss = criterion(u_pred_batch, batch_y)
            loss = raw_loss.mean()

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            epoch_train_loss_accum += current_loss_val * current_batch_size
            train_batch_loss_history.append(current_loss_val)

        scheduler.step()
        mean_train_loss = epoch_train_loss_accum / N_train
        train_epoch_history.append(mean_train_loss)

        # --- TEST PASS EVERY EPOCH ---
        model.eval()
        epoch_test_loss_accum = 0.0

        with torch.no_grad():
            for i in range(0, N_test, mini_batch_size):
                batch_test_x = test_x[i : i + mini_batch_size].to(device)
                batch_test_y = test_y[i : i + mini_batch_size].to(device)
                current_test_batch_size = len(batch_test_x)

                if hasattr(model, "reset_memory"):
                    model.reset_memory(
                        batch_size=current_test_batch_size, device=device
                    )

                u_test_pred = model(batch_test_x)
                raw_test_loss = criterion(u_test_pred, batch_test_y)
                epoch_test_loss_accum += (
                    raw_test_loss.mean().item() * current_test_batch_size
                )

        mean_test_loss = (
            epoch_test_loss_accum / N_test if N_test > 0 else 0.0
        )
        test_epoch_history.append(mean_test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"✨ Epoch {epoch+1} Summary | LR: {current_lr:.6e} | Train Loss: {mean_train_loss:.6f} | Test Loss: {mean_test_loss:.6f}"
        )

        # Checkpoint / Early Stopping on Test Set Performance
        if mean_test_loss < (best_test_loss - min_delta):
            best_test_loss = mean_test_loss
            patience_counter = 0
            save_model(
                model,
                dirname=dirname,
                hyperparam_config=hyperparam_config,
                filename="best_model",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}.")
                break

    # --- 4. LOSS PLOTS ---
    plot_signals(
        t=np.arange(len(train_batch_loss_history)),
        signals=[np.array(train_batch_loss_history)],
        labels=["Batch Training Loss"],
        xlabel="Optimization Steps",
        ylabel="Loss",
        dirname=dirname,
        filename="granular_training_loss",
    )

    plot_signals(
        t=np.arange(1, len(train_epoch_history) + 1),
        signals=[np.array(train_epoch_history), np.array(test_epoch_history)],
        labels=["Avg Train Loss", "Avg Test Loss"],
        xlabel="Epochs",
        ylabel="Loss",
        dirname=dirname,
        filename="epoch_loss_curves",
    )

    # --- SUMMARY METRICS ---
    summary_metrics = {
        "Metric": ["Best Training Loss", "Best Testing Loss"],
        "Value": [min(train_epoch_history), best_test_loss],
    }
    summary_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 TRAINING SUMMARY")
    print("==========================================")
    print(summary_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(summary_df, dirname=dirname, filename="training_summary")

    history = {
        "train_loss": train_epoch_history,
        "test_loss": test_epoch_history,
    }

    return model, history, summary_df




def train_controller_closed_loop_online(
    model,
    Y_trajectories,
    U_trajectories,
    X_states,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_sim_with_plots=False
):
    """
    Trains an inverse neural controller online timestep-by-timestep using closed-loop tracking loss.
    
    At each step k, the controller forecasts u_k, steps the plant to compute y_{k+1}, 
    evaluates step loss against y_ref, backpropagates gradients, and updates model weights immediately.
    """
    # --- 1. HYPERPARAMETER & CONFIG EXTRACTION ---
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]
    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]
    lookback_offset = train_cfg["lookback_offset"]

    input_dim = plant_cfg["input_dim"]    # Control signal dimension (u)
    output_dim = plant_cfg["output_dim"]  # Measured output dimension (y)
    
    u_min = torch.tensor(plant_cfg["u_1_hard_min"], device=device, dtype=torch.float32)
    u_max = torch.tensor(plant_cfg["u_1_hard_max"], device=device, dtype=torch.float32)

    total_trajectories = Y_trajectories.shape[0]
    total_trajectory_len = Y_trajectories.shape[1]

    # Feature dimension formula matching model input_size: (1 target + 1 current + n_y past)*dim_y + n_u*dim_u
    feature_dim = (2 + n_y) * output_dim + n_u * input_dim

    # --- 2. K-FOLD CV SPLIT ---
    all_indices = np.arange(total_trajectories)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING CLOSED-LOOP ONLINE FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction='none')
        else:
            criterion = torch.nn.MSELoss(reduction='none')

        # Instantiate Plant for this fold
        plant_instance = plant(hyperparam_config) if isinstance(plant, type) else plant

        # --- 3. FIT STANDARD SCALERS ON TRAINING SPLIT ---
        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        Y_train_flat = Y_trajectories[train_idx_arr].reshape(-1, output_dim)
        U_train_flat = U_trajectories[train_idx_arr].reshape(-1, input_dim)

        scaler_y = StandardScaler().fit(Y_train_flat)
        scaler_u = StandardScaler().fit(U_train_flat)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")
        save_scaler_object(scaler_u, dirname=fold_dir, filename="scaler_u")

        # Scaler PyTorch tensors for differentiable graph operations
        scaler_y_mean_t = torch.tensor(scaler_y.mean_, device=device, dtype=torch.float32)
        scaler_y_std_t = torch.tensor(scaler_y.scale_, device=device, dtype=torch.float32)
        scaler_u_mean_t = torch.tensor(scaler_u.mean_, device=device, dtype=torch.float32)
        scaler_u_std_t = torch.tensor(scaler_u.scale_, device=device, dtype=torch.float32)

        # Tile scaler tensors for concatenated feature vector (1 target + 1 current + n_y past y's)
        y_next_y_hist_mean = scaler_y_mean_t.repeat(2 + n_y)
        y_next_y_hist_std = scaler_y_std_t.repeat(2 + n_y)

        u_hist_mean = scaler_u_mean_t.repeat(n_u)
        u_hist_std = scaler_u_std_t.repeat(n_u)

        best_val_loss = float('inf')
        patience_counter = 0

def train_controller_closed_loop_online(
    model,
    Y_trajectories,
    U_trajectories,
    X_states,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_sim_with_plots=False,
):
    """Train an inverse controller online in a closed-loop setting using K-Fold cross-validation.

    This function performs online gradient updates at every simulation time step by 
    unrolling closed-loop trajectories through the plant model. Parameters are updated 
    step-by-step using instant output loss. The model is trained across K-folds with 
    independent feature and target standard scalers fitted on each training fold.

    :param model: The PyTorch neural network model representing the inverse controller.
    :type model: torch.nn.Module
    :param Y_trajectories: Tensor containing output target/reference trajectories of shape 
        ``(n_samples, sequence_length, output_dim)``.
    :type Y_trajectories: torch.Tensor
    :param U_trajectories: Tensor containing control input trajectories of shape 
        ``(n_samples, sequence_length, input_dim)``.
    :type U_trajectories: torch.Tensor
    :param X_states: Tensor containing plant state trajectories of shape 
        ``(n_samples, sequence_length, state_dim)``.
    :type X_states: torch.Tensor
    :param hyperparam_config: Configuration dictionary containing training, data, and plant hyperparameters.
        Must include keys ``'train'``, ``'training_data_cfg'``, and ``'plant'``.
    :type hyperparam_config: dict
    :param plant: The physical plant class or instance used to compute dynamic steps during trajectory unrolling.
    :type plant: type | object
    :param dirname: Directory path where fold checkpoints, scalers, configs, and loss plots will be saved.
    :type dirname: str
    :param show_plots: Whether to interactively display generated loss plots, defaults to False.
    :type show_plots: bool, optional
    :param run_sim_with_plots: Flag indicating whether closed-loop simulation figures should be saved/shown, defaults to False.
    :type run_sim_with_plots: bool, optional

    :returns: A tuple containing four elements:
        - **fold_histories** (*dict*): Dictionary mapping fold indices to training and validation loss histories per epoch.
        - **avg_metrics** (*dict*): Dictionary summarizing aggregated best train loss, validation loss, and closed-loop MSE across folds.
        - **avg_best_val_loss** (*float*): The average best validation loss across all K-folds.
        - **summary_results_df** (*pd.DataFrame*): Summary DataFrame containing averaged performance metrics across folds.
    :rtype: tuple[dict, dict, float, pandas.DataFrame]

    .. note::
        - Gradient computation flows back through plant dynamics step-by-step: 
          ``Loss(y_sim_{k+1}, y_ref_{k+1}) -> Plant Dynamics -> Scaler_y -> Controller -> Scaler_x``.
        - To prevent exploding gradients during online unrolling, state and action tensors 
          are detached at each step after applying the gradient update (truncated BPTT at step :math:`k`).
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]

    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]
    lookback_offset = train_cfg.get("lookback_offset", max(n_y, n_u))

    input_dim = plant_cfg["input_dim"]    # Control u dimension
    output_dim = plant_cfg["output_dim"]  # Output y dimension

    # --- 1. GENERATE SLIDING WINDOW DATASET ---
    print(f"🔄 Slicing trajectories with sliding windows (n_y={n_y}, n_u={n_u})...")
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u,
    )
    print("X_raw shape:", X_raw.shape)
    print("Y_raw shape:", Y_raw.shape)

    total_sequences = X_raw.shape[0]
    total_trajectory_len = Y_trajectories.shape[1]
    
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # Initialize plant instance
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING ONLINE CLOSED-LOOP FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=train_cfg.get("lr_decay_rate", 0.99)
        )

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction="none")
        else:
            criterion = torch.nn.MSELoss(reduction="none")

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")

        dim_x = train_x_raw.shape[-1]
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        mean_x_t = torch.tensor(scaler_x.mean_, dtype=torch.float32, device=device)
        scale_x_t = torch.tensor(scaler_x.scale_, dtype=torch.float32, device=device)
        mean_y_t = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        scale_y_t = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)

        train_size = len(train_idx_arr)
        val_size = len(val_idx_arr)

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            shuffled_train_indices = np.random.permutation(train_size)

            print(f"\n🎬 Online Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")

            for i in range(0, train_size, mini_batch_size):
                batch_train_seq_idx = train_idx_arr[
                    shuffled_train_indices[i : i + mini_batch_size]
                ]
                current_batch_size = len(batch_train_seq_idx)

                target_y_batch = Y_trajectories[batch_train_seq_idx].to(device)
                target_u_batch = U_trajectories[batch_train_seq_idx].to(device)
                init_states_batch = X_states[batch_train_seq_idx, 0].to(device)

                sim_outputs = torch.zeros(
                    current_batch_size, total_trajectory_len, output_dim, device=device
                )
                sim_controls = torch.zeros(
                    current_batch_size, total_trajectory_len, input_dim, device=device
                )

                sim_outputs[:, 0] = target_y_batch[:, 0]
                sim_controls[:, 0] = target_u_batch[:, 0]
                current_sim_state = init_states_batch.clone()

                if hasattr(model, "reset_memory"):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                batch_accumulated_seq_loss = 0.0
                active_steps_count = 0

                # --- ONLINE CLOSED-LOOP UNROLLING & STEP-BY-STEP GRADIENT UPDATE ---
                for k in range(total_trajectory_len - 1):
                    if k < lookback_offset:
                        # Warm-up phase: ground-truth actions to build historical context
                        u_action = target_u_batch[:, k]
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action, t=k * dt, dt=dt
                        )
                        sim_controls[:, k + 1] = u_action
                        sim_outputs[:, k + 1] = y_next_sim
                    else:
                        optimizer.zero_grad()

                        # 1. Construct raw feature vector
                        y_next_ref = target_y_batch[:, k + 1]
                        
                        y_hist = sim_outputs[:, k - n_y : k + 1]
                        y_hist_rev = torch.flip(y_hist, dims=[1]).reshape(current_batch_size, -1)
                        
                        u_hist = sim_controls[:, k - n_u : k]
                        u_hist_rev = torch.flip(u_hist, dims=[1]).reshape(current_batch_size, -1)

                        v_k_raw = torch.cat([y_next_ref, y_hist_rev, u_hist_rev], dim=-1)

                        # 2. Normalize input features
                        v_k_scaled = (v_k_raw - mean_x_t) / scale_x_t
                        v_k_input = v_k_scaled.unsqueeze(1)

                        # 3. Forecast control u
                        u_pred_norm = model(v_k_input).squeeze(1)
                        u_pred_raw = u_pred_norm * scale_y_t + mean_y_t

                        # 4. Saturation limits
                        u_action = torch.clamp(
                            u_pred_raw,
                            plant_cfg["u_1_hard_min"],
                            plant_cfg["u_1_hard_max"],
                        )

                        # 5. Forward step through plant dynamics
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action, t=k * dt, dt=dt
                        )

                        # 6. Immediate single-step loss computation
                        step_loss_raw = criterion(y_next_sim, y_next_ref)
                        step_loss = step_loss_raw.mean()

                        # 7. ONLINE GRADIENT UPDATE STEP
                        step_loss.backward()
                        optimizer.step()

                        # 8. Accumulate loss for epoch statistics
                        batch_accumulated_seq_loss += step_loss.item()
                        active_steps_count += 1

                        # 9. Store and detach variables for truncated BPTT at step k
                        current_sim_state = current_sim_state.detach()
                        sim_controls[:, k + 1] = u_action.detach()
                        sim_outputs[:, k + 1] = y_next_sim.detach()

                avg_seq_loss = batch_accumulated_seq_loss / max(1, active_steps_count)
                epoch_train_loss_accum += avg_seq_loss * current_batch_size
                fold_train_batch_loss.append(avg_seq_loss)
                fold_train_batch_indices.append(global_batch_counter)
                global_batch_counter += 1

            scheduler.step()

            # --- VALIDATION PASS (INFERENCE / EVALUATION MODE) ---
            model.eval()
            epoch_val_loss_accum = 0.0

            with torch.no_grad():
                for i in range(0, val_size, mini_batch_size):
                    batch_val_seq_idx = val_idx_arr[i : i + mini_batch_size]
                    current_val_batch_size = len(batch_val_seq_idx)

                    target_y_val = Y_trajectories[batch_val_seq_idx].to(device)
                    target_u_val = U_trajectories[batch_val_seq_idx].to(device)
                    init_states_val = X_states[batch_val_seq_idx, 0].to(device)

                    sim_outputs_val = torch.zeros(
                        current_val_batch_size, total_trajectory_len, output_dim, device=device
                    )
                    sim_controls_val = torch.zeros(
                        current_val_batch_size, total_trajectory_len, input_dim, device=device
                    )

                    sim_outputs_val[:, 0] = target_y_val[:, 0]
                    sim_controls_val[:, 0] = target_u_val[:, 0]
                    current_sim_state_val = init_states_val.clone()

                    if hasattr(model, "reset_memory"):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    for k in range(total_trajectory_len - 1):
                        if k < lookback_offset:
                            u_action = target_u_val[:, k]
                            current_sim_state_val, y_next_sim = plant_instance.step(
                                state=current_sim_state_val, u=u_action, t=k * dt, dt=dt
                            )
                            sim_controls_val[:, k + 1] = u_action
                            sim_outputs_val[:, k + 1] = y_next_sim
                        else:
                            y_next_ref = target_y_val[:, k + 1]
                            y_hist = sim_outputs_val[:, k - n_y : k + 1]
                            y_hist_rev = torch.flip(y_hist, dims=[1]).reshape(current_val_batch_size, -1)
                            u_hist = sim_controls_val[:, k - n_u : k]
                            u_hist_rev = torch.flip(u_hist, dims=[1]).reshape(current_val_batch_size, -1)

                            v_k_raw = torch.cat([y_next_ref, y_hist_rev, u_hist_rev], dim=-1)

                            v_k_scaled = (v_k_raw - mean_x_t) / scale_x_t
                            v_k_input = v_k_scaled.unsqueeze(1)

                            u_pred_norm = model(v_k_input).squeeze(1)
                            u_pred_raw = u_pred_norm * scale_y_t + mean_y_t

                            u_action = torch.clamp(
                                u_pred_raw,
                                plant_cfg["u_1_hard_min"],
                                plant_cfg["u_1_hard_max"],
                            )

                            current_sim_state_val, y_next_sim = plant_instance.step(
                                state=current_sim_state_val, u=u_action, t=k * dt, dt=dt
                            )
                            sim_controls_val[:, k + 1] = u_action
                            sim_outputs_val[:, k + 1] = y_next_sim

                    val_raw_loss = criterion(
                        sim_outputs_val[:, lookback_offset:],
                        target_y_val[:, lookback_offset:],
                    )
                    epoch_val_loss_accum += val_raw_loss.mean().item() * current_val_batch_size

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0

            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {"train_loss": [], "val_loss": [], "val_epochs": []}

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Online Train Loss (MSE_y): {mean_train_loss:.6f} | Closed-Loop Val Loss: {mean_val_loss:.6f}")

            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0
                save_model(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    break

        # --- PLOT LOSS CURVES ---
        epoch_axis = np.array(fold_histories[fold]["val_epochs"])

        plot_signals(
            t=np.array(fold_train_batch_indices),
            signals=[np.array(fold_train_batch_loss)],
            labels=[f"Fold {fold+1} Total Loss"],
            xlabel="Optimization Steps",
            ylabel="Loss",
            dirname=fold_dir,
            filename="granular_training_loss"
        )

        plot_signals(
            t=epoch_axis,
            signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
            labels=["Avg Train Loss", "Avg Val Loss"],
            xlabel="Epochs",
            ylabel="Loss",
            dirname=fold_dir,
            filename="epoch_validation_loss"
        )

    # --- AGGREGATE RESULTS ACROSS ALL FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))

    summary_metrics = {
        "Metric": [
            "Avg Best Online Closed-Loop Train Loss (MSE_y)",
            "Avg Best Closed-Loop Val Loss (MSE_y)",
        ],
        "Value": [
            avg_best_train_loss,
            avg_best_val_loss,
        ],
    }

    summary_results_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 ONLINE CLOSED-LOOP K-FOLD CROSS-VALIDATION SUMMARY")
    print("==========================================")
    print(summary_results_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(
        summary_results_df,
        dirname=dirname,
        filename="k_fold_cross_validation_summary"
    )

    return (
        fold_histories,
        {
            "train_loss": avg_best_train_loss,
            "val_loss": avg_best_val_loss,
            "closed_loop_mse": avg_best_val_loss,
        },
        avg_best_val_loss,
        summary_results_df,
    )
    



def train_controller_closed_loop_offline(
    model,
    train_data,
    test_data,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation=True,
    run_sim_with_plots=False,
):
    """Train an inverse neural network controller in offline closed-loop mode.

    This function simulates the interaction between the controller and the plant model 
    step-by-step over the full sequence duration. Unlike online training, loss is 
    accumulated over the full time sequence before triggering backpropagation to update 
    the controller parameters.

    :param model: The PyTorch neural network model representing the inverse controller.
    :type model: torch.nn.Module
    :param train_data: A tuple/list containing raw training trajectory arrays ``(Y_train, U_train)``.
    :type train_data: tuple[np.ndarray, np.ndarray]
    :param test_data: A tuple/list containing raw testing trajectory arrays ``(Y_test, U_test)``.
    :type test_data: tuple[np.ndarray, np.ndarray]
    :param hyperparam_config: Configuration dictionary containing training, data, and plant hyperparameters.
        Must include keys ``'train'``, ``'training_data_cfg'``, and ``'plant'``.
    :type hyperparam_config: dict
    :param plant: The physical plant or dynamic system object interacted with during sequence unrolling.
    :type plant: object
    :param dirname: Path to the directory where model checkpoints, scalers, configs, and plots will be saved.
    :type dirname: str
    :param show_plots: Whether to interactively display generated loss plots, defaults to False.
    :type show_plots: bool, optional
    :param run_simulation: Flag indicating whether closed-loop simulation should run post-training, defaults to True.
    :type run_simulation: bool, optional
    :param run_sim_with_plots: Flag indicating whether simulation results should be plotted, defaults to False.
    :type run_sim_with_plots: bool, optional

    :returns: A tuple containing:
        - **model** (*torch.nn.Module*): The trained model (loaded with the best performing checkpoint).
        - **history** (*dict*): Dictionary mapping keys ``'train_loss'`` and ``'test_loss'`` to lists of mean losses per epoch.
        - **summary_df** (*pd.DataFrame*): Summary DataFrame containing the overall best training and testing losses.
    :rtype: tuple[torch.nn.Module, dict, pandas.DataFrame]

    .. note::
        - Features and target data are scaled independently using ``StandardScaler`` instances fitted strictly on training data.
        - Backpropagation Through Time (BPTT) is performed across full sequence lengths via step-by-step unrolling inside ``_run_closed_loop_batch``.
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    patience = train_cfg.get("val_patience_epochs", 10)
    min_delta = train_cfg.get("val_min_delta", 1e-4)

    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]

    # Helper function to extract dataset split
    def _prepare_split(split_data, split_name):
        arg1, arg2 = split_data
        print(
            f"🔄 Slicing {split_name} trajectories with sliding windows"
            f" (n_y={n_y}, n_u={n_u})..."
        )
        return create_inverse_controller_dataset(
            Y_trajectories=arg1, U_trajectories=arg2, n_y=n_y, n_u=n_u
        )

    # --- 1. PREPARE SPLITS ---
    train_x_raw, train_y_raw = _prepare_split(train_data, "train")
    test_x_raw, test_y_raw = _prepare_split(test_data, "test")

    print(
        f"📊 Dataset shapes -> Train: {train_x_raw.shape} | Test:"
        f" {test_x_raw.shape}"
    )

    # --- 2. FIT SCALERS ON TRAIN DATA ONLY ---
    print("⚖️ Fitting StandardScalers on Training Data...")
    N_train, seq_len_train, dim_x = train_x_raw.shape
    dim_y = train_y_raw.shape[-1]

    train_x_flat = train_x_raw.reshape(-1, dim_x)
    train_y_flat = train_y_raw.reshape(-1, dim_y)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_x_flat)
    scaler_y.fit(train_y_flat)

    # Transform Train
    train_x = torch.tensor(
        scaler_x.transform(train_x_flat).reshape(
            N_train, seq_len_train, dim_x
        ),
        dtype=torch.float32,
    )
    train_y = torch.tensor(
        scaler_y.transform(train_y_flat).reshape(
            N_train, seq_len_train, dim_y
        ),
        dtype=torch.float32,
    )

    # Transform Test
    N_test, seq_len_test, _ = test_x_raw.shape
    test_x_flat = test_x_raw.reshape(-1, dim_x)
    test_y_flat = test_y_raw.reshape(-1, dim_y)
    test_x = torch.tensor(
        scaler_x.transform(test_x_flat).reshape(N_test, seq_len_test, dim_x),
        dtype=torch.float32,
    )
    test_y = torch.tensor(
        scaler_y.transform(test_y_flat).reshape(N_test, seq_len_test, dim_y),
        dtype=torch.float32,
    )

    # Save Config and Scalers
    save_to_json(hyperparam_config, dirname, "hyperparam_config")
    save_scaler_object(scaler_x, dirname=dirname, filename="scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="scaler_y")

    # --- SETUP MODEL, OPTIMIZER, & LOSS ---
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=train_cfg["lr_decay_rate"]
    )

    loss_name = train_cfg["loss_function"].replace("()", "")
    if loss_name == "NormalizedRMSELoss":
        criterion = NormalizedRMSELoss(reduction="none")
    elif loss_name == "MSELoss":
        criterion = torch.nn.MSELoss(reduction="none")

    train_epoch_history = []
    test_epoch_history = []
    train_batch_loss_history = []

    best_test_loss = float("inf")
    patience_counter = 0

    # --- HELPER FUNCTION FOR CLOSED-LOOP BATCH RUN ---
    def _run_closed_loop_batch(batch_x, batch_y_target, is_training=True):
        """Simule la boucle fermée pas à pas et calcule la perte sur la séquence."""
        batch_size, seq_len, _ = batch_x.shape

        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=batch_size, device=device)

        # Initialisation de l'état de la plante [batch_size, 2] -> [x1, x2]
        state = plant.get_initial_state(
            batch_size=batch_size, randomize=is_training
        )

        u_preds = []

        # Boucle temporelle étape par étape
        for t_step in range(seq_len):
            current_time = t_step * dt
            x_t = batch_x[
                :, t_step : t_step + 1, :
            ]  # Entrée au temps t: [batch_size, 1, dim_x]

            # 1. Le contrôleur prédit u_t: [batch_size, 1, dim_u]
            u_t = model(x_t)
            u_preds.append(u_t)

            # Formatage de u_t pour la plante: [batch_size, dim_u]
            u_t_step = u_t.squeeze(1) if u_t.dim() == 3 else u_t

            # 2. Intégration RK45 de la plante: step(state, u, t, dt)
            state, _ = plant.step(
                state=state, u=u_t_step, t=current_time, dt=dt
            )

        # Reconstitution de la trajectoire de commande prédite [batch_size, seq_len, dim_u]
        u_pred_seq = torch.cat(u_preds, dim=1)

        # Calcul de la perte globale sur la séquence
        raw_loss = criterion(u_pred_seq, batch_y_target)
        return raw_loss.mean()

    # --- 3. TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        epoch_train_loss_accum = 0.0

        print(f"\n🎬 Starting Epoch {epoch+1}/{epochs}")
        shuffled_train_indices = torch.randperm(N_train)

        # Batch Training Pass
        for i in range(0, N_train, mini_batch_size):
            batch_indices = shuffled_train_indices[i : i + mini_batch_size]
            current_batch_size = len(batch_indices)

            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)

            optimizer.zero_grad()

            loss = _run_closed_loop_batch(batch_x, batch_y, is_training=True)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            epoch_train_loss_accum += current_loss_val * current_batch_size
            train_batch_loss_history.append(current_loss_val)

        scheduler.step()
        mean_train_loss = epoch_train_loss_accum / N_train
        train_epoch_history.append(mean_train_loss)

        # --- TEST PASS EVERY EPOCH ---
        model.eval()
        epoch_test_loss_accum = 0.0

        with torch.no_grad():
            for i in range(0, N_test, mini_batch_size):
                batch_test_x = test_x[i : i + mini_batch_size].to(device)
                batch_test_y = test_y[i : i + mini_batch_size].to(device)
                current_test_batch_size = len(batch_test_x)

                test_loss = _run_closed_loop_batch(
                    batch_test_x, batch_test_y, is_training=False
                )
                epoch_test_loss_accum += (
                    test_loss.item() * current_test_batch_size
                )

        mean_test_loss = (
            epoch_test_loss_accum / N_test if N_test > 0 else 0.0
        )
        test_epoch_history.append(mean_test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"✨ Epoch {epoch+1} Summary | LR: {current_lr:.6e} | Train Loss:"
            f" {mean_train_loss:.6f} | Test Loss: {mean_test_loss:.6f}"
        )

        # Checkpoint / Early Stopping
        if mean_test_loss < (best_test_loss - min_delta):
            best_test_loss = mean_test_loss
            patience_counter = 0
            save_model(
                model,
                dirname=dirname,
                hyperparam_config=hyperparam_config,
                filename="best_model",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}.")
                break

    # --- 4. LOSS PLOTS ---
    plot_signals(
        t=np.arange(len(train_batch_loss_history)),
        signals=[np.array(train_batch_loss_history)],
        labels=["Batch Training Loss"],
        xlabel="Optimization Steps",
        ylabel="Loss",
        dirname=dirname,
        filename="granular_training_loss",
    )

    plot_signals(
        t=np.arange(1, len(train_epoch_history) + 1),
        signals=[np.array(train_epoch_history), np.array(test_epoch_history)],
        labels=["Avg Train Loss", "Avg Test Loss"],
        xlabel="Epochs",
        ylabel="Loss",
        dirname=dirname,
        filename="epoch_loss_curves",
    )

    # --- SUMMARY METRICS ---
    summary_metrics = {
        "Metric": ["Best Training Loss", "Best Testing Loss"],
        "Value": [min(train_epoch_history), best_test_loss],
    }
    summary_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 TRAINING SUMMARY")
    print("==========================================")
    print(summary_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(summary_df, dirname=dirname, filename="training_summary")

    history = {
        "train_loss": train_epoch_history,
        "test_loss": test_epoch_history,
    }

    return model, history, summary_df



def train_full_dataset(model, Y_trajectories, U_trajectories, hyperparam_config, dirname="./final_model"):
    """Retrain the final inverse controller model on the entire dataset for production deployment.

    This function slices full trajectory arrays using sliding windows, fits global feature and target 
    ``StandardScaler`` objects on all available data, and trains the inverse neural network model across 
    the total dataset for a specified number of epochs. Global scalers, hyperparameter configurations, 
    and the final model weights are saved to disk upon completion.

    :param model: The PyTorch neural network model representing the inverse controller.
    :type model: torch.nn.Module
    :param Y_trajectories: Array or tensor containing complete reference/output trajectories of shape 
        ``(n_samples, sequence_length, output_dim)``.
    :type Y_trajectories: numpy.ndarray | torch.Tensor
    :param U_trajectories: Array or tensor containing complete control input trajectories of shape 
        ``(n_samples, sequence_length, input_dim)``.
    :type U_trajectories: numpy.ndarray | torch.Tensor
    :param hyperparam_config: Configuration dictionary containing training, data, and plant hyperparameters.
        Must include keys ``'train'`` and ``'plant'``.
    :type hyperparam_config: dict
    :param dirname: Output directory path where the final model checkpoint, global scalers, 
        and best hyperparameter config JSON will be saved, defaults to ``"./final_model"``.
    :type dirname: str, optional

    :returns: The fully retrained production PyTorch model instance.
    :rtype: torch.nn.Module

    .. note::
        - No cross-validation or validation splits are performed in this function as the objective is to maximize 
          data usage for the production model.
        - Memory states are reset prior to forward passes if the model defines a ``reset_memory`` method (e.g., recurrent architectures).
    """
    train_cfg = hyperparam_config["train"]
    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    batch_size = train_cfg["mini_batch_size"]
    
    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]
    
    os.makedirs(dirname, exist_ok=True)
    
    print("\n==========================================")
    print("🚀 RETRAINING FINAL MODEL ON FULL DATASET")
    print("==========================================")
    
    # 1. Slicing full dataset
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u
    )
    
    N_total, seq_len, dim_x = X_raw.shape
    dim_y = Y_raw.shape[-1]
    
    # 2. Fit global scalers on entire dataset
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_flat = X_raw.reshape(-1, dim_x)
    Y_flat = Y_raw.reshape(-1, dim_y)
    
    scaler_x.fit(X_flat)
    scaler_y.fit(Y_flat)
    
    train_x = torch.tensor(scaler_x.transform(X_flat).reshape(N_total, seq_len, dim_x), dtype=torch.float32)
    train_y = torch.tensor(scaler_y.transform(Y_flat).reshape(N_total, seq_len, dim_y), dtype=torch.float32)
    
    # Save production scalers
    save_scaler_object(scaler_x, dirname=dirname, filename="final_scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="final_scaler_y")
    save_to_json(hyperparam_config, dirname, "best_hyperparam_config")
    
    # 3. Setup training components
    device = "cuda"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])
    
    loss_name = train_cfg["loss_function"].replace("()", "")
    criterion = NormalizedRMSELoss(reduction='none') if loss_name == "NormalizedRMSELoss" else getattr(nn, loss_name)(reduction='none')
    
    # 4. Optimization loop across whole dataset
    for epoch in range(epochs):
        model.train()
        epoch_loss_accum = 0.0
        shuffled_indices = torch.randperm(N_total)
        
        for i in range(0, N_total, batch_size):
            batch_indices = shuffled_indices[i : i + batch_size]
            current_bs = len(batch_indices)
            
            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)
            
            if hasattr(model, 'reset_memory'):
                model.reset_memory(batch_size=current_bs, device=device)
                
            optimizer.zero_grad()
            u_pred = model(batch_x)
            loss = criterion(u_pred, batch_y).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss_accum += loss.item() * current_bs
            
        scheduler.step()
        mean_epoch_loss = epoch_loss_accum / N_total
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"🔥 [Full Retrain] Epoch {epoch+1}/{epochs} | Loss: {mean_epoch_loss:.6f}")
            
    # Save fully retrained model
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="final_retrained_model")
    print(f"✅ Production Model & Scalers saved successfully in: {dirname}")
    return model





def train_controller_esn(
    model,
    X_raw,          # Shape: [Total_Seqs, Seq_Len, input_dim * 2] (y_t and y_next)
    Y_raw,          # Shape: [Total_Seqs, Seq_Len, output_dim]
    hyperparam_config,
    plant,
    dirname,
    run_simulation=False
):
    """Train an Echo State Network (ESN) inverse controller using K-Fold Cross-Validation.

    This function performs analytical Ridge Regression fitting on an ESN (e.g., ReservoirPy model) 
    across K-folds. In each fold, independent standard scalers are fitted on the training split, 
    instant readout weights are computed, and validation performance is evaluated. Optional plant 
    simulation rollouts and detailed tracking plots (for control signals and plant outputs) are 
    generated for validation sequences.

    :param model: The Echo State Network (ESN) reservoir model instance supporting ReservoirPy-style 
        ``fit()``, ``forward()``, and parameter saving methods.
    :type model: object
    :param X_raw: Array of input trajectory sequences (concatenated current and next reference states/outputs) 
        of shape ``(total_sequences, sequence_length, input_dim * 2)``.
    :type X_raw: numpy.ndarray
    :param Y_raw: Array of target control input sequences of shape 
        ``(total_sequences, sequence_length, output_dim)``.
    :type Y_raw: numpy.ndarray
    :param hyperparam_config: Configuration dictionary containing training, data, and plant hyperparameters.
        Must include keys ``'train'``, ``'training_data_cfg'``, and ``'plant'``.
    :type hyperparam_config: dict
    :param plant: Physical plant class or instance used for optional forward simulation rollout verification.
    :type plant: type | object
    :param dirname: Base directory path where fold-specific checkpoints, scalers, CSV logs, 
        and validation plots will be stored.
    :type dirname: str
    :param run_simulation: Flag indicating whether to execute closed-loop plant dynamic simulations 
        on all validation sequences for each fold, defaults to False.
    :type run_simulation: bool, optional

    :returns: Dictionary mapping fold indices (0 to K-1) to performance metrics history, 
        including overall training/validation MSE and channel-specific MSE values.
    :rtype: dict[int, dict[str, list[float]]]

    .. note::
        - Processing is maintained entirely in native NumPy arrays for optimal compatibility 
          with ReservoirPy reservoir computing workflows.
        - Training uses an exact analytical solution (Ridge Regression), completing in a single pass per fold without iterative epoch loops.
        - When ``run_simulation=True``, full trajectory simulation logs and signal comparison CSV files 
          are saved in fold-specific subdirectories.
    """
    # --- EXTRACT HYPERPARAMETERS ---
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = hyperparam_config["train"]["k_folds"]

    # --- MIMO-SPECIFIC CONFIG ---
    esn_cfg = hyperparam_config["plant"]
    input_dim = esn_cfg["input_dim"]    # Number of plant outputs
    output_dim = esn_cfg["output_dim"]  # Number of control inputs

    # --- SET UP K-FOLD INDICES ---
    total_sequences = X_raw.shape[0]
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    fold_histories = {}

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds} (Dedicated ESN Analytical Fit)")
        print(f"==========================================")

        # Clear internal memory states for the new fold
        model.load_state_dict(None)

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        # Isolate raw splits for this specific fold
        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")
        N_train, seq_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        # Keep everything native NumPy for ReservoirPy processing
        train_x = scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x)
        train_y = scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y)

        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)

        val_x = scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x)
        val_y = scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y)

        fold_dir = f"{dirname}/fold_{fold+1}"
        os.makedirs(fold_dir, exist_ok=True)
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        # --- ⚡ ANALYTICAL RIDGE REGRESSION TRAINING ---
        print(f"⚡ Executing instant weight computation via Ridge Regression...")

        # Convert full array matrices into sequence lists for ReservoirPy compatibility
        X_train_list = [train_x[i] for i in range(N_train)]
        Y_train_list = [train_y[i] for i in range(N_train)]

        # Train linear readout matrix instantly
        model.fit(X_train_list, Y_train_list)

        # --- 📊 EVALUATION METRICS COLLECTION ---
        # Generate predictions across train traces
        train_all_preds = np.array([model.forward(train_x[i]) for i in range(N_train)])

        # Generate predictions across validation traces
        val_all_preds_arr = np.array([model.forward(val_x[i]) for i in range(N_val)])
        val_all_trues_arr = val_y

        # Calculate Mean Squared Error performance evaluation bounds
        mean_train_loss = np.mean((train_all_preds - train_y) ** 2)
        mean_val_loss = np.mean((val_all_preds_arr - val_all_trues_arr) ** 2)

        # Populate history dictionaries to match validation summary targets
        fold_histories[fold] = {
            "train_loss": [mean_train_loss],
            "val_loss": [mean_val_loss],
            "val_epochs": [1],
            **{f"train_loss_ch{ch+1}": [np.mean((train_all_preds[..., ch] - train_y[..., ch]) ** 2)] for ch in range(output_dim)},
            **{f"val_loss_ch{ch+1}": [np.mean((val_all_preds_arr[..., ch] - val_all_trues_arr[..., ch]) ** 2)] for ch in range(output_dim)}
        }
     
        print(f"✨ [Fold {fold+1}] Performance Complete:")
        model.save_parameters(f"{fold_dir}/parameters")
        print(f"   ↳ Total Train MSE: {mean_train_loss:.6f} | Total Val MSE: {mean_val_loss:.6f}")

        # Persist trained parameters to disk
        save_model_esn(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")

        # --- 📊 VALIDATION PLOTS: CONTROL INPUTS + OUTPUT SIGNALS (FIRST 5 SEQUENCES) ---
        if N_val > 0:
            print(f"📊 Generating validation plots for first 5 sequences of Fold {fold+1}...")

            t_axis_val = np.arange(seq_len) * dt
            plots_dir = f"{fold_dir}/validation_control_and_output_plots"
            os.makedirs(plots_dir, exist_ok=True)

            for seq_idx in range(min(5, N_val)):
                # Unscaled predictions and ground truth for control inputs
                seq_pred_unscaled = scaler_y.inverse_transform(val_all_preds_arr[seq_idx])
                seq_true_unscaled = scaler_y.inverse_transform(val_all_trues_arr[seq_idx])

                # Unscaled output signals (y_t and y_next) from X_raw
                seq_x_unscaled = scaler_x.inverse_transform(val_x[seq_idx])
                y_t = seq_x_unscaled[:, :input_dim]  # First half: y_t
                y_next = seq_x_unscaled[:, input_dim:]  # Second half: y_next

                # --- PLOT 1: CONTROL INPUTS (u) ---
                for ch in range(output_dim):
                    actual_u = seq_true_unscaled[:, ch]
                    predicted_u = seq_pred_unscaled[:, ch]

                    plot_signals(
                        t=t_axis_val,
                        signals=[actual_u, predicted_u],
                        labels=[
                            rf"Actual Control ($u_{ch+1}$)",
                            rf"Predicted Control ($\hat{{u}}_{ch+1}$)"
                        ],
                        title=f"Fold {fold+1} - Seq {seq_idx+1}: Control Input (Channel {ch+1})",
                        xlabel="Time [s]",
                        ylabel="Control Input",
                        figsize=(7, 5),
                        filename=f"control_tracking_fold_{fold+1}_seq_{seq_idx+1}_ch{ch+1}",
                        dirname=plots_dir
                    )

                # --- PLOT 2: OUTPUT SIGNALS (y) ---
                for out_ch in range(input_dim):
                    plot_signals(
                        t=t_axis_val,
                        signals=[y_t[:, out_ch], y_next[:, out_ch]],
                        labels=[
                            rf"Original Output ($y_{out_ch+1}$)",
                            rf"Next Output ($y_{out_ch+1,next}$)"
                        ],
                        title=f"Fold {fold+1} - Seq {seq_idx+1}: Output Signal (Channel {out_ch+1})",
                        xlabel="Time [s]",
                        ylabel="Output Signal",
                        figsize=(7, 5),
                        filename=f"output_tracking_fold_{fold+1}_seq_{seq_idx+1}_ch{out_ch+1}",
                        dirname=plots_dir
                    )

            print(f"✅ Validation plots (control + output) generated for first 5 sequences of Fold {fold+1}.")

        # --- ⏳ PLANT SIMULATION ROLLOUT FOR VALIDATION SEQUENCES (OPTIONAL) ---
        if run_simulation and N_val > 0:
            print(f"📊 Simulating plant dynamics across ALL ({N_val}) validation profiles...")

            t_axis_val = np.arange(seq_len) * dt
            pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
            os.makedirs(pred_curves_dir, exist_ok=True)

            # Instantiate or use the plant instance
            plant_instance = plant(hyperparam_config) if isinstance(plant, type) else plant
            device = plant_instance.device

            for seq_idx in range(N_val):
                seq_pred_unscaled = scaler_y.inverse_transform(val_all_preds_arr[seq_idx])
                seq_true_unscaled = scaler_y.inverse_transform(val_all_trues_arr[seq_idx])
                seq_x_unscaled = scaler_x.inverse_transform(val_x[seq_idx])

                # Get starting state profile: [1, 2]
                current_sim_state = plant_instance.get_initial_state(batch_size=1)
                state_dim = current_sim_state.shape[-1]

                simulated_states_history = {st: [] for st in range(state_dim)}
                simulated_outputs_history = {out: [] for out in range(input_dim)}

                # 🌀 CRITICAL CORRECTION: Calculate and record initial output at t = 0
                y_init = plant_instance.get_y(current_sim_state, t_axis_val[0])
                for out in range(input_dim):
                    simulated_outputs_history[out].append(y_init[0, out].item())

                for step in range(seq_len):
                    # Record the current state components before stepping forward
                    for st in range(state_dim):
                        simulated_states_history[st].append(current_sim_state[0, st].item())

                    # Package predicted controller output u into a Tensor for the step
                    u_pred_step = torch.from_numpy(seq_pred_unscaled[step:step+1]).to(device=device, dtype=torch.float32)
                    t_curr = t_axis_val[step]

                    # Execute plant step -> advances state to t + dt
                    current_sim_state, y_next_pred = plant_instance.step(
                        current_sim_state,
                        u_pred_step,
                        t_curr,
                        dt
                    )

                    # Only capture the subsequent steps up to step < seq_len - 1 to match timeline bounds
                    if step < (seq_len - 1):
                        for out in range(input_dim):
                            simulated_outputs_history[out].append(y_next_pred[0, out].item())

                # --- PLOT 3: ORIGINAL VS. SIMULATED OUTPUTS (ONLY FOR FIRST 5 SEQUENCES) ---
                if seq_idx < 5:
                    for out_ch in range(input_dim):
                        # Ensure arrays match length exactly
                        sim_y_track = np.array(simulated_outputs_history[out_ch])
                        original_y_track = seq_x_unscaled[:, out_ch] # y_t from dataset

                        plot_signals(
                            t=t_axis_val,
                            signals=[
                                original_y_track,  # Original ground truth path
                                sim_y_track        # Pure output driven by ESN predicted control sequence
                            ],
                            labels=[
                                rf"Original Dataset Output ($y_{out_ch+1}$)",
                                rf"Simulated Output from Predicted $u$ ($\hat{{y}}_{out_ch+1}$)"
                            ],
                            title=f"Fold {fold+1} - Seq {seq_idx+1}: Dataset vs. Predicted Control Output (Channel {out_ch+1})",
                            xlabel="Time [s]",
                            ylabel="Output Signal [Growth Rate]",
                            figsize=(7, 5),
                            filename=f"output_comparison_fold_{fold+1}_seq_{seq_idx+1}_ch{out_ch+1}",
                            dirname=pred_curves_dir
                        )

                # Save simulation data to CSV
                log_data = {"Time (s)": t_axis_val}
                for out_idx in range(input_dim):
                    log_data[f"Target_y{out_idx+1}_t"] = seq_x_unscaled[:, out_idx]
                    log_data[f"Target_y{out_idx+1}_next"] = seq_x_unscaled[:, input_dim + out_idx]
                    log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs_history[out_idx]

                for ch in range(output_dim):
                    log_data[f"Actual_u{ch+1}"] = seq_true_unscaled[:, ch]
                    log_data[f"Predicted_u{ch+1}"] = seq_pred_unscaled[:, ch]
                    log_data[f"Control_Error_u{ch+1}"] = seq_true_unscaled[:, ch] - seq_pred_unscaled[:, ch]

                for st in range(state_dim):
                    log_data[f"Simulated_State_x{st+1}"] = simulated_states_history[st]

                val_profile_df = pd.DataFrame(log_data)
                save_df_to_csv(val_profile_df, dirname=pred_curves_dir, filename=f"val_plant_simulation_fold_{fold+1}_seq_{seq_idx+1}")

            print(f"✅ All {N_val} validation trajectory simulation logs dumped. Sample diagrams generated for Fold {fold + 1}.")

    # --- FINAL SUMMARY RECORD GENERATION ---
    print("\n💾 Packing overarching metadata curves...")
    summary_records = []
    for f in fold_histories:
        record = {"fold": f+1, "best_recorded_total_val_loss": fold_histories[f]["val_loss"][0]}
        for ch in range(output_dim):
            record[f"best_val_loss_u{ch+1}"] = fold_histories[f][f"val_loss_ch{ch+1}"][0]
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    print("✅ Dedicated ESN optimization finalized successfully.")

    return fold_histories