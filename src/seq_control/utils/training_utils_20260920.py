"""
Training Utility Functions
==========================

This module contains utility functions for the training of inverse controllers. 
"""

# Import standard libraries
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Import custom utility functions
from seq_control.config import *
from seq_control.decorators.general_decorators import *
from seq_control.utils.loss_utils import *
from seq_control.utils.plotting_utils import *
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.general_utils import *
from seq_control.utils.data_generation_utils import *

#=== FUNCTION TO TRAIN SEQUENCE MODEL WITH K-FOLD CROSS VALIDATION ===#
def train_sequence_model(
    model,
    plant,
    sw_dataset,
    hyperparam_config,
    dirname,
    show_plots=False
):
    # Save dataset and hyperparameter configuration
    save_dataset(sw_dataset, dirname = dirname, filename = "sw_ic_dataset")
    save_to_json(hyperparam_config, dirname,"hyperparam_config") 

    # --- EXTRACT HYPERPARAMETERS ---
    training_data_cfg = hyperparam_config["training_data_cfg"]
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    dt = training_data_cfg["dt"]

    device = train_cfg["device"]
    k_folds = train_cfg["k_folds"]
    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    mini_batch_size = train_cfg["mini_batch_size"]
    test_patience = train_cfg["patience_epochs"]
    patience_min_improvement = train_cfg["patience_min_improvement"]

    input_dim = training_data_cfg["input_dim"]
    output_dim = training_data_cfg["output_dim"]

    X_raw = sw_dataset["X_raw"]
    Y_raw = sw_dataset["Y_raw"]

    total_sequences = X_raw.shape[0]

    all_indices = np.arange(total_sequences)
    # Randomly shuffle the indices of the sequences
    np.random.shuffle(all_indices)
    # Split data into k folds
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # Set loss function
    loss_name = train_cfg["loss_function"].replace("()", "")
    if loss_name == "NormalizedRMSELoss":
        criterion = NormalizedRMSELoss(reduction='None')
    elif loss_name == "MSELoss":
        criterion = torch.nn.MSELoss(reduction='none')

    # --- K-FOLD CROSS VALIDATION LOOP ---
    # Loop across folds. In case of 5-fold cross validation, the code in this for loop will run 5 times for different partitions of the data into training and validation data.
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        
    
        model.load_state_dict(initial_model_state)
        model.to(device)
    
        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Split data into training and testing data
        test_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, test_idx_arr)

        train_x_raw, test_x_raw = X_raw[train_idx_arr], X_raw[test_idx_arr]
        train_y_raw, test_y_raw = Y_raw[train_idx_arr], Y_raw[test_idx_arr]

        # Apply standard scaling
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

        N_test = test_x_raw.shape[0]
        test_x_flat = test_x_raw.reshape(-1, dim_x)
        test_y_flat = test_y_raw.reshape(-1, dim_y)

        test_x = torch.tensor(scaler_x.transform(test_x_flat).reshape(N_test, seq_len, dim_x), dtype=torch.float32)
        test_y = torch.tensor(scaler_y.transform(test_y_flat).reshape(N_test, seq_len, dim_y), dtype=torch.float32)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        train_size = train_x.shape[0]
        test_size = test_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_train_channel_batch_loss = {ch: [] for ch in range(output_dim)}
        fold_train_channel_epoch_history = {ch: [] for ch in range(output_dim)}
        fold_test_channel_epoch_history = {ch: [] for ch in range(output_dim)}
        fold_test_epoch_history = []
        fold_train_epoch_history = []

        best_test_loss = float('inf')
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

                current_loss_test = loss.item()
                epoch_train_loss_accum += current_loss_test * current_batch_size
                fold_train_batch_loss.append(current_loss_test)
                fold_train_batch_indices.append(global_batch_counter)

                for ch in range(input_dim):
                    ch_loss_test = raw_loss[:, :, ch].mean().item()
                    epoch_train_channel_accum[ch] += ch_loss_test * current_batch_size
                    fold_train_channel_batch_loss[ch].append(ch_loss_test)

                global_batch_counter += 1

            # --- 3. TEST PASS ---
            model.eval()
            epoch_test_loss_accum = 0.0
            epoch_test_channel_accum = {ch: 0.0 for ch in range(output_dim)}

            all_test_preds = []
            all_test_trues = []

            with torch.no_grad():
                for i in range(0, test_size, mini_batch_size):
                    batch_test_x = test_x[i : i + mini_batch_size].to(device)
                    batch_test_y = test_y[i : i + mini_batch_size].to(device)
                    current_test_batch_size = len(batch_test_x)

                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_test_batch_size, device=device)

                    u_test_pred = model(batch_test_x)
                    raw_test_loss = criterion(u_test_pred, batch_test_y)

                    test_loss = raw_test_loss.mean()
                    epoch_test_loss_accum += test_loss.item() * current_test_batch_size

                    for ch in range(input_dim):
                        ch_test_loss_test = raw_test_loss[:, :, ch].mean().item()
                        epoch_test_channel_accum[ch] += ch_test_loss_test * current_test_batch_size

                    all_test_preds.append(u_test_pred.cpu().numpy())
                    all_test_trues.append(batch_test_y.cpu().numpy())

            test_all_preds_arr = np.concatenate(all_test_preds, axis=0)
            test_all_trues_arr = np.concatenate(all_test_trues, axis=0)

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_test_loss = (epoch_test_loss_accum / test_size) if test_size > 0 else 0.0

            fold_test_epoch_history.append(mean_test_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {
                    "train_loss": [], 
                    "test_loss": [], 
                    "test_epochs": [],
                    **{f"train_loss_ch{ch+1}": [] for ch in range(output_dim)},
                    **{f"test_loss_ch{ch+1}": [] for ch in range(output_dim)}
                }

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["test_loss"].append(mean_test_loss)
            fold_histories[fold]["test_epochs"].append(epoch + 1)

            for ch in range(output_dim):
                mean_train_ch = epoch_train_channel_accum[ch] / train_size
                mean_test_ch = (epoch_test_channel_accum[ch] / test_size) if test_size > 0 else 0.0

                fold_train_channel_epoch_history[ch].append(mean_train_ch)
                fold_test_channel_epoch_history[ch].append(mean_test_ch)

                fold_histories[fold][f"train_loss_ch{ch+1}"].append(mean_train_ch)
                fold_histories[fold][f"test_loss_ch{ch+1}"].append(mean_test_ch)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Total Train Loss: {mean_train_loss:.6f} | Total Test Loss: {mean_test_loss:.6f}")

            if mean_test_loss < (best_test_loss - patience_min_improvement):
                best_test_loss = mean_test_loss
                patience_counter = 0
                save_model(model, 
                           dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= test_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    early_stopped = True
                    break
        
        # --- 4. PLOT EXTENDED LOSS CURVES ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"
        epoch_axis = np.array(fold_histories[fold]["test_epochs"])

        
        plot_signals(t=np.array(fold_train_batch_indices),
                     signals=[np.array(fold_train_batch_loss)],
                     labels=[f"Fold {fold+1} Total Loss"],
                     xlabel= r"$n_{seq}$",
                     ylabel= loss_name,
                     dirname=fold_dir,
                     filename="granular_training_loss",
                     asp=0.3)

        plot_signals(t=epoch_axis,
                     signals=[np.array(fold_train_epoch_history), np.array(fold_test_epoch_history)],
                     labels=[r"$\mathcal{L}_{\mathrm{tr}}$", r"$\mathcal{L}_{\mathrm{val}}$"],
                     xlabel= "Epochs",
                     ylabel= r"\text{MSE} [-]", 
                     dirname=fold_dir,
                     filename="epoch_validation_loss",
                     asp=0.3)

        # --- PLOT ALL VALIDATION PREDICTIONS FOR THIS FOLD ---
        print(f"📈 Plotting all validation predictions for Fold {fold + 1}...")
        pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
        t_axis_test = np.arange(seq_len) * dt

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
        if show_plots:
            for seq_idx in range(len(test_all_preds_arr)):
                seq_pred_scaled = test_all_preds_arr[seq_idx]
                seq_true_scaled = test_all_trues_arr[seq_idx]

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
                        t=t_axis_test,
                        signals=[seq_true_unscaled[:, ch], seq_pred_unscaled[:, ch]],
                        labels=[f"True {base_label}", f"Predicted {base_label}"],
                        xlabel=xlabel_str,
                        ylabel=ylabel_str,
                        dirname=pred_curves_dir,
                        filename=f"test_prediction_seq{seq_idx+1}_u{ch+1}",
                        asp=0.3
                    )

    # --- METRIC AGGREGATION ACROSS FOLDS ---
    fold_best_train_losses = []
    fold_best_test_losses = []

    for f in range(k_folds):
        best_test_epoch_idx = np.argmin(fold_histories[f]["test_loss"])
        fold_best_test_losses.append(fold_histories[f]["test_loss"][best_test_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_test_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_test_loss = float(np.mean(fold_best_test_losses))
    mean_cv_loss = avg_best_test_loss  # Open-loop mean CV loss

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
            "test_loss": avg_best_test_loss,
        },
        mean_cv_loss,
        summary_results_df,
    )

#=== FUNCTION TO TRAIN SEQUENCE MODEL ON A DATASET ===#
def train_full_dataset(model, 
                       dataset,
                       hyperparam_config, 
                       dirname="./final_model"):
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
    nu_y = train_cfg["nu_y"]
    nu_u = train_cfg["nu_u"]
    batch_size = train_cfg["mini_batch_size"]
    
    plant_cfg = hyperparam_config["plant"]
    
    #os.makedirs(dirname, exist_ok=True)
    
    print("\n==========================================")
    print("🚀 RETRAINING FINAL MODEL ON FULL DATASET")
    print("==========================================")
    
    # 1. Slicing full dataset
    X_raw, Y_raw = dataset["X_raw"], dataset["Y_raw"]
    
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
            
        mean_epoch_loss = epoch_loss_accum / N_total
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"🔥 [Full Retrain] Epoch {epoch+1}/{epochs} | Loss: {mean_epoch_loss:.6f}")
            
    # Save fully retrained model
    save_model(model, 
               dirname=dirname, 
               hyperparam_config=hyperparam_config, filename="final_retrained_model")
    print(f"✅ Production Model & Scalers saved successfully in: {dirname}")
    return model

#=== FUNCTION TO TRAIN AN ESN WITH K-FOLD CROSS-VALIDATION ===#
@track_resources
def train_controller_esn(
    model,
    X_raw,                # Shape: [Total_Seqs, Seq_Len, feature_dim]
    Y_raw,                # Shape: [Total_Seqs, Seq_Len, num_control_inputs]
    hyperparam_config,
    dirname,
    plant=None,           # Optional plant instance with get_plot_config()
    save_test_plots=False
):
    """Train an Echo State Network (ESN) inverse controller using K-Fold Cross-Validation."""
    
    # --- EXTRACT HYPERPARAMETERS ---
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = hyperparam_config["train"]["k_folds"]

    # --- DYNAMICALLY DERIVE ACTUAL ARRAY DIMENSIONS ---
    num_control_inputs = Y_raw.shape[-1]              # Actual target control channels (u)
    feature_dim = X_raw.shape[-1]                     # Total feature vector size
    num_plant_outputs = hyperparam_config["plant"].get("input_dim", 1)  # Plant measurement channels (y)

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

        model.load_state_dict(None)

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

        train_x = scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x)
        train_y = scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y)

        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)

        val_x = scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x)
        val_y = scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y)

        fold_dir = f"{dirname}/fold_{fold+1}"
        
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        # --- ANALYTICAL RIDGE REGRESSION TRAINING ---
        print(f"⚡ Executing instant weight computation via Ridge Regression...")
        X_train_list = [train_x[i] for i in range(N_train)]
        Y_train_list = [train_y[i] for i in range(N_train)]

        model.fit(X_train_list, Y_train_list)

        # --- EVALUATION METRICS COLLECTION ---
        train_all_preds = np.array([model.forward(train_x[i]) for i in range(N_train)])
        val_all_preds_arr = np.array([model.forward(val_x[i]) for i in range(N_val)])
        val_all_trues_arr = val_y

        mean_train_loss = np.mean((train_all_preds - train_y) ** 2)
        mean_val_loss = np.mean((val_all_preds_arr - val_all_trues_arr) ** 2)

        fold_histories[fold] = {
            "train_loss": [mean_train_loss],
            "val_loss": [mean_val_loss],
            "val_epochs": [1],
            **{f"train_loss_ch{ch+1}": [np.mean((train_all_preds[..., ch] - train_y[..., ch]) ** 2)] for ch in range(num_control_inputs)},
            **{f"val_loss_ch{ch+1}": [np.mean((val_all_preds_arr[..., ch] - val_all_trues_arr[..., ch]) ** 2)] for ch in range(num_control_inputs)}
        }
     
        print(f"✨ [Fold {fold+1}] Performance Complete:")
        print(f"   ↳ Total Train MSE: {mean_train_loss:.6f} | Total Val MSE: {mean_val_loss:.6f}")

        save_model_esn(model, 
                       dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")

        # --- 📈 PLOT ALL VALIDATION PREDICTIONS FOR THIS FOLD ---
        if save_test_plots and N_val > 0:
            print(f"📈 Plotting all validation predictions for Fold {fold + 1}...")
            pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
            #os.makedirs(pred_curves_dir, exist_ok=True)
            t_axis_test = np.arange(seq_len) * dt

            # Extract dynamic metadata from plant if provided
            u_cfg, t_cfg = None, None
            if plant is not None and hasattr(plant, "get_plot_config"):
                plot_cfg = plant.get_plot_config()
                u_cfg = next((cfg for cfg in plot_cfg if "u" in cfg.get("cols", [])), None)
                t_cfg = next((cfg for cfg in plot_cfg if "t" in cfg.get("cols", [])), None)

            # Resolve dynamic x-axis label
            xlabel_str = "Time [s]"
            if t_cfg and "xlabel" in t_cfg:
                xlabel_str = t_cfg["xlabel"][0] if isinstance(t_cfg["xlabel"], list) else t_cfg["xlabel"]

            # Plot prediction curves across all validation sequences
            for seq_idx in range(N_val):
                seq_pred_scaled = val_all_preds_arr[seq_idx]
                seq_true_scaled = val_all_trues_arr[seq_idx]

                seq_pred_unscaled = scaler_y.inverse_transform(seq_pred_scaled)
                seq_true_unscaled = scaler_y.inverse_transform(seq_true_scaled)

                for ch in range(num_control_inputs):
                    # Extract y-axis and curve labels dynamically per control channel
                    if u_cfg:
                        if isinstance(u_cfg.get("ylabel"), list):
                            ylabel_str = u_cfg["ylabel"][ch] if ch < len(u_cfg["ylabel"]) else f"Control Unit u_{ch+1}"
                        else:
                            ylabel_str = u_cfg.get("ylabel", f"Control Unit u_{ch+1}")

                        if "labels" in u_cfg and isinstance(u_cfg["labels"], list) and ch < len(u_cfg["labels"]):
                            base_label = u_cfg["labels"][ch]
                        else:
                            base_label = f"u_{ch+1}"
                    else:
                        ylabel_str = f"Control Unit u_{ch+1}"
                        base_label = f"u_{ch+1}"

                    plot_signals(
                        t=t_axis_test,
                        signals=[seq_true_unscaled[:, ch], seq_pred_unscaled[:, ch]],
                        labels=[f"True {base_label}", f"Predicted {base_label}"],
                        xlabel=xlabel_str,
                        ylabel=ylabel_str,
                        dirname=pred_curves_dir,
                        filename=f"validation_prediction_seq{seq_idx+1}_u{ch+1}",
                        asp=0.3
                    )

            print(f"✅ Saved all validation prediction plots to '{pred_curves_dir}'.")

    # --- FINAL SUMMARY RECORD GENERATION ---
    print("\n💾 Packing overarching metadata curves...")
    summary_records = []
    for f in fold_histories:
        record = {"fold": f+1, "best_recorded_total_val_loss": fold_histories[f]["val_loss"][0]}
        for ch in range(num_control_inputs):
            record[f"best_val_loss_u{ch+1}"] = fold_histories[f][f"val_loss_ch{ch+1}"][0]
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    print("✅ Dedicated ESN optimization finalized successfully.")

    return fold_histories

#=== FUNCTION TO DEFINE OBJECTIVE FUNCTION FOR OPTUNA-BASED HYPERPARAMETER OPTIMIZATION ===#
def objective(trial, 
              model_class, 
              dataset,
              base_config, 
              plant, 
              dirname, 
              param_space):
    """
    Modular Optuna objective function sampling hyperparameters from a space dict.
    """
    def set_nested_value(d, keys, value):
        """Helper to set a value in a nested dictionary given a list/path of keys."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    config = copy.deepcopy(base_config)

    # --- 1. DYNAMICALLY SAMPLE HYPERPARAMETERS FROM PARAM_SPACE ---
    for param_path, spec in param_space.items():
        # Split path string (e.g., "mamba.d_state") into list of keys
        keys = param_path.split(".") if isinstance(param_path, str) else param_path
        
        # Determine parameter name for Optuna logs
        param_name = keys[-1] if isinstance(keys, list) else keys
        
        param_type = spec["type"]
        
        # Sample using appropriate Optuna method
        if param_type == "int":
            val = trial.suggest_int(param_name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
        elif param_type == "float":
            val = trial.suggest_float(param_name, spec["low"], spec["high"], step=spec.get("step", None), log=spec.get("log", False))
        elif param_type == "categorical":
            val = trial.suggest_categorical(param_name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        # Insert sampled value into the nested config dictionary
        set_nested_value(config, keys, val)

    # Dynamic directory for each trial
    trial_dirname = f"{dirname}/trial_{trial.number}"

    # Re-instantiate model with updated config
    model = model_class(config)

    # --- 2. EXECUTE CONTROLLER TRAINING ---
    # Add these 2 lines in objective() right before calling train_sequence_model:
    import inspect
    print("🔍 EXECUTING train_sequence_model FROM:", inspect.getfile(train_sequence_model), "LINE:", inspect.getsourcelines(train_sequence_model)[1])

    

    res = train_sequence_model(
        model=model,
        plant=plant,
        sw_dataset=dataset,
        hyperparam_config=config,
        dirname=trial_dirname,
        show_plots=False
    )

    print("📍 FILE PATH:", inspect.getfile(train_sequence_model))
    print("📍 STARTING LINE:", inspect.getsourcelines(train_sequence_model)[1])
    print("📍 RETURN CONTENTS:", [type(x) for x in res])

    print("🔍 RETURNED TYPE & LENGTH:", type(res), len(res) if isinstance(res, tuple) else "Not a tuple")

    _, _, mean_cv_val_loss, _ = train_sequence_model(
        model=model,
        plant=plant,
        sw_dataset=dataset,
        hyperparam_config=config,
        dirname=trial_dirname,
        show_plots=False
    )

    return mean_cv_val_loss


#=== FUNCTION TO RUN HYPERPARAMETER OPTIMIZATION STUDY USING OPTUNA ===#
@track_resources
def run_optuna_study(
        model_class,
        dataset,
        hyperparam_config,
        param_space,
        plant,
        dirname,
        n_trials
    ):
    # Setup directories
    optuna_dir = f"{dirname}/optuna_trials"
    final_dir = f"{dirname}/final_production_model"
    n_trials = n_trials  # Set desired trials limit
    
    print("🛠️ Initializing Optuna Study (TPE Sampler)...")
    study = optuna.create_study(
        study_name="controller_hyperparam_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 1. Run Bayesian Optimization via Optuna
    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_class=model_class, 
            dataset=dataset,
            base_config=hyperparam_config,
            plant=plant,
            dirname=optuna_dir,
            param_space=param_space
        ),
        n_trials=n_trials
    )

    # 2. Print Summary Results
    print("\n==========================================")
    print("🏆 OPTUNA HYPERPARAMETER TUNING COMPLETE")
    print("==========================================")
    print(f"Best Trial Number   : {study.best_trial.number}")
    print(f"Best Mean CV Loss   : {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")

    # 4. Retrain Final Model on Full Dataset
    # Prepare best configuration dictionary
    best_config = copy.deepcopy(hyperparam_config)
    best_config[model_class.__name__].update(study.best_params)

    print("best config", best_config)

    # Instantiate fresh model with the best parameters
    best_model = model_class(best_config) 

    # Train on complete dataset
    final_model = train_full_dataset(
        model=best_model,
        dataset= dataset,
        hyperparam_config=best_config,
        dirname=final_dir
    )
    
    return study, final_model