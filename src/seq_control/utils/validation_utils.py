"""
Validation Utility Functions
============================

This module contains utilities for the validation of sequence-model-based controller performance.
"""

# Import standard libraries
from typing import Dict, Any, List
import os
import torch
import numpy as np
import pandas as pd

# Import utility functions
from seq_control.utils.plotting_utils import *
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.general_utils import *

import numpy as np

def extract_mimo_control_bounds(hyperparam_config, control_dim):
    """
    Extracts per-channel control bounds u_min and u_max arrays of shape (control_dim,).
    Falls back to -inf / +inf if bounds are not specified.
    """
    
    training_data_cfg = hyperparam_config["training_data_cfg"]

    u_min_list = []
    u_max_list = []


    for ch in range(1, control_dim + 1):
        # 1. Look for channel-specific keys (e.g., 'u_1_hard_min')
        ch_min = training_data_cfg[f"u_{ch}_hard_min"]
        ch_max = training_data_cfg[f"u_{ch}_hard_max"]        
        print(ch_min, ch_max)
        # Use infinity for unconstrained channels
        u_min_list.append(ch_min if ch_min is not None else -np.inf)
        u_max_list.append(ch_max if ch_max is not None else np.inf)

    return np.array(u_min_list, dtype=np.float32), np.array(u_max_list, dtype=np.float32)

#=== FUNCTION TO VALIDATE MULTPILE CONTROLLERS ===#
def validate_multiple_controllers(
    models_dict,          # Dict[str, Dict[str, Any]] containing "Model", "x_scaler", "y_scaler"
    plant,
    dataset_io,
    hyperparam_config,
    dirname,
    start_idx,
    window_len=None,
    mode="closed_loop",
    show_plots=False
):
    """
    Sequence-based validation routine that evaluates multiple inverse control models 
    side-by-side from a nested model configuration dictionary.
    
    Each entry in `models_dict` expects:
    {
        "Controller_Name": {
            "Model": model_instance,
            "x_scaler": scaler_x_instance,
            "y_scaler": scaler_y_instance
        }
    }
    """
    #os.makedirs(dirname, exist_ok=True)
    
    device = hyperparam_config["train"]["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    nu_y = hyperparam_config["training_data_cfg"]["nu_y"]
    nu_u = hyperparam_config["training_data_cfg"]["nu_u"]
    


    # --- LOAD REFERENCE DATASET ---
    u_ref_raw = dataset_io["u"].to(dtype=torch.float32)
    y_ref_raw = dataset_io["y"].to(dtype=torch.float32)
    states_raw = dataset_io["states"].to(dtype=torch.float32)
    
    N, total_seq_len, output_dim = y_ref_raw.shape
    control_dim = u_ref_raw.shape[-1]
    end_idx = total_seq_len - 1
    
    
    y_ref_np = y_ref_raw.numpy()
    u_ref_np = u_ref_raw.numpy()
    has_state_ref = (states_raw.ndim == 3)
    states_ref_np = states_raw.numpy() if has_state_ref else None

    initial_state_base = states_raw[:, 0, :] if has_state_ref else states_raw.clone()

    # --- STORAGE BUFFERS FOR COMPARATIVE PLOTS ---
    u_applied_dict = {}
    y_achieved_dict = {}
    states_achieved_dict = {}
    summary_records = []

    u_min_arr, u_max_arr = extract_mimo_control_bounds(
        hyperparam_config, 
        control_dim)
    # --- SIMULATION LOOP OVER EACH MODEL CONFIGURATION ---
    for model_name, model_cfg in models_dict.items():
        print(f"\n🧪 Simulating Model: '{model_name}' ({mode.upper()} Mode)...")
        if model_name == "TransformerInverseController" or plant.__class__.__name__ == "IndForProteinProductionPlant":
            window_len = 100
        # Extract model and scalers directly from sub-dictionary
        model = model_cfg.get("Model", model_cfg.get("model"))
        sx = model_cfg.get("x_scaler", model_cfg.get("scaler_x"))
        sy = model_cfg.get("y_scaler", model_cfg.get("scaler_y"))

        if model is None or sx is None or sy is None:
            raise KeyError(
                f"Model entry '{model_name}' must contain 'Model', 'x_scaler', and 'y_scaler' keys."
            )

        current_state = initial_state_base.clone().to(device)
        state_dim = current_state.shape[-1]

        y_achieved = np.zeros((N, total_seq_len, output_dim), dtype=np.float32)
        u_applied = np.zeros((N, total_seq_len, control_dim), dtype=np.float32)
        states_achieved = np.zeros((N, total_seq_len, state_dim), dtype=np.float32)

        # Log initial conditions
        states_achieved[:, 0, :] = current_state.cpu().numpy()
        y_0 = plant.get_y(current_state, 0.0)
        if y_0.ndim == 1:
            y_0 = y_0.unsqueeze(-1)
        y_achieved[:, 0, :] = y_0.cpu().numpy()

        v_frames_scaled = []

        # --- 1. PHASE 1: WARM-UP (k = 0 to start_idx - 1) ---
        for k in range(0, start_idx):
            t_current = k * dt
            
            y_src = y_ref_np if mode == "open_loop" else y_achieved
            u_src = u_ref_np if mode == "open_loop" else u_applied
            
            v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
            v_k_scaled = sx.transform(v_k)
            v_frames_scaled.append(v_k_scaled)

            u_k_np = u_ref_np[:, k, :]
            u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)
            u_applied[:, k, :] = u_k_np

            u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

            next_state, _ = plant.step(current_state, u_k_tensor, t=t_current)
            current_state = next_state

            states_achieved[:, k + 1, :] = current_state.cpu().numpy()
            y_next = plant.get_y(current_state, (k + 1) * dt)
            if y_next.ndim == 1:
                y_next = y_next.unsqueeze(-1)
            y_achieved[:, k + 1, :] = y_next.cpu().numpy()

        # --- 2. PHASE 2: MODEL TAKEOVER (k = start_idx to end_idx - 1) ---
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(device)

        for k in range(start_idx, end_idx):
            t_current = k * dt

            y_src = y_ref_np if mode == "open_loop" else y_achieved
            u_src = u_ref_np if mode == "open_loop" else u_applied

            v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
            v_k_scaled = sx.transform(v_k)
            v_frames_scaled.append(v_k_scaled)
            

            v_seq_np = np.stack(v_frames_scaled, axis=1)
            if window_len is not None:
                v_seq_np = np.stack(v_frames_scaled[-window_len:], axis=1)
            v_seq_tensor = torch.tensor(v_seq_np, dtype=torch.float32, device=device)

            if hasattr(model, "forward") and isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    u_pred_seq_scaled = model(v_seq_tensor)
                u_pred_last_scaled = u_pred_seq_scaled[:, -1, :].cpu().numpy()
            else:
                # Analytical fit (e.g. ESN) or direct object call
                u_pred_seq_scaled = np.array([model.forward(v_seq_np[i]) for i in range(N)])
                u_pred_last_scaled = u_pred_seq_scaled[:, -1, :]

            u_k_np = sy.inverse_transform(u_pred_last_scaled)

            u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)
            u_applied[:, k, :] = u_k_np
            u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

            next_state, _ = plant.step(current_state, u_k_tensor, t=t_current)
            current_state = next_state

            states_achieved[:, k + 1, :] = current_state.cpu().numpy()
            y_next_achieved = plant.get_y(current_state, (k + 1) * dt)
            if y_next_achieved.ndim == 1:
                y_next_achieved = y_next_achieved.unsqueeze(-1)
            y_achieved[:, k + 1, :] = y_next_achieved.cpu().numpy()

        # --- COMPUTE METRICS ---
        y_ref_eval = y_ref_np[:, start_idx:end_idx, :]
        y_achieved_eval = y_achieved[:, start_idx:end_idx, :]
        u_ref_eval = u_ref_np[:, start_idx:end_idx, :]
        u_applied_eval = u_applied[:, start_idx:end_idx, :]

        total_mse = float(np.mean((y_ref_eval - y_achieved_eval) ** 2))
        total_rmse = float(np.sqrt(total_mse))
        control_mse = float(np.mean((u_ref_eval - u_applied_eval) ** 2))

        summary_records.append({
            "Model": model_name,
            "Tracking MSE": total_mse,
            "Tracking RMSE": total_rmse,
            "Control MSE": control_mse
        })

        u_applied_dict[model_name] = u_applied[:, :end_idx, :]
        y_achieved_dict[model_name] = y_achieved[:, :end_idx, :]
        states_achieved_dict[model_name] = states_achieved[:, :end_idx, :]

    # --- SUMMARY TABLE GENERATION ---
    summary_df = pd.DataFrame(summary_records)
    print(f"\n==========================================")
    print(f"🔁 {mode.upper()} MULTI-MODEL VALIDATION SUMMARY")
    print(f"==========================================")
    print(summary_df.to_string(index=False))
    print(f"==========================================\n")
    save_df_to_csv(summary_df, 
                   dirname=dirname, 
                   filename=f"multi_model_{mode}_summary.csv")

    # --- RENDER STACKED PLOTS ---
    print(f"📊 Rendering multi-model trajectory plots using `plot_stacked`...")
    t_full = np.arange(0, end_idx) * dt

    plot_closed_loop_trajectories_multi(
        t=t_full,
        u_ref=u_ref_np[:, :end_idx, :],
        u_applied_dict=u_applied_dict,
        y_ref=y_ref_np[:, :end_idx, :],
        y_achieved_dict=y_achieved_dict,
        states_achieved_dict=states_achieved_dict,
        states_ref=states_ref_np[:, :end_idx, :] if has_state_ref else None,
        plant=plant,
        dirname=dirname,
        show=show_plots
    )

    return {
        "summary_df": summary_df,
        "u_applied_dict": u_applied_dict,
        "y_achieved_dict": y_achieved_dict,
        "states_achieved_dict": states_achieved_dict
    }

#=== FUNCTION TO VALIDATE MULTIPLE ON CONTROLLERS ON GIVEN TRAJECTORIES ===#
def validate_controller_ext_ref_multi(
    models_dict,
    plant,
    y_ref,
    hyperparam_config,
    dirname="./plots_ref_multi",
    start_idx=10,
    window_len=None,
    mode="closed_loop",
    show_plots=False
):
    """
    Evaluates and compares multiple neural network controllers against an external 
    user-supplied reference trajectory y_ref.

    Parameters:
    - models_dict (dict): Dictionary mapping model names to dicts containing 'model',
      'x_scaler', and 'y_scaler' (or 'Model', 'scaler_x', 'scaler_y').
    - plant: Plant class instance.
    - y_ref (Tensor or np.ndarray): Reference target shaped [steps], [steps, nu_y], 
      or [N, steps, nu_y].
    - hyperparam_config (dict): Configuration containing dt, nu_y, nu_u, device, etc.
    - dirname (str): Folder path to store summary CSV and trajectory figures.
    - start_idx (int): Warm-up step index where models take over control.
    - window_len (int): Length of the window for validation.
    - mode (str): 'closed_loop' or 'open_loop'.
    - show_plots (bool): Whether to display Matplotlib figures interactively.
    """
    #os.makedirs(dirname, exist_ok=True)

    device = hyperparam_config["train"]["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    nu_y = hyperparam_config["training_data_cfg"]["nu_y"]
    nu_u = hyperparam_config["training_data_cfg"]["nu_u"]



    u_min_arr, u_max_arr = extract_mimo_control_bounds(hyperparam_config, control_dim)

    # --- 1. FORMAT REFERENCE TARGET (y_ref) ---
    def _to_3d_ref(ref):
        """Standardizes an individual target trajectory into shape [N, total_seq_len, feature_dim]."""
        if isinstance(ref, torch.Tensor):
            t = ref.detach().cpu().float()
        else:
            t = torch.tensor(ref, dtype=torch.float32)

        if t.ndim == 1:
            # [steps] -> [1, steps, 1]
            return t.unsqueeze(0).unsqueeze(-1)
        elif t.ndim == 2:
            # [steps, feature_dim] -> [1, steps, feature_dim]
            # E.g., [2001, 1] becomes [1, 2001, 1]
            if t.shape[1] == 1 and t.shape[0] > 1:
                return t.unsqueeze(0)
            # [N, steps] -> [N, steps, 1]
            elif t.shape[0] == 1:
                return t.unsqueeze(-1)
            else:
                # Default assumption for 2D trajectories [steps, output_dim]
                return t.unsqueeze(0)
        elif t.ndim == 3:
            return t
        else:
            raise ValueError(f"Unsupported reference tensor shape: {t.shape}")

    if isinstance(y_ref, (list, tuple)):
        # Format each channel trajectory and concatenate along the output feature dimension
        formatted_refs = [_to_3d_ref(ref) for ref in y_ref]
        y_ref_raw = torch.cat(formatted_refs, dim=-1)
    else:
        y_ref_raw = _to_3d_ref(y_ref)

    N, total_seq_len, output_dim = y_ref_raw.shape
    control_dim = hyperparam_config["training_data_cfg"]["input_dim"]
    end_idx = total_seq_len - 1
    y_ref_np = y_ref_raw.numpy()

    # --- 2. FORMAT CONTROL REFERENCE (u_ref) ---
    
    u_ref_np = np.zeros((N, total_seq_len, control_dim), dtype=np.float32)

    # --- 3. GET BASE INITIAL PLANT STATE ---
    config_x0 = get_initial_state_from_config(hyperparam_config, batch_size=N, device=device)
    print(f"🌱 Base initial plant state loaded (Batch size={N})")

    u_applied_dict = {}
    y_achieved_dict = {}
    states_achieved_dict = {}
    summary_records = []

    # --- 4. MULTI-MODEL SIMULATION LOOP ---
    for model_name, model_cfg in models_dict.items():
        print(f"\n==========================================")
        print(f"🧪 Simulating '{model_name}' ({mode.upper()} Mode)")
        print(f"==========================================")

        # Extract model and scalers dynamically supporting common key naming variants
        model = model_cfg["model"]
        scaler_x = model_cfg["x_scaler"]
        scaler_y = model_cfg["y_scaler"]
        if model_name == "TransformerInverseController" or plant.__class__.__name__ == "IndForProteinProductionPlant":
            window_len = 100
        # Reset plant initial state independently for each controller
        current_state = config_x0.clone().to(device)
        state_dim = current_state.shape[-1]

        # Allocate simulation buffers
        y_achieved = np.zeros((N, total_seq_len, output_dim), dtype=np.float32)
        u_applied = np.zeros((N, total_seq_len, control_dim), dtype=np.float32)
        states_achieved = np.zeros((N, total_seq_len, state_dim), dtype=np.float32)

        # Record initial condition k=0
        states_achieved[:, 0, :] = current_state.cpu().numpy()
        y_0 = plant.get_y(current_state, 0.0)
        if y_0.ndim == 1:
            y_0 = y_0.unsqueeze(-1)
        y_achieved[:, 0, :] = y_0.cpu().numpy()

        v_frames_scaled = []

        # PHASE 1: WARM-UP (k = 0 to start_idx - 1)
        for k in range(0, start_idx):
            t_current = k * dt

            y_src = y_ref_np if mode == "open_loop" else y_achieved
            u_src = u_ref_np if mode == "open_loop" else u_applied

            v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
            v_k_scaled = scaler_x.transform(v_k)
            v_frames_scaled.append(v_k_scaled)

            u_k_np = u_ref_np[:, k, :]
            u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)
            u_applied[:, k, :] = u_k_np
            u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

            next_state, _ = plant.step(current_state, u_k_tensor, t=t_current)
            current_state = next_state

            states_achieved[:, k + 1, :] = current_state.cpu().numpy()
            y_next = plant.get_y(current_state, (k + 1) * dt)
            if y_next.ndim == 1:
                y_next = y_next.unsqueeze(-1)
            y_achieved[:, k + 1, :] = y_next.cpu().numpy()

        # PHASE 2: MODEL TAKEOVER (k = start_idx to end_idx - 1)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(device)

        for k in range(start_idx, end_idx):
            t_current = k * dt

            y_src = y_ref_np if mode == "open_loop" else y_achieved
            u_src = u_ref_np if mode == "open_loop" else u_applied

            v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
            v_k_scaled = scaler_x.transform(v_k)
            v_frames_scaled.append(v_k_scaled)

            v_seq_np = np.stack(v_frames_scaled, axis=1)
            if window_len is not None:
                v_seq_np = np.stack(v_frames_scaled[-window_len:], axis=1)
            v_seq_tensor = torch.tensor(v_seq_np, dtype=torch.float32, device=device)

            if hasattr(model, "forward") and isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    u_pred_seq_scaled = model(v_seq_tensor)
                u_pred_last_scaled = u_pred_seq_scaled[:, -1, :].cpu().numpy()
            else:
                u_pred_seq_scaled = np.array([model.forward(v_seq_np[i]) for i in range(N)])
                u_pred_last_scaled = u_pred_seq_scaled[:, -1, :]

            u_k_np = scaler_y.inverse_transform(u_pred_last_scaled)
            u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)
            
            u_applied[:, k, :] = u_k_np
            u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

            next_state, _ = plant.step(current_state, u_k_tensor, t=t_current)
            current_state = next_state

            states_achieved[:, k + 1, :] = current_state.cpu().numpy()
            y_next_achieved = plant.get_y(current_state, (k + 1) * dt)
            if y_next_achieved.ndim == 1:
                y_next_achieved = y_next_achieved.unsqueeze(-1)
            y_achieved[:, k + 1, :] = y_next_achieved.cpu().numpy()

        # METRICS COMPUTATION FOR CURRENT MODEL
        y_ref_eval = y_ref_np[:, start_idx:end_idx, :]
        y_achieved_eval = y_achieved[:, start_idx:end_idx, :]
        u_ref_eval = u_ref_np[:, start_idx:end_idx, :]
        u_applied_eval = u_applied[:, start_idx:end_idx, :]

        total_mse = float(np.mean((y_ref_eval - y_achieved_eval) ** 2))
        total_rmse = float(np.sqrt(total_mse))
        control_mse = float(np.mean((u_ref_eval - u_applied_eval) ** 2))

        summary_records.append({
            "Model": model_name,
            "Tracking MSE": total_mse,
            "Tracking RMSE": total_rmse,
            "Control MSE": control_mse
        })

        # Store trajectory results up to end_idx
        u_applied_dict[model_name] = u_applied[:, :end_idx, :]
        y_achieved_dict[model_name] = y_achieved[:, :end_idx, :]
        states_achieved_dict[model_name] = states_achieved[:, :end_idx, :]

    # --- 5. SUMMARY DATAFRAME GENERATION ---
    summary_df = pd.DataFrame(summary_records).sort_values(by="Tracking RMSE").reset_index(drop=True)
    os.makedirs(dirname, exist_ok=True)
    save_df_to_csv(summary_df, 
                   dirname=dirname,             filename=f"multi_model_{mode}_summary.csv")

    print(f"\n==================================================")
    print(f"🏆 MULTI-MODEL {mode.upper()} EVALUATION RANKINGS")
    print(f"==================================================")
    print(summary_df.to_string(index=False))
    print(f"==================================================\n")

    # --- 6. PLOT COMPARATIVE TRAJECTORIES ---
    t_full = np.arange(0, end_idx) * dt

    plot_closed_loop_trajectories_multi(
        t=t_full,
        u_ref=u_ref_np[:, :end_idx, :],
        u_applied_dict=u_applied_dict,
        y_ref=y_ref_np[:, :end_idx, :],
        y_achieved_dict=y_achieved_dict,
        states_achieved_dict=states_achieved_dict,
        states_ref=None,
        plant=plant,
        dirname=dirname,
        show=show_plots
    )

    return {
        "summary_df": summary_df,
        "u_applied_dict": u_applied_dict,
        "y_achieved_dict": y_achieved_dict,
        "states_achieved_dict": states_achieved_dict
    }









def construct_feature_vector(k, y_ref, y_src, u_src, nu_y, nu_u):
    """
    Constructs feature frame v_k at step k with zero-padding 
    if history length is smaller than nu_y or nu_u.
    """
    N = y_ref.shape[0]
    y_next_ref = y_ref[:, k + 1, :]

    # Extract or pad y history (length nu_y + 1)
    if k >= nu_y:
        y_hist = y_src[:, k - nu_y : k + 1, :][:, ::-1, :].reshape(N, -1)
    else:
        available_y = y_src[:, 0 : k + 1, :][:, ::-1, :]
        pad_len = (nu_y + 1) - available_y.shape[1]
        pad = np.zeros((N, pad_len, y_ref.shape[-1]), dtype=np.float32)
        y_hist = np.concatenate([available_y, pad], axis=1).reshape(N, -1)

    # Extract or pad u history (length nu_u)
    if k >= nu_u:
        u_hist = u_src[:, k - nu_u : k, :][:, ::-1, :].reshape(N, -1)
    else:
        if k > 0:
            available_u = u_src[:, 0 : k, :][:, ::-1, :]
            pad_len = nu_u - available_u.shape[1]
            pad = np.zeros((N, pad_len, u_src.shape[-1]), dtype=np.float32)
            u_hist = np.concatenate([available_u, pad], axis=1).reshape(N, -1)
        else:
            u_hist = np.zeros((N, nu_u * u_src.shape[-1]), dtype=np.float32)

    return np.concatenate([y_next_ref, y_hist, u_hist], axis=1)


def get_initial_state_from_config(hyperparam_config, batch_size=1, device="cuda"):
    """
    Parses the initial state list (e.g. [x10, x20, ...]) from hyperparam_config["plant"]["x0"].

    Returns:
    - x0_tensor (torch.Tensor or None): Shaped [batch_size, state_dim] if found, else None.
    """
    plant_cfg = hyperparam_config.get("plant", {})
    
    # Check for 'x0' inside plant dict, or fallback to top-level dict
    x0_list = plant_cfg["initial_state"]

    if x0_list is None:
        return None

    x0_tensor = torch.tensor(x0_list, 
                             dtype=torch.float32, 
                             device=device)
    
    # Reshape from [state_dim] to [batch_size, state_dim]
    return x0_tensor.unsqueeze(0).repeat(batch_size, 1)




#=== FUNCTION TO VALIDATE CONTROLLER ON GIVEN TRAJECTORY ===#
def validate_controller_ext_ref(
    model,
    plant,
    y_ref,
    scaler_x,
    scaler_y,
    hyperparam_config,
    dirname="./plots_ref",
    start_idx=10,
    u_ref=None,
    mode="closed_loop",
    show_plots=False
):
    """
    Evaluates controller tracking against an arbitrary user-supplied reference 
    trajectory y_ref (e.g. generated by `generate_reference_trajectory`).

    Parameters:
    - model: Neural network controller model.
    - plant: Plant class instance.
    - y_ref (Tensor or np.ndarray): Reference output target shaped [steps, output_dim],
      [steps], or [N, steps, output_dim].
    - scaler_x, scaler_y: Feature/target scalers (e.g. StandardScaler / MinMaxScaler).
    - hyperparam_config (dict): Configuration containing dt, nu_y, nu_u, device, etc.
    - dirname (str): Folder path to store resulting plot figures.
    - start_idx (int): Warm-up step index where model takes over control.
    - initial_state (Tensor or np.ndarray, optional): Starting state x_0 of the system.
    - u_ref (Tensor or np.ndarray, optional): Nominal control inputs for warm-up phase (defaults to 0).
    - mode (str): 'closed_loop' or 'open_loop'.
    - show_plots (bool): Whether to display Matplotlib figures interactively.
    """
    #os.makedirs(dirname, exist_ok=True)

    device = hyperparam_config["train"]["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    nu_y = hyperparam_config["training_data_cfg"]["nu_y"]
    nu_u = hyperparam_config["training_data_cfg"]["nu_u"]

    

    # --- 1. FORMAT REFERENCE TARGET (y_ref) ---
    if isinstance(y_ref, torch.Tensor):
        y_ref_raw = y_ref.detach().cpu()
    else:
        y_ref_raw = torch.tensor(y_ref, dtype=torch.float32)

    # Standardize shape to [N, total_seq_len, output_dim]
    if y_ref_raw.ndim == 1:
        y_ref_raw = y_ref_raw.unsqueeze(0).unsqueeze(-1)
    elif y_ref_raw.ndim == 2:
        y_ref_raw = y_ref_raw.unsqueeze(0)

    N, total_seq_len, output_dim = y_ref_raw.shape
    control_dim = hyperparam_config["training_data_cfg"].get("control_dim", 1)
    end_idx = total_seq_len - 1
    y_ref_np = y_ref_raw.numpy()

    u_min_arr, u_max_arr = extract_mimo_control_bounds(hyperparam_config, control_dim)
    # --- 2. FORMAT CONTROL REFERENCE (u_ref) ---
    if u_ref is not None:
        if isinstance(u_ref, torch.Tensor):
            u_ref_np = u_ref.detach().cpu().numpy()
        else:
            u_ref_np = np.array(u_ref, dtype=np.float32)
        if u_ref_np.ndim == 1:
            u_ref_np = np.expand_dims(u_ref_np, axis=(0, -1))
        elif u_ref_np.ndim == 2:
            u_ref_np = np.expand_dims(u_ref_np, axis=0)
    else:
        u_ref_np = np.zeros((N, total_seq_len, control_dim), dtype=np.float32)

    # --- 3. FORMAT INITIAL PLANT STATE ---

    config_x0 = get_initial_state_from_config(hyperparam_config, batch_size=N, device=device)


    current_state = config_x0
    print(f"🌱 Initialized plant state from config: {current_state[0].cpu().tolist()}")


    print("Initial state:", current_state)
    state_dim = current_state.shape[-1]

    # --- 4. ALLOCATE STORAGE ARRAYS ---
    y_achieved = np.zeros((N, total_seq_len, output_dim), dtype=np.float32)
    u_applied = np.zeros((N, total_seq_len, control_dim), dtype=np.float32)
    states_achieved = np.zeros((N, total_seq_len, state_dim), dtype=np.float32)

    # Log initial step k=0
    states_achieved[:, 0, :] = current_state.cpu().numpy()
    y_0 = plant.get_y(current_state, 0.0)
    if y_0.ndim == 1:
        y_0 = y_0.unsqueeze(-1)
    y_achieved[:, 0, :] = y_0.cpu().numpy()

    v_frames_scaled = []

    # --- 5. PHASE 1: WARM-UP (k = 0 to start_idx - 1) ---
    print(f"🔄 Executing warm-up (k=0 to {start_idx - 1}) using nominal inputs...")
    for k in range(0, start_idx):
        t_current = k * dt

        y_src = y_ref_np if mode == "open_loop" else y_achieved
        u_src = u_ref_np if mode == "open_loop" else u_applied

        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        u_k_np = u_ref_np[:, k, :]
        u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)
        u_applied[:, k, :] = u_k_np
        u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

        next_state, _ = plant.step(current_state, u_k_tensor, t=t_current, dt=dt)
        current_state = next_state

        states_achieved[:, k + 1, :] = current_state.cpu().numpy()
        y_next = plant.get_y(current_state, (k + 1) * dt)
        if y_next.ndim == 1:
            y_next = y_next.unsqueeze(-1)
        y_achieved[:, k + 1, :] = y_next.cpu().numpy()

    # --- 6. PHASE 2: MODEL TAKEOVER (k = start_idx to end_idx - 1) ---
    model.eval()
    model.to(device)

    print(f"🚀 Starting sequence model takeover (k={start_idx} to {end_idx - 1})...")
    for k in range(start_idx, end_idx):
        t_current = k * dt

        y_src = y_ref_np if mode == "open_loop" else y_achieved
        u_src = u_ref_np if mode == "open_loop" else u_applied

        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        v_seq_np = np.stack(v_frames_scaled, axis=1)
        v_seq_tensor = torch.tensor(v_seq_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred_seq_scaled = model(v_seq_tensor)

        u_pred_last_scaled = u_pred_seq_scaled[:, -1, :]
        u_k_np = scaler_y.inverse_transform(u_pred_last_scaled.cpu().numpy())

        
        u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)

        u_applied[:, k, :] = u_k_np
        u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

        next_state, _ = plant.step(current_state, u_k_tensor, t=t_current, dt=dt)
        current_state = next_state

        states_achieved[:, k + 1, :] = current_state.cpu().numpy()
        y_next_achieved = plant.get_y(current_state, (k + 1) * dt)
        if y_next_achieved.ndim == 1:
            y_next_achieved = y_next_achieved.unsqueeze(-1)
        y_achieved[:, k + 1, :] = y_next_achieved.cpu().numpy()

    # --- 7. METRICS COMPUTATION ---
    y_ref_eval = y_ref_np[:, start_idx:end_idx, :]
    y_achieved_eval = y_achieved[:, start_idx:end_idx, :]

    total_mse = float(np.mean((y_ref_eval - y_achieved_eval) ** 2))
    total_rmse = float(np.sqrt(total_mse))

    summary_df = pd.DataFrame({
        "Metric": ["Tracking MSE", "Tracking RMSE"],
        "Value": [total_mse, total_rmse]
    })

    print(f"\n==========================================")
    print(f"🔁 {mode.upper()} REFERENCE TRACKING SUMMARY")
    print(f"==========================================")
    print(summary_df.to_string(index=False))
    print(f"==========================================\n")

    # --- 8. PLOTTING TRAJECTORIES ---
    print(f"📊 Generating trajectory plots using plant configurations...")
    t_full = np.arange(0, end_idx) * dt

    plot_closed_loop_trajectories(
        t=t_full,
        u_ref=u_ref_np[:, :end_idx, :],
        u_applied=u_applied[:, :end_idx, :],
        y_ref=y_ref_np[:, :end_idx, :],
        y_achieved=y_achieved[:, :end_idx, :],
        states_achieved=states_achieved[:, :end_idx, :],
        states_ref=None,
        plant=plant,
        dirname=dirname,
        show=show_plots
    )

    return {
        "summary_df": summary_df,
        "metrics": {"tracking_mse": total_mse, "tracking_rmse": total_rmse},
        "simulated_trajectories": {
            "y_ref": y_ref_np,
            "y_achieved": y_achieved,
            "u_applied": u_applied,
            "states_achieved": states_achieved
        }
    }

#=== FUNCTION TO VALIDATE CONTROLLER ON VALIDATION DATA SET ===#
def validate_controller(
    model,
    plant,
    dataset_io,
    scaler_x,
    scaler_y,
    hyperparam_config,
    dirname,
    start_idx,
    mode="closed_loop",
    show_plots=False
):
    """
    Sequence-based validation routine that passes a growing history tensor 
    [N, k + 1, feature_dim] to the model at each step k and drives the plant.
    Plots all sequence trajectories using `plot_closed_loop_trajectories`.
    """
    #os.makedirs(dirname, exist_ok=True)
    
    device = hyperparam_config["train"]["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    nu_y = hyperparam_config["training_data_cfg"]["nu_y"]
    nu_u = hyperparam_config["training_data_cfg"]["nu_u"]
    
    plant_cfg = hyperparam_config.get("plant", {})
    u_min = plant_cfg.get("u_1_hard_min", None)
    u_max = plant_cfg.get("u_1_hard_max", None)

    # Load trajectory dataset
    u_ref_raw = dataset_io["u"].to(dtype=torch.float32)
    y_ref_raw = dataset_io["y"].to(dtype=torch.float32)
    states_raw = dataset_io["states"].to(dtype=torch.float32)

    N, total_seq_len, output_dim = y_ref_raw.shape
    control_dim = u_ref_raw.shape[-1]
    end_idx = total_seq_len - 1

    y_ref_np = y_ref_raw.numpy()
    u_ref_np = u_ref_raw.numpy()
    has_state_ref = (states_raw.ndim == 3)
    states_ref_np = states_raw.numpy() if has_state_ref else None

    # Storage arrays
    y_achieved = np.zeros((N, total_seq_len, output_dim), dtype=np.float32)
    u_applied = np.zeros((N, total_seq_len, control_dim), dtype=np.float32)
    
    initial_state = states_raw[:, 0, :] if has_state_ref else states_raw.clone()
    current_state = initial_state.to(device)
    state_dim = current_state.shape[-1]
    states_achieved = np.zeros((N, total_seq_len, state_dim), dtype=np.float32)

    # Initial state logging
    states_achieved[:, 0, :] = current_state.cpu().numpy()
    y_0 = plant.get_y(current_state, 0.0)
    if y_0.ndim == 1:
        y_0 = y_0.unsqueeze(-1)
    y_achieved[:, 0, :] = y_0.cpu().numpy()

    v_frames_scaled = []

    u_min_arr, u_max_arr = extract_mimo_control_bounds(hyperparam_config, control_dim)

    # --- 1. PHASE 1: WARM-UP (k = 0 to start_idx - 1) ---
    print(f"🔄 Executing warm-up (k=0 to {start_idx - 1}) using reference inputs...")
    for k in range(0, start_idx):
        t_current = k * dt
        
        y_src = y_ref_np if mode == "open_loop" else y_achieved
        u_src = u_ref_np if mode == "open_loop" else u_applied
        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        u_k_np = u_ref_np[:, k, :]
        u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)
        u_applied[:, k, :] = u_k_np
        u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

        next_state, _ = plant.step(current_state, u_k_tensor, t=t_current, dt=dt)
        current_state = next_state

        states_achieved[:, k + 1, :] = current_state.cpu().numpy()
        y_next = plant.get_y(current_state, (k + 1) * dt)
        if y_next.ndim == 1:
            y_next = y_next.unsqueeze(-1)
        y_achieved[:, k + 1, :] = y_next.cpu().numpy()

    # --- 2. PHASE 2: MODEL TAKEOVER (k = start_idx to end_idx - 1) ---
    model.eval()
    model.to(device)

    print(f"🚀 Starting sequence model execution (k={start_idx} to {end_idx - 1})...")

    for k in range(start_idx, end_idx):
        t_current = k * dt

        y_src = y_ref_np if mode == "open_loop" else y_achieved
        u_src = u_ref_np if mode == "open_loop" else u_applied

        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, nu_y, nu_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        v_seq_np = np.stack(v_frames_scaled, axis=1)
        v_seq_tensor = torch.tensor(v_seq_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred_seq_scaled = model(v_seq_tensor)

        u_pred_last_scaled = u_pred_seq_scaled[:, -1, :]

        u_k_np = scaler_y.inverse_transform(u_pred_last_scaled.cpu().numpy())
        
        u_k_np = np.clip(u_k_np, a_min=u_min_arr, a_max=u_max_arr)

        u_applied[:, k, :] = u_k_np
        u_k_tensor = torch.tensor(u_k_np, dtype=torch.float32, device=device)

        next_state, _ = plant.step(current_state, u_k_tensor, t=t_current, dt=dt)
        current_state = next_state

        states_achieved[:, k + 1, :] = current_state.cpu().numpy()
        y_next_achieved = plant.get_y(current_state, (k + 1) * dt)
        if y_next_achieved.ndim == 1:
            y_next_achieved = y_next_achieved.unsqueeze(-1)
        y_achieved[:, k + 1, :] = y_next_achieved.cpu().numpy()

    # --- 3. METRICS COMPUTATION ---
    y_ref_eval = y_ref_np[:, start_idx:end_idx, :]
    y_achieved_eval = y_achieved[:, start_idx:end_idx, :]
    u_ref_eval = u_ref_np[:, start_idx:end_idx, :]
    u_applied_eval = u_applied[:, start_idx:end_idx, :]

    total_mse = float(np.mean((y_ref_eval - y_achieved_eval) ** 2))
    total_rmse = float(np.sqrt(total_mse))
    control_mse = float(np.mean((u_ref_eval - u_applied_eval) ** 2))

    summary_df = pd.DataFrame({
        "Metric": ["Tracking MSE", "Tracking RMSE", "Control MSE"],
        "Value": [total_mse, total_rmse, control_mse]
    })

    print(f"\n==========================================")
    print(f"🔁 {mode.upper()} SEQUENCE VALIDATION SUMMARY")
    print(f"==========================================")
    print(summary_df.to_string(index=False))
    print(f"==========================================\n")

    # --- 4. PLOTTING ALL SEQUENCES ---
    print(f"📊 Generating plots for all {N} sequences using `plot_closed_loop_trajectories`...")
    t_full = np.arange(0, end_idx) * dt

    plot_closed_loop_trajectories(
        t=t_full,
        u_ref=u_ref_np[:, :end_idx, :],
        u_applied=u_applied[:, :end_idx, :],
        y_ref=y_ref_np[:, :end_idx, :],
        y_achieved=y_achieved[:, :end_idx, :],
        states_achieved=states_achieved[:, :end_idx, :],
        states_ref=states_ref_np[:, :end_idx, :] if has_state_ref else None,
        plant=plant,
        dirname=dirname,
        show=show_plots
    )

    return {
        "summary_df": summary_df,
        "metrics": {"tracking_mse": total_mse, "tracking_rmse": total_rmse, "control_mse": control_mse},
        "simulated_trajectories": {
            "y_ref": y_ref_np,
            "y_achieved": y_achieved,
            "u_ref": u_ref_np,
            "u_applied": u_applied,
            "states_ref": states_ref_np,
            "states_achieved": states_achieved
        }
    }



def simulate_tracking_esn(
    model,
    plant,
    r_trajectories,  
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    plot_individual_plots=False
):
    """
    Simulates a controlled MIMO plant using torchdiffeq adaptive-step integration 
    over a specified time horizon while tracking separate reference trajectories using an ESN.
    """
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    esn_cfg = hyperparam_config.get("esn", hyperparam_config.get("plant", {}))
    plant_cfg = hyperparam_config["plant"]

    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    
    # ESN runs on CPU via NumPy, but torchdiffeq can still use your configured device
    device = hyperparam_config["train"]["device"]
    
    input_dim = plant_cfg["input_dim"]    
    output_dim = plant_cfg["output_dim"]  

    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")

    r_trajectory = torch.stack(r_trajectories, dim=1).to(device)  
    r_np = r_trajectory.cpu().numpy()  

    all_y = torch.zeros((steps, batch_size, input_dim), device=device)  
    all_u = torch.zeros((steps, batch_size, output_dim), device=device)  
    
    sample_state = plant.get_initial_state(1)
    state_dim = sample_state.shape[-1]
    all_states = torch.zeros((steps, batch_size, state_dim), device=device)  

    # Force initial state to float32 to protect torchdiffeq execution graph
    state = plant.get_initial_state(batch_size).to(device=device, dtype=torch.float32)

    print(f"📈 Testing ESN MIMO Trajectory Tracking: {batch_size} trajectories across {steps} steps...")

    # ReservoirPy processes historical feedback paths step-by-step per batch index
    history_pairs = [[] for _ in range(batch_size)]

    # Instantiate our localized adaptive step wrapper
   

    # --- 🌀 STEP-BY-STEP CLOSED-LOOP ROLLOUT LOOP ---
    for i in range(steps):
        t_start = i * dt
        t_end = t_start + dt
        
        # 1. Obtain current tracking performance observable
        with torch.no_grad():
            y_current = plant.get_y(state, t_start).to(dtype=torch.float32)  

        target_r = r_trajectory[i].expand(batch_size, input_dim)  
        y_curr_np = y_current.cpu().numpy()  
        tgt_r_np = target_r.cpu().numpy()  

        # 2. Sequential Normalization Processing
        u_unscaled_batch = []
        for b_idx in range(batch_size):
            input_pair = np.hstack([y_curr_np[b_idx], tgt_r_np[b_idx]])  
            input_normalized = x_scaler.transform([input_pair])[0] if x_scaler else input_pair
            
            # ESN expects the historical trajectory up to this time step
            history_pairs[b_idx].append(input_normalized)
            curr_history = np.array(history_pairs[b_idx])  

            # 3. Model inference through ESN forward interface
            # Pass the complete historical trace for this index; take the final output step
            u_seq_norm = model.forward(curr_history)
            u_norm_step = u_seq_norm[-1, :]  # Shape: [output_dim]

            
            u_unscaled = y_scaler.inverse_transform([u_norm_step])[0]
                
            u_unscaled_batch.append(u_unscaled)

        # Convert back to array for hard-clipping and tensor conversion
        u_unscaled_arr = np.array(u_unscaled_batch)

        # 4. Enforce actuator hard clipping profiles 
        u_unscaled_arr = np.clip(u_unscaled_arr, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])  
        u = torch.tensor(u_unscaled_arr, dtype=torch.float32, device=device)  

        # 5. Logging current conditions BEFORE step integration update
        all_y[i] = y_current  
        all_u[i] = u  
        all_states[i] = state  

        # 6. INTEGRATION STEP 
        
        
        with torch.no_grad():
            state, _ = plant.step(state, u, t_start, dt)

        
    # --- PLOTTING & EXPORT BLOCKS (Fully Untouched) ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    total_stacked_blocks = input_dim + output_dim
    
    plot_metadata = plant.get_plot_config()
    
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        #os.makedirs(state_dirname, exist_ok=True)

        y_traj = all_y[:, b, :].cpu().numpy()  
        u_traj = all_u[:, b, :].cpu().numpy()  
        states_traj = all_states[:, b, :].cpu().numpy()  

        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)] + [f"u_{i+1}" for i in range(output_dim)],
                steps
            ),
            "value": np.concatenate([y_traj[:, i] for i in range(input_dim)] + [u_traj[:, i] for i in range(output_dim)]),
        }
        
        for i in range(states_traj.shape[1]):
            df_data[f"state_{i+1}"] = np.tile(states_traj[:, i], total_stacked_blocks)

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        if plot_individual_plots:
            u_meta = next((item for item in plot_metadata if "u" in item["cols"]), {})
            for i in range(output_dim):
                label = u_meta.get("labels", [f"u_{i+1}"])[0] if i == 0 else f"Control Input (u_{i+1})"
                title = u_meta.get("title", "Control Profile") if i == 0 else f"Control Input Profile (u_{i+1})"
                plot_signals(t=time_axis, signals=[u_traj[:, i]], labels=[label], title=f"Trajectory {b}: {title}", xlabel="Time (h)", ylabel=u_meta.get("ylabel", "Action Value"), dirname=state_dirname, filename=f"plot_control_signal_u_{i+1}")

            y_meta = next((item for item in plot_metadata if "y" in item["cols"]), {})
            meta_labels_ind = y_meta.get("labels", [])
            for i in range(input_dim):
                title = y_meta.get("title", "Tracking Performance")
                if len(meta_labels_ind) > (i * 2 + 1):
                    ind_y_label = meta_labels_ind[i * 2]
                    ind_r_label = meta_labels_ind[i * 2 + 1]
                elif len(meta_labels_ind) > i:
                    ind_y_label = meta_labels_ind[i]
                    ind_r_label = f"Target (r_{i+1})"
                else:
                    ind_y_label = f"Output (y_{i+1})"
                    ind_r_label = f"Target (r_{i+1})"
                
                plot_signals(t=time_axis, signals=[y_traj[:, i], r_np[:, i]], labels=[ind_y_label, ind_r_label], title=f"Trajectory {b}: {title} (y_{i+1})", xlabel="Time (h)", ylabel=y_meta.get("ylabel", "Signal Value"), dirname=state_dirname, filename=f"plot_output_tracking_y_{i+1}")

            for i in range(states_traj.shape[1]):
                x_meta = next((item for item in plot_metadata if "x" in item["cols"]), {})
                label = x_meta.get("labels", [f"State x_{i+1}"])[0]
                title = x_meta.get("title", f"Internal Plant State (x_{i+1})")
                plot_signals(t=time_axis, signals=[states_traj[:, i]], labels=[label], title=f"Trajectory {b}: {title}", xlabel="Time (h)", ylabel=x_meta.get("ylabel", "State Magnitude"), dirname=state_dirname, filename=f"plot_plant_state_x_{i+1}")

    y_np = all_y.cpu().numpy()       
    u_np = all_u.cpu().numpy()       
    s_np = all_states.cpu().numpy()  

    y_meta = plot_metadata[2] if len(plot_metadata) > 2 else {}
    meta_labels = y_meta.get("labels", [])
    for i in range(input_dim):
        summary_signals = [y_np[:, j, i] for j in range(batch_size)] + [r_np[:, i]]
        base_y_label = meta_labels[0] if len(meta_labels) > 0 else f"y_{i+1}"
        base_r_label = meta_labels[1] if len(meta_labels) > 1 else f"r_{i+1}"
        plot_signals(t=time_axis, signals=summary_signals, labels=[f"Traj {j} ({base_y_label})" for j in range(batch_size)] + [f"Target ({base_r_label})"], title=f"Batch Convergence ({base_y_label}) - {batch_size} Trajectories Overview", xlabel="Time (h)", ylabel=y_meta.get("ylabel", "System Output"), dirname=dirname, filename=f"batch_summary_y_{i+1}")

    u_meta = plot_metadata[3] if len(plot_metadata) > 3 else {}
    for i in range(output_dim):
        label_base = u_meta.get("labels", [f"u_{i+1}"])[0] if i == 0 else f"u_{i+1}"
        title_base = u_meta.get("title", "Control Profile") if i == 0 else f"Control Input Profile (u_{i+1})"
        summary_inputs = [u_np[:, j, i] for j in range(batch_size)]
        plot_signals(t=time_axis, signals=summary_inputs, labels=[f"Traj {j} ({label_base})" for j in range(batch_size)], title=f"Batch Profile: {title_base} - Overlaid Actions", xlabel="Time (h)", ylabel=u_meta.get("ylabel", "Action Value"), dirname=dirname, filename=f"batch_summary_u_{i+1}")

    for i in range(s_np.shape[2]):
        x_meta = plot_metadata[i] if i < len(plot_metadata) else {}
        label_base = x_meta.get("labels", [f"x_{i+1}"])[0]
        title_base = x_meta.get("title", f"State x_{i+1}")
        summary_states = [s_np[:, j, i] for j in range(batch_size)]
        plot_signals(t=time_axis, signals=summary_states, labels=[f"Traj {j} ({label_base})" for j in range(batch_size)], title=f"Batch Trajectories: {title_base} Ensembles", xlabel="Time (h)", ylabel=x_meta.get("ylabel", "State Magnitude"), dirname=dirname, filename=f"batch_summary_x_{i+1}")

    tracking_metrics = {}
    for i in range(input_dim):
        y_traj = y_np[:, :, i]  
        r_traj = r_np[:, i]  
        metrics = compute_and_save_tracking_metrics(y_traj, r_traj, dt, dirname, suffix=f"y_{i+1}")
        tracking_metrics[f"y_{i+1}"] = metrics

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }





#=== FUNCTION TO GENERATE SMOOTH PROFILE REFERENCE TRAJECTORY ===#
def generate_smooth_profile_trajectory(time_axis, config):
    """
    Generates a reference trajectory that starts flat, rises via a sigmoid,
    stays constant at a peak plateau, sinks via a sigmoid, and stays flat at the floor.
    
    Args:
        time_axis (torch.Tensor): 1D tensor representing the simulation time timeline.
        config (dict): Parameters containing:
            - 'y_floor': Baseline value (e.g., initial concentration)
            - 'y_peak': Upper plateau value
            - 't_rise_center': Time at the midpoint of the rising ramp
            - 'k_rise': Slope/steepness coefficient for the rise phase
            - 't_sink_center': Time at the midpoint of the sinking ramp
            - 'k_sink': Slope/steepness coefficient for the sinking phase
    Returns:
        r_t (torch.Tensor): Smooth trajectory evaluated at each timestep.
    """
    y_floor = config['y_floor']
    y_peak = config['y_peak']
    delta_y = y_peak - y_floor
    
    # 📈 Rising Sigmoid Component
    # Centers the transition around t_rise_center
    sigma_rise = 1.0 / (1.0 + torch.exp(-config['k_rise'] * (time_axis - config['t_rise_center'])))
    
    # 📉 Sinking Sigmoid Component
    # Centers the transition around t_sink_center
    sigma_sink = 1.0 / (1.0 + torch.exp(-config['k_sink'] * (time_axis - config['t_sink_center'])))
    
    # Combined smooth blending expression
    r_t = y_floor + (delta_y * sigma_rise) - (delta_y * sigma_sink)
    
    return r_t

#=== FUNCTION TO GENERATE EXPONENTIALLY DECAYING REFERENCE TRAJECTORY ===#
def generate_exponential_decay_trajectory(steps, 
                                          dt, 
                                          y_start, 
                                          y_target, 
                                          tau, 
                                          device="cuda"):
    """
    Generates a smooth reference trajectory that starts at y_start and 
    exponentially transitions down to y_target governed by time constant tau.
    """
    # Create the time axis
    t_axis = torch.arange(steps, device=device, dtype=torch.float32) * dt
    
    # Compute the exponential curve
    r = y_target + (y_start - y_target) * torch.exp(-t_axis / tau)
    
    # Return with shape [steps, 1] to keep consistency with your other generators
    return r.unsqueeze(-1)

#=== FUNCTION TO GENERATE CONSTANT OR SINUSOIDAL REFERENCE TRAJECTORY ===#
def generate_reference_trajectory(steps, 
                                  dt, 
                                  constant_val, 
                                  amplitude=None,
                                  period=None,
                                  gain=1.0, 
                                  mode="constant",
                                  device="cuda" ):
    """
    Generates reference target trajectories for physical control system tracking simulations.

    Parameters:
    - steps (int): Total sequence length of the reference trajectory.
    - dt (float): Sampling time increment between sequential steps.
    - device (str/torch.device): Target execution hardware device mapping for the output tensor.
    - mode (str): Style of reference generated; choices are "constant" or "dynamic" (default: "constant").
    - constant_val (float): Fixed tracking setpoint used if mode is "constant" (default: 0.3).
    - gain (float): Tuning scaling modifier to sharpen tracking switches if mode is "dynamic" (default: 1.0).
    - period (float): Cyclical hour sequence timeframe for oscillations if mode is "dynamic" (default: 20.0).

    Returns:
    - r_trajectory (torch.Tensor): Reference target values mapped to the execution device, shaped [steps, 1].

    The function acts as an isolated generator for processing setpoints. Depending on the requested 
    mode, it outputs either a uniform static tensor filled with a base setpoint value or computes 
    a dynamic, bounded time-varying profile utilizing transcendental mathematical operators to yield 
    smooth rectangular transitions.
    """
    if mode == "constant":
        # Create a tensor of shape [steps, 1] filled with a static target value
        r_trajectory = torch.full((steps, 1), 
                                  constant_val, 
                                  device=device, 
                                  dtype=torch.float32)
    
    elif mode == "dynamic":
        # Generate time steps array
        time_axis = np.arange(steps) * dt
        
        # Calculate a smooth time-varying waveform base
        sine_base = np.sin(2 * np.pi * time_axis / period)
        
        # Use tanh to sharpen the curve into a "soft" rectangle with an upward linear drift

        noise = 0 #np.random.uniform(-0.005, 0.005, size=time_axis.shape)

        r_trajectory_np = constant_val + amplitude * np.tanh(gain * sine_base) + noise  # Add small random noise for realism
        
        # Convert the structural numpy baseline into a target PyTorch tensor array
        r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)
    
    else:
        raise ValueError(f"Unknown reference mode selection: '{mode}'. Choose 'constant' or 'dynamic'.")

    return r_trajectory