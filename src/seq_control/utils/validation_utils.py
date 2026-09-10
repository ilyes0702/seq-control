"""
Validation Utility Functions
============================

This module contains utilities for 
"""

# Import standard libraries
from typing import Dict, Any, List
import os
import torch
import numpy as np
import pandas as pd

# Import utility functions
from seq_control.utils.plotting_utils import plot_stacked
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.general_utils import *


def _get_channel_labels(cfg, idx, default_prefix, total_dim):
    """
    Extracts (ylabel, legend_base_label) for channel index `idx` from a plot_config section.
    """
    if not cfg:
        return f"${default_prefix}_{{{idx+1}}}$", f"${default_prefix}_{{{idx+1}}}$"

    # 1. Extract y-axis title
    ylabel_val = cfg.get("ylabel", None)
    if isinstance(ylabel_val, list) and idx < len(ylabel_val):
        y_label = ylabel_val[idx]
    elif isinstance(ylabel_val, str):
        y_label = ylabel_val if total_dim == 1 else f"{ylabel_val}_{{{idx+1}}}"
    else:
        y_label = f"${default_prefix}_{{{idx+1}}}$"

    # 2. Extract legend label base
    labels_val = cfg.get("labels", None)
    if isinstance(labels_val, list) and idx < len(labels_val):
        base_label = labels_val[idx]
    elif isinstance(labels_val, str):
        base_label = labels_val
    else:
        base_label = y_label

    return y_label, base_label


def plot_closed_loop_trajectories(
    t,
    u_ref,
    u_applied,
    y_ref,
    y_achieved,
    states_achieved,
    states_ref,
    plant,
    dirname="./plots",
    show=False
):
    """
    Renders a single stacked trajectory comparison plot combining Inputs (u), 
    Outputs (y), and States (x) for ALL sequences in the dataset using `plot_stacked`.
    Dynamically uses axis labels and units from `plant.get_plot_config()`.
    """
    num_sequences = u_ref.shape[0]
    control_dim = u_ref.shape[-1]
    output_dim = y_ref.shape[-1]
    state_dim = states_achieved.shape[-1]

    # --- Extract Plant Plot Configuration ---
    plot_config = plant.get_plot_config() if (plant is not None and hasattr(plant, "get_plot_config")) else None

    xlabel = "Time [s]"
    u_cfg, y_cfg, x_cfg = {}, {}, {}

    if plot_config:
        for item in plot_config:
            cols = item.get("cols", [])
            if any(c == "t" for c in cols):
                xl = item.get("xlabel", item.get("labels", ["Time [s]"]))
                xlabel = xl[0] if isinstance(xl, list) else xl
            elif any(c == "u" or c.startswith("u") for c in cols):
                u_cfg = item
            elif any(c == "y" or c.startswith("y") for c in cols):
                y_cfg = item
            elif any(c == "x" or c.startswith("x") for c in cols):
                x_cfg = item

    # --- Render Plots for Each Sequence ---
    for trace_idx in range(num_sequences):
        trace_tag = f"trace_{trace_idx + 1}"

        signals = []
        labels = []
        ylabels = []

        # ----------------------------------------------------
        # 1. CONTROL INPUTS (u)
        # ----------------------------------------------------
        for ch in range(control_dim):
            y_title, base_lbl = _get_channel_labels(u_cfg, ch, "u", control_dim)
            signals.append([u_ref[trace_idx, :, ch], u_applied[trace_idx, :, ch]])
            labels.append([f"Reference", f"Model prediction"])
            ylabels.append(y_title)

        # ----------------------------------------------------
        # 2. PLANT OUTPUTS (y)
        # ----------------------------------------------------
        for ch in range(output_dim):
            y_title, base_lbl = _get_channel_labels(y_cfg, ch, "y", output_dim)
            signals.append([y_ref[trace_idx, :, ch], y_achieved[trace_idx, :, ch]])
            labels.append([f"{base_lbl} (ref)", f"{base_lbl} (achieved)"])
            ylabels.append(y_title)

        # ----------------------------------------------------
        # 3. SYSTEM STATES (x)
        # ----------------------------------------------------
        for ch in range(state_dim):
            y_title, base_lbl = _get_channel_labels(x_cfg, ch, "x", state_dim)
            if states_ref is not None:
                signals.append([states_ref[trace_idx, :, ch], states_achieved[trace_idx, :, ch]])
                labels.append([f"{base_lbl} (ref)", f"{base_lbl} (achieved)"])
            else:
                signals.append([states_achieved[trace_idx, :, ch]])
                labels.append([f"{base_lbl} (achieved)"])
            
            ylabels.append(y_title)

        # ----------------------------------------------------
        # RENDER COMBINED STACKED PLOT
        # ----------------------------------------------------
        plot_stacked(
            t=t,
            signals=signals,
            labels=labels,
            ylabel=ylabels,
            xlabel=xlabel,
            filename=f"closed_loop_trajectories_{trace_tag}.png",
            dirname=dirname,
            show=show
        )

def construct_feature_vector(k, y_ref, y_src, u_src, n_y, n_u):
    """
    Constructs feature frame v_k at step k with zero-padding 
    if history length is smaller than n_y or n_u.
    """
    N = y_ref.shape[0]
    y_next_ref = y_ref[:, k + 1, :]

    # Extract or pad y history (length n_y + 1)
    if k >= n_y:
        y_hist = y_src[:, k - n_y : k + 1, :][:, ::-1, :].reshape(N, -1)
    else:
        available_y = y_src[:, 0 : k + 1, :][:, ::-1, :]
        pad_len = (n_y + 1) - available_y.shape[1]
        pad = np.zeros((N, pad_len, y_ref.shape[-1]), dtype=np.float32)
        y_hist = np.concatenate([available_y, pad], axis=1).reshape(N, -1)

    # Extract or pad u history (length n_u)
    if k >= n_u:
        u_hist = u_src[:, k - n_u : k, :][:, ::-1, :].reshape(N, -1)
    else:
        if k > 0:
            available_u = u_src[:, 0 : k, :][:, ::-1, :]
            pad_len = n_u - available_u.shape[1]
            pad = np.zeros((N, pad_len, u_src.shape[-1]), dtype=np.float32)
            u_hist = np.concatenate([available_u, pad], axis=1).reshape(N, -1)
        else:
            u_hist = np.zeros((N, n_u * u_src.shape[-1]), dtype=np.float32)

    return np.concatenate([y_next_ref, y_hist, u_hist], axis=1)


def get_initial_state_from_config(hyperparam_config, batch_size=1, device="cpu"):
    """
    Dynamically parses initial state parameters (e.g. x10, x20, x30) from hyperparam_config["plant"].

    Returns:
    - x0_tensor (torch.Tensor or None): Shaped [batch_size, state_dim] if found, else None.
    """
    plant_cfg = hyperparam_config.get("plant", {})

    # Extract and sort keys matching pattern 'x10', 'x20', 'x30'...
    x0_keys = sorted(
        [k for k in plant_cfg.keys() if k.startswith("x") and k.endswith("0") and k[1:-1].isdigit()],
        key=lambda k: int(k[1:-1])
    )

    if x0_keys:
        x0_values = [float(plant_cfg[k]) for k in x0_keys]
        x0_tensor = torch.tensor(x0_values, dtype=torch.float32, device=device)
        return x0_tensor.unsqueeze(0).repeat(batch_size, 1)

    # Fallback to 'x0' or 'initial_state' explicit key if present
    for key in ["x0", "initial_state"]:
        if key in plant_cfg and plant_cfg[key] is not None:
            x0_tensor = torch.tensor(plant_cfg[key], dtype=torch.float32, device=device)
            return x0_tensor.unsqueeze(0).repeat(batch_size, 1)

    return None



import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch

def evaluate_and_plot_mpc(
    mpc_controller,
    plant,
    dataset_dict,
    Y_trajectories,
    U_trajectories,
    dt,
    trace_idx=0,
    plot_config=None,
    dirname="mpc_plots"
):
    """
    Evaluates MPC performance on a specific dataset trace and plots ground-truth 
    vs MPC-predicted outputs, control actions, and internal plant states (TrophophasePlant).
    """
    device = getattr(mpc_controller, "device", "cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(mpc_controller, "model"):
        mpc_controller.model.eval()

    # --- 1. EXTRACT DIMENSIONS & PLANT HELPERS ---
    # Determine dims from wrapper or plant instance safely
    output_dim = getattr(plant, "output_dim", 1)
    control_dim = getattr(plant, "control_dim", getattr(plant, "input_dim", 1))

    # Convert trajectory inputs to NumPy arrays if they are PyTorch tensors
    if isinstance(Y_trajectories, torch.Tensor):
        Y_trajectories = Y_trajectories.detach().cpu().numpy()
    if isinstance(U_trajectories, torch.Tensor):
        U_trajectories = U_trajectories.detach().cpu().numpy()

    # Y_trajectories shape: [Num_Traces, Seq_Len, output_dim]
    # U_trajectories shape: [Num_Traces, Seq_Len, input_dim]
    y_gt_full = Y_trajectories[trace_idx]  # Shape: [Seq_Len, output_dim]
    u_gt_full = U_trajectories[trace_idx]  # Shape: [Seq_Len, control_dim]

    total_len = len(y_gt_full)
    n_y = mpc_controller.n_y
    n_u = mpc_controller.n_u
    start_idx = max(n_y, n_u)
    eval_steps = total_len - start_idx - 1

    # Save original horizon to restore after loop
    H_orig = mpc_controller.H

    # Time vector for horizontal axis [hours]
    t_vec = np.arange(eval_steps) * dt

    # Containers for logging MPC results
    y_mpc = np.zeros((eval_steps, output_dim), dtype=np.float32)
    u_mpc = np.zeros((eval_steps, control_dim), dtype=np.float32)
    x_mpc = []

    # Ground truth evaluation windows
    y_gt = y_gt_full[start_idx + 1 : start_idx + 1 + eval_steps]
    u_gt = u_gt_full[start_idx : start_idx + eval_steps]

    # --- 2. INITIALIZE PLANT STATE ---
    # TrophophasePlant uses get_initial_state instead of reset()
    if hasattr(plant, "get_initial_state"):
        current_state = plant.get_initial_state(batch_size=1, randomize=False)
    elif hasattr(plant, "plant") and hasattr(plant.plant, "get_initial_state"):
        current_state = plant.plant.get_initial_state(batch_size=1, randomize=False)
    elif hasattr(plant, "reset"):
        current_state = plant.reset(batch_size=1)
    else:
        raise AttributeError("Plant instance does not expose `get_initial_state` or `reset`.")

    if not isinstance(current_state, torch.Tensor):
        current_state = torch.tensor(current_state, dtype=torch.float32, device=device)
    else:
        current_state = current_state.to(device)

    # Warm-up feature history vector with ground truth initial steps
    v_history_raw = []
    for k in range(start_idx):
        y_hist = y_gt_full[k - n_y : k + 1].flatten()
        u_hist = u_gt_full[k - n_u : k + 1].flatten()
        v_k = np.concatenate([y_hist, u_hist])
        v_history_raw.append(v_k)

    u_prev_tensor = torch.tensor(
        u_gt_full[start_idx - 1 : start_idx], dtype=torch.float32, device=device
    )
    if u_prev_tensor.ndim == 1:
        u_prev_tensor = u_prev_tensor.unsqueeze(0)

    # --- 3. CLOSED-LOOP MPC EXECUTION LOOP ---
    print(f"🚀 Running MPC evaluation on trace {trace_idx} ({eval_steps} steps)...")

    try:
        for step_i in range(eval_steps):
            k = start_idx + step_i
            t_curr = step_i * dt

            # A. Compute dynamic horizon near end of sequence
            H_actual = min(H_orig, total_len - 1 - k)
            mpc_controller.H = H_actual

            # B. Extract reference target horizon: y_ref [1, H_actual, output_dim]
            y_ref_horizon_np = y_gt_full[k + 1 : k + 1 + H_actual, :][np.newaxis, ...]
            y_ref_horizon = torch.tensor(y_ref_horizon_np, dtype=torch.float32, device=device)

            # C. Build history tensor: [1, seq_len, feature_dim]
            hist_tensor = torch.tensor(
                np.array(v_history_raw), dtype=torch.float32, device=device
            ).unsqueeze(0)

            # D. Solve MPC optimization problem
            u_optimal_k = mpc_controller.solve(
                history_frames=hist_tensor,
                y_ref_horizon=y_ref_horizon,
                u_prev=u_prev_tensor
            )  # Output shape: [control_dim] or [1, control_dim]

            # Standardize action shape to [1, control_dim]
            if u_optimal_k.ndim == 1:
                u_optimal_k_batch = u_optimal_k.unsqueeze(0)
            else:
                u_optimal_k_batch = u_optimal_k

            u_k_np = u_optimal_k_batch.detach().cpu().numpy().flatten()
            u_mpc[step_i] = u_k_np

            # E. Step physical Trophophase plant dynamics (RK45 step with explicit time t)
            next_state, y_k_tensor = plant.step(
                state=current_state, 
                u=u_optimal_k_batch, 
                t=t_curr, 
                dt=dt
            )

            # Log internal biomass (x1) and substrate (x2) mass states
            x_mpc.append(current_state.detach().cpu().numpy().flatten())

            # Log growth rate tracking output
            y_k = y_k_tensor.detach().cpu().numpy().flatten()
            y_mpc[step_i] = y_k

            # F. Update plant state and dynamic feature history for next iteration
            current_state = next_state

            y_hist = np.concatenate([y_gt_full[k - n_y + 1 : k + 1], y_k[np.newaxis, :]], axis=0).flatten()
            u_hist = np.concatenate([u_gt_full[k - n_u + 1 : k + 1], u_k_np[np.newaxis, :]], axis=0).flatten()
            v_next = np.concatenate([y_hist, u_hist])
            v_history_raw.append(v_next)

            u_prev_tensor = u_optimal_k_batch

    finally:
        # Restore controller horizon state
        mpc_controller.H = H_orig

    x_mpc = np.array(x_mpc)

    # --- 4. FORMAT SIGNALS AND LABELS FOR PLOTTING ---
    signals = []
    labels = []
    ylabels = []

    # Growth Rate Output y [1/h]
    for dim in range(output_dim):
        signals.append([y_gt[:, dim], y_mpc[:, dim]])
        labels.append([f"Target Growth Rate $y_{{{dim+1}}}$", f"MPC Tracked $y_{{{dim+1}}}$"])
        ylabels.append(r"Growth Rate $y$ [$\mathrm{h}^{-1}$]")

    # Control Input u (Dilution Rate / Substrate Feed) [1/h]
    for dim in range(control_dim):
        signals.append([u_gt[:, dim], u_mpc[:, dim]])
        labels.append([f"Dataset Action $u_{{{dim+1}}}$", f"MPC Action $u_{{{dim+1}}}$"])
        ylabels.append(r"Feed/Dilution $u$ [$\mathrm{h}^{-1}$]")

    # Internal Plant States: x1 (Biomass Mass [g]) and x2 (Substrate Mass [mg])
    if len(x_mpc) > 0 and x_mpc.ndim > 1:
        state_names = ["Biomass Mass $x_1$", "Substrate Mass $x_2$"]
        state_units = [r"Mass $x_1$ [$\mathrm{g}$]", r"Mass $x_2$ [$\mathrm{mg}$]"]
        num_states = x_mpc.shape[-1]

        for s_dim in range(num_states):
            s_name = state_names[s_dim] if s_dim < len(state_names) else f"State $x_{{{s_dim+1}}}$"
            s_unit = state_units[s_dim] if s_dim < len(state_units) else f"State $x_{{{s_dim+1}}}$"
            
            signals.append([x_mpc[:, s_dim]])
            labels.append([f"Plant {s_name}"])
            ylabels.append(s_unit)

    # Use plant configuration plot overrides if provided
    if plot_config is None and hasattr(plant, "get_plot_config"):
        plot_config = plant.get_plot_config()

    # --- 5. CALL PLOT_STACKED ---
    img = plot_stacked(
        t=t_vec,
        signals=signals,
        plot_config=plot_config,
        labels=labels,
        title=f"Trophophase Plant MPC Tracking vs Ground Truth (Trace {trace_idx})",
        xlabel=r"Time $t$ [$\mathrm{h}$]",
        ylabel=ylabels,
        filename=f"mpc_eval_trace_{trace_idx}.png",
        dirname=dirname,
        show=True
    )

    return img, {
        "y_gt": y_gt, 
        "y_mpc": y_mpc, 
        "u_gt": u_gt, 
        "u_mpc": u_mpc, 
        "x_mpc": x_mpc, 
        "t": t_vec
    }


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
    - hyperparam_config (dict): Configuration containing dt, n_y, n_u, device, etc.
    - dirname (str): Folder path to store resulting plot figures.
    - start_idx (int): Warm-up step index where model takes over control.
    - initial_state (Tensor or np.ndarray, optional): Starting state x_0 of the system.
    - u_ref (Tensor or np.ndarray, optional): Nominal control inputs for warm-up phase (defaults to 0).
    - mode (str): 'closed_loop' or 'open_loop'.
    - show_plots (bool): Whether to display Matplotlib figures interactively.
    """
    os.makedirs(dirname, exist_ok=True)

    device = hyperparam_config["train"]["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    n_y = hyperparam_config["training_data_cfg"]["n_y"]
    n_u = hyperparam_config["training_data_cfg"]["n_u"]

    plant_cfg = hyperparam_config.get("plant", {})
    u_min = plant_cfg.get("u_1_hard_min", None)
    u_max = plant_cfg.get("u_1_hard_max", None)

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

        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, n_y, n_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        u_k_np = u_ref_np[:, k, :]
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

        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, n_y, n_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        v_seq_np = np.stack(v_frames_scaled, axis=1)
        v_seq_tensor = torch.tensor(v_seq_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred_seq_scaled = model(v_seq_tensor)

        u_pred_last_scaled = u_pred_seq_scaled[:, -1, :]
        u_k_np = scaler_y.inverse_transform(u_pred_last_scaled.cpu().numpy())

        if u_min is not None or u_max is not None:
            u_k_np = np.clip(u_k_np, a_min=u_min, a_max=u_max)

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
    os.makedirs(dirname, exist_ok=True)
    
    device = hyperparam_config["train"]["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    n_y = hyperparam_config["training_data_cfg"]["n_y"]
    n_u = hyperparam_config["training_data_cfg"]["n_u"]
    
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

    # --- 1. PHASE 1: WARM-UP (k = 0 to start_idx - 1) ---
    print(f"🔄 Executing warm-up (k=0 to {start_idx - 1}) using reference inputs...")
    for k in range(0, start_idx):
        t_current = k * dt
        
        y_src = y_ref_np if mode == "open_loop" else y_achieved
        u_src = u_ref_np if mode == "open_loop" else u_applied
        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, n_y, n_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        u_k_np = u_ref_np[:, k, :]
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

        v_k = construct_feature_vector(k, y_ref_np, y_src, u_src, n_y, n_u)
        v_k_scaled = scaler_x.transform(v_k)
        v_frames_scaled.append(v_k_scaled)

        v_seq_np = np.stack(v_frames_scaled, axis=1)
        v_seq_tensor = torch.tensor(v_seq_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred_seq_scaled = model(v_seq_tensor)

        u_pred_last_scaled = u_pred_seq_scaled[:, -1, :]

        u_k_np = scaler_y.inverse_transform(u_pred_last_scaled.cpu().numpy())
        if u_min is not None or u_max is not None:
            u_k_np = np.clip(u_k_np, a_min=u_min, a_max=u_max)

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

def simulate_tracking_stateful(
    model,
    plant,
    val_data,  # Tuple (Y_val, U_val, X_val) or Dict {"y": ..., "u": ..., "x": ...}
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    mode="closed_loop",  # Options: "open_loop" or "closed_loop"
    plot_individual_plots=True,
):
    """Simulates trajectory tracking using validation data sequences in either

    Open-Loop or Closed-Loop mode, compares outputs, inputs, and state
    trajectories, and generates stacked comparison plots.

    Parameters:
    -----------
    model : nn.Module
        Trained inverse controller model.
    plant : class or object instance
        Physical plant model simulation environment.
    val_data : tuple or dict
        Validation set containing reference trajectories Y_val, controls U_val,
        and state trajectories X_val.
    hyperparam_config : dict
        Hyperparameter configuration dictionary.
    x_scaler, y_scaler : StandardScaler
        Fitted scalers for inputs (v_k features) and outputs (control signals
        u).
    dirname : str
        Directory path to save reports and figures.
    mode : str
        'closed_loop' (iterative prediction with real-time plant feedback) or
        'open_loop' (direct batch sequence prediction).
    plot_individual_plots : bool
        Whether to generate stacked comparison plots for every validation
        sequence.
    """
    # --- 1. UNPACK VALIDATION DATA ---
    X_val = None
    if isinstance(val_data, dict):
        Y_val = val_data["y"]
        U_val = val_data.get("u", None)
        X_val = val_data.get("x", None)
    elif isinstance(val_data, (tuple, list)):
        Y_val = val_data[0]
        U_val = val_data[1] if len(val_data) > 1 else None
        X_val = val_data[2] if len(val_data) > 2 else None
    else:
        raise ValueError(
            "val_data must be a tuple/list (Y_val, U_val, X_val) or dict with 'y', 'u', 'x' keys."
        )

    # Ensure Y_val tensor format: [batch_size, steps, input_dim]
    if not isinstance(Y_val, torch.Tensor):
        Y_val = torch.tensor(Y_val, dtype=torch.float32)

    batch_size, steps, input_dim = Y_val.shape

    # Format U_val and X_val if provided
    if U_val is not None and not isinstance(U_val, torch.Tensor):
        U_val = torch.tensor(U_val, dtype=torch.float32)

    if X_val is not None and not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)

    # Extract configuration
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    training_data_cfg = hyperparam_config["training_data_cfg"]

    dt = training_data_cfg["dt"]
    device = train_cfg["device"]
    output_dim = plant_cfg["output_dim"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]

    model.eval()

    # --- 2. HANDLE PLANT INSTANTIATION & INITIAL STATE ---
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # Initialize initial state x_0 using X_val[:, 0, :] if available
    if X_val is not None:
        init_state = X_val[:, 0, :].to(device=device, dtype=torch.float32)
    else:
        init_state = plant_instance.get_initial_state(batch_size)

    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]

    print(f"\n🚀 Running Validation Tracking ({mode.upper()} MODE)")
    print(
        f"📊 Evaluating {batch_size} validation trajectories over {steps} time steps..."
    )
    if X_val is not None:
        print(f"🔍 Ground truth states detected (state_dim = {state_dim})")

    # =========================================================================
    # MODE A: OPEN-LOOP VALIDATION
    # =========================================================================
    if mode.lower() == "open_loop":
        if hasattr(model, "core") and hasattr(model.core, "return_bc"):
            model.core.return_bc = False

        # Build windowed input sequences from reference data
        v_seqs_raw = []
        for b in range(batch_size):
            y_ref = Y_val[b].cpu().numpy()
            u_gt = (
                U_val[b].cpu().numpy()
                if U_val is not None
                else np.zeros((steps, output_dim))
            )

            v_traj = []
            for i in range(steps):
                next_idx = min(i + 1, steps - 1)
                r_target = y_ref[next_idx]

                y_window = y_ref[max(0, i - n_y) : i + 1]
                if len(y_window) < (n_y + 1):
                    pad_len = (n_y + 1) - len(y_window)
                    y_window = np.pad(
                        y_window, ((pad_len, 0), (0, 0)), mode="edge"
                    )

                y_hist_rev = y_window[::-1].flatten()

                if n_u > 0:
                    u_window = u_gt[max(0, i - n_u) : i]
                    if len(u_window) < n_u:
                        pad_len = n_u - len(u_window)
                        u_window = np.pad(
                            u_window, ((pad_len, 0), (0, 0)), mode="edge"
                        )
                    u_hist_rev = u_window[::-1].flatten()
                else:
                    u_hist_rev = np.array([])

                v_k = np.concatenate([r_target, y_hist_rev, u_hist_rev])
                v_traj.append(v_k)

            v_seqs_raw.append(np.array(v_traj))

        v_seqs_raw = np.array(v_seqs_raw)  # [batch_size, steps, v_dim]

        # Standardize features
        b_idx, s_idx, d_idx = v_seqs_raw.shape
        v_flat = v_seqs_raw.reshape(-1, d_idx)
        v_scaled = x_scaler.transform(v_flat).reshape(b_idx, s_idx, d_idx)
        v_tensor = torch.tensor(v_scaled, dtype=torch.float32, device=device)

        # Batch sequence prediction
        with torch.no_grad():
            if hasattr(model, "reset_memory"):
                model.reset_memory(batch_size=batch_size, device=device)

            u_pred_norm = model(v_tensor)
            u_pred_np = u_pred_norm.cpu().numpy()

        u_pred_flat = u_pred_np.reshape(-1, output_dim)
        u_unscaled_flat = y_scaler.inverse_transform(u_pred_flat)
        u_unscaled = u_unscaled_flat.reshape(batch_size, steps, output_dim)
        u_unscaled = np.clip(
            u_unscaled,
            plant_cfg["u_1_hard_min"],
            plant_cfg["u_1_hard_max"],
        )

        # Simulate state and output responses in open-loop
        all_y = torch.zeros((steps, batch_size, input_dim), device=device)
        all_states = torch.zeros(
            (steps, batch_size, state_dim), device=device
        )
        all_u = torch.tensor(
            u_unscaled, dtype=torch.float32, device=device
        ).permute(1, 0, 2)

        state = init_state.clone()
        with torch.no_grad():
            for i in range(steps):
                t = i * dt
                y_curr = plant_instance.get_y(state, t)
                all_y[i] = y_curr
                all_states[i] = state

                u_step = all_u[i]
                if output_dim == 1:
                    state, _ = plant_instance.step(
                        state=state, u=u_step[:, 0:1], t=t, dt=dt
                    )
                else:
                    state, _ = plant_instance.step(
                        state=state, u=u_step, t=t, dt=dt
                    )

    # =========================================================================
    # MODE B: CLOSED-LOOP VALIDATION
    # =========================================================================
    elif mode.lower() == "closed_loop":
        if hasattr(model, "core") and hasattr(model.core, "return_bc"):
            model.core.return_bc = True

        all_y = torch.zeros((steps, batch_size, input_dim), device=device)
        all_u = torch.zeros((steps, batch_size, output_dim), device=device)
        all_states = torch.zeros(
            (steps, batch_size, state_dim), device=device
        )

        state = init_state.clone()

        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=batch_size, device=device)

        warmup_steps = 10
        initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()
        y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
        u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

        with torch.no_grad():
            for i in range(steps):
                t = i * dt
                y_current = plant_instance.get_y(state, t)
                y_curr_np = y_current.cpu().numpy()

                for b in range(batch_size):
                    y_histories[b].append(y_curr_np[b])
                    if len(y_histories[b]) > (n_y + 1):
                        y_histories[b].pop(0)

                next_idx = min(i + 1, steps - 1)
                tgt_r_np = Y_val[:, next_idx, :].cpu().numpy()

                if i < warmup_steps:
                    u_unscaled = 0.5 * np.ones((batch_size, output_dim))
                    u = torch.tensor(
                        u_unscaled, dtype=torch.float32, device=device
                    )
                else:
                    v_k_batch_raw = []
                    for b in range(batch_size):
                        y_window = np.array(y_histories[b])
                        y_hist_reversed = y_window[::-1].flatten()

                        u_window = (
                            np.array(u_histories[b])
                            if n_u > 0
                            else np.array([])
                        )
                        u_hist_reversed = (
                            u_window[::-1].flatten()
                            if n_u > 0
                            else np.array([])
                        )

                        v_k_single = np.concatenate(
                            [tgt_r_np[b], y_hist_reversed, u_hist_reversed]
                        )
                        v_k_batch_raw.append(v_k_single)

                    v_k_batch_raw = np.array(v_k_batch_raw)
                    v_k_scaled = x_scaler.transform(v_k_batch_raw)
                    v_k_tensor = torch.tensor(
                        v_k_scaled, dtype=torch.float32, device=device
                    )

                    u_norm_tensor = model.step(v_k_tensor)
                    u_norm_np = u_norm_tensor.cpu().numpy()
                    u_unscaled = y_scaler.inverse_transform(u_norm_np)

                    u_unscaled = np.clip(
                        u_unscaled,
                        plant_cfg["u_1_hard_min"],
                        plant_cfg["u_1_hard_max"],
                    )
                    u = torch.tensor(
                        u_unscaled, dtype=torch.float32, device=device
                    )

                for b in range(batch_size):
                    u_histories[b].append(u_unscaled[b])
                    if len(u_histories[b]) > n_u:
                        u_histories[b].pop(0)

                if output_dim == 1:
                    state, _ = plant_instance.step(
                        state=state, u=u[:, 0:1], t=t, dt=dt
                    )
                else:
                    try:
                        state, _ = plant_instance.step(
                            state=state, u=u, t=t, dt=dt
                        )
                    except TypeError:
                        kwargs = {
                            f"u{j+1}": u[:, j : j + 1] for j in range(output_dim)
                        }
                        state, _ = plant_instance.step(
                            state=state, t=t, dt=dt, **kwargs
                        )

                all_y[i] = y_current
                all_u[i] = u
                all_states[i] = state
    else:
        raise ValueError("mode parameter must be 'open_loop' or 'closed_loop'.")

    # =========================================================================
    # --- 3. METRICS, PLOTTING, & EXPORT ---
    # =========================================================================
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    trajectory_images = []

    y_np = all_y.cpu().numpy()  # [steps, batch_size, input_dim]
    u_np = all_u.cpu().numpy()  # [steps, batch_size, output_dim]
    s_np = all_states.cpu().numpy()  # [steps, batch_size, state_dim]

    # Reorder to [batch_size, steps, dim]
    s_sim_batch = np.transpose(s_np, (1, 0, 2))
    u_sim_batch = np.transpose(u_np, (1, 0, 2))
    y_sim_batch = np.transpose(y_np, (1, 0, 2))

    r_gt_batch = Y_val.cpu().numpy()  # [batch_size, steps, input_dim]
    u_gt_batch = U_val.cpu().numpy() if U_val is not None else None
    x_gt_batch = X_val.cpu().numpy() if X_val is not None else None

    # --- Generate DataFrames & Plots for each sequence ---
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"{mode}_val_sequence_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj_sim = y_sim_batch[b]  # [steps, input_dim]
        u_traj_sim = u_sim_batch[b]  # [steps, output_dim]
        s_traj_sim = s_sim_batch[b]  # [steps, state_dim]

        total_stacked_blocks = input_dim + output_dim
        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)]
                + [f"u_{i+1}" for i in range(output_dim)],
                steps,
            ),
            "value": np.concatenate(
                [y_traj_sim[:, i] for i in range(input_dim)]
                + [u_traj_sim[:, i] for i in range(output_dim)]
            ),
        }

        # Append simulated states
        for i in range(state_dim):
            df_data[f"state_sim_{i+1}"] = np.tile(
                s_traj_sim[:, i], total_stacked_blocks
            )

        # Append ground-truth states
        if x_gt_batch is not None:
            s_traj_gt = x_gt_batch[b]
            for i in range(state_dim):
                df_data[f"state_gt_{i+1}"] = np.tile(
                    s_traj_gt[:, i], total_stacked_blocks
                )

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(
            df_traj, dirname=state_dirname, filename="state_report"
        )
        trajectory_reports.append(df_traj)

        # --- Stacked Plot Generation via plot_stacked ---
        if plot_individual_plots:
            signals = []
            labels = []
            ylabels = []

            # 1. Fetch plot configuration from plant (or model) if available
            plot_config = None
            if hasattr(plant, "get_plot_config"):
                plot_config = plant.get_plot_config()
            elif hasattr(model, "get_plot_config"):
                plot_config = model.get_plot_config()

            def get_config_ylabel(var_prefix, idx, fallback):
                if not plot_config:
                    return fallback
                for cfg in plot_config:
                    cols = cfg.get("cols", [])
                    if (
                        var_prefix in cols
                        or f"{var_prefix}_{idx+1}" in cols
                        or any(c.startswith(var_prefix) for c in cols)
                    ):
                        yl = cfg.get("ylabel")
                        if isinstance(yl, (list, tuple)):
                            return yl[idx] if idx < len(yl) else " / ".join(yl)
                        elif isinstance(yl, str):
                            return yl
                return fallback

            xlabel = "Time [s]"
            if plot_config:
                for cfg in plot_config:
                    if "xlabel" in cfg:
                        xl = cfg["xlabel"]
                        xlabel = xl[0] if isinstance(xl, (list, tuple)) else xl
                        break

            # 2. Output Signals
            for i in range(input_dim):
                signals.append([y_traj_sim[:, i], r_gt_batch[b, :, i]])
                labels.append(["Simulated", "Reference"])
                ylabels.append(get_config_ylabel("y", i, f"$y_{{{i+1}}}$"))

            # 3. Control Signals
            for j in range(output_dim):
                if u_gt_batch is not None:
                    signals.append([u_traj_sim[:, j], u_gt_batch[b, :, j]])
                    labels.append(["Simulated", "Ground Truth"])
                else:
                    signals.append(u_traj_sim[:, j])
                    labels.append(["Simulated"])
                ylabels.append(get_config_ylabel("u", j, f"$u_{{{j+1}}}$"))

            # 4. State Signals
            for k in range(state_dim):
                if x_gt_batch is not None:
                    signals.append([s_traj_sim[:, k], x_gt_batch[b, :, k]])
                    labels.append(["Simulated", "Ground Truth"])
                else:
                    signals.append(s_traj_sim[:, k])
                    labels.append(["Simulated"])
                ylabels.append(get_config_ylabel("x", k, f"$x_{{{k+1}}}$"))

            plot_filename = f"tracking_stacked_plot_seq_{b}"
            img = plot_stacked(
                t=time_axis,
                signals=signals,
                labels=labels,
                xlabel=xlabel,
                ylabel=ylabels,
                filename=plot_filename,
                dirname=state_dirname,
                asp=0.33,
                hspace=0.08,
            )
            trajectory_images.append(img)

    # =========================================================================
    # --- 4. COMPUTE PER-SEQUENCE METRICS AND DATASET AVERAGES ---
    # =========================================================================
    per_sequence_results = []

    for b in range(batch_size):
        seq_metrics = {
            "sequence_id": b,
            "outputs": {},
            "controls": {},
            "states": {},
        }

        # <<< PASTE / REPLACE HERE >>>
        # 1. Output tracking metrics (y_sim vs y_ref)
        for i in range(input_dim):
            y_sim = y_sim_batch[b, :, i][:, None]  # Reshape from (steps,) to (steps, 1)
            y_ref = r_gt_batch[b, :, i][:, None]
            seq_metrics["outputs"][f"y_{i+1}"] = compute_and_save_tracking_metrics(
                y_sim, y_ref, dt, dirname=None, suffix=f"y_{i+1}_seq_{b}"
            )

        # 2. Control signal metrics (u_sim vs u_gt)
        if u_gt_batch is not None:
            for j in range(output_dim):
                u_sim = u_sim_batch[b, :, j][:, None]  # Reshape to (steps, 1)
                u_gt = u_gt_batch[b, :, j][:, None]
                seq_metrics["controls"][f"u_{j+1}"] = compute_and_save_tracking_metrics(
                    u_sim, u_gt, dt, dirname=None, suffix=f"u_{j+1}_seq_{b}"
                )

        # 3. State tracking metrics (x_sim vs x_gt)
        if x_gt_batch is not None:
            for k in range(state_dim):
                x_sim = s_sim_batch[b, :, k][:, None]  # Reshape to (steps, 1)
                x_gt = x_gt_batch[b, :, k][:, None]
                seq_metrics["states"][f"state_{k+1}"] = compute_and_save_tracking_metrics(
                    x_sim, x_gt, dt, dirname=None, suffix=f"state_{k+1}_seq_{b}"
                )
        # <<< END PASTE >>>

        per_sequence_results.append(seq_metrics)

    # Calculate dataset-wide average metrics across all sequences
    dataset_averages = {"outputs": {}, "controls": {}, "states": {}}

    for category in ["outputs", "controls", "states"]:
        # Find all signal keys present for this category
        signal_keys = set()
        for seq in per_sequence_results:
            signal_keys.update(seq[category].keys())

        for sig_key in signal_keys:
            # Collect metric values for this signal across all batches
            all_metric_keys = per_sequence_results[0][category][sig_key].keys()
            avg_metrics = {}
            for m_key in all_metric_keys:
                vals = [
                    seq[category][sig_key][m_key]
                    for seq in per_sequence_results
                    if sig_key in seq[category] and m_key in seq[category][sig_key]
                ]
                avg_metrics[m_key] = float(np.mean(vals)) if len(vals) > 0 else np.nan

            dataset_averages[category][sig_key] = avg_metrics

    # =========================================================================
    # --- 5. BUILD & SAVE DATAFRAMES ---
    # =========================================================================
    # 1. Detailed per-sequence metrics DataFrame
    per_seq_rows = []
    for seq in per_sequence_results:
        seq_id = seq["sequence_id"]
        for category in ["outputs", "controls", "states"]:
            for signal_name, metrics in seq[category].items():
                per_seq_rows.append(
                    {"sequence_id": seq_id, "category": category, "signal": signal_name, **metrics}
                )

    df_per_sequence = pd.DataFrame(per_seq_rows)

    # 2. Overall dataset averages DataFrame
    avg_rows = []
    for category, category_dict in dataset_averages.items():
        for signal_name, metrics in category_dict.items():
            avg_rows.append(
                {"category": category, "signal": signal_name, **metrics}
            )

    df_averages = pd.DataFrame(avg_rows)

    # 3. Save CSV summary reports to the main mode directory (`dirname`)
    save_df_to_csv(
        df=df_per_sequence,
        dirname=dirname,
        filename=f"tracking_metrics_per_sequence_{mode}",
    )

    save_df_to_csv(
        df=df_averages,
        dirname=dirname,
        filename=f"tracking_metrics_dataset_averages_{mode}",
    )

    return {
        "mode": mode,
        "trajectory_dataframes": trajectory_reports,
        "trajectory_images": trajectory_images,
        "per_sequence_metrics": per_sequence_results,
        "dataset_averages": dataset_averages,
        "simulated_outputs": y_sim_batch,
        "simulated_controls": u_sim_batch,
        "simulated_states": s_sim_batch,
        "gt_states": x_gt_batch,
    }


def simulate_tracking_stateful_external_ref_trajectory(
    model,
    plant,
    r_trajectories,  # List of reference trajectories, one for each output dimension. Shape: [steps] for each trajectory.
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    plot_individual_plots=False
):
    """
    Simulates a controlled MIMO plant over a specified time horizon using stateful,
    step-by-step inference (carrying forward Mamba hidden states) while tracking a
    separate reference trajectory for each output dimension.
    
    Aligned with the robust plant instantiation and plotting routines of the validation rollout.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    plant_cfg = hyperparam_config["plant"]

    model.core.return_bc = True  # Enable returning B and C
    
    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]
    input_dim = plant_cfg["input_dim"]    # Number of plant outputs (y1, y2, ...)
    output_dim = plant_cfg["output_dim"]  # Number of control inputs (u1, u2, ...)

    # 🌟 1. HANDLE PLANT INSTANTIATION (Class vs. Instance)
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # Validate r_trajectories
    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")

    # Convert r_trajectories to a tensor of shape [steps, input_dim]
    r_trajectory = torch.stack(r_trajectories, dim=1)  # Shape: [steps, input_dim]
    r_np = r_trajectory.cpu().numpy()  # Shape: [steps, input_dim]

    # Initialize GPU tensor buffers
    all_y = torch.zeros((steps, batch_size, input_dim), device=device)  
    all_u = torch.zeros((steps, batch_size, output_dim), device=device)  
    
    # Dynamically query plant state dimensions to avoid hardcoding
    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]
    all_states = torch.zeros((steps, batch_size, state_dim), device=device)  

    state = plant_instance.get_initial_state(batch_size)

    # Prepare model for evaluation mode
    model.eval()
    ssm_history = {
        "step": [], "time": [],
        "A_bar": [], "B_bar": [], "C": [], "dt": []
    }
    print(f"📈 Testing Stateful MIMO Trajectory Tracking: {batch_size} trajectories across {steps} steps...")

    # CORRECT MEMORY INITIALIZATION
    model.reset_memory(batch_size=batch_size, device=device)

    # INITIALIZE SLIDING WINDOW RUNNING BUFFERS FOR THE BATCH
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]

    # 🌟 Calculate the physical lookback threshold
    # Since we need (n_y + 1) past outputs and n_u past controls:
    warmup_steps = 10 # max(n_y + 1, n_u)

    # Seed history buffers with only the very first step instead of dummy-repeating them
    initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()  # [batch_size, input_dim]
    y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
    u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

    # Execute forward tracking simulation
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant_instance.get_y(state, t)  # Shape: [batch_size, input_dim]
            y_curr_np = y_current.cpu().numpy()

            # 1. Update running history with the newly observed plant output
            for b in range(batch_size):
                y_histories[b].append(y_curr_np[b])
                if len(y_histories[b]) > (n_y + 1):
                    y_histories[b].pop(0)

            # Look-ahead: Target reference state for the NEXT time-step (i+1)
            next_idx = min(i + 1, steps - 1)
            target_r = r_trajectory[next_idx].expand(batch_size, input_dim)
            tgt_r_np = target_r.cpu().numpy()

            # 🌟 Determine if we have enough physical history to start model control
            if i < warmup_steps:
                # --- WARMUP PHASE ---
                # Model does not act yet. Use safe default control actions (zeros)
                u_unscaled = 0.5* np.ones((batch_size, output_dim))
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)
                
            else:
                # --- ACTIVE CONTROL PHASE ---
                # We construct the input vector v_k using only genuine accumulated histories
                v_k_batch_raw = []
                for b in range(batch_size):
                    y_window = np.array(y_histories[b])
                    y_hist_reversed = y_window[::-1].flatten()

                    u_window = np.array(u_histories[b]) if n_u > 0 else np.array([])
                    u_hist_reversed = u_window[::-1].flatten() if n_u > 0 else np.array([])

                    # Combine: [Target_Future, Past_Outputs, Past_Controls]
                    v_k_single = np.concatenate([tgt_r_np[b], y_hist_reversed, u_hist_reversed])
                    v_k_batch_raw.append(v_k_single)

                # Convert, scale, and infer
                v_k_batch_raw = np.array(v_k_batch_raw)
                v_k_scaled = x_scaler.transform(v_k_batch_raw)
                v_k_tensor = torch.tensor(v_k_scaled, dtype=torch.float32, device=device)

                u_norm_tensor = model.step(v_k_tensor)
                
                u_norm_np = u_norm_tensor.cpu().numpy() 
                u_unscaled = y_scaler.inverse_transform(u_norm_np)

                # Force physical actuator limits
                u_unscaled = np.clip(u_unscaled, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])  
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)  

            # 2. Update control history with the chosen action
            for b in range(batch_size):
                u_histories[b].append(u_unscaled[b])
                if len(u_histories[b]) > n_u:
                    u_histories[b].pop(0)

            # Step the physical plant forward
            if output_dim == 1:
                state, _ = plant_instance.step(state=state, u=u[:, 0:1], t=t, dt=dt)
            else:
                try:
                    state, _ = plant_instance.step(state=state, u=u, t=t, dt=dt)
                except TypeError:
                    kwargs = {f"u{j+1}": u[:, j:j+1] for j in range(output_dim)}
                    state, _ = plant_instance.step(state=state, t=t, dt=dt, **kwargs)

            # Logging metrics
            all_y[i] = y_current  
            all_u[i] = u  
            all_states[i] = state 

    # --- PLOTTING & EXPORT CONFIGURATION ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    total_stacked_blocks = input_dim + output_dim
    
    # Retrieve the metadata configuration blocks from the plant safely
    plot_metadata = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []

    # Safe lookup logic mirrored from validation sequence
    state_meta = next((c for c in plot_metadata if any(col.startswith("x") for col in c["cols"])), {})
    output_meta = next((c for c in plot_metadata if any(col.startswith("y") for col in c["cols"])), {})
    control_meta = next((c for c in plot_metadata if any(col.startswith("u") for col in c["cols"])), {})
    
    save_to_json(
        data=ssm_history,
        dirname=dirname,          
        filename="ssm_matrices_history"
    )
    
    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj = all_y[:, b, :].cpu().numpy()            # Shape: [steps, input_dim]
        u_traj = all_u[:, b, :].cpu().numpy()            # Shape: [steps, output_dim]
        states_traj = all_states[:, b, :].cpu().numpy()  # Shape: [steps, state_dim]

        # Save DataFrame for this trajectory
        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)] + [f"u_{i+1}" for i in range(output_dim)],
                steps
            ),
            "value": np.concatenate([
                y_traj[:, i] for i in range(input_dim)
            ] + [
                u_traj[:, i] for i in range(output_dim)
            ]),
        }
        
        for i in range(states_traj.shape[1]):
            df_data[f"state_{i+1}"] = np.tile(states_traj[:, i], total_stacked_blocks)

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        if plot_individual_plots:
            # 1. Individual Plot: Control Signals
            for i in range(output_dim):
                labels_list = control_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"Control Input (u_{i+1})"
                title = control_meta.get("title", "Control Profile")
                
                plot_signals(
                    t=time_axis,
                    signals=[u_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=control_meta.get("ylabel", "Action Value"),
                    dirname=state_dirname,
                    filename=f"plot_control_signal_u_{i+1}"
                )

            # 2. Individual Plot: Output tracking performance
            for i in range(input_dim):
                labels_list = output_meta.get("labels", [])
                ind_y_label = labels_list[0] if len(labels_list) > 0 else f"Output (y_{i+1})"
                ind_r_label = labels_list[1] if len(labels_list) > 1 else f"Target (r_{i+1})"
                title = output_meta.get("title", "Tracking Performance")
                
                plot_signals(
                    t=time_axis,
                    signals=[y_traj[:, i], r_np[:, i]],  
                    labels=[ind_y_label, ind_r_label],   
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=output_meta.get("ylabel", "Signal Value"),
                    dirname=state_dirname,
                    filename=f"plot_output_tracking_y_{i+1}"
                )
                
            # 3. Individual Plot: Internal plant states
            for i in range(states_traj.shape[1]):
                labels_list = state_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"State x_{i+1}"
                title = state_meta.get("title", "Internal Plant States")
                
                plot_signals(
                    t=time_axis,
                    signals=[states_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title} - {label}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=state_meta.get("ylabel", "State Magnitude"),
                    dirname=state_dirname,
                    filename=f"plot_plant_state_x_{i+1}"
                )

    # =========================================================================
    # UNIFIED BATCH OVERLAY STACKED PLOT (Outputs + Controls + States)
    # =========================================================================
    y_np = all_y.cpu().numpy()       # Shape: [steps, batch_size, input_dim]
    u_np = all_u.cpu().numpy()       # Shape: [steps, batch_size, output_dim]
    s_np = all_states.cpu().numpy()  # Shape: [steps, batch_size, state_dim]

    # Helper function to query label/ylabel safely from plot_metadata
    def get_meta_info(prefix, index, default_label, default_ylabel):
        for block in plot_metadata:
            cols = block.get("cols", [])
            # Match block by column naming standard (e.g. 'y', 'y_1', 'x_2')
            if any(col == prefix or col.startswith(f"{prefix}_") or col.startswith(f"{prefix}") for col in cols):
                labels = block.get("labels", [])
                ylabel = block.get("ylabel", default_ylabel)
                
                # Fetch output label
                label_val = labels[index] if index < len(labels) else default_label
                
                # Handle cases where 'ylabel' is either a list or a string
                if isinstance(ylabel, list):
                    ylabel_val = ylabel[index] if index < len(ylabel) else default_ylabel
                else:
                    ylabel_val = ylabel if index == 0 else default_ylabel
                
                return label_val, ylabel_val
        
        return default_label, default_ylabel

    # Helper function to extract x-axis label safely
    time_meta = next((c for c in plot_metadata if "t" in c.get("cols", [])), {})
    time_xlabel_list = time_meta.get("xlabel", [r"$t \; / \; \mathrm{s}$"])
    xlabel_str = time_xlabel_list[0] if isinstance(time_xlabel_list, list) and time_xlabel_list else time_meta.get("xlabel", r"$t \; / \; \mathrm{s}$")

    signals_to_plot = []
    labels_to_plot = []
    ylabels_to_plot = []

    # -------------------------------------------------------------------------
    # 1. Stack System Outputs (y)
    # -------------------------------------------------------------------------
    for i in range(input_dim):
        ref_label, y_label = get_meta_info("y", i, f"Output y_{i+1}", f"$y_{i+1}$")
        
        row_signals = [y_np[:, j, i] for j in range(batch_size)] + [r_np[:, i]]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)] + [f"Reference ({ref_label})"]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(y_label)

    # -------------------------------------------------------------------------
    # 2. Stack Control Actions (u)
    # -------------------------------------------------------------------------
    for i in range(output_dim):
        act_label, u_label = get_meta_info("u", i, f"Control u_{i+1}", f"$u_{i+1}$")
        
        row_signals = [u_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(u_label)

    # -------------------------------------------------------------------------
    # 3. Stack Internal Plant States (x)
    # -------------------------------------------------------------------------
    for i in range(s_np.shape[2]):
        st_label, x_label = get_meta_info("x", i, f"State x_{i+1}", f"$x_{i+1}$")
        
        row_signals = [s_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(x_label)

    # -------------------------------------------------------------------------
    # 4. Render Unified Stacked Figure
    # -------------------------------------------------------------------------
    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=xlabel_str,
        ylabel=ylabels_to_plot,
        asp=[0.25] * len(signals_to_plot),  # Scaled aspect ratio for larger stack
        dirname=dirname,
        filename="batch_summary_system_trajectory_stacked.png",
        show=True
    )

    # --- METRICS COMPILATION ---
    tracking_metrics = {}
    for i in range(input_dim):
        y_traj = y_np[:, :, i]  # Shape: [steps, batch_size]
        r_traj = r_np[:, i]     # Shape: [steps]
        metrics = compute_and_save_tracking_metrics(y_traj, r_traj, dt, dirname, suffix=f"y_{i+1}")
        tracking_metrics[f"y_{i+1}"] = metrics

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
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
        os.makedirs(state_dirname, exist_ok=True)

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


def simulate_tracking_stateful_multi_model(
    models_dict: Dict[str, Dict[str, Any]],  # {"Model_Name": {"model": m, "x_scaler": x_s, "y_scaler": y_s}}
    plant: Any,
    r_trajectories: List[torch.Tensor],       # List of reference trajectories [steps] for each output dim
    hyperparam_config: Dict[str, Any],
    dirname: str,
    plot_individual_plots: bool = False
) -> Dict[str, Any]:
    """
    Simulates and compares multiple controlled MIMO plant inverse models over a 
    specified time horizon using stateful, step-by-step inference on identical initial conditions.
    """
    # Extract global configurations
    train_cfg = hyperparam_config["train"]
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    plant_cfg = hyperparam_config["plant"]

    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]
    input_dim = plant_cfg["input_dim"]    # Plant outputs (y1, y2, ...)
    output_dim = plant_cfg["output_dim"]  # Plant inputs / control actions (u1, u2, ...)

    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    warmup_steps = 10

    # Instantiate Plant
    plant_instance = plant(hyperparam_config) if isinstance(plant, type) else plant

    # Validate and prepare reference trajectory
    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")
    
    r_trajectory = torch.stack(r_trajectories, dim=1).to(device)  # [steps, input_dim]
    r_np = r_trajectory.cpu().numpy()

    # Capture canonical baseline initial state across ALL model runs for absolute fairness
    baseline_initial_state = plant_instance.get_initial_state(batch_size)
    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]

    # Data structures to store results for every model
    model_results = {}

    # =========================================================================
    # 1. RUN SIMULATION FOR EACH MODEL SEPARATELY
    # =========================================================================
    for model_name, model_meta in models_dict.items():
        print(f"\n🚀 Simulating Model: [{model_name}] ({batch_size} trajectories x {steps} steps)...")
        
        model = model_meta["model"]
        x_scaler = model_meta["x_scaler"]
        y_scaler = model_meta["y_scaler"]

        model.eval()
        
        # Agnostic state/memory reset call
        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=batch_size, device=device)
        elif hasattr(model, "reset_hidden_states"):
            model.reset_hidden_states(batch_size=batch_size, device=device)

        if hasattr(model, "core") and hasattr(model.core, "return_bc"):
            model.core.return_bc = True

        # Copy identical initial plant state
        state = baseline_initial_state.clone() if torch.is_tensor(baseline_initial_state) else baseline_initial_state.copy()

        # Allocate storage buffers for this model
        all_y = torch.zeros((steps, batch_size, input_dim), device=device)
        all_u = torch.zeros((steps, batch_size, output_dim), device=device)
        all_states = torch.zeros((steps, batch_size, state_dim), device=device)

        # Seed signal history buffers
        initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()
        y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
        u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

        with torch.no_grad():
            for i in range(steps):
                t = i * dt
                y_current = plant_instance.get_y(state, t)
                y_curr_np = y_current.cpu().numpy()

                # Update output window history
                for b in range(batch_size):
                    y_histories[b].append(y_curr_np[b])
                    if len(y_histories[b]) > (n_y + 1):
                        y_histories[b].pop(0)

                # Look-ahead target reference
                next_idx = min(i + 1, steps - 1)
                target_r = r_trajectory[next_idx].expand(batch_size, input_dim)
                tgt_r_np = target_r.cpu().numpy()

                if i < warmup_steps:
                    # Warmup Phase
                    u_unscaled = 0.5 * np.ones((batch_size, output_dim))
                    u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)
                else:
                    # Model Control Phase
                    v_k_batch_raw = []
                    for b in range(batch_size):
                        y_window = np.array(y_histories[b])
                        y_hist_reversed = y_window[::-1].flatten()

                        u_window = np.array(u_histories[b]) if n_u > 0 else np.array([])
                        u_hist_reversed = u_window[::-1].flatten() if n_u > 0 else np.array([])

                        v_k_single = np.concatenate([tgt_r_np[b], y_hist_reversed, u_hist_reversed])
                        v_k_batch_raw.append(v_k_single)

                    v_k_batch_raw = np.array(v_k_batch_raw)
                    v_k_scaled = x_scaler.transform(v_k_batch_raw)
                    v_k_tensor = torch.tensor(v_k_scaled, dtype=torch.float32, device=device)

                    # Model Step (Works for Mamba, LSTMs, Transformers, MLPs)
                    if hasattr(model, "step"):
                        u_norm_tensor = model.step(v_k_tensor)
                    else:
                        u_norm_tensor = model(v_k_tensor)

                    u_norm_np = u_norm_tensor.cpu().numpy()
                    u_unscaled = y_scaler.inverse_transform(u_norm_np)

                    # Clip to physical limits
                    u_unscaled = np.clip(u_unscaled, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])
                    u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)

                # Update control action history window
                for b in range(batch_size):
                    u_histories[b].append(u_unscaled[b])
                    if len(u_histories[b]) > n_u:
                        u_histories[b].pop(0)

                # Step physical plant
                if output_dim == 1:
                    state, _ = plant_instance.step(state=state, u=u[:, 0:1], t=t, dt=dt)
                else:
                    try:
                        state, _ = plant_instance.step(state=state, u=u, t=t, dt=dt)
                    except TypeError:
                        kwargs = {f"u{j+1}": u[:, j:j+1] for j in range(output_dim)}
                        state, _ = plant_instance.step(state=state, t=t, dt=dt, **kwargs)

                # Store sequence history
                all_y[i] = y_current
                all_u[i] = u
                all_states[i] = state

        # Save model trajectory results
        model_results[model_name] = {
            "y": all_y.cpu().numpy(),       # [steps, batch_size, input_dim]
            "u": all_u.cpu().numpy(),       # [steps, batch_size, output_dim]
            "states": all_states.cpu().numpy()  # [steps, batch_size, state_dim]
        }

    # =========================================================================
    # 2. OVERLAYED PLOTTING ACROSS MODELS
    # =========================================================================
    time_axis = np.arange(steps) * dt
    plot_metadata = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []

    output_meta = next((c for c in plot_metadata if any(col.startswith("y") for col in c["cols"])), {})
    control_meta = next((c for c in plot_metadata if any(col.startswith("u") for col in c["cols"])), {})
    state_meta = next((c for c in plot_metadata if any(col.startswith("x") for col in c["cols"])), {})

    # --- 2A. Stacked Output Tracking Comparison Plot ---
    signals_to_plot, labels_to_plot, ylabels_to_plot = [], [], []

    for i in range(input_dim):
        row_signals, row_labels = [], []
        
        # Add Reference Curve
        row_signals.append(r_np[:, i])
        row_labels.append("Reference Target")

        # Add mean response trajectory for each model
        for model_name, res in model_results.items():
            mean_y_traj = res["y"][:, :, i].mean(axis=1)  # Average over batch
            row_signals.append(mean_y_traj)
            row_labels.append(f"{model_name}")

        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        
        y_labels_list = output_meta.get("ylabel", [])
        row_ylabel = y_labels_list[i] if isinstance(y_labels_list, list) and i < len(y_labels_list) else f"Output {i+1}"
        ylabels_to_plot.append(row_ylabel)

    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=rf"$t \; / \; \mathrm{{s}}$",
        ylabel=ylabels_to_plot,
        asp=[0.33] * len(signals_to_plot),
        dirname=dirname,
        filename="models_comparison_outputs_stacked.png",
        show=True
    )

    # --- 2B. Stacked Control Actions Comparison Plot ---
    signals_to_plot, labels_to_plot, ylabels_to_plot = [], [], []

    for i in range(output_dim):
        row_signals, row_labels = [], []

        for model_name, res in model_results.items():
            mean_u_traj = res["u"][:, :, i].mean(axis=1)
            row_signals.append(mean_u_traj)
            row_labels.append(f"{model_name}")

        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)

        y_labels_list = control_meta.get("ylabel", [])
        row_ylabel = y_labels_list[i] if isinstance(y_labels_list, list) and i < len(y_labels_list) else f"Action {i+1}"
        ylabels_to_plot.append(row_ylabel)

    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=rf"$t \; / \; \mathrm{{s}}$",
        ylabel=ylabels_to_plot,
        asp=[0.33] * len(signals_to_plot),
        dirname=dirname,
        filename="models_comparison_controls_stacked.png",
        show=True
    )

    # =========================================================================
    # 3. METRICS COMPILATION FOR ALL MODELS
    # =========================================================================
    comparison_metrics = {}
    for model_name, res in model_results.items():
        comparison_metrics[model_name] = {}
        for i in range(input_dim):
            y_traj = res["y"][:, :, i]
            r_traj = r_np[:, i]
            metrics = compute_and_save_tracking_metrics(
                y_traj, r_traj, dt, dirname=os.path.join(dirname, model_name), suffix=f"y_{i+1}"
            )
            comparison_metrics[model_name][f"y_{i+1}"] = metrics

    return {
        "model_results": model_results,
        "metrics": comparison_metrics
    }

def simulate_tracking_stateful_external_ref_trajectory(
    model,
    plant,
    r_trajectories,  # List of reference trajectories, one for each output dimension. Shape: [steps] for each trajectory.
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    plot_individual_plots=False
):
    """
    Simulates a controlled MIMO plant over a specified time horizon using stateful,
    step-by-step inference (carrying forward Mamba hidden states) while tracking a
    separate reference trajectory for each output dimension.
    
    Aligned with the robust plant instantiation and plotting routines of the validation rollout.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    plant_cfg = hyperparam_config["plant"]

    model.core.return_bc = True  # Enable returning B and C
    
    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]
    input_dim = plant_cfg["input_dim"]    # Number of plant outputs (y1, y2, ...)
    output_dim = plant_cfg["output_dim"]  # Number of control inputs (u1, u2, ...)

    # 🌟 1. HANDLE PLANT INSTANTIATION (Class vs. Instance)
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # Validate r_trajectories
    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")

    # Convert r_trajectories to a tensor of shape [steps, input_dim]
    r_trajectory = torch.stack(r_trajectories, dim=1)  # Shape: [steps, input_dim]
    r_np = r_trajectory.cpu().numpy()  # Shape: [steps, input_dim]

    # Initialize GPU tensor buffers
    all_y = torch.zeros((steps, batch_size, input_dim), device=device)  
    all_u = torch.zeros((steps, batch_size, output_dim), device=device)  
    
    # Dynamically query plant state dimensions to avoid hardcoding
    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]
    all_states = torch.zeros((steps, batch_size, state_dim), device=device)  

    state = plant_instance.get_initial_state(batch_size)

    # Prepare model for evaluation mode
    model.eval()
    ssm_history = {
        "step": [], "time": [],
        "A_bar": [], "B_bar": [], "C": [], "dt": []
    }
    print(f"📈 Testing Stateful MIMO Trajectory Tracking: {batch_size} trajectories across {steps} steps...")

    # CORRECT MEMORY INITIALIZATION
    model.reset_memory(batch_size=batch_size, device=device)

    # INITIALIZE SLIDING WINDOW RUNNING BUFFERS FOR THE BATCH
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]

    # 🌟 Calculate the physical lookback threshold
    # Since we need (n_y + 1) past outputs and n_u past controls:
    warmup_steps = 10 # max(n_y + 1, n_u)

    # Seed history buffers with only the very first step instead of dummy-repeating them
    initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()  # [batch_size, input_dim]
    y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
    u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

    # Execute forward tracking simulation
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant_instance.get_y(state, t)  # Shape: [batch_size, input_dim]
            y_curr_np = y_current.cpu().numpy()

            # 1. Update running history with the newly observed plant output
            for b in range(batch_size):
                y_histories[b].append(y_curr_np[b])
                if len(y_histories[b]) > (n_y + 1):
                    y_histories[b].pop(0)

            # Look-ahead: Target reference state for the NEXT time-step (i+1)
            next_idx = min(i + 1, steps - 1)
            target_r = r_trajectory[next_idx].expand(batch_size, input_dim)
            tgt_r_np = target_r.cpu().numpy()

            # 🌟 Determine if we have enough physical history to start model control
            if i < warmup_steps:
                # --- WARMUP PHASE ---
                # Model does not act yet. Use safe default control actions (zeros)
                u_unscaled = 0.5* np.ones((batch_size, output_dim))
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)
                
            else:
                # --- ACTIVE CONTROL PHASE ---
                # We construct the input vector v_k using only genuine accumulated histories
                v_k_batch_raw = []
                for b in range(batch_size):
                    y_window = np.array(y_histories[b])
                    y_hist_reversed = y_window[::-1].flatten()

                    u_window = np.array(u_histories[b]) if n_u > 0 else np.array([])
                    u_hist_reversed = u_window[::-1].flatten() if n_u > 0 else np.array([])

                    # Combine: [Target_Future, Past_Outputs, Past_Controls]
                    v_k_single = np.concatenate([tgt_r_np[b], y_hist_reversed, u_hist_reversed])
                    v_k_batch_raw.append(v_k_single)

                # Convert, scale, and infer
                v_k_batch_raw = np.array(v_k_batch_raw)
                v_k_scaled = x_scaler.transform(v_k_batch_raw)
                v_k_tensor = torch.tensor(v_k_scaled, dtype=torch.float32, device=device)

                u_norm_tensor = model.step(v_k_tensor)
                
                u_norm_np = u_norm_tensor.cpu().numpy() 
                u_unscaled = y_scaler.inverse_transform(u_norm_np)

                # Force physical actuator limits
                u_unscaled = np.clip(u_unscaled, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])  
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)  

            # 2. Update control history with the chosen action
            for b in range(batch_size):
                u_histories[b].append(u_unscaled[b])
                if len(u_histories[b]) > n_u:
                    u_histories[b].pop(0)

            # Step the physical plant forward
            if output_dim == 1:
                state, _ = plant_instance.step(state=state, u=u[:, 0:1], t=t, dt=dt)
            else:
                try:
                    state, _ = plant_instance.step(state=state, u=u, t=t, dt=dt)
                except TypeError:
                    kwargs = {f"u{j+1}": u[:, j:j+1] for j in range(output_dim)}
                    state, _ = plant_instance.step(state=state, t=t, dt=dt, **kwargs)

            # Logging metrics
            all_y[i] = y_current  
            all_u[i] = u  
            all_states[i] = state 

    # --- PLOTTING & EXPORT CONFIGURATION ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    total_stacked_blocks = input_dim + output_dim
    
    # Retrieve the metadata configuration blocks from the plant safely
    plot_metadata = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []

    # Safe lookup logic mirrored from validation sequence
    state_meta = next((c for c in plot_metadata if any(col.startswith("x") for col in c["cols"])), {})
    output_meta = next((c for c in plot_metadata if any(col.startswith("y") for col in c["cols"])), {})
    control_meta = next((c for c in plot_metadata if any(col.startswith("u") for col in c["cols"])), {})
    
    save_to_json(
        data=ssm_history,
        dirname=dirname,          
        filename="ssm_matrices_history"
    )
    
    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj = all_y[:, b, :].cpu().numpy()            # Shape: [steps, input_dim]
        u_traj = all_u[:, b, :].cpu().numpy()            # Shape: [steps, output_dim]
        states_traj = all_states[:, b, :].cpu().numpy()  # Shape: [steps, state_dim]

        # Save DataFrame for this trajectory
        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)] + [f"u_{i+1}" for i in range(output_dim)],
                steps
            ),
            "value": np.concatenate([
                y_traj[:, i] for i in range(input_dim)
            ] + [
                u_traj[:, i] for i in range(output_dim)
            ]),
        }
        
        for i in range(states_traj.shape[1]):
            df_data[f"state_{i+1}"] = np.tile(states_traj[:, i], total_stacked_blocks)

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        if plot_individual_plots:
            # 1. Individual Plot: Control Signals
            for i in range(output_dim):
                labels_list = control_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"Control Input (u_{i+1})"
                title = control_meta.get("title", "Control Profile")
                
                plot_signals(
                    t=time_axis,
                    signals=[u_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=control_meta.get("ylabel", "Action Value"),
                    dirname=state_dirname,
                    filename=f"plot_control_signal_u_{i+1}"
                )

            # 2. Individual Plot: Output tracking performance
            for i in range(input_dim):
                labels_list = output_meta.get("labels", [])
                ind_y_label = labels_list[0] if len(labels_list) > 0 else f"Output (y_{i+1})"
                ind_r_label = labels_list[1] if len(labels_list) > 1 else f"Target (r_{i+1})"
                title = output_meta.get("title", "Tracking Performance")
                
                plot_signals(
                    t=time_axis,
                    signals=[y_traj[:, i], r_np[:, i]],  
                    labels=[ind_y_label, ind_r_label],   
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=output_meta.get("ylabel", "Signal Value"),
                    dirname=state_dirname,
                    filename=f"plot_output_tracking_y_{i+1}"
                )
                
            # 3. Individual Plot: Internal plant states
            for i in range(states_traj.shape[1]):
                labels_list = state_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"State x_{i+1}"
                title = state_meta.get("title", "Internal Plant States")
                
                plot_signals(
                    t=time_axis,
                    signals=[states_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title} - {label}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=state_meta.get("ylabel", "State Magnitude"),
                    dirname=state_dirname,
                    filename=f"plot_plant_state_x_{i+1}"
                )

    # =========================================================================
    # UNIFIED BATCH OVERLAY STACKED PLOT (Outputs + Controls + States)
    # =========================================================================
    y_np = all_y.cpu().numpy()       # Shape: [steps, batch_size, input_dim]
    u_np = all_u.cpu().numpy()       # Shape: [steps, batch_size, output_dim]
    s_np = all_states.cpu().numpy()  # Shape: [steps, batch_size, state_dim]

    # Helper function to query label/ylabel safely from plot_metadata
    def get_meta_info(prefix, index, default_label, default_ylabel):
        for block in plot_metadata:
            cols = block.get("cols", [])
            # Match block by column naming standard (e.g. 'y', 'y_1', 'x_2')
            if any(col == prefix or col.startswith(f"{prefix}_") or col.startswith(f"{prefix}") for col in cols):
                labels = block.get("labels", [])
                ylabel = block.get("ylabel", default_ylabel)
                
                # Fetch output label
                label_val = labels[index] if index < len(labels) else default_label
                
                # Handle cases where 'ylabel' is either a list or a string
                if isinstance(ylabel, list):
                    ylabel_val = ylabel[index] if index < len(ylabel) else default_ylabel
                else:
                    ylabel_val = ylabel if index == 0 else default_ylabel
                
                return label_val, ylabel_val
        
        return default_label, default_ylabel

    # Helper function to extract x-axis label safely
    time_meta = next((c for c in plot_metadata if "t" in c.get("cols", [])), {})
    time_xlabel_list = time_meta.get("xlabel", [r"$t \; / \; \mathrm{s}$"])
    xlabel_str = time_xlabel_list[0] if isinstance(time_xlabel_list, list) and time_xlabel_list else time_meta.get("xlabel", r"$t \; / \; \mathrm{s}$")

    signals_to_plot = []
    labels_to_plot = []
    ylabels_to_plot = []

    # -------------------------------------------------------------------------
    # 1. Stack System Outputs (y)
    # -------------------------------------------------------------------------
    for i in range(input_dim):
        ref_label, y_label = get_meta_info("y", i, f"Output y_{i+1}", f"$y_{i+1}$")
        
        row_signals = [y_np[:, j, i] for j in range(batch_size)] + [r_np[:, i]]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)] + [f"Reference ({ref_label})"]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(y_label)

    # -------------------------------------------------------------------------
    # 2. Stack Control Actions (u)
    # -------------------------------------------------------------------------
    for i in range(output_dim):
        act_label, u_label = get_meta_info("u", i, f"Control u_{i+1}", f"$u_{i+1}$")
        
        row_signals = [u_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(u_label)

    # -------------------------------------------------------------------------
    # 3. Stack Internal Plant States (x)
    # -------------------------------------------------------------------------
    for i in range(s_np.shape[2]):
        st_label, x_label = get_meta_info("x", i, f"State x_{i+1}", f"$x_{i+1}$")
        
        row_signals = [s_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(x_label)

    # -------------------------------------------------------------------------
    # 4. Render Unified Stacked Figure
    # -------------------------------------------------------------------------
    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=xlabel_str,
        ylabel=ylabels_to_plot,
        asp=[0.25] * len(signals_to_plot),  # Scaled aspect ratio for larger stack
        dirname=dirname,
        filename="batch_summary_system_trajectory_stacked.png",
        show=True
    )

    # --- METRICS COMPILATION ---
    tracking_metrics = {}
    for i in range(input_dim):
        y_traj = y_np[:, :, i]  # Shape: [steps, batch_size]
        r_traj = r_np[:, i]     # Shape: [steps]
        metrics = compute_and_save_tracking_metrics(y_traj, r_traj, dt, dirname, suffix=f"y_{i+1}")
        tracking_metrics[f"y_{i+1}"] = metrics

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }



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


def generate_exponential_decay_trajectory(steps, dt, y_start, y_target, tau, device="cpu"):
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

#=== FUNCTION TO GENERATE REFERENCE TRAJECTORY ===#
def generate_reference_trajectory(steps, dt, device, mode="constant", constant_val=0.3, gain=1.0, period=20.0):
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
        r_trajectory = torch.full((steps, 1), constant_val, device=device, dtype=torch.float32)
    
    elif mode == "dynamic":
        # Generate time steps array
        time_axis = np.arange(steps) * dt
        
        # Calculate a smooth time-varying waveform base
        sine_base = np.sin(2 * np.pi * time_axis / period)
        
        # Use tanh to sharpen the curve into a "soft" rectangle with an upward linear drift

        noise = 0 #np.random.uniform(-0.005, 0.005, size=time_axis.shape)

        #r_trajectory_np = 0.25 + 0.04 * np.tanh(gain * sine_base) - 0.00 * time_axis   # Add small random noise for realism #chemostat

        r_trajectory_np = 0.02 + 0.005 * np.tanh(gain * sine_base) - 0.00 * time_axis +noise  # Add small random noise for realism
        
        # Convert the structural numpy baseline into a target PyTorch tensor array
        r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)
    
    else:
        raise ValueError(f"Unknown reference mode selection: '{mode}'. Choose 'constant' or 'dynamic'.")

    return r_trajectory