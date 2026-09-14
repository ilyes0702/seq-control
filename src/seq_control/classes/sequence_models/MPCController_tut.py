import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from seq_control.utils.plotting_utils import *

import matplotlib.pyplot as plt
plt.style.use("src/seq_control/style.mplstyle")

class SysIdFeatureBuffer:

    def __init__(self, n_y, n_u, output_dim, input_dim, device="cuda"):
        """Maintains dynamic sliding history buffers for online MPC evaluation.

        Mirrors create_sliced_window_dataset_sysid indexing:
        - y_hist: [y_{k-n_y}, ..., y_k]      -> (n_y + 1) * output_dim
        - u_hist: [u_{k-n_u}, ..., u_k]      -> (n_u + 1) * input_dim
        """
        self.n_y = n_y
        self.n_u = n_u
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.device = device

        self.y_buffer = None  # Stores past (n_y + 1) outputs
        self.u_buffer = None  # Stores past n_u inputs

    def init_from_trace(self, y_trace, u_trace, start_idx):
        """Warm-starts the buffer using historical dataset up to start_idx

        (start_idx >= max(n_y, n_u)).
        """
        if torch.is_tensor(y_trace):
            y_trace = y_trace.detach().cpu().numpy()
        if torch.is_tensor(u_trace):
            u_trace = u_trace.detach().cpu().numpy()

        # Extract exact slices matching offline start window
        # y_buffer receives y[start_idx - n_y : start_idx + 1] -> length (n_y + 1)
        y_slice = y_trace[start_idx - self.n_y : start_idx + 1]
        # u_buffer receives u[start_idx - n_u : start_idx]     -> length n_u (u_k appended during get_feature_tensor)
        u_slice = u_trace[start_idx - self.n_u : start_idx]

        self.y_buffer = torch.tensor(
            y_slice, dtype=torch.float32, device=self.device
        )
        self.u_buffer = torch.tensor(
            u_slice, dtype=torch.float32, device=self.device
        )

    def get_feature_tensor(self, u_k):
        """Assembles the feature vector v_k = [y_hist, u_hist] for candidate

        action u_k.

        :param u_k: Candidate control input tensor [batch_size, input_dim] or
        [input_dim]
        :return: v_k feature tensor matching training layout
        """
        if not torch.is_tensor(u_k):
            u_k = torch.tensor(u_k, dtype=torch.float32, device=self.device)

        # Standardize u_k shape to [batch_size, input_dim]
        if u_k.dim() == 1:
            u_k = u_k.unsqueeze(0)

        batch_size = u_k.shape[0]

        # 1. Prepare y_hist: [y_{k-n_y}, ..., y_k] flattened -> shape: [batch_size, (n_y + 1) * output_dim]
        y_flat = self.y_buffer.view(-1)  # Flatten past history
        y_hist = y_flat.repeat(batch_size, 1)

        # 2. Prepare u_hist: [u_{k-n_u}, ..., u_{k-1}, u_k] -> shape: [batch_size, (n_u + 1) * input_dim]
        u_past_flat = self.u_buffer.view(-1)
        u_past_batch = u_past_flat.repeat(batch_size, 1)
        u_hist = torch.cat([u_past_batch, u_k], dim=-1)

        # 3. Concatenate [y_hist, u_hist] -> shape: [batch_size, feature_dim]
        v_k = torch.cat([y_hist, u_hist], dim=-1)
        return v_k

    def update(self, latest_y, applied_u):
        """Slides the window forward by 1 step after step prediction/execution:

        - Drop oldest y_{k-n_y}, append new y_{k+1}
        - Drop oldest u_{k-n_u}, append applied u_k
        """
        if not torch.is_tensor(latest_y):
            latest_y = torch.tensor(
                latest_y, dtype=torch.float32, device=self.device
            )
        if not torch.is_tensor(applied_u):
            applied_u = torch.tensor(
                applied_u, dtype=torch.float32, device=self.device
            )

        latest_y = latest_y.view(1, self.output_dim)
        applied_u = applied_u.view(1, self.input_dim)

        # Slide y_buffer: Keep last n_y entries, append latest_y
        self.y_buffer = torch.cat([self.y_buffer[1:], latest_y], dim=0)

        # Slide u_buffer: Keep last n_u - 1 entries + applied_u
        if self.n_u > 0:
            self.u_buffer = torch.cat([self.u_buffer[1:], applied_u], dim=0)

    def clone(self):
        """Deep copy of the buffer for isolated MPC forward rollouts."""
        new_buf = SysIdFeatureBuffer(
            self.n_y, self.n_u, self.output_dim, self.input_dim, self.device
        )
        new_buf.y_buffer = self.y_buffer.clone()
        new_buf.u_buffer = self.u_buffer.clone()
        return new_buf

import numpy as np

from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import torch
import torch.nn as nn


def gradient_mpc_compute_action(
    model,
    buffer,
    y_ref_slice,
    H=15,
    num_opt_steps=20,
    lr=0.05,
    u_min=-1.0,
    u_max=1.0,
    u_prev=None,
    u_init=None,
    weight_tracking=1.0,
    weight_rate=0.01,
    device="cuda",
):
    """Computes optimal control action using Gradient-Based MPC (Backpropagation

    Through Time).

    :param model: Surrogate neural network plant model.
    :param buffer: SysIdFeatureBuffer tracking past (y, u) window.
    :param y_ref_slice: Reference trajectory target over horizon H [H,
    output_dim].
    :param H: Prediction horizon steps.
    :param num_opt_steps: Number of Adam gradient descent steps per control
    step.
    :param lr: Learning rate for control parameter optimization.
    :param u_min: Minimum physical actuator boundary.
    :param u_max: Maximum physical actuator boundary.
    :param u_prev: Control input applied at step k-1 (for rate penalty).
    :param u_init: Initial unconstrained control tensor (for warm-starting).
    :return: (u_opt, u_next_init) -> Executable control tensor and warm-start
    state for step k+1.
    """
    input_dim = buffer.input_dim if hasattr(buffer, "input_dim") else 1

    # 1. Initialize or warm-start control sequence parameter U_raw
    if u_init is not None:
        u_raw_tensor = u_init.clone().detach().to(device)
    else:
        u_raw_tensor = torch.zeros(
            (H, input_dim), dtype=torch.float32, device=device
        )

    U_raw = nn.Parameter(u_raw_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([U_raw], lr=lr)

    if not isinstance(y_ref_slice, torch.Tensor):
        y_ref_slice = torch.tensor(
            y_ref_slice, dtype=torch.float32, device=device
        )
    else:
        y_ref_slice = y_ref_slice.to(device)

    model.eval()  # Freeze dropout/batchnorm during rollout

    # 2. Gradient Optimization Loop
    for _ in range(num_opt_steps):
        optimizer.zero_grad()

        # Clone buffer state to isolate optimization rollout from real system history
        temp_buffer = (
            buffer.clone() if hasattr(buffer, "clone") else buffer.copy()
        )

        # Reset recurrent memory if model is stateful (Mamba / LSTM)
        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=1, device=device)

        # Apply smooth tanh bounding: U_raw -> [u_min, u_max]
        U_bounded = u_min + 0.5 * (u_max - u_min) * (torch.tanh(U_raw) + 1.0)

        predicted_outputs = []

        # 3. Unroll Autoregressive Horizon H
        for h in range(H):
            u_h = U_bounded[h : h + 1]  # Shape: [1, input_dim]

            # --- FIX HERE: Extract single feature vector v_h ---
            v_h = temp_buffer.get_feature_tensor(u_h)

            # Pass only v_h to model.step
            step_out = model.step(v_h)

            y_next = (
                step_out[0]
                if isinstance(step_out, (tuple, list))
                else step_out
            )

            y_next_flat = y_next.squeeze(0) if y_next.dim() > 1 else y_next
            predicted_outputs.append(y_next_flat)

            # Update working buffer for step h+1
            temp_buffer.update(latest_y=y_next, applied_u=u_h)

        Y_pred = torch.stack(predicted_outputs, dim=0)  # Shape: [H, output_dim]

        # 4. Compute Loss J
        tracking_loss = torch.mean((Y_pred - y_ref_slice[:H]) ** 2)

        # Slew rate penalty: sum(||u_h - u_{h-1}||^2)
        u_prev_tensor = (
            u_prev if u_prev is not None else U_bounded[0].detach()
        )
        u_diffs = U_bounded[1:] - U_bounded[:-1]
        u_first_diff = U_bounded[0] - u_prev_tensor
        rate_loss = torch.mean(u_first_diff**2) + torch.mean(u_diffs**2)

        total_loss = weight_tracking * tracking_loss + weight_rate * rate_loss

        # 5. Backpropagate Gradients to U_raw
        total_loss.backward()
        optimizer.step()

    # 6. Extract optimal control action u_0* and prepare warm-start for k+1
    with torch.no_grad():
        U_final = u_min + 0.5 * (u_max - u_min) * (torch.tanh(U_raw) + 1.0)
        u_opt = U_final[0].detach()

        # Warm-start shift: shift sequence by 1 step, repeat final element
        u_next_init = torch.cat([U_raw[1:], U_raw[-1:]], dim=0).detach()

    return u_opt, u_next_init


import numpy as np
import torch


def run_dataset_validation(
    model,
    Y_trajectories,
    U_trajectories,
    X_trajectories=None,
    n_y=2,
    n_u=2,
    H=15,
    K=500,
    u_min=-1.0,
    u_max=1.0,
    plot_results=True,
    plot_config=None,
    device="cuda",
):
    """Validates MPC controller tracking accuracy using a surrogate model instead

    of a physical plant simulator and visualizes dataset vs. MPC rollout.
    """
    num_traces, total_seq_len, output_dim = Y_trajectories.shape
    input_dim = U_trajectories.shape[-1]
    has_states = X_trajectories is not None

    start_idx = max(n_y, n_u)
    results = []

    model.eval()

    for trace_idx in range(num_traces):
        y_target_trace = Y_trajectories[trace_idx]  # Target output trajectory
        u_ground_truth = U_trajectories[
            trace_idx
        ]  # Dataset reference input trajectory
        x_ground_truth = X_trajectories[trace_idx] if has_states else None

        # Reset recurrent state if model supports it (e.g., Mamba/RNN)
        if hasattr(model, "reset_memory"):
            model.reset_memory()

        # 1. Initialize feature buffer from historical warm-up window
        buffer = SysIdFeatureBuffer(
            n_y, n_u, output_dim, input_dim, device=device
        )
        buffer.init_from_trace(y_target_trace, u_ground_truth, start_idx)

        y_tracked = []
        u_applied = []
        x_tracked = [] if has_states else None

        u_init = None
        u_prev = None

        # 2. Closed-loop control rollout
        for k in range(start_idx, total_seq_len - 1):
            # --- FIX 1: Extract slice into raw_ref_slice BEFORE tensor check ---
            end_ref = min(k + 1 + H, total_seq_len)
            raw_ref_slice = y_target_trace[k + 1 : end_ref]

            if torch.is_tensor(raw_ref_slice):
                y_ref_slice = raw_ref_slice.clone().detach().to(
                    dtype=torch.float32, device=device
                )
            else:
                y_ref_slice = torch.tensor(
                    raw_ref_slice, dtype=torch.float32, device=device
                )

            # Pad horizon if near boundary
            if y_ref_slice.shape[0] < H:
                pad_count = H - y_ref_slice.shape[0]
                y_ref_slice = torch.cat(
                    [y_ref_slice, y_ref_slice[-1:].repeat(pad_count, 1)], dim=0
                )

            # Compute optimal control action
            u_k, u_init = gradient_mpc_compute_action(
                model=model,
                buffer=buffer,
                y_ref_slice=y_ref_slice,
                H=H,
                num_opt_steps=20,
                lr=0.05,
                u_min=u_min,
                u_max=u_max,
                u_prev=u_prev,
                u_init=u_init,
                device=device,
            )

            u_prev = u_k.detach()

            # Execute step on surrogate model
            with torch.no_grad():
                v_k = buffer.get_feature_tensor(u_k)
                step_out = model.step(v_k)

                y_next = (
                    step_out[0]
                    if isinstance(step_out, (tuple, list))
                    else step_out
                )

            # --- FIX 2: Save predictions and actions for tracking analysis ---
            y_tracked.append(y_next.detach().cpu().numpy().squeeze())
            u_applied.append(u_k.detach().cpu().numpy().squeeze())

            # Update history buffer
            buffer.update(latest_y=y_next, applied_u=u_k)

        # --- FIX 3: Removed stray invalid line 'raw_ref_slice = y_trace[k + 1 : k + 1 + H]' ---

        # 3. Calculate performance metrics for this trace
        y_tracked_arr = np.array(y_tracked)
        u_applied_arr = np.array(u_applied)

        y_ref_arr = y_target_trace[start_idx + 1 :]
        u_ref_arr = u_ground_truth[start_idx + 1 :]

        if torch.is_tensor(y_ref_arr):
            y_ref_arr = y_ref_arr.detach().cpu().numpy()
        if torch.is_tensor(u_ref_arr):
            u_ref_arr = u_ref_arr.detach().cpu().numpy()

        # Compute RMSE
        rmse = np.sqrt(np.mean((y_tracked_arr - y_ref_arr) ** 2))

        trace_result = {
            "trace_idx": trace_idx,
            "rmse": rmse,
            "y_tracked": y_tracked_arr,
            "y_ref": y_ref_arr,
            "u_applied": u_applied_arr,
            "u_ref": u_ref_arr,
        }

        if has_states:
            x_ref_arr = x_ground_truth[start_idx + 1 :]
            if torch.is_tensor(x_ref_arr):
                x_ref_arr = x_ref_arr.detach().cpu().numpy()
            x_tracked_arr = (
                np.array(x_tracked) if x_tracked else np.zeros_like(x_ref_arr)
            )
            trace_result["x_tracked"] = x_tracked_arr
            trace_result["x_ref"] = x_ref_arr

        # 4. Generate visual comparison
        plot_image = None
        if plot_results:
            t = np.arange(start_idx + 1, total_seq_len)

            signals = [
                [u_ref_arr, u_applied_arr],
                [y_ref_arr, y_tracked_arr],
            ]
            default_labels = [
                ["Dataset Input (U)", "MPC Control (U)"],
                ["Dataset Target (Y)", "MPC Output (Y)"],
            ]

            if has_states:
                signals.append([x_ref_arr, trace_result["x_tracked"]])
                default_labels.append(["Dataset State (X)", "MPC State (X)"])

            plot_image = plot_stacked(
                t=t,
                signals=signals,
                plot_config=plot_config,
                labels=default_labels if plot_config is None else None,
                title=f"Trace {trace_idx + 1} - Dataset vs. MPC Surrogate Validation (RMSE: {rmse:.5f})",
                xlabel="Time Step (k)",
                ylabel=[
                    "Input (U)",
                    "Output (Y)",
                    *(["State (X)"] if has_states else []),
                ],
                show=False,
                filename=f"trace_{trace_idx + 1}_validation.png",
            )
            trace_result["plot_image"] = plot_image

        results.append(trace_result)
        print(
            f"Trace {trace_idx + 1}/{num_traces} - Tracking RMSE: {rmse:.5f}"
        )

    return results


from src.seq_control.classes.plants.TrophophasePlant import *
from src.seq_control.classes.sequence_models.MambaInverseController import *
from src.seq_control.utils.saving_and_loading_utils import *

model = load_model(MambaSurrogateModel,"src/seq_control/results/2026-09-11/2026-09-11_16-00-43/fChemostatPlant_sysid/type/fold_1/2026-09-11_16-00-43_best_fold_model.pt")
plant_sim = TrophophasePlant(hyperparam_config_TrophophasePlant)
dataset = torch.load("src/seq_control/results/2026-09-11/2026-09-11_15-30-02/TrophophasePlant/io/dataset/2026-09-11_15-30-02_val_io_data.pt", weights_only=True)
print(dataset.keys())

Y_trajectories = dataset["y"]
U_trajectories = dataset["u"]
X_trajectories = dataset["states"]

run_dataset_validation(
    model,
    Y_trajectories,
    U_trajectories,
    X_trajectories
)