# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams
from src.seq_control.classes.controllers.MPCController import *

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaSysIDModel(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        Mamba-based forward system identification model (NARX-style).
        
        Given a regressor window of past outputs and past+current inputs,
        predicts the plant output at the current time step.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector w_k.
                       If not provided, it will be calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["plant"]["input_dim"]   # Dimension of plant output y (e.g. 2)
        self.output_dim = hyperparam_config["plant"]["output_dim"] # Dimension of plant control u (e.g. 2)
        self.d_state = hyperparam_config["mamba"]["d_state"]
        self.expand = hyperparam_config["mamba"]["expand"]
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["train"]["n_y"]
            n_u = hyperparam_config["train"]["n_u"]
            # w_k = [u_k, u_{k-1} ... u_{k-n_u}, y_{k-1} ... y_{k-n_y}]
            print(self.input_dim)
            print(self.output_dim)
            self.d_model = (n_u + 1) * self.output_dim + n_y * self.input_dim
        print("d_model: ", self.d_model)
        print(f"🛠️ Initializing Mamba core with d_model (feature_dim) = {self.d_model}")
        
        # 3. Instantiate Mamba Core
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=4,
            expand=self.expand
        )
        
        # 4. Map latent features to predicted plant output dimension
        self.output_proj = nn.Linear(self.d_model, self.input_dim)
        
        # Inference memory state buffers
        self.conv_state = None
        self.ssm_state = None

    def forward(self, w_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - w_seq: Tensor of shape [Batch, Seq_Len, d_model] (Contains stacked w_k regressor sequences)
        
        Returns:
        - predicted_y: Tensor of shape [Batch, Seq_Len, input_dim] (Target plant outputs y_k)
        """
        x = self.core(w_seq)  # Shape: [Batch, Seq_Len, d_model]
        return self.output_proj(x)  # Shape: [Batch, Seq_Len, input_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Allocates or resets zero-filled tracking memory tensors.
        Essential for stateful step-by-step rolling simulation.
        """
        d_inner = self.d_model * self.expand
        self.conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        self.ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)

    def step(self, w_k_single):
        """
        Closed-loop / rolling-simulation evaluation step. Consumes the current
        regressor vector (past y's + current & past u's) to predict y_k while
        updating recurrent states.
        
        Parameters:
        - w_k_single: Tensor of shape [Batch, d_model] (or [Batch, 1, d_model])
        
        Returns:
        - y_out: Tensor of shape [Batch, input_dim] (Predicted plant output)
        """
        if self.conv_state is None or self.ssm_state is None:
            raise RuntimeError("Inference states are uninitialized. Please call reset_memory() first.")
            
        if w_k_single.dim() == 2:
            x_3d = w_k_single.unsqueeze(1)
        else:
            x_3d = w_k_single
            
        x_out_3d, self.conv_state, self.ssm_state = self.core.step(
            x_3d, self.conv_state, self.ssm_state
        )
        
        x_out = x_out_3d.squeeze(1)
        return self.output_proj(x_out)  # Shape: [Batch, input_dim]

    # --- PROPERTIES FOR SYSTEM MODEL ANALYSIS ---
    @property
    def A(self):
        return self.core.A
        
    @property
    def B(self):
        return self.core.B

    @property
    def C(self):
        return self.core.C
    
    @property
    def D(self):
        return getattr(self.core, "extracted_D", getattr(self.core, "D", None))
    
    @property
    def mamba_dt(self):
        return getattr(self.core, "extracted_dt", getattr(self.core, "dt", None))



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def prepare_narx_dataset(y_data, u_data, n_y, n_u):
    """
    Constructs 3D NARX regressor sequence tensors from 2D continuous trajectories.
    
    Parameters:
    - y_data: Tensor [Time_Steps, input_dim] (Plant outputs y)
    - u_data: Tensor [Time_Steps, output_dim] (Plant inputs u)
    - n_y: Number of past output lags
    - n_u: Number of past input lags
    
    Returns:
    - w_seq: Tensor [1, Valid_Steps, d_model]
    - y_target: Tensor [1, Valid_Steps, input_dim]
    """
    T, dim_y = y_data.shape
    _, dim_u = u_data.shape
    
    start_idx = max(n_y, n_u)
    valid_steps = T - start_idx
    
    w_list = []
    y_target_list = []
    
    for k in range(start_idx, T):
        # 1. Input lags: u_k down to u_{k - n_u}
        u_lags = [u_data[k - i] for i in range(n_u + 1)]
        u_flat = torch.cat(u_lags, dim=-1)
        
        # 2. Output lags: y_{k-1} down to y_{k - n_y}
        y_lags = [y_data[k - j] for j in range(1, n_y + 1)]
        y_flat = torch.cat(y_lags, dim=-1)
        
        # 3. Concatenate regressor vector w_k
        w_k = torch.cat([u_flat, y_flat], dim=-1)
        
        w_list.append(w_k)
        y_target_list.append(y_data[k])
        
    w_seq = torch.stack(w_list, dim=0).unsqueeze(0)        # Shape: [1, Valid_Steps, d_model]
    y_target = torch.stack(y_target_list, dim=0).unsqueeze(0) # Shape: [1, Valid_Steps, input_dim]
    
    return w_seq, y_target


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold


def train_mamba_sysid_kfold(
    model_cls,
    y_data: torch.Tensor,
    u_data: torch.Tensor,
    hyperparam_config: dict,
    n_splits: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,
    seed: int = 42,
    plot_validation: bool = True,
):
  """Trains MambaSysIDModel using K-Fold Cross-Validation on contiguous sequence chunks

  and plots predicted vs. ground truth outputs for each validation sequence segment.
  """
  device = torch.device(hyperparam_config["train"]["device"])
  n_y = hyperparam_config["train"]["n_y"]
  n_u = hyperparam_config["train"]["n_u"]

  # --- 1. Dataset Formatting & Sliding Window Construction ---
  if y_data.dim() == 2:
    print(
        "⚙️ Preparing single continuous trajectory into NARX regressor"
        " sequence..."
    )
    w_seq, y_target = prepare_narx_dataset(y_data, u_data, n_y, n_u)
    w_seq = w_seq.squeeze(0)  # Shape: [Total_Steps, d_model]
    y_target = y_target.squeeze(0)  # Shape: [Total_Steps, input_dim]
  else:
    print(f"⚙️ Preparing multi-trace dataset ({y_data.shape[0]} traces)...")
    w_list, y_target_list = [], []
    for t in range(y_data.shape[0]):
      w_sub, y_sub = prepare_narx_dataset(y_data[t], u_data[t], n_y, n_u)
      w_list.append(w_sub.squeeze(0))
      y_target_list.append(y_sub.squeeze(0))
    w_seq = torch.cat(w_list, dim=0)
    y_target = torch.cat(y_target_list, dim=0)

  num_samples = w_seq.shape[0]

  # DISABLE shuffle to preserve chronological sequential integrity across validation folds
  kf = KFold(n_splits=n_splits, shuffle=False)

  fold_histories = []
  best_overall_model = None
  best_overall_val_loss = float("inf")

  print(
      f"\n🚀 Starting {n_splits}-Fold Cross-Validation on {num_samples} sequence"
      " timesteps..."
  )

  # --- 2. K-Fold Cross Validation Loop ---
  for fold, (train_idx, val_idx) in enumerate(kf.split(w_seq), start=1):
    print(f"\n==================== FOLD {fold}/{n_splits} ====================")

    # Unsqueeze(0) restores 3D shape [1, Seq_Len, Feature_Dim] expected by Mamba core
    w_train = w_seq[train_idx].unsqueeze(0).to(device)
    y_target_train = y_target[train_idx].unsqueeze(0).to(device)

    w_val = w_seq[val_idx].unsqueeze(0).to(device)
    y_target_val = y_target[val_idx].unsqueeze(0).to(device)

    model = model_cls(hyperparam_config).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {"train_loss": [], "val_loss": []}
    best_fold_val_loss = float("inf")
    best_fold_weights = None

    # --- 3. Training Loop ---
    for epoch in range(1, epochs + 1):
      model.train()
      optimizer.zero_grad()

      y_pred_train = model(w_train)
      loss = criterion(y_pred_train, y_target_train)
      loss.backward()

      if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_grad_norm
        )

      optimizer.step()
      scheduler.step()

      train_loss = loss.item()

      # Validation step
      model.eval()
      with torch.no_grad():
        y_pred_val = model(w_val)
        val_loss = criterion(y_pred_val, y_target_val).item()

      history["train_loss"].append(train_loss)
      history["val_loss"].append(val_loss)

      if val_loss < best_fold_val_loss:
        best_fold_val_loss = val_loss
        best_fold_weights = copy.deepcopy(model.state_dict())

      if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.6f} |"
            f" Val Loss: {val_loss:.6f}"
        )

    print(
        f"📌 Fold {fold} Complete. Best Val Loss: {best_fold_val_loss:.6f}"
    )
    fold_histories.append(history)

    # --- 4. Plot Validation Sequence for Current Fold ---
    if plot_validation:
      # Load the best model weights checkpoint for this fold
      model.load_state_dict(best_fold_weights)
      model.eval()

      with torch.no_grad():
        y_pred_val_seq = model(w_val)

      # Extract continuous validation trajectory arrays
      y_true_seq = y_target_val.squeeze(0).cpu().numpy()
      y_pred_seq = y_pred_val_seq.squeeze(0).cpu().numpy()

      # Select primary channel if multi-dimensional output
      if y_true_seq.ndim > 1:
        y_true_seq = y_true_seq[:, 0]
        y_pred_seq = y_pred_seq[:, 0]

      # Continuous sequence timeline array for the validation fold
      t_val_seq = np.arange(len(val_idx))

      # Render predicted vs real sequence output
      plot_signals(
          t=t_val_seq,
          signals=[y_true_seq, y_pred_seq],
          labels=[
              "Real Output ($y_{val}$)",
              "Predicted Output ($\hat{y}_{val}$)",
          ],
          title=f"Fold {fold}/{n_splits} Validation Sequence Response (Val"
          f" Loss: {best_fold_val_loss:.6f})",
          xlabel="Sequence Step [k]",
          ylabel="Plant Output (y)",
          show=True,
      )

    # Preserve overall champion model across all K folds
    if best_fold_val_loss < best_overall_val_loss:
      best_overall_val_loss = best_fold_val_loss
      best_overall_model = model_cls(hyperparam_config)
      best_overall_model.load_state_dict(best_fold_weights)

  # --- 5. Cross-Validation Summary ---
  val_losses = [np.min(h["val_loss"]) for h in fold_histories]
  print("\n==================================================")
  print(f"✅ {n_splits}-Fold Cross-Validation Complete!")
  print(
      f"   ↳ Average Val Loss: {np.mean(val_losses):.6f} ±"
      f" {np.std(val_losses):.6f}"
  )
  print(f"   ↳ Top Performing Model Val Loss: {best_overall_val_loss:.6f}")
  print("==================================================")

  return best_overall_model, fold_histories


# 1. Hyperparameter Configuration
hyperparam_config = {
    "train": {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "n_y": 2,
        "n_u": 2
    },
    "plant": {
        "input_dim": 1,    # Dimension of y
        "output_dim": 1    # Dimension of u
    },
    "mamba": {
        "d_state": 16,
        "expand": 2
    }
}

# 2. System Identification Synthetic Data Generation
time_steps = 1000
u_data = torch.randn(time_steps, hyperparam_config["plant"]["output_dim"])
y_data = torch.sin(u_data) * 0.5 + torch.randn(time_steps, hyperparam_config["plant"]["input_dim"]) * 0.05

# 3. Execute 5-Fold Cross Validation
best_model, histories = train_mamba_sysid_kfold(
    model_cls=MambaSysIDModel,
    y_data=y_data,
    u_data=u_data,
    hyperparam_config=hyperparam_config,
    n_splits=5,
    epochs=150,
    lr=2e-3
)


import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Import BasePlant from your framework
from src.seq_control.classes.plants.BasePlant import BasePlant

class MambaPlantWrapper(BasePlant):
    def __init__(self, mamba_model: torch.nn.Module, hyperparam_config: dict, device="cpu"):
        super().__init__()
        self.model = mamba_model.to(device)
        self.model.eval()
        
        # Freeze model weights so MPC gradients only flow to control inputs
        for param in self.model.parameters():
            param.requires_grad = False
            
        self._device = torch.device(device)
        self.n_y = hyperparam_config["train"]["n_y"]
        self.n_u = hyperparam_config["train"]["n_u"]
        self._input_dim = hyperparam_config["plant"]["input_dim"]   # Output y dim
        self._output_dim = hyperparam_config["plant"]["output_dim"] # Control u dim
        
        # Internal state tracking buffers for lookback history
        self.reset_history()

    @property
    def state_dim(self) -> int:
        # Dummy dimension representation for the state history tuple
        return (self.n_u + 1) * self._output_dim + self.n_y * self._input_dim

    @property
    def control_dim(self) -> int:
        return self._output_dim

    @property
    def output_dim(self) -> int:
        return self._input_dim

    @property
    def device(self):
        return self._device

    def reset_history(self, batch_size: int = 1):
        """Initializes stateful history lookback buffers with zeros."""
        self.u_history = [
            torch.zeros(batch_size, self._output_dim, device=self._device)
            for _ in range(self.n_u + 1)
        ]
        self.y_history = [
            torch.zeros(batch_size, self._input_dim, device=self._device)
            for _ in range(self.n_y)
        ]
        self.model.reset_memory(batch_size=batch_size, device=self._device)

    def step(self, state: torch.Tensor, u: torch.Tensor, t: float = 0.0, dt: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single step forward using the NARX feature vector w_k.
        """
        # Update u history with current action u_k at index 0
        u_lags = [u] + self.u_history[:-1]
        u_flat = torch.cat(u_lags, dim=-1)
        
        # Output history y_{k-1} ... y_{k-n_y}
        y_flat = torch.cat(self.y_history, dim=-1)
        
        # Construct regressor vector w_k = [u_k, u_{k-1}..., y_{k-1}...]
        w_k = torch.cat([u_flat, y_flat], dim=-1)
        
        # Pass through Mamba (preserves PyTorch computation graph for MPC)
        y_pred = self.model(w_k.unsqueeze(1)).squeeze(1)
        
        # Prepare next state representation (updated histories)
        next_state = w_k.detach().clone()
        return next_state, y_pred

# Assume 'best_model' is your trained MambaSysIDModel instance
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. Wrap trained model
wrapped_plant = MambaPlantWrapper(best_model, hyperparam_config, device=device)

# 2. Instantiate MPC Controller
mpc = MPCController(
    plant=wrapped_plant,
    horizon=15,
    lr=0.05,
    num_sim_steps=25,
    u_min=-2.0,
    u_max=2.0,
    w_y=10.0,
    w_u=0.01,
    w_du=0.5
)

# 3. Define Simulation Parameters and Reference Trajectory
sim_time = 10.0
dt = 0.1
steps = int(sim_time / dt)
t_grid = np.linspace(0, sim_time, steps)

# Example Square-Wave Reference Trajectory for y (Batch=1, Dim=1)
y_ref_full = torch.zeros(steps, 1, wrapped_plant.output_dim, device=device)
for k in range(steps):
    val = 1.0 if (k // 25) % 2 == 0 else -0.5
    y_ref_full[k] = val

# 4. Run Closed-Loop Control Loop
history_y = []
history_u = []
history_ref = []

current_state = torch.zeros(1, wrapped_plant.state_dim, device=device)
last_action = torch.zeros(1, wrapped_plant.control_dim, device=device)

wrapped_plant.reset_history(batch_size=1)

print("🚀 Starting Closed-Loop MPC Simulation...")
for k in range(steps):
    # Slice dynamic receding horizon reference [Horizon, Batch, Output_Dim]
    if k + mpc.horizon <= steps:
        y_ref_horizon = y_ref_full[k : k + mpc.horizon]
    else:
        # Pad with final reference point if near end of simulation
        pad_len = mpc.horizon - (steps - k)
        y_ref_horizon = torch.cat([y_ref_full[k:], y_ref_full[-1:].repeat(pad_len, 1, 1)], dim=0)

    # Compute optimal control action
    u_control = mpc.get_action(current_state, y_ref_horizon, dt=dt, last_action=last_action)
    
    # Step actual plant environment forward
    current_state, y_pred = wrapped_plant.step(current_state, u_control, t=t_grid[k], dt=dt)
    
    # Log step results
    history_u.append(u_control.squeeze(0).cpu().numpy())
    history_y.append(y_pred.squeeze(0).cpu().numpy())
    history_ref.append(y_ref_full[k].squeeze(0).cpu().numpy())
    
    last_action = u_control

print("✅ Closed-loop simulation finished.")

# Format data into NumPy arrays
u_arr = np.array(history_u)      # Shape: [Steps, Control_Dim]
y_arr = np.array(history_y)      # Shape: [Steps, Output_Dim]
ref_arr = np.array(history_ref)  # Shape: [Steps, Output_Dim]



# Format signals to match plot_stacked layout: list of subplot row lists
signals = [
    [y_arr.squeeze(), ref_arr.squeeze()],  # Subplot 1: Output vs Reference
    [u_arr.squeeze()]                      # Subplot 2: Control Action
]

labels = [
    ["System Output (y)", "Reference (y_ref)"],
    ["Control Input (u)"]
]

ylabels = [
    "Output Response",
    "Control Input"
]

# Execute stacked plotting function
img = plot_stacked(
    t=t_grid,
    signals=signals,
    labels=labels,
    ylabel=ylabels,
    title="MPC Closed-Loop Control on Learned Mamba Plant",
    xlabel="Time [s]",
    show=True
)