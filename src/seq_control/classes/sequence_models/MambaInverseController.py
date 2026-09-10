# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaInverseController(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        Mamba Inverse Controller supporting arbitrary sliding window inputs.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector v_k. 
                       If not provided, it will be calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["plant"]["input_dim"]  
        self.output_dim = hyperparam_config["plant"]["output_dim"] 
        self.d_state = hyperparam_config["mamba"]["d_state"]
        self.expand = hyperparam_config["mamba"]["expand"]
        self.d_conv = hyperparam_config["mamba"]["d_conv"]
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["train"]["n_y"]
            n_u = hyperparam_config["train"]["n_u"]
            # v_k = [y_{k+1}, y_k ... y_{k-n_y}, u_{k-1} ... u_{k-n_u}]
            
            self.d_model = n_u * self.input_dim + (n_y+2) * self.output_dim
        
        print(f"🛠️ Initializing Mamba core with d_model (feature_dim) = {self.d_model}")
        
        # 3. Instantiate Mamba Core
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        
        # 4. Map latent features back to actuator control dimensions
        self.output_proj = nn.Linear(self.d_model, self.input_dim)
        
        # Inference memory state buffers
        self.conv_state = None
        self.ssm_state = None

    def forward(self, v_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - v_seq: Tensor of shape [Batch, Seq_Len, d_model] (Contains stacked v_k sequences)
        
        Returns:
        - predicted_u: Tensor of shape [Batch, Seq_Len, output_dim] (Target control actions u_k)
        """
        # Pass sequence through Mamba S6 engine
        x = self.core(v_seq)  # Shape: [Batch, Seq_Len, d_model]
        return self.output_proj(x)  # Shape: [Batch, Seq_Len, output_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Allocates or resets zero-filled tracking memory tensors.
        Essential for stateful step-by-step rolling simulation.
        """
        d_inner = self.d_model * self.expand
        self.conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        self.ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)

    def step(self, v_k_single):
        """
        Closed-loop evaluation step. Seamlessly consumes current lookback vector 
        to output the next control instruction while updating recurring states.
        
        Parameters:
        - v_k_single: Tensor of shape [Batch, d_model] (or [Batch, 1, d_model])
        
        Returns:
        - u_out: Tensor of shape [Batch, output_dim] (Unprojected control action)
        """
        if self.conv_state is None or self.ssm_state is None:
            raise RuntimeError("Inference states are uninitialized. Please call reset_memory() first.")
            
        # Standardize input to 3D tensor layout required by mamba_ssm.step: [Batch, 1, d_model]
        if v_k_single.dim() == 2:
            x_3d = v_k_single.unsqueeze(1)
        else:
            x_3d = v_k_single
            
        # Recurrent state computation
        x_out_3d, self.conv_state, self.ssm_state = self.core.step(
            x_3d, self.conv_state, self.ssm_state
        )
        
        # Flatten back sequence step to 2D
        x_out = x_out_3d.squeeze(1)
        
        # Project to physical actuator outputs
        return self.output_proj(x_out) # Shape: [Batch, output_dim]

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





class MambaSurrogateModel(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        Mamba Inverse Controller supporting arbitrary sliding window inputs.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector v_k. 
                       If not provided, it will be calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["plant"]["input_dim"]  
        self.output_dim = hyperparam_config["plant"]["output_dim"] 
        self.d_state = hyperparam_config["mamba"]["d_state"]
        self.expand = hyperparam_config["mamba"]["expand"]
        self.d_conv = hyperparam_config["mamba"]["d_conv"]
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["train"]["n_y"]
            n_u = hyperparam_config["train"]["n_u"]
            # v_k = [y_{k+1}, y_k ... y_{k-n_y}, u_{k-1} ... u_{k-n_u}]
            
            self.d_model = (n_u+1) * self.input_dim + (n_y+1) * self.output_dim
        
        print(f"🛠️ Initializing Mamba core with d_model (feature_dim) = {self.d_model}")
        
        # 3. Instantiate Mamba Core
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        
        # 4. Map latent features back to actuator control dimensions
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # Inference memory state buffers
        self.conv_state = None
        self.ssm_state = None

    def forward(self, v_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - v_seq: Tensor of shape [Batch, Seq_Len, d_model] (Contains stacked v_k sequences)
        
        Returns:
        - predicted_u: Tensor of shape [Batch, Seq_Len, output_dim] (Target control actions u_k)
        """
        # Pass sequence through Mamba S6 engine
        x = self.core(v_seq)  # Shape: [Batch, Seq_Len, d_model]
        return self.output_proj(x)  # Shape: [Batch, Seq_Len, output_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Allocates or resets zero-filled tracking memory tensors.
        Essential for stateful step-by-step rolling simulation.
        """
        d_inner = self.d_model * self.expand
        self.conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        self.ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)

    def step(self, v_k_single):
        """
        Closed-loop evaluation step. Seamlessly consumes current lookback vector 
        to output the next control instruction while updating recurring states.
        
        Parameters:
        - v_k_single: Tensor of shape [Batch, d_model] (or [Batch, 1, d_model])
        
        Returns:
        - u_out: Tensor of shape [Batch, output_dim] (Unprojected control action)
        """
        if self.conv_state is None or self.ssm_state is None:
            raise RuntimeError("Inference states are uninitialized. Please call reset_memory() first.")
            
        # Standardize input to 3D tensor layout required by mamba_ssm.step: [Batch, 1, d_model]
        if v_k_single.dim() == 2:
            x_3d = v_k_single.unsqueeze(1)
        else:
            x_3d = v_k_single
            
        # Recurrent state computation
        x_out_3d, self.conv_state, self.ssm_state = self.core.step(
            x_3d, self.conv_state, self.ssm_state
        )
        
        # Flatten back sequence step to 2D
        x_out = x_out_3d.squeeze(1)
        
        # Project to physical actuator outputs
        return self.output_proj(x_out) # Shape: [Batch, output_dim]

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

class MambaGradientMPC:
    """
    Gradient-based Model Predictive Controller using a trained Mamba surrogate model.
    Optimizes future control sequences via backpropagation through time (BPTT).
    """
    def __init__(
        self,
        surrogate_model: nn.Module,
        scaler_x=None,
        scaler_y=None,
        n_y: int = 1,
        n_u: int = 1,
        horizon: int = 15,
        num_iters: int = 20,
        lr: float = 0.03,
        u_min: float = -1.0,
        u_max: float = 1.0,
        weight_y: float = 15.0,
        weight_u: float = 0.01,
        weight_du: float = 0.2,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = surrogate_model.to(self.device)
        self.model.eval()  # Freeze surrogate model layers (disable dropout/batchnorm updates)

        self.n_y = n_y
        self.n_u = n_u
        self.H = horizon
        self.num_iters = num_iters
        self.lr = lr
        self.u_min = u_min
        self.u_max = u_max

        # Loss function weights
        self.weight_y = weight_y
        self.weight_u = weight_u
        self.weight_du = weight_du

        # Extract scaling parameters into PyTorch tensors for GPU/CPU tensor operations
        if scaler_x is not None and hasattr(scaler_x, "mean_"):
            self.x_mean = torch.tensor(scaler_x.mean_, dtype=torch.float32, device=self.device)
            self.x_scale = torch.tensor(scaler_x.scale_, dtype=torch.float32, device=self.device)
        else:
            self.x_mean, self.x_scale = None, None

        if scaler_y is not None and hasattr(scaler_y, "mean_"):
            self.y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=self.device)
            self.y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=self.device)
        else:
            self.y_mean, self.y_scale = None, None

    def _scale_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_mean is not None:
            return (x - self.x_mean) / self.x_scale
        return x

    def _unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        if self.y_mean is not None:
            return y_scaled * self.y_scale + self.y_mean
        return y_scaled

    def _rollout_surrogate(self, history_frames: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """
        Autoregressively rolls out the Mamba surrogate model over the prediction horizon H.

        Args:
            history_frames: [1, seq_len, d_x] Raw lookback feature sequence.
            u_seq: [1, H, d_u] Optimizable future control inputs.

        Returns:
            y_pred: [1, H, d_y] Predicted unscaled outputs over horizon H.
        """
        batch_size, seq_len, d_x = history_frames.shape
        d_u = u_seq.shape[-1]
        d_y = self.y_mean.shape[0] if self.y_mean is not None else 1

        curr_history = history_frames.clone()
        y_preds = []

        # Step forward H time steps into the future
        for h in range(self.H):
            u_h = u_seq[:, h:h+1, :]  # Current candidate action [1, 1, d_u]

            # Scale history frames and pass through the Mamba surrogate model
            scaled_history = self._scale_x(curr_history)
            y_pred_scaled = self.model(scaled_history)  # Model outputs [1, seq_len, d_y]
            
            # Extract step output (last element in time sequence)
            y_next_scaled = y_pred_scaled[:, -1:, :]
            y_next = self._unscale_y(y_next_scaled)
            y_preds.append(y_next)

            # Construct next feature frame v_{k+1} by shifting history window
            # Assuming feature vector structure: [y_{k-n_y+1}...y_k, u_{k-n_u+1}...u_k]
            y_hist_prev = curr_history[:, -1, : (self.n_y * d_y)]
            u_hist_prev = curr_history[:, -1, (self.n_y * d_y) :]

            # Shift output history window
            y_hist_new = torch.cat([y_hist_prev[:, d_y:], y_next.squeeze(1)], dim=-1)
            # Shift input history window
            u_hist_new = torch.cat([u_hist_prev[:, d_u:], u_h.squeeze(1)], dim=-1)

            # Combine into new history state frame
            v_next = torch.cat([y_hist_new, u_hist_new], dim=-1).unsqueeze(1)
            curr_history = torch.cat([curr_history[:, 1:, :], v_next], dim=1)

        return torch.cat(y_preds, dim=1)  # [1, H, d_y]

    def solve(
        self,
        history_frames: torch.Tensor,
        y_ref_horizon: torch.Tensor,
        u_prev: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Runs gradient descent optimization to find optimal control action u_k.

        Args:
            history_frames: [1, seq_len, d_x] Past context window.
            y_ref_horizon: [1, H, d_y] Target trajectory over horizon H.
            u_prev: [1, d_u] Previous executed control action (for slew rate penalty).

        Returns:
            u_optimal: [d_u] First action step to execute in physical plant.
        """
        d_u = u_prev.shape[-1] if u_prev is not None else 1
        
        # 1. Initialize action sequence variable (zeros or warm-started)
        u_seq = torch.zeros((1, self.H, d_u), dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = optim.Adam([u_seq], lr=self.lr)

        # 2. Gradient optimization loop over the horizon
        for iteration in range(self.num_iters):
            optimizer.zero_grad()

            # Predict system outputs using Mamba surrogate model rollouts
            y_pred = self._rollout_surrogate(history_frames, u_seq)

            # A. Output Tracking Error Loss
            loss_y = self.weight_y * torch.mean((y_pred - y_ref_horizon) ** 2)

            # B. Control Input Magnitude Penalty
            loss_u = self.weight_u * torch.mean(u_seq ** 2)

            # C. Slew Rate / Control Smoothness Penalty (du)
            if u_prev is not None:
                u_first_diff = u_seq[:, 0:1, :] - u_prev.unsqueeze(1)
                u_rest_diff = u_seq[:, 1:, :] - u_seq[:, :-1, :]
                u_diff = torch.cat([u_first_diff, u_rest_diff], dim=1)
            else:
                u_diff = u_seq[:, 1:, :] - u_seq[:, :-1, :]

            loss_du = self.weight_du * torch.mean(u_diff ** 2)

            # Total MPC Objective function
            total_loss = loss_y + loss_u + loss_du

            # Backpropagate gradients into u_seq
            total_loss.backward()
            optimizer.step()

            # Enforce hard actuator bounds via projection / clamping
            with torch.no_grad():
                if self.u_min is not None or self.u_max is not None:
                    u_seq.clamp_(min=self.u_min, max=self.u_max)

        # Return optimal action for immediate execution at step k (Receding Horizon Principle)
        return u_seq.detach()[0, 0, :]