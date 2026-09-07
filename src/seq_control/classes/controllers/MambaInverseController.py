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
    def __init__(
        self,
        surrogate_model,
        horizon=15,
        num_iters=20,
        lr=0.05,
        u_min=None,
        u_max=None,
        weight_y=10.0,
        weight_u=0.01,
        weight_du=0.1,
        device="cuda"
    ):
        """
        Gradient-Based MPC controller using PyTorch autograd over a Mamba surrogate model.

        Parameters:
        - surrogate_model: Trained MambaSurrogateModel instance.
        - horizon (H): Lookahead prediction horizon steps.
        - num_iters: Gradient optimization steps per control cycle.
        - lr: Learning rate for updating control action tensor.
        - u_min, u_max: Hard bounds for control actuators.
        - weight_y, weight_u, weight_du: Penalty weights for tracking error, input magnitude, and slew rate.
        """
        self.model = surrogate_model
        self.H = horizon
        self.num_iters = num_iters
        self.lr = lr
        self.u_min = u_min
        self.u_max = u_max
        
        self.w_y = weight_y
        self.w_u = weight_u
        self.w_du = weight_du
        self.device = device
        
        # Freeze surrogate model parameters (we only optimize control inputs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def solve(self, history_seq, y_ref_horizon, u_prev=None):
        """
        Solves for the optimal control sequence over horizon H.

        Parameters:
        - history_seq: Tensor [1, history_len, d_model] containing warm-up feature history.
        - y_ref_horizon: Tensor [1, H, output_dim] reference trajectory for the next H steps.
        - u_prev: Tensor [1, control_dim] control action applied in the previous timestep (k-1).

        Returns:
        - u_optimal_next: Tensor [control_dim] control action to apply at current timestep k.
        """
        control_dim = self.model.output_dim
        
        # 1. Warm-start control trajectory over horizon H (shift previous solution or initialize zeros)
        u_traj = torch.zeros((1, self.H, control_dim), device=self.device, requires_grad=True)
        optimizer = optim.Adam([u_traj], lr=self.lr)

        best_u_traj = u_traj.detach().clone()
        min_cost = float("inf")

        # 2. Optimization loop over control actions
        for _ in range(self.num_iters):
            optimizer.zero_grad()
            
            # Clamp candidate controls to physical bounds
            if self.u_min is not None or self.u_max is not None:
                u_clamped = torch.clamp(u_traj, self.u_min, self.u_max)
            else:
                u_clamped = u_traj

            # 3. Roll out Mamba model predictions over horizon H
            # Predict future outputs y_pred using Mamba (either full sequence forward or recurrent step)
            y_pred = self._rollout(history_seq, u_clamped)

            # 4. Compute MPC Cost Function
            # Tracking Loss
            cost_y = self.w_y * torch.mean((y_pred - y_ref_horizon) ** 2)
            
            # Control Effort Loss
            cost_u = self.w_u * torch.mean(u_clamped ** 2)
            
            # Smoothness / Slew Rate Loss (u_k - u_{k-1})
            if u_prev is not None:
                u_full = torch.cat([u_prev.unsqueeze(1), u_clamped], dim=1)
                du = u_full[:, 1:, :] - u_full[:, :-1, :]
            else:
                du = u_clamped[:, 1:, :] - u_clamped[:, :-1, :]
            cost_du = self.w_du * torch.mean(du ** 2)

            total_loss = cost_y + cost_u + cost_du

            # 5. Backpropagate gradients to control inputs
            total_loss.backward()
            optimizer.step()

            if total_loss.item() < min_cost:
                min_cost = total_loss.item()
                best_u_traj = u_clamped.detach().clone()

        # Return the first control action in the optimized sequence (Receding Horizon)
        u_optimal_next = best_u_traj[0, 0, :]
        return u_optimal_next

    def _rollout(self, history_seq, u_horizon):
        """
        Simulates future trajectory predictions across the horizon.
        """
        # If your model accepts full sequences directly:
        # Concatenate history features with candidate control sequences
        # Note: Adapt feature vector construction to match your surrogate's exact input format
        full_seq = torch.cat([history_seq, u_horizon], dim=1)
        y_pred_full = self.model(full_seq)
        
        # Return only the future horizon segment [1, H, output_dim]
        return y_pred_full[:, -self.H:, :]