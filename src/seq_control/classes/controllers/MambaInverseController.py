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