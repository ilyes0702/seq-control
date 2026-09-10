"""
General Utility Functions
=========================

This module provides essential utility functions and system helpers supporting sequence-based 
control systems, experimental reproducibility, and model introspection.

Key Features
------------
* **Tracking Performance Evaluation**: Computes dynamic plant output tracking metrics 
  (MAE, MAPE, MSE, RMSE, IAE, ISE, Max Error, Time-in-Band) and writes detailed CSV reports.
* **Global Determinism & Seeding**: Configures seed states across Python, NumPy, PyTorch CPU/GPU, 
  and cuDNN backends to ensure reproducible training and evaluation runs.
* **SSM State Extraction**: Inspects selective state-space neural architectures (e.g., Mamba) 
  to extract continuous-time matrices (:math:`A`, :math:`B`, :math:`C`, :math:`D`) and discretized 
  system matrices (:math:`\\bar{A}`, :math:`\\bar{B}`).
"""

# Import standard libraries
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# import machine learning modules
from seq_control.decorators.general_decorators import *
from seq_control.utils.saving_and_loading_utils import *
from seq_control.config import *
from seq_control.utils.plotting_utils import *

#=== FUNCTION TO COUNT THE PARAMETERS OF A SEQUENCE MODEL ===#
def count_seq_model_params(model):
    
        # Total parameters (trainable + non-trainable)
        total_params = sum(p.numel() for p in model.parameters())

        # Trainable parameters only
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

#=== FUNCTION TO COMPUTE AND SAVE TRACKING METRICS ===#
def compute_and_save_tracking_metrics(
    y_np,        # Actual output [steps, batch_size]
    ref_np,      # Reference trajectory [steps, batch_size]
    dt,
    dirname,
    settle_tol=0.05, # Band for "tracking error"
    suffix=None
):
    """
    Computes trajectory tracking metrics against a reference signal and saves performance reports.

    Calculates integral and dynamic metrics (MAE, MAPE, MSE, RMSE, IAE, ISE, Max Error, 
    and Time-in-Band Percentage) across multiple batch trajectories and saves both 
    per-trajectory and summary statistics as CSV files.

    :param y_np: Actual plant output trajectories array of shape ``[steps, batch_size]``.
    :type y_np: numpy.ndarray
    :param ref_np: Dynamic or static reference trajectory of shape ``[steps, batch_size]``, 
        ``[steps]``, or a scalar value.
    :type ref_np: numpy.ndarray or float
    :param dt: Sampling time interval between consecutive discrete time steps.
    :type dt: float
    :param dirname: Directory path where output CSV files will be written.
    :type dirname: str or pathlib.Path
    :param settle_tol: Relative error threshold defining the tracking tolerance band 
        (e.g., 0.05 corresponds to 5%), defaults to 0.05.
    :type settle_tol: float, optional
    :param suffix: Optional string tag appended to the generated CSV filenames, 
        defaults to None.
    :type suffix: str or None, optional

    :returns: DataFrame indexed by trajectory containing calculated tracking metrics for 
        each batch trace.
    :rtype: pandas.DataFrame
    """
    steps, batch_size = y_np.shape
    
    # Ensure ref_np is the same shape as y_np
    if np.isscalar(ref_np):
        ref_np = np.full_like(y_np, ref_np)
    elif ref_np.ndim == 1:
        ref_np = np.tile(ref_np[:, np.newaxis], (1, batch_size))

    error = y_np - ref_np
    abs_error = np.abs(error)

    # Small epsilon to prevent division by zero for near-zero references
    denom = np.abs(ref_np) + 1e-8

    # --- Integral & Percentage Metrics ---
    mae = abs_error.mean(axis=0)
    mape = (abs_error / denom).mean(axis=0) * 100.0  # MAPE in %
    mse = (error ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    iae = abs_error.sum(axis=0) * dt
    ise = (error ** 2).sum(axis=0) * dt

    # --- Dynamic Tracking Metrics ---
    max_error = np.max(abs_error, axis=0)
    
    # Time spent within tolerance band (%)
    within_band = abs_error <= (settle_tol * denom)
    time_in_band_pct = (np.sum(within_band, axis=0) / steps) * 100.0

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "trajectory": np.arange(batch_size),
        "MAE": mae,
        "MAPE_%": mape,
        "MSE": mse,
        "RMSE": rmse,
        "IAE": iae,
        "ISE": ise,
        "Max_Error": max_error,
        "TimeInBand_%": time_in_band_pct
    })

    # Summary Statistics
    summary_df = df.describe().loc[['mean', 'std', 'min', 'max']].T.reset_index()
    summary_df.columns = ['metric', 'mean', 'std', 'min', 'max']

    # --- Save ---
    file_ext = f"_{suffix}" if suffix else ""
    save_df_to_csv(df, dirname, f"tracking_metrics_per_trajectory{file_ext}")
    save_df_to_csv(summary_df, dirname, f"tracking_metrics_summary{file_ext}")

    return df

#=== FUNCTION TO SEED EVERYTHING FOR REPRODUCIBILITY ===#
def seed_everything(seed=42):
    """
    Seeds all relevant libraries to ensure reproducible results.

    Enforces absolute determinism across various execution contexts by explicitly 
    binding the seed to Python's core ``random`` module, environment variables, 
    NumPy, and both CPU and GPU tensor variants in PyTorch. Additionally, it 
    overrides standard CUDA Deep Neural Network (cuDNN) runtime configurations to 
    deactivate dynamic kernel auto-tuning, eliminating stochastic variance across 
    identical runs.

    :param seed: The numerical seed value used to initialize all random number 
        generators, defaults to 42.
    :type seed: int, optional

    :returns: None
    :rtype: None
    """
    # Seed native Python behaviors and system environments
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed third-party matrix execution engines
    np.random.seed(seed)
    
    # Seed deep learning core frameworks across standard processing hardware
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Force synchronization across multi-GPU environments
    
    # Critical for CUDA reproducibility: override cuDNN runtime optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")

#=== FUNCTION TO EXTRACT SSM MATRICES ===#
def extract_ssm_matrices_at_step(model, mamba_block, y_t_tensor, y_next_tensor):
    """
    Extracts the analytical SSM matrices (A, B, C, D) and discretized matrices (A_bar, B_bar)
    at the CURRENT (latest) step of the simulation sequence.
    
    Args:
        model: Your outer wrapper model class instance.
        mamba_block: The raw Mamba module instance (model.core).
        y_t_tensor: Tensor of shape [Batch, Seq_Len, input_dim]
        y_next_tensor: Tensor of shape [Batch, Seq_Len, input_dim]
    """
    # 1. Replicate how your outer wrapper converts raw plant signals to hidden_states.
    # If your wrapper has a custom forward pass, we mirror its embedding step here:
    
    # Example A: If your model concatenates or adds inputs and pushes them through a linear layer:
    if hasattr(model, 'input_projection'): # Change 'input_projection' to your wrapper's actual embedding layer name
        # If your model processes y_t and y_next together
        hidden_states = torch.cat([y_t_tensor, y_next_tensor], dim=-1)
        hidden_states = model.input_projection(hidden_states)
    else:
        # Fallback/Default: Combine them and check if we need to manually map them to d_model
        hidden_states = y_t_tensor + y_next_tensor 
        
        # If they are still raw plant dimensions (e.g., feature dim = 1 instead of 64)
        if hidden_states.shape[-1] != mamba_block.d_model:
            # Look for an embedding layer in your outer model dynamically
            embedding_layer = None
            for module in model.modules():
                if isinstance(module, nn.Linear) and module.out_features == mamba_block.d_model:
                    embedding_layer = module
                    break
            
            if embedding_layer is not None:
                # If your embedding layer expects concatenated inputs:
                if embedding_layer.in_features == (y_t_tensor.shape[-1] + y_next_tensor.shape[-1]):
                    hidden_states = torch.cat([y_t_tensor, y_next_tensor], dim=-1)
                hidden_states = embedding_layer(hidden_states)
            else:
                raise AttributeError(
                    f"Could not automatically find the embedding layer projecting raw features "
                    f"to d_model ({mamba_block.d_model}). Please pass hidden_states after your "
                    f"outer model's embedding layer."
                )

    batch, seqlen, dim = hidden_states.shape

    # 2. Project using F.linear (Now safely guaranteed to be Batch x SeqLen x 64)
    xz = F.linear(hidden_states, mamba_block.in_proj.weight, mamba_block.in_proj.bias)
    xz = rearrange(xz, "b l d -> b d l")

    x, z = xz.chunk(2, dim=1)

    # 3. Compute short convolution
    x_conv = mamba_block.act(mamba_block.conv1d(x)[..., :seqlen])
    
    # 4. Pull dynamic projections for the entire sequence
    x_dbl = mamba_block.x_proj(rearrange(x_conv, "b d l -> (b l) d"))
    dt_proj, B_seq, C_seq = torch.split(
        x_dbl, [mamba_block.dt_rank, mamba_block.d_state, mamba_block.d_state], dim=-1
    )

    dt_seq = F.linear(dt_proj, mamba_block.dt_proj.weight)
    dt_seq = rearrange(dt_seq, "(b l) d -> b d l", l=seqlen)
    
    B_seq = rearrange(B_seq, "(b l) dstate -> b dstate l", l=seqlen)
    C_seq = rearrange(C_seq, "(b l) dstate -> b dstate l", l=seqlen)

    # 5. ISOLATE THE CURRENT SIMULATION STEP
    dt_current = dt_seq[..., -1]         
    B_current = B_seq[..., -1]           
    C_current = C_seq[..., -1]           

    # 6. Extract static variables
    A_static = -torch.exp(mamba_block.A_log.float())  
    D_static = mamba_block.D.float()                  

    # 7. Discretize
    dt_current = F.softplus(dt_current + mamba_block.dt_proj.bias.float())
    
    A_bar = torch.exp(torch.einsum("bd,dn->bdn", dt_current, A_static))
    B_bar = torch.einsum("bd,bn->bdn", dt_current, B_current)

    return {
        "A": A_static,
        "B": B_current,
        "C": C_current,
        "D": D_static,
        "dt": dt_current,
        "A_bar": A_bar,
        "B_bar": B_bar
    }