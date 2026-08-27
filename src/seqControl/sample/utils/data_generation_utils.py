"""
Data Generation Utilities
=========================

This module contains utilities for the in-silico generation of 
training signals and dynamic MIMO datasets.
"""

import os
import numpy as np
import torch
import pandas as pd
from seqControl.sample.utils.saving_and_loading_utils import save_df_to_csv, save_training_dataset
from seqControl.sample.utils.plotting_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_signal_single(training_data_cfg, channel_idx=1):
    """Generates smooth, band-limited control signals using an Active Shielding methodology.

    This function guarantees that each unique Multi-Input Multi-Output (MIMO) channel stays completely
    within its hard boundaries. It utilizes independent, channel-specific bandwidth (:math:`\\lambda`)
    and amplitude (:math:`p`) settings.

    :param dict training_data_cfg: Configuration dictionary containing global and channel-specific operational parameters:

        * **batch_size** (*int*) -- Number of trajectories to generate in parallel.
        * **seq_len** (*int*) -- Length of the signal trajectory sequence.
        * **dt** (*float*) -- Sampling time step for frequency analysis.
        * **lambd** (*float*) -- Global fallback wavelength/bandwidth parameter.
        * **p** (*float*) -- Global fallback target amplitude parameter.
        * **u_{channel_idx}_lambd** (*float, optional*) -- Channel-specific bandwidth cutoff limit.
        * **u_{channel_idx}_p** (*float, optional*) -- Channel-specific target amplitude.
        * **u_{channel_idx}_D_center_min** (*float*) -- Lower bound for baseline center distribution.
        * **u_{channel_idx}_D_center_max** (*float*) -- Upper bound for baseline center distribution.
        * **u_{channel_idx}_hard_min** (*float*) -- Strict lower physical threshold limit.
        * **u_{channel_idx}_hard_max** (*float*) -- Strict upper physical threshold limit.

    :param int channel_idx: Index specifying the target MIMO channel (default is ``1``).

    :return: A tuple containing:

        * **u_buffer** (*torch.Tensor*) -- Active-shielded, band-limited control signal tensor of shape ``(batch_size, seq_len)``.
        * **u_center** (*torch.Tensor*) -- Generated baseline centers for each trajectory of shape ``(batch_size, 1)``.

    :rtype: Tuple[torch.Tensor, torch.Tensor]

    Process Flow
    ------------

    1. **Dynamic Configuration Extraction**
       Extracts channel-specific parameters (``lambd``, ``p``, bounds) with immediate fallbacks to global defaults if channel overrides are omitted.

    2. **Frequency Filtering**
       Generates uniform noise, converts to frequency domain via Real Fast Fourier Transform (``rfft``), clips frequencies above cutoff frequency :math:`f_c = \\frac{1}{\\lambda}`, and transforms back via Inverse RFFT (``irfft``).

    3. **Normalization**
       Scales the smooth trajectories to the normalized range :math:`[-1, 1]`.

    4. **Active Shielding Safeguard**
       * Calculates distance from randomly assigned centers (:math:`u_{\\text{center}}`) to hard boundaries (:math:`u_{\\text{hard\\_min}}`, :math:`u_{\\text{hard\\_max}}`).
       * Sets safe amplitude :math:`p_{\\text{adaptive}} = \\min(p_{\\text{configured}}, 0.98 \\cdot p_{\\text{max\\_safe}})`.
       * Clamps output signal to guarantee zero hard-limit breaches due to floating-point imprecision.

    Example Usage
    -------------

    .. code-block:: python

        import torch
        from seqControl.sample.utils.data_generation_utils import generate_signal_single

        cfg = {
            "batch_size": 32,
            "seq_len": 100,
            "dt": 0.01,
            "lambd": 0.5,
            "p": 1.0,
            "u_1_lambd": 0.2,
            "u_1_p": 0.5,
            "u_1_D_center_min": -0.5,
            "u_1_D_center_max": 0.5,
            "u_1_hard_min": -2.0,
            "u_1_hard_max": 2.0,
        }

        u_buffer, u_center = generate_signal_single(cfg, channel_idx=1)
    """
    batch_size = training_data_cfg["batch_size"]
    seq_len = training_data_cfg["seq_len"]
    device = "cuda"

    # 🎯 NEW: Dynamic Channel-Specific Lambda (Bandwidth) Extraction
    lambd = training_data_cfg[f"u_{channel_idx}_lambd"]
    if lambd is None:
        lambd = training_data_cfg["lambd"]  # Global fallback

    # 🎯 NEW: Dynamic Channel-Specific Configured Amplitude Extraction
    configured_p = training_data_cfg[f"u_{channel_idx}_p"]
    if configured_p is None:
        configured_p = training_data_cfg["p"]  # Global fallback

    # Step 1: Sample values from a uniform distribution [-1, 1]
    raw = torch.rand((batch_size, seq_len), device=device) * 2 - 1

    # Step 2: Fourier-transform to frequency domain
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=training_data_cfg["dt"])

    # Step 3: Drop frequencies above 1/lambda using the channel-specific lambda
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff] = 0

    # Step 4: Inverse-Fourier-transform
    v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)

    # Step 5: Normalize to [-1, 1]
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1

    # Dynamic Channel-Specific Center Value Extraction
    c_min = training_data_cfg[f"u_{channel_idx}_D_center_min"]
    c_max = training_data_cfg[f"u_{channel_idx}_D_center_max"]

    # Read channel-specific hard boundaries or fall back to system defaults
    u_hard_min = training_data_cfg[f"u_{channel_idx}_hard_min"]
    u_hard_max = training_data_cfg[f"u_{channel_idx}_hard_max"]

    # Generate random baseline centers across the channel-specific configured range
    u_center = torch.rand((batch_size, 1), device=device) * (c_max - c_min) + c_min

    # 🎯 ACTIVE SHIELDING: Calculate exact allowable deviation limits per trajectory row
    dist_to_max = u_hard_max - u_center
    dist_to_min = u_center - u_hard_min
    max_safe_p = torch.minimum(dist_to_max, dist_to_min)

    # Adapt amplitude: use the channel-specific configured p, but scale down if it approaches boundaries
    adaptive_p = torch.minimum(torch.full_like(max_safe_p, configured_p), max_safe_p * 0.98)

    u_buffer = u_center + (v_norm * adaptive_p)

    # Guard clamp for precision floating point margins
    u_buffer = torch.clamp(u_buffer, u_hard_min, u_hard_max)

    return u_buffer, u_center


def generate_signals(training_data_cfg):
    """Generates independent MIMO control vectors using index-aware tracking.

    Iterates through each input channel defined in the configuration, invokes
    single-channel signal generation using a 1-based index, and aggregates the
    resulting control signals and delay center values into unified tensors.

    :param dict training_data_cfg: Configuration dictionary containing training parameters.
        Must include the key ``"input_dim"`` specifying the number of input channels,
        alongside any parameters required by :func:`generate_signal_single`.

    :return: A tuple containing:

        * **u_buffer** (*torch.Tensor*) -- Concatenated MIMO control vectors across all input channels of shape ``(..., input_dim)``.
        * **D_center** (*torch.Tensor*) -- Stacked delay center values across all input channels of shape ``(..., input_dim)``.

    :rtype: Tuple[torch.Tensor, torch.Tensor]

    .. note::
       This function passes a 1-based index (``channel_idx = i + 1``) to :func:`generate_signal_single`
       to ensure proper channel configuration mapping.
    """
    # batch_size = training_data_cfg["batch_size"]

    # output_dim = training_data_cfg["output_dim"]
    input_dim = training_data_cfg["input_dim"]

    u_buffer = []
    D_center_list = []

    for i in range(input_dim):
        # Pass 1-based index to resolve channel configurations cleanly
        u_single, D_center = generate_signal_single(training_data_cfg, channel_idx=i + 1)
        u_buffer.append(u_single.unsqueeze(-1))
        D_center_list.append(D_center)

    u_buffer = torch.cat(u_buffer, dim=-1)
    D_center = torch.stack(D_center_list, dim=-1)

    return u_buffer, D_center


def generate_signals_mix(hyperparam_config):
    """Generates independent MIMO control vectors mixing Canaday signals with constant step signals.

    Iterates through each output channel defined in the configuration and generates standard active-shielded
    Canaday signals. A portion of the batch elements (determined by a probabilistic threshold) are then
    randomly overwritten with flat, constant step signals sampled uniformly within physical channel boundaries.

    :param dict hyperparam_config: Nested configuration dictionary containing setup subsections:

        * **train** (*dict*) -- Includes ``"batch_size"`` (*int*) and ``"constant_signal_probability"`` (*float*, probability in :math:`[0, 1]` to replace a sequence with a constant line).
        * **mamba** (*dict*) -- Includes ``"output_dim"`` (*int*, number of control output channels).
        * **signal** (*dict*) -- Includes ``"seq_len"`` (*int*, sequence length per batch item).
        * **plant** (*dict*) -- Includes channel boundary limits ``"u_{i}_min"`` and ``"u_{i}_max"`` (*float*, defaults to ``0.0`` and ``1.0`` if omitted).

    :return: A tuple containing:

        * **u_buffer** (*torch.Tensor*) -- Active-shielded MIMO control vector tensor mixed with constant step signals. Shape: ``(batch_size, seq_len, output_dim)``.
        * **D_center** (*torch.Tensor*) -- Delay baseline center values across channels (zeroed out for constant step overrides). Shape: ``(batch_size, 1, output_dim)``.

    :rtype: Tuple[torch.Tensor, torch.Tensor]

    .. note::
       Constant step values are generated per batch item and per channel independently using uniform
       sampling: :math:`u_{\\text{val}} \\sim \\mathcal{U}(u_{\\text{min}}, u_{\\text{max}})`.
    """
    train_cfg = hyperparam_config["train"]
    mamba_cfg = hyperparam_config["mamba"]
    sig_cfg = hyperparam_config["signal"]

    batch_size = train_cfg["batch_size"]
    output_dim = mamba_cfg["output_dim"] 
    seq_len = int(sig_cfg["seq_len"])
    
    # Probability of a batch item being a constant signal (e.g., 0.3 = 30%)
    # Default to 0.0 if not specified in config to remain backward compatible
    constant_prob = train_cfg["constant_signal_probability"]

    u_buffer = []
    D_center_list = []

    for i in range(output_dim):
        # 1. Generate the standard baseline Canaday signal for the channel
        u_single, D_center = generate_signal_single(hyperparam_config, channel_idx=i+1)
        
        # u_single shape: [batch_size, seq_len]
        # 2. Determine which batch items will be replaced with constants
        # We perform this independently per channel or globally per batch element
        for b in range(batch_size):
            if np.random.rand() < constant_prob:
                # Pick a random constant value within your physical input range
                # For the Trophophase plant, u1 bounds are [0.0, 1.0]
                u_min = hyperparam_config["plant"].get(f"u_{i+1}_min", 0.0)
                u_max = hyperparam_config["plant"].get(f"u_{i+1}_max", 1.0)
                
                # Sample a random continuous step value
                constant_val = u_min + (u_max - u_min) * np.random.rand()
                
                # Overwrite this specific batch sequence index with a flat line
                u_single[b, :] = constant_val
                
                # (Optional) If D_center tracking matters for the constant, zero it out or preserve it
                D_center[b, :] = 0.0 

        u_buffer.append(u_single.unsqueeze(-1))
        D_center_list.append(D_center)

    u_buffer = torch.cat(u_buffer, dim=-1)  # Shape: [batch_size, seq_len, output_dim]
    D_center = torch.stack(D_center_list, dim=-1) 

    return u_buffer, D_center

def generate_training_batch(plant, training_data_cfg):
    """Simulates the physical system plant and generates raw, continuous time-series arrays.

    Generates smooth MIMO input control signals via :func:`generate_signals` and integrates the plant dynamics
    step-by-step over the defined sequence length. No sliding window slicing or derivative calculations are applied.

    :param object plant: Dynamical system instance providing `get_initial_state`, `get_y`, and `step` methods.
    :param dict training_data_cfg: Configuration dictionary containing setup parameters:

        * **dt** (*float*) -- Simulation time step size.
        * **seq_len** (*int*) -- Total number of continuous time steps to simulate per sequence.
        * **batch_size** (*int*) -- Number of trajectory instances to simulate in parallel.

    :return: A 4-element tuple containing raw trajectory tensors moved to the target CUDA device:

        * **raw_u** (*torch.Tensor*) -- Control input sequences tensor of shape ``(batch_size, seq_len, output_dim)``.
        * **raw_y** (*torch.Tensor*) -- System output observation sequences tensor of shape ``(batch_size, seq_len, input_dim)``.
        * **raw_states** (*torch.Tensor*) -- Internal state trajectory history tensor of shape ``(batch_size, seq_len, state_dim)``.
        * **D_center** (*torch.Tensor*) -- Delay baseline center values returned by signal generation of shape ``(batch_size, 1, output_dim)``.

    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    .. note::
       State trajectories are detached (``state = state.detach()``) at every integration step to prevent unrolling computation graphs across time steps.
    """
    
    dt = training_data_cfg["dt"]
    seq_len = training_data_cfg["seq_len"]
    
    batch_size = int(training_data_cfg["batch_size"])
    device = "cuda"

    # Initialize plant state
    state = plant.get_initial_state(batch_size)

    # Generate smooth input signals (u)
    u_buffer, D_center = generate_signals(training_data_cfg)

    raw_y_history = []
    raw_u_history = []
    raw_state_history = []

    # Simulate step-by-step
    for t_idx in range(seq_len):
        t = t_idx * dt
        u_signal = u_buffer[:, t_idx, :]  # [batch_size, output_dim]
        y_t = plant.get_y(state, t)       # [batch_size, input_dim]

        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)
        raw_state_history.append(state.clone())

        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # Stack raw arrays along the time dimension (dim=1)
    raw_u = torch.stack(raw_u_history, dim=1).to(device)       # [batch_size, seq_len, output_dim]
    raw_y = torch.stack(raw_y_history, dim=1).to(device)       # [batch_size, seq_len, input_dim]
    raw_states = torch.stack(raw_state_history, dim=1).to(device) # [batch_size, seq_len, state_dim]

    return raw_u, raw_y, raw_states, D_center








def generate_and_save_dataset(
    plant,
    training_data_cfg,
    dirname,
    show_plots=False,
    show_overlay_plot=False,
    save_logs=False
):
    """
    Generates, validates, and exports a raw continuous MIMO dataset.

    Simulates a batch of system trajectories and filters out sequences that
    violate defined hard state, input, or output boundaries, as well as those 
    failing minimum Pearson correlation thresholds. Successfully validated 
    sequences are stored as raw PyTorch tensors, optionally exported as CSV 
    logs, and visualized.

    :param plant: The dynamical system plant instance containing system 
        dynamics and plotting configuration via ``get_plot_config()``.
    :type plant: object
    :param training_data_cfg: Configuration dictionary containing setup parameters:

        * **dt** (*float*): Time step size.
        * **input_dim** (*int*): Number of control input variables (:math:`u`).
        * **output_dim** (*int*): Number of dynamic output variables (:math:`y`).
        * **batch_size** (*int*): Total number of sequences to simulate.
        * **min_correlation_threshold** (*float*): Minimum absolute Pearson 
        correlation required between :math:`u` and :math:`y` channels.
        * **y_{i}_hard_min/max** (*float, optional*): Hard boundary limits for output :math:`y_i`.
        * **f_u_{i}_hard_min/max** or **u_{i}_hard_min/max** (*float, optional*): 
        Hard boundary limits for control input :math:`u_i`.
        * **x_{i}_hard_min/max** (*float, optional*): Hard boundary limits for state variable :math:`x_i`.

    :type training_data_cfg: dict
    :param dirname: Root directory path where dataset tensors, logs, and plots 
        will be saved.
    :type dirname: str
    :param show_plots: If ``True``, generates and displays individual stacked 
        trajectory plots (:math:`u`, :math:`y`, states) for every valid sequence. 
        Defaults to ``False``.
    :type show_plots: bool, optional
    :param show_overlay_plot: If ``True``, generates an overlaid trajectory plot 
        for all accepted :math:`u` and :math:`y` sequences. Defaults to ``False``.
    :type show_overlay_plot: bool, optional
    :param save_logs: If ``True``, exports per-sequence trajectory data as individual 
        CSV files in a ``dataset_logs`` subfolder. Defaults to ``False``.
    :type save_logs: bool, optional

    :return: A dictionary containing the accepted raw dataset tensors mapped to CPU, 
        or ``None`` if 100% of generated sequences fail validation:

        * **"u"** (*torch.Tensor*): Control inputs tensor of shape 
        ``[Valid_Seqs, Seq_Len, input_dim]``.
        * **"y"** (*torch.Tensor*): Dynamic outputs tensor of shape 
        ``[Valid_Seqs, Seq_Len, output_dim]``.
        * **"states"** (*torch.Tensor*): Internal states tensor of shape 
        ``[Valid_Seqs, Seq_Len, state_dim]``.

    :rtype: dict[str, torch.Tensor] | None
    """

    

    device = "cuda"
    dt = training_data_cfg["dt"]
    
    input_dim = training_data_cfg["input_dim"]   # e.g., (y1, y2)
    output_dim = training_data_cfg["output_dim"] # e.g., (u1, u2)
    total_sequences = int(training_data_cfg["batch_size"])
    
    logs_dir = os.path.join(dirname, "dataset_logs")
    plots_dir = os.path.join(dirname, "dataset_plots")

    print(f"🚀 Running batch simulation for {total_sequences} sequences...")
    
    # 1. Fetch raw, un-sliced continuous histories from the simulation engine
    u_raw, y_raw, state_raw, batch_d_centers = generate_training_batch(plant, training_data_cfg)

    u_np = u_raw.cpu().numpy()          # Shape: [Total_Seqs, Seq_Len, output_dim]
    y_np = y_raw.cpu().numpy()          # Shape: [Total_Seqs, Seq_Len, input_dim]
    state_np = state_raw.cpu().numpy()  # Shape: [Total_Seqs, Seq_Len, state_dim]
    batch_d_centers_np = batch_d_centers.cpu().numpy()  

    violated_sequences_count = 0
    valid_u_list, valid_y_list, valid_states_list, valid_dfs = [], [], [], []
    per_sequence_correlations = []
    valid_idx_counter = 0

    # 2. Iterate through and validate each simulated sequence
    for s_idx in range(total_sequences):
        u = u_np[s_idx]             # [Seq_Len, output_dim]
        y_t = y_np[s_idx]           # [Seq_Len, input_dim]
        states = state_np[s_idx]     # [Seq_Len, state_dim]

        seq_has_violation = False

        # 📊 DYNAMIC OUTPUTS (y) BOUNDS CHECK
        for i in range(input_dim):
            single_seq_y = y_t[:, i]
            h_min = training_data_cfg.get(f"y_{i+1}_hard_min")
            h_max = training_data_cfg.get(f"y_{i+1}_hard_max")
            
            if h_min is not None and np.min(single_seq_y) < h_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Output y_{i+1} dropped below limit! "
                      f"Bound: {h_min}, Min Found: {np.min(single_seq_y):.4f}\033[0m")
                seq_has_violation = True
                
            if h_max is not None and np.max(single_seq_y) > h_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Output y_{i+1} exceeded limit! "
                      f"Bound: {h_max}, Max Found: {np.max(single_seq_y):.4f}\033[0m")
                seq_has_violation = True

        # 🕹️ DYNAMIC CONTROL INPUTS (u) BOUNDS CHECK
        for i in range(input_dim):
            single_seq_u = u[:, i]
            h_min = training_data_cfg.get(f"f_u_{i+1}_hard_min") or training_data_cfg.get(f"u_{i+1}_hard_min")
            h_max = training_data_cfg.get(f"f_u_{i+1}_hard_max") or training_data_cfg.get(f"u_{i+1}_hard_max")
            
            if h_min is not None and np.min(single_seq_u) < h_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Input u_{i+1} dropped below limit! "
                      f"Bound: {h_min}, Min Found: {np.min(single_seq_u):.4f}\033[0m")
                seq_has_violation = True
                
            if h_max is not None and np.max(single_seq_u) > h_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Input u_{i+1} exceeded limit! "
                      f"Bound: {h_max}, Max Found: {np.max(single_seq_u):.4f}\033[0m")
                seq_has_violation = True

        # 🛡️ DYNAMIC STATE VARIABLES HARD BOUNDS CHECK
        state_dim = states.shape[-1] 
        for i in range(state_dim):
            single_state_seq = states[:, i]
            x_min = training_data_cfg.get(f"x_{i+1}_hard_min")
            x_max = training_data_cfg.get(f"x_{i+1}_hard_max")
            
            if x_min is not None and np.min(single_state_seq) < x_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] State variable x_{i+1} dropped below limit! "
                      f"Bound: {x_min}, Min Found: {np.min(single_state_seq):.4f}\033[0m")
                seq_has_violation = True
                
            if x_max is not None and np.max(single_state_seq) > x_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] State variable x_{i+1} exceeded limit! "
                      f"Bound: {x_max}, Max Found: {np.max(single_state_seq):.4f}\033[0m")
                seq_has_violation = True

        # Rejection handling
        if seq_has_violation:
            violated_sequences_count += 1
            print(f"\033[91m🛑 Excluding [Seq {s_idx}] from final dataset structures.\033[0m")
            continue

        # 📈 Calculate Pearson Correlation for filtering
        seq_corr_metrics = {"sequence_index": valid_idx_counter}
        drop_due_to_correlation = False
        min_correlation_threshold = training_data_cfg["min_correlation_threshold"]
        
        for u_idx in range(input_dim):
            single_u_curve = u[:, u_idx]
            for y_idx in range(input_dim):
                single_y_curve = y_t[:, y_idx]
                
                if np.std(single_u_curve) == 0 or np.std(single_y_curve) == 0:
                    corr_val = 0.0
                else:
                    corr_val = np.corrcoef(single_u_curve, single_y_curve)[0, 1]
                
                if np.abs(corr_val) < min_correlation_threshold:
                    drop_due_to_correlation = True
                    print(f"\033[94mℹ️ INFO: [Seq {s_idx}] Rejected. Low correlation on u_{u_idx+1}──y_{y_idx+1} ({corr_val:+.4f})\033[0m")
                
                seq_corr_metrics[f"corr_u{u_idx+1}_y{y_idx+1}"] = corr_val

        if drop_due_to_correlation:
            violated_sequences_count += 1
            print(f"\033[91m🛑 Excluding [Seq {s_idx}] due to weak u-y behavior mapping.\033[0m")
            continue

        # Save valid references to our clean tracking lists
        valid_u_list.append(u_raw[s_idx])
        valid_y_list.append(y_raw[s_idx])
        valid_states_list.append(state_raw[s_idx])
        per_sequence_correlations.append(seq_corr_metrics)

        # Construct CSV log files dynamically
        time_axis = np.arange(len(u)) * dt
        columns, values = ["t"], [time_axis]

        for i in range(output_dim):
            columns.append(f"y_{i+1}")
            values.append(y_t[:, i])
        for i in range(input_dim):
            columns.append(f"u_{i+1}")
            values.append(u[:, i])
        for i in range(state_dim):
            columns.append(f"x_{i+1}")
            values.append(states[:, i])

        d_center = np.squeeze(batch_d_centers_np[s_idx])
        for i in range(input_dim):
            columns.append(f"D_center_u_{i+1}")
            d_val = float(d_center[i]) if output_dim > 1 else float(d_center)
            values.append(np.full(len(time_axis), d_val))

        seq_df = pd.DataFrame({col: val for col, val in zip(columns, values)})
        valid_dfs.append(seq_df)
        
        filename_base = f"sequence_{valid_idx_counter}.csv"
        valid_idx_counter += 1
        
        if save_logs:
            save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)

        if show_plots:
            # Reuses your plotting routine using y_t as the base trajectory
            plot_configs = plant.get_plot_config()
            u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
            y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
                
            signals_to_plot = []
            labels_to_plot = []
            ylabels_to_plot = []

            for idx in range(u.shape[1]):
                signals_to_plot.append(u[:, idx])
                labels_to_plot.append([None])
                ylabels_to_plot.append(u_config["labels"][idx] if u_config else rf"Input $u_{{{idx+1}}}$")

            for idx in range(y_t.shape[1]):
                signals_to_plot.append(y_t[:, idx])
                labels_to_plot.append([None])
                ylabels_to_plot.append(y_config["labels"][idx] if y_config else rf"Output $y_{{{idx+1}}}$")

            dynamic_asp = [0.33] * len(signals_to_plot)
            plot_stacked(
                t=time_axis,
                signals=signals_to_plot,
                labels=labels_to_plot,
                xlabel=rf"$t$ [$\mathrm{{h}}$]",
                ylabel=ylabels_to_plot,
                asp=dynamic_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_plot.png",
                show=True
            )
            # =================================================================
            # 3. SEPARATE STACKED PLOT FOR STATE VARIABLES (No legends needed)
            # =================================================================
            state_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)
            
            state_signals = []
            state_labels = []
            state_ylabels = []

            num_states = states.shape[1]  
            for idx in range(num_states):
                state_signals.append(states[:, idx])
                state_labels.append([None])  # Prevents redundant legend
                
                if state_config and idx < len(state_config["labels"]):
                    state_ylabels.append(state_config["labels"][idx])
                else:
                    state_ylabels.append(rf"State $x_{{{idx+1}}}$")

            state_asp = [0.33] * len(state_signals)
            
            plot_stacked(
                t=time_axis,
                signals=state_signals,
                labels=state_labels,
                xlabel=rf"$t$ [$\mathrm{{h}}$]",
                ylabel=state_ylabels,
                asp=state_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_states_plot.png",
                show=True
            )

            # =================================================================
            # 3. CONSOLIDATED STACKED PLOT FOR ALL SIGNALS (u, y, and x)
            # =================================================================
            all_signals = signals_to_plot + state_signals
            all_labels = labels_to_plot + state_labels
            all_ylabels = ylabels_to_plot + state_ylabels
            all_asp = [0.33] * len(all_signals)

            plot_stacked(
                t=time_axis,
                signals=all_signals,
                labels=all_labels,
                xlabel=rf"$t$ [$\mathrm{{h}}$]",
                ylabel=all_ylabels,
                asp=all_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_all_stacked_plot.png",
                show=True
            )

    # 🚨 FINAL BOUNDS VIOLATION SUMMARY
    violation_percentage = (violated_sequences_count / total_sequences) * 100
    valid_sequences_count = total_sequences - violated_sequences_count

    print("\n" + "="*60)
    print("📊 BOUNDS CHECK SUMMARY REPORT (y, u, & all states)")
    print(f"Total Raw Sequences Evaluated : {total_sequences}")
    if violated_sequences_count > 0:
        print(f"\033[91m\033[1m❌ Out-of-Bounds Curves Found : {violated_sequences_count} curves ({violation_percentage:.2f}%)\033[0m")
        print(f"\033[32m\033[1m✓ Clean Saved Dataset Curves   : {valid_sequences_count} curves accepted\033[0m")
    else:
        print("\033[92m\033[1m   All generated curves are within the defined hard boundaries! (100% accepted)\033[0m")
    print("="*60 + "\n")

    if valid_sequences_count == 0:
        print("\033[91m\033[1mCRITICAL ERROR: 100% of generated data curves violated bounds. No files exported.\033[0m")
        return

    # 3. Stack only validated raw tensors together 
    final_u_tensor = torch.stack(valid_u_list, dim=0).to(device)
    final_y_tensor = torch.stack(valid_y_list, dim=0).to(device)
    final_state_tensor = torch.stack(valid_states_list, dim=0).to(device)
    plot_config = plant.get_plot_config()
    # 📈 NEW: Plot all validated u trajectories in one figure
    if show_overlay_plot and valid_sequences_count > 0:
        plot_all_signals_overlay(
            u_tensor=final_u_tensor,  # Control Inputs u (subplot 1)
            y_tensor=final_y_tensor,  # System Outputs y (subplot 2)
            dt=plant.dt,
            dirname="plots",
            plot_config=plot_config,
            show_plot=True,
        )

        
    
    # Save global stats
    if per_sequence_correlations:
        per_seq_df = pd.DataFrame(per_sequence_correlations)
        save_df_to_csv(per_seq_df, dirname=dirname, filename="mimo_per_sequence_correlations.csv")
    
    data_to_save = {
        "u": final_u_tensor.cpu(),
        "y": final_y_tensor.cpu(),
        "states": final_state_tensor.cpu()
    }
    save_training_dataset(data_to_save, dirname=dirname)
    
    

    return data_to_save

