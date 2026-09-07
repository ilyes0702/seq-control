"""
Plotting Utility Functions
=========================

This module contains utilities for plotting.
"""


from seq_control.utils.saving_and_loading_utils import save_plot_image

import optuna

import numpy as np
# IMPORT VISUALIZATION MODULES
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os



def plot_all_signals_overlay(
    u_tensor,
    y_tensor,
    dt,
    dirname,
    x_tensor=None,
    plot_config=None,
    xlim=None,
    show_plot=True,
    filename="all_signals_overlay.png",
):
    """Plots input ($u$) and output ($y$) overlay sequences stacked vertically,

    sharing a unified horizontal time axis.

    Parameters:
    -----------
    u_tensor : torch.Tensor or np.ndarray
        Control inputs tensor of shape [Num_Seqs, Seq_Len, u_Channels] or [Num_Seqs, Seq_Len].
    y_tensor : torch.Tensor or np.ndarray
        System outputs tensor of shape [Num_Seqs, Seq_Len, y_Channels] or [Num_Seqs, Seq_Len].
    dt : float
        Sampling time step.
    dirname : str
        Directory where the figure will be saved.
    x_tensor : torch.Tensor or np.ndarray, optional
        Optional state tensor of shape [Num_Seqs, Seq_Len, x_Channels].
    plot_config : list of dict, optional
        Config dictionary array returned by plant.get_plot_config().
    xlim : tuple of (float, float), optional
        Limits for the shared x-axis (e.g., (-1, 25)).
    show_plot : bool, optional
        Whether to display the plot via plt.show() or close the figure.
    filename : str, optional
        Name of the saved image file.
    """

    # Helper function to convert PyTorch Tensors to 3D NumPy arrays [Seqs, Len, Channels]
    def _to_numpy(tensor):
        if hasattr(tensor, "cpu"):
            arr = tensor.cpu().numpy()
        else:
            arr = np.asarray(tensor)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr

    u_np = _to_numpy(u_tensor)
    y_np = _to_numpy(y_tensor)

    num_seqs, seq_len, u_channels = u_np.shape
    time_axis = np.arange(seq_len) * dt

    # Helper function to extract channel y-labels from plant.get_plot_config()
    def _get_labels(type_key, num_channels):
        if plot_config is not None:
            for cfg in plot_config:
                if any(type_key in col.lower() for col in cfg["cols"]):
                    ylabels = cfg["ylabel"]
                    if isinstance(ylabels, list):
                        return ylabels
                    return [ylabels]
        return [rf"${type_key}_{{{i+1}}}$" for i in range(num_channels)]

    # Build ordered list of (data_channel_slice, y_label) tuples
    # Order: Inputs (u) -> Outputs (y) -> States (x, if provided)
    plot_channels = []

    # 1. Inputs (u)
    u_labels = _get_labels("u", u_channels)
    for c in range(u_channels):
        plot_channels.append((u_np[:, :, c], u_labels[c]))

    # 2. Outputs (y)
    y_channels = y_np.shape[2]
    y_labels = _get_labels("y", y_channels)
    for c in range(y_channels):
        plot_channels.append((y_np[:, :, c], y_labels[c]))

    # 3. States (x) - optional
    if x_tensor is not None:
        x_np = _to_numpy(x_tensor)
        x_channels = x_np.shape[2]
        x_labels = _get_labels("x", x_channels)
        for c in range(x_channels):
            plot_channels.append((x_np[:, :, c], x_labels[c]))

    total_subplots = len(plot_channels)

    # Create vertically stacked subplots sharing the time axis
    fig, axes = plt.subplots(
        total_subplots, 1, figsize=(10, 2.8 * total_subplots), sharex=True
    )
    if total_subplots == 1:
        axes = [axes]

    # Plot each channel
    for idx, (sig_data, ylabel) in enumerate(plot_channels):
        ax = axes[idx]

        # Overlay individual sequence trajectories
        for s_idx in range(num_seqs):
            label = "Validated Sequences" if s_idx == 0 else None
            ax.plot(
                time_axis,
                sig_data[s_idx, :],
                color="tab:blue",
                alpha=0.25,
                linewidth=1.0,
                label=label,
            )

        # Highlight ensemble mean trajectory
        mean_sig = np.mean(sig_data, axis=0)
        ax.plot(
            time_axis,
            mean_sig,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Ensemble Mean",
        )

        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.legend(loc="upper right", framealpha=0.9)

    # Set x-label and x-limits on the bottom shared axis
    axes[-1].set_xlabel(r"$t \; [\mathrm{h}]$")
    if xlim is not None:
        axes[-1].set_xlim(xlim)

    plt.tight_layout()

    # Save to disk
    os.makedirs(dirname, exist_ok=True)
    save_path = os.path.join(dirname, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"🖼️ Stacked overlay plot saved to: {save_path}")
    
#=== FUNCTION TO PLOT SIGNALS ===#
def plot_signals(
    t,
    signals,
    labels=None,
    title=None,
    xlabel="Time",
    ylabel="Value",
    save_path=None,
    show=False,
    filename=None,
    dirname=None,
    asp=1.0,
    
):
    """
    Generic plotting function for time-series or x-y signals.

    The function acts as a uniform parsing layout for multi-trace system tracking data. 
    It sets up the canvas grid framework, sequentially maps input arrays onto a scatter 
    axis topology, and forces a strict normalized 1:1 rectangular aspect ratio overlay. 
    It maps peripheral aesthetic labels, serializes raw canvas elements into an in-memory 
    high-density byte stream buffer, and wraps the result into a PIL image construct for 
    flexible external logging or automated directory routing.

    :param t: The independent variable timeline array for the horizontal x-axis.
    :type t: numpy.ndarray | list
    :param signals: Collection of dependent signal arrays to map along the vertical y-axis.
    :type signals: list[numpy.ndarray]
    :param labels: Text labels matched by index sequence to identify each unique signal, 
        defaults to None.
    :type labels: list[str], optional
    :param title: Overarching header title text displayed at the top of the plot grid, 
        defaults to None.
    :type title: str, optional
    :param xlabel: Explicit label tracking the horizontal x-axis context, 
        defaults to "Time".
    :type xlabel: str, optional
    :param ylabel: Explicit label tracking the vertical y-axis context, 
        defaults to "Value".
    :type ylabel: str, optional
    :param figsize: Specific scale layout dimensions for the graphic asset canvas, 
        defaults to (5, 5).
    :type figsize: tuple[float, float], optional
    :param save_path: A targeted physical file path to write the initial vector plot graphic, 
        defaults to None.
    :type save_path: str, optional
    :param show: Toggle flag which forces standard Matplotlib UI canvas rendering if ``True``, 
        defaults to ``False``.
    :type show: bool, optional
    :param filename: Defined target image label for processing custom raster export sequences, 
        defaults to None.
    :type filename: str, optional
    :param dirname: System subfolder location targeted for writing custom raster image files, 
        defaults to None.
    :type dirname: str, optional

    :return: A high-resolution raster image object version of the finalized signal canvas.
    :rtype: PIL.Image.Image
    """
    base_width = 7
    fig, ax = plt.subplots(figsize=(base_width, base_width * asp),layout="constrained")

    # Iterate and render each signal track onto the common subplot grid
    for i, sig in enumerate(signals):
        if labels is not None:
            ax.plot(t, sig, label=labels[i])
        else:
            ax.plot(t, sig)

    # === FIXED ASPECT RATIO === #
    # Standardize scale boundaries to maintain geometric proportions
    x_range = np.diff(ax.get_xlim())[0]
    y_range = np.diff(ax.get_ylim())[0]
    range_ratio = x_range / y_range
    ax.set_aspect(asp * range_ratio)
    #ax.set_box_aspect(asp)

    # Apply axis descriptors and grid annotations
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if labels:
        ax.legend()

    # Handle standard initial vector export if path properties are configured
    if save_path:
        fig.savefig(save_path, 
                    dpi=300, 
                    bbox_inches="tight", 
                    pad_inches=0.0)

    if show:
        plt.show()

    # plt.tight_layout()

    # Serialize the complete plot canvas into an active memory buffer stream
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the memory stream into an editable PIL Image representation
    image = Image.open(buf)
    
    # Process custom external tracking image storage
    save_plot_image(image=image, filename=filename, dirname=dirname)
    
    return image


def plot_all_signals_overlay(
    u_tensor,
    y_tensor,
    dt,
    dirname,
    x_tensor=None,
    plot_config=None,
    xlim=None,
    show_plot=True,
    filename="all_signals_overlay.png",
):
    """
    Plots input (:math:`u`) and output (:math:`y`) overlay sequences stacked vertically,
    sharing a unified horizontal time axis.

    :param u_tensor: Control inputs tensor of shape ``[Num_Seqs, Seq_Len, u_Channels]`` 
        or ``[Num_Seqs, Seq_Len]``.
    :type u_tensor: torch.Tensor | numpy.ndarray
    :param y_tensor: System outputs tensor of shape ``[Num_Seqs, Seq_Len, y_Channels]`` 
        or ``[Num_Seqs, Seq_Len]``.
    :type y_tensor: torch.Tensor | numpy.ndarray
    :param dt: Sampling time step.
    :type dt: float
    :param dirname: Directory where the figure will be saved.
    :type dirname: str
    :param x_tensor: Optional state tensor of shape ``[Num_Seqs, Seq_Len, x_Channels]``, 
        defaults to None.
    :type x_tensor: torch.Tensor | numpy.ndarray, optional
    :param plot_config: Config dictionary array returned by ``plant.get_plot_config()``, 
        defaults to None.
    :type plot_config: list[dict], optional
    :param xlim: Limits for the shared x-axis (e.g., ``(-1, 25)``), defaults to None.
    :type xlim: tuple[float, float], optional
    :param show_plot: Whether to display the plot via ``plt.show()`` or close the figure, 
        defaults to ``False``.
    :type show_plot: bool, optional
    :param filename: Name of the saved image file, defaults to None.
    :type filename: str, optional
    """

    # Helper function to convert PyTorch Tensors to 3D NumPy arrays [Seqs, Len, Channels]
    def _to_numpy(tensor):
        if hasattr(tensor, "cpu"):
            arr = tensor.cpu().numpy()
        else:
            arr = np.asarray(tensor)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr

    u_np = _to_numpy(u_tensor)
    y_np = _to_numpy(y_tensor)

    num_seqs, seq_len, u_channels = u_np.shape
    time_axis = np.arange(seq_len) * dt

    # Helper function to extract channel y-labels from plant.get_plot_config()
    def _get_labels(type_key, num_channels):
        if plot_config is not None:
            for cfg in plot_config:
                if any(type_key in col.lower() for col in cfg["cols"]):
                    ylabels = cfg["ylabel"]
                    if isinstance(ylabels, list):
                        return ylabels
                    return [ylabels]
        return [rf"${type_key}_{{{i+1}}}$" for i in range(num_channels)]

    # Build ordered list of (data_channel_slice, y_label) tuples
    # Order: Inputs (u) -> Outputs (y) -> States (x, if provided)
    plot_channels = []

    # 1. Inputs (u)
    u_labels = _get_labels("u", u_channels)
    for c in range(u_channels):
        plot_channels.append((u_np[:, :, c], u_labels[c]))

    # 2. Outputs (y)
    y_channels = y_np.shape[2]
    y_labels = _get_labels("y", y_channels)
    for c in range(y_channels):
        plot_channels.append((y_np[:, :, c], y_labels[c]))

    # 3. States (x) - optional
    if x_tensor is not None:
        x_np = _to_numpy(x_tensor)
        x_channels = x_np.shape[2]
        x_labels = _get_labels("x", x_channels)
        for c in range(x_channels):
            plot_channels.append((x_np[:, :, c], x_labels[c]))

    total_subplots = len(plot_channels)

    # Create vertically stacked subplots sharing the time axis
    fig, axes = plt.subplots(
        total_subplots, 1, figsize=(10, 2.8 * total_subplots), sharex=True
    )
    if total_subplots == 1:
        axes = [axes]

    # Plot each channel
    for idx, (sig_data, ylabel) in enumerate(plot_channels):
        ax = axes[idx]

        # Overlay individual sequence trajectories
        for s_idx in range(num_seqs):
            label = "Validated Sequences" if s_idx == 0 else None
            ax.plot(
                time_axis,
                sig_data[s_idx, :],
                color="tab:blue",
                alpha=0.25,
                linewidth=1.0,
                label=label,
            )

        # Highlight ensemble mean trajectory
        mean_sig = np.mean(sig_data, axis=0)
        ax.plot(
            time_axis,
            mean_sig,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Ensemble Mean",
        )

        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.legend(loc="upper right", framealpha=0.9)

    # Set x-label and x-limits on the bottom shared axis
    axes[-1].set_xlabel(r"$t \; [\mathrm{h}]$")
    if xlim is not None:
        axes[-1].set_xlim(xlim)

    plt.tight_layout()

    # Save to disk
    os.makedirs(dirname, exist_ok=True)
    save_path = os.path.join(dirname, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"🖼️ Stacked overlay plot saved to: {save_path}")


def plot_param_heatmap(
    study,
    param_x,
    param_y,
    title=None,
    figsize=(6, 6),
    filename=None,
    dirname=None,
    asp=1.0,
    show=False,
):
    """
    Generates a 2D objective surface heatmap for any pair of hyperparameters
    from an Optuna study, following the project canvas rendering pipeline.

    :param study: Completed or ongoing Optuna study object.
    :type study: optuna.study.Study
    :param param_x: Name of the hyperparameter mapped to the horizontal x-axis.
    :type param_x: str
    :param param_y: Name of the hyperparameter mapped to the vertical y-axis.
    :type param_y: str
    :param title: Custom title header (defaults to ``"Optuna Loss Heatmap: {param_y} vs {param_x}"``), 
        defaults to None.
    :type title: str, optional
    :param figsize: Canvas size layout dimensions, defaults to (6, 6).
    :type figsize: tuple[float, float], optional
    :param filename: Output image name (defaults to ``"{param_y}_vs_{param_x}_heatmap"``), 
        defaults to None.
    :type filename: str, optional
    :param dirname: Directory path for saving the raster image, defaults to None.
    :type dirname: str, optional
    :param asp: Aspect ratio scalar modifier, defaults to 1.0.
    :type asp: float, optional
    :param show: Toggle UI canvas display, defaults to ``False``.
    :type show: bool, optional

    :return: A high-resolution raster image object version of the generated heatmap.
    :rtype: PIL.Image.Image

    :raises ValueError: If no completed trials in the study contain both requested 
        hyperparameters ``param_x`` and ``param_y``.
    """
    # 1. Extract all unique sampled values for param_x and param_y
    x_vals = set()
    y_vals = set()

    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if param_x in trial.params and param_y in trial.params:
                x_vals.add(trial.params[param_x])
                y_vals.add(trial.params[param_y])

    if not x_vals or not y_vals:
        raise ValueError(
            f"No completed trials found containing both hyperparameters: '{param_x}' and '{param_y}'."
        )

    # Sort tick labels for clean axis ordering
    x_ticks = sorted(list(x_vals))
    y_ticks = sorted(list(y_vals))

    x_to_idx = {val: idx for idx, val in enumerate(x_ticks)}
    y_to_idx = {val: idx for idx, val in enumerate(y_ticks)}

    # 2. Initialize grid with NaNs
    grid = np.full((len(y_ticks), len(x_ticks)), np.nan)

    # 3. Populate matrix (aggregating via minimum if duplicates exist)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            px = trial.params.get(param_x)
            py = trial.params.get(param_y)
            val = trial.value

            if px in x_to_idx and py in y_to_idx and val is not None:
                row_idx = y_to_idx[py]
                col_idx = x_to_idx[px]
                if np.isnan(grid[row_idx, col_idx]):
                    grid[row_idx, col_idx] = val
                else:
                    grid[row_idx, col_idx] = min(grid[row_idx, col_idx], val)

    # 4. Render canvas
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    cax = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Mean Validation Loss")

    # Set dynamic axis tick labels
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_yticklabels([str(y) for y in y_ticks])

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)

    if title is None:
        ax.set_title(f"Optuna Loss Heatmap: {param_y} vs {param_x}")
    elif title:
        ax.set_title(title)

    # Overlay numerical cell values
    mean_val = np.nanmean(grid)
    for i in range(len(y_ticks)):
        for j in range(len(x_ticks)):
            val = grid[i, j]
            if not np.isnan(val):
                # Format floating point numbers cleanly
                text_val = f"{val:.4f}" if val < 1.0 else f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    text_val,
                    ha="center",
                    va="center",
                    color="white" if val > mean_val else "black",
                    fontsize=8,
                )

    # Dynamic aspect ratio scaling based on grid dimensions
    range_ratio = len(x_ticks) / len(y_ticks)
    ax.set_aspect(asp * range_ratio)

    if show:
        plt.show()

    # 5. Pipeline export via PIL buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    image = Image.open(buf)

    out_filename = (
        filename if filename else f"{param_y}_vs_{param_x}_heatmap"
    )
    save_plot_image(image=image, filename=out_filename, dirname=dirname)

    return image


def plot_stacked(
    t,
    signals,
    plot_config=None,  # Accepts output from get_plot_config()
    labels=None,
    title=None,
    xlabel="Time",
    ylabel="Value",
    save_path=None,
    show=False,
    filename=None,
    dirname=None,
    asp=0.33,
    hspace=0.05,
):
    """
    Generic plotting function that stacks various subplots vertically sharing a unified x-axis.

    Parses optional configuration structures to configure labels, line legends, aspect ratios, 
    and y-axis titles per subplot. Converts the generated figure into an in-memory 
    PIL image object for logging, optional file saving, or interactive rendering.

    :param t: The independent timeline variable array mapped to the shared horizontal x-axis.
    :type t: numpy.ndarray | list
    :param signals: Collection of signal arrays or groups of signal arrays. Each entry 
        corresponds to a subplot row and can contain single or multiple trace arrays.
    :type signals: list[numpy.ndarray | list[numpy.ndarray]]
    :param plot_config: Subplot configuration list (e.g., returned by plant/model 
        configuration methods) used to extract axis labels and subplot titles automatically, 
        defaults to None.
    :type plot_config: list[dict], optional
    :param labels: Explicit list of legend label sequences corresponding to each subplot 
        row, defaults to None.
    :type labels: list[list[str] | str], optional
    :param title: Main header title displayed above the top subplot row, defaults to None.
    :type title: str, optional
    :param xlabel: Label text tracking the horizontal x-axis context, defaults to "Time".
    :type xlabel: str, optional
    :param ylabel: Label or collection of labels tracking the vertical y-axis context for 
        individual subplots, defaults to "Value".
    :type ylabel: str | list[str], optional
    :param save_path: Targeted file system path to export the high-resolution output figure, 
        defaults to None.
    :type save_path: str, optional
    :param show: If ``True``, displays the Matplotlib UI plot window before closing, 
        defaults to ``False``.
    :type show: bool, optional
    :param filename: Target filename for automated image logging utilities, defaults to None.
    :type filename: str, optional
    :param dirname: Subdirectory path targeted for writing automated image exports, 
        defaults to None.
    :type dirname: str, optional
    :param asp: Fixed box aspect ratio (height-to-width ratio) or a list of ratios applied 
        to each individual subplot, defaults to 0.33.
    :type asp: float | list[float], optional
    :param hspace: Vertical padding space separating adjacent stacked subplots, 
        defaults to 0.05.
    :type hspace: float, optional

    :return: A high-resolution raster image object version of the stacked multi-trace plot.
    :rtype: PIL.Image.Image
    """
    # === 1. PARSE PLOT CONFIG IF PROVIDED ===
    if plot_config is not None:
        # Separate time/x-axis config from subplot configs
        x_cfg = next(
            (c for c in plot_config if "xlabel" in c or "t" in c.get("cols", [])),
            None,
        )
        sub_cfgs = [c for c in plot_config if c != x_cfg]

        # Extract xlabel
        if x_cfg and "xlabel" in x_cfg:
            raw_xl = x_cfg["xlabel"]
            xlabel = raw_xl[0] if isinstance(raw_xl, (list, tuple)) else raw_xl

        # Extract legend labels and ylabels for each subplot row
        labels = [c.get("labels") for c in sub_cfgs]

        extracted_ylabels = []
        for c in sub_cfgs:
            raw_yl = c.get("ylabel", "")
            if isinstance(raw_yl, (list, tuple)):
                raw_yl = " / ".join(raw_yl)  # Combines e.g. "$x_1$ [g] / $x_2$ [mg]"
            extracted_ylabels.append(raw_yl)

        ylabel = extracted_ylabels

    num_subplots = len(signals)

    fig, axes = plt.subplots(
        nrows=num_subplots, ncols=1, sharex=True, figsize=(7, 1 * num_subplots)
    )

    if num_subplots == 1:
        axes = [axes]

    fig.subplots_adjust(hspace=hspace)

    # Iterate and render each subplot row
    for i, sig_group in enumerate(signals):
        ax = axes[i]

        is_multi = isinstance(sig_group, (list, tuple)) and not isinstance(
            sig_group[0], (int, float)
        )
        curves = sig_group if is_multi else [sig_group]
        row_labels = labels[i] if labels is not None else None

        if row_labels is not None:
            if not isinstance(row_labels, (list, tuple)):
                row_labels = [row_labels]

        for j, sig in enumerate(curves):
            lbl = (
                row_labels[j]
                if row_labels is not None and j < len(row_labels)
                else None
            )
            ax.plot(t, sig, label=lbl)

        # === LEGEND ONLY ON TOP SUBPLOT ===
        if (
            i == 0
            and row_labels is not None
            and any(lbl is not None for lbl in row_labels)
        ):
            ax.legend(loc="upper right")

        # === SET Y-AXIS LABEL ===
        if isinstance(ylabel, (list, tuple)) and len(ylabel) == num_subplots:
            current_ylabel = ylabel[i]
        elif isinstance(ylabel, str):
            current_ylabel = (
                ylabel if num_subplots == 1 else f"{ylabel} {i+1}"
            )
        else:
            current_ylabel = f"Signal {i+1}"

        ax.set_ylabel(current_ylabel)

        # === ASPECT RATIO ===
        if isinstance(asp, (list, tuple)) and len(asp) == num_subplots:
            current_asp = asp[i]
        else:
            current_asp = asp if isinstance(asp, (int, float)) else 0.33

        if current_asp is not None and hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(current_asp)

        # === CLEAN UP X-TICK LABELS ===
        if i < num_subplots - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel(xlabel)

    if title:
        axes[0].set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()

    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=600, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    image = Image.open(buf)
    if "save_plot_image" in globals():
        save_plot_image(image=image, filename=filename, dirname=dirname)

    return image
    



