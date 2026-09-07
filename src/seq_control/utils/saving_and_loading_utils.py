"""
Saving and Loading Utility Functions
=========================

This module contains utilities for saving and loading trained models, datasets and scalers.
"""


import os
import torch
import pandas as pd
import json
import numpy as np

from PIL import Image
from seq_control.config import date as default_date
from seq_control.config import date_and_time as default_date_and_time
import pickle

def save_model_esn(model, dirname, hyperparam_config, filename="best_fold_model"):
    """
    Save the Echo State Network (ESN) model state and configuration to disk.

    This function serializes the reservoir model's state dictionary—including
    the ReservoirPy pipeline structure and trained readout weights—alongside
    its hyperparameter configuration dictionary into a single pickle binary file.

    :param model: The wrapper or container holding the ReservoirPy ESN model.
                  Must expose a ``model`` attribute containing the underlying
                  pipeline object.
    :type model: object
    :param dirname: Path to the target directory where the model checkpoint
                    will be saved. Created recursively if it does not exist.
    :type dirname: str
    :param hyperparam_config: Dictionary containing the hyperparameters and
                              architectural settings used to configure the ESN.
    :type hyperparam_config: dict
    :param filename: Base filename for the saved pickle checkpoint without the
                     file extension. Defaults to ``"best_fold_model"``.
    :type filename: str, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If directory creation or file writing fails due to permissions
                     or invalid path syntax.
    :raises AttributeError: If the provided ``model`` object lacks a ``model``
                            attribute.

    .. note::
       The saved checkpoint contains both the model architecture state dict and
       the hyperparameter configuration, making it compatible with matching reload
       routines for inference or further evaluation.
    """
    os.makedirs(dirname, exist_ok=True)
    checkpoint_path = os.path.join(dirname, f"{filename}.pkl")
    
    checkpoint = {
        'config': hyperparam_config,
        'model_state_dict': model.model  # Saves the ReservoirPy pipeline layout + trained readout
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"💾 ESN Model successfully saved to: {checkpoint_path}")

def save_esn_parameters_to_csv(model, fold_dir):
    """
    Helper function to safely extract and save ESN weight matrices 
    (both trained readout weights and untrained reservoir weights) to CSV.
    """
    print(f"💾 Exporting ESN weight matrices to CSV in: {fold_dir}")
    
    # 1. Handle ReservoirPy Model or Node structure
    if hasattr(model, "nodes"):
        for node in model.nodes:
            node_name = node.name.lower()
            # Untrained Reservoir Parameters
            if "reservoir" in node_name:
                if hasattr(node, "Win") and node.Win is not None:
                    pd.DataFrame(node.Win).to_csv(os.path.join(fold_dir, "esn_W_in_untrained.csv"), index=False, header=False)
                if hasattr(node, "W") and node.W is not None:
                    pd.DataFrame(node.W).to_csv(os.path.join(fold_dir, "esn_W_reservoir_untrained.csv"), index=False, header=False)
                if hasattr(node, "bias") and node.bias is not None:
                    pd.DataFrame(node.bias).to_csv(os.path.join(fold_dir, "esn_bias_untrained.csv"), index=False, header=False)
            
            # Trained Readout Parameters
            elif "ridge" in node_name or "readout" in node_name:
                # ReservoirPy stores readout weights either in 'Wout' or 'W' depending on version/configuration
                w_out = getattr(node, "Wout", getattr(node, "W", None))
                if w_out is not None:
                    pd.DataFrame(w_out).to_csv(os.path.join(fold_dir, "esn_W_out_trained.csv"), index=False, header=False)
                if hasattr(node, "bias") and node.bias is not None:
                    pd.DataFrame(node.bias).to_csv(os.path.join(fold_dir, "esn_readout_bias_trained.csv"), index=False, header=False)

    # 2. Fallback for custom or flat objects (e.g., model.W_in, model.W, model.W_out)
    else:
        # Untrained parameters
        for attr_name, file_name in [("W_in", "esn_W_in_untrained.csv"), 
                                     ("Win", "esn_W_in_untrained.csv"),
                                     ("W", "esn_W_reservoir_untrained.csv"), 
                                     ("W_res", "esn_W_reservoir_untrained.csv"),
                                     ("bias", "esn_bias_untrained.csv")]:
            if hasattr(model, attr_name):
                weights = getattr(model, attr_name)
                if weights is not None:
                    pd.DataFrame(np.asarray(weights)).to_csv(os.path.join(fold_dir, file_name), index=False, header=False)
        
        # Trained parameters
        for attr_name, file_name in [("W_out", "esn_W_out_trained.csv"), 
                                     ("Wout", "esn_W_out_trained.csv")]:
            if hasattr(model, attr_name):
                weights = getattr(model, attr_name)
                if weights is not None:
                    pd.DataFrame(np.asarray(weights)).to_csv(os.path.join(fold_dir, file_name), index=False, header=False)

#=== FUNCTION TO SAVE TRAINED MODEL ===#
def save_model(model, dirname, hyperparam_config, filename="trained_controller"):
    """
    Save the model state dictionary and configuration to a timestamped directory.

    This function serializes a PyTorch model's parameters along with its associated
    hyperparameter configuration dictionary into a single checkpoint file (`.pt`).
    The file is saved inside a structured subdirectory path based on global training
    session dates and timestamps. If the generated file path exceeds the operating
    system path limit, the filename is safely truncated.

    :param model: The PyTorch neural network model instance whose state dictionary
                  will be serialized. Must implement the ``state_dict()`` method.
    :type model: torch.nn.Module
    :param dirname: Subdirectory name or path within the timestamped output folder
                    where the checkpoint will be stored.
    :type dirname: str
    :param hyperparam_config: Dictionary containing training hyperparameters, architecture
                              settings, or contextual metadata to persist alongside weights.
    :type hyperparam_config: dict
    :param filename: Base label for the checkpoint file without the ``.pt`` extension.
                     Defaults to ``"trained_controller"``.
    :type filename: str, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If directory creation fails or file write permissions are denied.
    :raises AttributeError: If the provided ``model`` object does not implement ``state_dict()``.

    .. note::
       - The directory path is constructed as ``results/<global_date>/<timestamp>/<dirname>/``.
       - The final saved checkpoint file includes a prefix matching the training session's
         ``timestamp``.
       - Path lengths exceeding 255 characters are automatically truncated at the filename
         level to prevent file system operations from throwing path length errors.
    """
    # Assuming 'date' and 'date_and_time' are defined globally 
    # or extracted from your training session context
    global_date = default_date # e.g., "2026-04-29"
    timestamp = default_date_and_time # e.g., "2026-04-29_11-22"

    # Construct the directory and filename logic
    model_dir = f"src/seq_control/results/{global_date}/{timestamp}/{dirname}/"
    save_filename = f"{timestamp}_{filename}.pt"

    
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, save_filename)
    
    # Check path length (keeping consistent with your CSV function)
    max_path_length = 255
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(save_filename)
        allowed_len = max_path_length - len(os.path.join(model_dir, ext))
        save_filename = basename[:allowed_len] + ext
        full_path = os.path.join(model_dir, save_filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': hyperparam_config,
    }
    
    torch.save(checkpoint, full_path)
    print(f"💾 Model saved to: {full_path}")

#=== FUNCTION TO SAVE DATAFRAME AS CSV IN SPECIFIED DIRECTORY ===#
def save_df_to_csv(df, dirname, filename, max_path_length=255):
    """
    Save a pandas DataFrame as a CSV file to a timestamped directory structure.

    This function rounds numerical float columns in the DataFrame to six decimal
    places, constructs a standardized directory path using global date and timestamp
    context, and exports the data to disk. It handles automatic directory creation
    and truncates the target filename if the total file path exceeds a specified
    maximum character limit.

    :param df: The pandas DataFrame object to be exported to CSV.
    :type df: pandas.DataFrame
    :param dirname: Subdirectory segment within the timestamped output directory
                    where the output report folder will be nested.
    :type dirname: str
    :param filename: Base name for the CSV file (e.g., ``"evaluation_metrics"`` or
                     ``"results.csv"``). Note that a timestamp prefix and extension
                     are automatically managed during path construction.
    :type filename: str
    :param max_path_length: Maximum allowable character length for the target total
                            file path. Defaults to ``255``.
    :type max_path_length: int, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If folder creation fails or file write operations are denied by the OS.
    :raises AttributeError: If the input object lacks pandas-compatible ``round()`` or ``to_csv()`` methods.

    .. note::
       - Floating-point values are rounded using ``df.round(6)`` prior to writing.
       - The directory hierarchy is dynamically set to ``results/<default_date>/<default_date_and_time>/<dirname>/reports/``.
       - If the generated ``full_path`` exceeds ``max_path_length``, the filename stem is safely truncated to ensure the path fits within OS boundaries.
       - The DataFrame index is excluded from the output file (``index=False``).
    """
    # Round all float columns to 2 decimal places
    df = df.round(6)
    
     # Check full path length
    dirname = f"src/seq_control/results/{default_date}/{default_date_and_time}/{dirname}/reports/"
    filename = f"{default_date_and_time}_{filename}.csv"
    os.makedirs(dirname, exist_ok=True)
    full_path = os.path.join(dirname, filename)
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(filename)
        # Calculate how much to trim
        allowed_len = max_path_length - len(os.path.join(dirname, ext))
        new_basename = basename[:allowed_len]
        filename = new_basename + ext
        print(f"Filename was too long, truncated to: {filename}")
    df.to_csv(dirname+ filename, index=False)



#=== FUNCTION TO SAVE JSON FILE IN SPECIFIED DIRECTORY ===#
def save_to_json(data, dirname, filename, max_path_length=255):
    """
    Save a dictionary or pandas DataFrame as a formatted JSON file.

    This function serializes structured data into a JSON file with pretty-printed
    indentation (4 spaces) and UTF-8 encoding. If a ``pandas.DataFrame`` is passed,
    floating-point values are rounded to 4 decimal places before converting the table
    to a list of record dictionaries. A custom NumPy encoder (``NpEncoder``) is used
    to safely handle embedded NumPy scalars and arrays without serialization errors.

    :param data: The dataset or configuration structure to serialize. DataFrames
                 are automatically converted via ``to_dict(orient="records")``.
    :type data: dict or pandas.DataFrame
    :param dirname: Subdirectory segment within the timestamped output path
                    where the output report folder will be nested.
    :type dirname: str
    :param filename: Base name for the JSON file without extension or date prefixes.
                     The file extension ``.json`` and timestamp prefixes are applied
                     automatically during path construction.
    :type filename: str
    :param max_path_length: Maximum allowable character length for the total target
                            file path. Defaults to ``255``.
    :type max_path_length: int, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If folder creation fails or write permissions are denied by the OS.
    :raises TypeError: If the input object contains data types unsupported by the JSON
                       encoder or custom NumPy handler.

    .. note::
       - Target directory layout is structured as ``results/<default_date>/<default_date_and_time>/<dirname>/reports/``.
       - If a ``pandas.DataFrame`` is provided, floats are rounded using ``df.round(4)``.
       - If the fully constructed filepath exceeds ``max_path_length``, the filename stem is truncated.
       - Embedded NumPy arrays are serialized as Python lists (``obj.tolist()``), while NumPy scalar types
         are cast to standard Python ``int`` or ``float`` types.
    """
    
    # 1. Handle the input type
    # If it's a DataFrame, convert to a list of dicts (records)
    if isinstance(data, pd.DataFrame):
        data = data.round(4)
        data_to_save = data.to_dict(orient="records")
    else:
        data_to_save = data

    # 2. Construct paths (using your specific global date variables)
    # Ensure these variables (date, date_and_time) are defined in your script
    dirname = f"src/seq_control/results/{default_date}/{default_date_and_time}/{dirname}/reports/"
    filename = f"{default_date_and_time}_{filename}.json"
    
    os.makedirs(dirname, exist_ok=True)
    full_path = os.path.join(dirname, filename)

    # 3. Handle Path Length Truncation
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(filename)
        allowed_len = max_path_length - len(os.path.join(dirname, ext))
        filename = basename[:allowed_len] + ext
        print(f"⚠️ Filename truncated to: {filename}")

    final_path = os.path.join(dirname, filename)

    # 4. Write the file
    # Custom encoder to catch hidden numpy arrays or types automatically
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NpEncoder, self).default(obj)

    # 4. Write the file
    with open(final_path, 'w', encoding='utf-8') as f:
        # Added cls=NpEncoder to intercept and convert ndarrays safely
        json.dump(data_to_save, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    
    print(f"✅ JSON saved: {final_path}")



#=== FUNCTION TO SAVE PLOT IMAGE IN SPECIFIED DIRECTORY ===#
def save_plot_image(image, filename, dirname):
    """
    Save a PIL Image object to a structured plot directory path.

    This function validates that the input object is a valid ``PIL.Image.Image`` instance,
    constructs a timestamped nested directory path, ensures the extension ends with ``.png``,
    and writes the image file to disk.

    :param image: The Pillow image instance to be saved to disk.
    :type image: PIL.Image.Image
    :param filename: Base name for the saved image file without mandatory extension.
                     The extension ``.png`` and timestamp prefix are appended during path construction.
    :type filename: str
    :param dirname: Subdirectory segment within the timestamped output path
                    where the output plots directory will be nested.
    :type dirname: str

    :returns: None
    :rtype: NoneType

    :raises TypeError: If the provided ``image`` argument is not an instance of ``PIL.Image.Image``.
    :raises OSError: If folder creation fails or file write operations are denied by the OS.

    .. note::
       - The output directory hierarchy follows ``results/<default_date>/<default_date_and_time>/<dirname>/plots/``.
       - The final filename is automatically formatted as ``<default_date_and_time>_<filename>.png``.
    """
    # 1. Validation
    if not isinstance(image, Image.Image):
        raise TypeError("The 'image' argument must be a PIL Image object.")
    
    # 2. Construct the directory path
    # Using your specific format: results/{date}/{date_and_time}/{dirname}/plots
    path = f"src/seq_control/results/{default_date}/{default_date_and_time}/{dirname}/plots"
    
    # 3. Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # 4. Define full file path
    full_path = f"{path}/{default_date_and_time}_{filename}"
    if not full_path.endswith(".png"):
        full_path += ".png"
    
    # 5. Save and Log
    image.save(full_path)
    print(f"Image successfully saved to: {full_path}")
    
    return()

#=== FUNCTION TO SAVE TRAINING DATASET TENSORS ===#
def save_dataset(data_dict, dirname, filename):
    """
    Save training dataset tensors to disk using PyTorch serialization.

    This function serializes a dictionary of training dataset tensors (such as
    inputs and targets) into a PyTorch binary file (``.pt``). The file is stored
    within a timestamped nested directory layout and automatically enforces path
    length limits to prevent operating system path overflow errors.

    :param data_dict: Dictionary containing PyTorch tensors or tensor collections
                      intended for dataset persistence (e.g., ``{'x': x_tensor, 'y': y_tensor}``).
    :type data_dict: dict
    :param dirname: Subdirectory segment within the timestamped output path where
                    the output dataset directory will be created.
    :type dirname: str
    :param filename: Base name for the dataset checkpoint file without the ``.pt`` extension.
                     Defaults to ``"training_data"``.
    :type filename: str, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If directory creation fails or write permissions are denied by the OS.
    :raises RuntimeError: If PyTorch encounters a serialization error while writing tensors to disk.

    .. note::
       - Target directory layout follows ``results/<default_date>/<default_date_and_time>/<dirname>/dataset/``.
       - Output file is saved with a timestamp prefix as ``<default_date_and_time>_<filename>.pt``.
       - If total file path length exceeds 255 characters, the filename stem is automatically truncated.
    """
    # Construct directory logic consistent with your other functions
    target_dir = f"src/seq_control/results/{default_date}/{default_date_and_time}/{dirname}/dataset/"
    save_filename = f"{default_date_and_time}_{filename}.pt"
    
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, save_filename)
    
    # Path length safety check (255 chars)
    max_path_length = 255
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(save_filename)
        allowed_len = max_path_length - len(os.path.join(target_dir, ext))
        save_filename = basename[:allowed_len] + ext
        full_path = os.path.join(target_dir, save_filename)

    # Save the dictionary containing the tensors
    torch.save(data_dict, full_path)
    print(f"📦 Dataset Tensors saved to: {full_path}")


def save_dataset_with_csv(data_dict, dirname, filename):
    """
    Save training dataset tensors to disk using PyTorch serialization and CSV export.

    This function serializes a dictionary of training dataset tensors/arrays into
    a PyTorch binary file (.pt) and exports each entry to a CSV file. For 3D 
    sequence tensors [N, seq_len, features], data is flattened with explicit 
    'sample_id' and 'step_id' tracking columns.

    :param data_dict: Dictionary containing PyTorch tensors or NumPy arrays
                      (e.g., {'X_raw': x_tensor, 'Y_raw': y_tensor}).
    :type data_dict: dict
    :param dirname: Subdirectory segment within the timestamped output path.
    :type dirname: str
    :param filename: Base name for dataset files without extensions.
                      Defaults to "training_data".
    :type filename: str, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If directory creation fails or write permissions are denied.
    :raises RuntimeError: If PyTorch encounters a serialization error.
    """
    target_dir = f"src/seq_control/results/{default_date}/{default_date_and_time}/{dirname}/dataset/"
    os.makedirs(target_dir, exist_ok=True)
    max_path_length = 255

    # --- 1. SAVE PYTORCH BINARY FILE (.pt) ---
    pt_filename = f"{default_date_and_time}_{filename}.pt"
    full_pt_path = os.path.join(target_dir, pt_filename)

    if len(full_pt_path) > max_path_length:
        basename, ext = os.path.splitext(pt_filename)
        allowed_len = max_path_length - len(os.path.join(target_dir, ext))
        pt_filename = basename[:allowed_len] + ext
        full_pt_path = os.path.join(target_dir, pt_filename)

    torch.save(data_dict, full_pt_path)
    print(f"📦 Dataset Tensors saved to: {full_pt_path}")

    # --- 2. EXPORT CSV FILES (.csv) ---
    for key, val in data_dict.items():
        # Convert PyTorch Tensors or list primitives to NumPy
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().numpy()
        elif isinstance(val, np.ndarray):
            arr = val
        else:
            try:
                arr = np.array(val)
            except Exception as e:
                print(f"⚠️ Skipping CSV export for key '{key}': {e}")
                continue

        # Format tabular DataFrames based on array dimensionality
        if arr.ndim == 1:
            df = pd.DataFrame(arr, columns=[f"{key}_0"])
            
        elif arr.ndim == 2:
            cols = [f"{key}_f{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)
            
        elif arr.ndim == 3:
            # Flatten [N_samples, seq_len, N_features] into tabular rows
            n_samples, seq_len, n_features = arr.shape

            sample_ids = np.repeat(np.arange(n_samples), seq_len)
            step_ids = np.tile(np.arange(seq_len), n_samples)
            flat_data = arr.reshape(-1, n_features)

            feature_cols = [f"{key}_f{i}" for i in range(n_features)]
            df = pd.DataFrame(flat_data, columns=feature_cols)
            df.insert(0, "step_id", step_ids)
            df.insert(0, "sample_id", sample_ids)
            
        else:
            print(f"⚠️ Key '{key}' has unsupported dimension ({arr.ndim}D) for CSV export.")
            continue

        # Save CSV with OS path length safety check
        csv_filename = f"{default_date_and_time}_{filename}_{key}.csv"
        full_csv_path = os.path.join(target_dir, csv_filename)

        if len(full_csv_path) > max_path_length:
            basename, ext = os.path.splitext(csv_filename)
            allowed_len = max_path_length - len(os.path.join(target_dir, ext))
            csv_filename = basename[:allowed_len] + ext
            full_csv_path = os.path.join(target_dir, csv_filename)

        df.to_csv(full_csv_path, index=False)
        print(f"📄 Dataset CSV ({key}) saved to: {full_csv_path}")
#=== FUNCTION TO SAVE SCALER OBJECTS IN SPECIFIED DIRECTORY ===#
def save_scaler_object(scaler, dirname, filename, max_path_length=255):
    """
    Serialize and save a scikit-learn or custom scaler object to a pickle file.

    This function accepts a fitted data preprocessor (such as a ``StandardScaler``,
    ``MinMaxScaler``, or custom transformer object) and serializes it to disk via ``pickle``.
    The file is stored inside a structured, timestamped subdirectory layout. If the generated
    total path exceeds the operating system's maximum length, the filename stem is safely truncated.

    :param scaler: The fitted scaling object or transformer instance to be serialized.
    :type scaler: object
    :param dirname: Subdirectory segment within the timestamped output path where the
                    output scalers folder will be created.
    :type dirname: str
    :param filename: Base name for the scaler file (with or without the ``.pkl`` extension).
    :type filename: str
    :param max_path_length: Maximum allowable character length for the total target file path.
                            Defaults to ``255``.
    :type max_path_length: int, optional

    :returns: None
    :rtype: NoneType

    :raises OSError: If folder creation fails or write permissions are denied by the OS.
    :raises PicklingError: If the provided scaler object contains unpicklable references or state.

    .. note::
       - The output directory hierarchy follows ``results/<default_date>/<default_date_and_time>/<dirname>/scalers/``.
       - Output file is saved with a timestamp prefix as ``<default_date_and_time>_<filename>.pkl``.
       - File extensions are automatically validated to ensure a single ``.pkl`` suffix.
       - If total file path length exceeds ``max_path_length``, the filename stem is automatically truncated.
    """
    # 1. Standardize file extension
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    # 2. Construct paths using your specific global variables
    target_dir = f"src/seq_control/results/{default_date}/{default_date_and_time}/{dirname}/scalers/"
    save_filename = f"{default_date_and_time}_{filename}"
    
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, save_filename)

    # 3. Handle Path Length Truncation
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(save_filename)
        allowed_len = max_path_length - len(os.path.join(target_dir, ext))
        save_filename = basename[:allowed_len] + ext
        full_path = os.path.join(target_dir, save_filename)
        print(f"⚠️ Filename too long, truncated to: {save_filename}")

    # 4. Write the binary pickle file
    with open(full_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"💾 Scaler successfully saved to: {full_path}")


def load_scaler(scaler_dir):
    """
    Load and deserialize feature and target scalers from disk.

    This function reads a serialized pickle binary file containing fitted scaler
    objects (such as ``scikit-learn`` preprocessing transformers) previously saved
    for a specific dataset split or cross-validation fold.

    :param scaler_dir: Filepath pointing directly to the serialized pickle file
                       containing the scaler object(s).
    :type scaler_dir: str or path-like

    :returns: The deserialized scaler instance or tuple/dictionary of scaler objects
              (e.g., ``scaler_x``, ``scaler_y``) loaded from the pickle binary.
    :rtype: object or tuple[object, ...]

    :raises FileNotFoundError: If no file exists at the specified ``scaler_dir`` path.
    :raises UnpicklingError: If the file is corrupted or cannot be deserialized into a Python object.
    :raises OSError: If read permissions are denied for the target filepath.

    .. note::
       - While the parameter is named ``scaler_dir``, it expects a path pointing
         to a specific file (e.g., ``path/to/scalers.pkl``) rather than a directory.
       - Ensure that any custom classes serialized within the pickle file are imported
         and available in the execution environment prior to calling this function.
    """
    with open(scaler_dir, "rb") as f:
        scaler = pickle.load(f)
        
        
    print(f"✅ Successfully loaded scalers from {scaler_dir}")
    return scaler


#=== FUNCTION TO LOAD TRAINED MODEL ===#
def load_model(model_class, checkpoint_path, device="cuda"):
    """
    Load a trained PyTorch model state dictionary and configuration from a checkpoint file.

    This function deserializes a saved PyTorch checkpoint binary file (`.pt`), extracts
    the encapsulated hyperparameter configuration dictionary, dynamically instantiates
    the specified model class, and restores its trained weight state. The resulting model
    is mapped to the target hardware device and locked into evaluation mode (``.eval()``)
    to ensure deterministic inference.

    :param model_class: The class object or callable factory used to instantiate the model.
                        Must accept the hyperparameter configuration dictionary as its primary argument.
    :type model_class: type or callable
    :param checkpoint_path: The file system path pointing to the saved PyTorch checkpoint file.
    :type checkpoint_path: str or path-like
    :param device: The target computing device (e.g., ``'cpu'``, ``'cuda'``, or a ``torch.device`` instance)
                   where the model parameters will be loaded. Defaults to ``"cuda"``.
    :type device: str or torch.device, optional

    :returns: The instantiated model restored with its historical trained parameters and set to evaluation mode.
    :rtype: torch.nn.Module

    :raises FileNotFoundError: If the file at ``checkpoint_path`` does not exist.
    :raises KeyError: If the checkpoint dictionary lacks required keys (``'config'`` or ``'model_state_dict'``).
    :raises RuntimeError: If parameter loading fails due to mismatched tensor shapes or model key structures.

    .. note::
       - Internal loading defaults to remapping tensors via ``map_location="cuda"`` before transferring to the designated ``device``.
       - Uses ``weights_only=False`` during unpickling to support complex configuration objects stored in the checkpoint.
       - Automatically calls ``model.eval()`` prior to returning to disable layer-specific training behavior (such as Dropout or BatchNorm).
    """
    
    # Load the serialized checkpoint dictionary from disk
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    
    # Extract structural configurations and instantiate the network
    hyperparam_config = checkpoint['config']
    model = model_class(hyperparam_config)
    
    # Transfer the model parameters to the target processing device
    model = model.to(device)
    
    # Restore the historical weight state configurations
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze layers into evaluation mode for tracking inference
    model.eval()
    
    return model

def load_model_esn(model_class, checkpoint_path, device=None):
    """
    Load a trained Echo State Network (ESN) model and configuration from a pickle checkpoint.

    This function deserializes a saved ESN checkpoint file (`.pkl` or `.pt`), extracts the
    hyperparameter configuration dictionary, dynamically instantiates the specified wrapper class,
    and restores the trained underlying ReservoirPy model layout along with its trained readout
    weights.

    :param model_class: The class object or callable factory used to instantiate the model container
                        (e.g., ``ESNInverseController``). Must accept the hyperparameter configuration
                        dictionary as its primary initialization argument.
    :type model_class: type or callable
    :param checkpoint_path: System filepath pointing to the serialized pickle checkpoint file.
    :type checkpoint_path: str or path-like
    :param device: Hardware device target. This argument is unused for CPU/NumPy-backed ESN models,
                   but retained to maintain API signature compatibility across training and evaluation pipelines.
    :type device: str, torch.device, or None, optional

    :returns: The instantiated ESN model container restored to its trained internal state.
    :rtype: object

    :raises FileNotFoundError: If the checkpoint file at ``checkpoint_path`` cannot be located.
    :raises KeyError: If the checkpoint dictionary is missing required keys (e.g., ``'config'`` or ``'model_state_dict'``).
    :raises UnpicklingError: If the checkpoint file is corrupted or cannot be safely deserialized.

    .. note::
       - Restores the underlying ``ReservoirPy`` model pipeline directly via the ``model.model`` attribute,
         recovering the analytically computed readout weights (e.g., Ridge regression readout).
       - Safely invokes ``load_state_dict(None)`` if available on the wrapper instance to reset or normalize
         internal parameter state boundaries upon loading.
    """
    # Load the serialized checkpoint dictionary using pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract the embedded hyperparameter configuration dictionary
    hyperparam_config = checkpoint['config']
    
    # Instantiate a fresh, uninitialized ESN network structure
    model = model_class(hyperparam_config)
    
    # Restore the underlying trained ReservoirPy Model object instance directly
    # This recovers the analytically derived Ridge readout weight matrix
    model.model = checkpoint['model_state_dict']
    
    # Safe state boundary check reset
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(None)
        
    return model



