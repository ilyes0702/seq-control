"""
General Decorators
==================

This module provides reusable general-purpose Python decorators for performance monitoring,
resource tracking, and utility enhancements across the project.

Functions:
    * :func:`track_resources`: Measures execution time and peak PyTorch GPU VRAM consumption.
"""

import functools
import time
import pandas as pd
import torch

from seq_control.config import *
from seq_control.utils.saving_and_loading_utils import save_df_to_csv

#=== DECORATOR TO TRACK GPU RESOURCES ===#
def track_resources(func):
    """
    Decorator to track execution time and peak GPU VRAM consumption of a function.

    Monitors PyTorch CUDA memory allocation and runtime during function execution.
    It synchronizes CUDA operations before and after execution, records peak VRAM 
    allocated in gigabytes (GB), prints a resource report to stdout, and exports 
    the results to a CSV file.

    :param func: The target function or routine to monitor.
    :type func: callable

    :Keyword Arguments:
        * **resource_filename** (*str*, optional): 
          Base filename for saving the CSV metrics. Defaults to ``"<func.__name__>_resource_stats"``.
        * **resource_dirname** (*str*, optional): 
          Directory where the CSV file will be saved. Defaults to ``"resource_stats"``.

    :returns: A tuple containing:
        - **result**: The return value of the wrapped function.
        - **metrics** (*dict* or *float*): A dictionary with keys ``'gpu_minutes'`` and 
          ``'peak_vram_gb'`` if CUDA is available, otherwise ``0.0``.
    :rtype: tuple

    .. note::
       If CUDA is unavailable (e.g., running on CPU), the wrapped function will execute 
       normally and return ``(result, 0.0)`` without saving resource reports.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs), 0.0
            
        # 1. Prepare GPU
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats() # Reset the "high-water mark"
        start_time = time.perf_counter()
        
        # 2. Execute Training
        result = func(*args, **kwargs)
        
        # 3. Finalize
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 4. Extract Metrics
        total_sec = end_time - start_time
        gpu_min = total_sec / 60
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3) # Convert bytes to Gigabytes
        
        print("\n" + "🚀" + " ="*20)
        print(f"RESOURCE REPORT: {func.__name__}")
        print(f"⏱️  Time Used:  {gpu_min:.4f} GPU-minutes")
        print(f"💾 Peak VRAM:  {peak_gb:.2f} GB")
        print(" ="*20 + "\n")
        
        # Return results + a dictionary of metrics for easy logging
        metrics = {
            "gpu_minutes": gpu_min,
            "peak_vram_gb": peak_gb
        }

        
        resource_df = pd.DataFrame([metrics])
        csv_filename = kwargs.get("resource_filename", f"{func.__name__}_resource_stats")
        csv_dirname = kwargs.get("resource_dirname", "resource_stats")
        save_df_to_csv(resource_df, filename=csv_filename, dirname=csv_dirname)

        return result, metrics
        
    return wrapper

