import time
import pandas as pd
from seq_control.config import *
#from dictionaries.variable_names import axis_label_mapping

from seq_control.utils.saving_and_loading_utils import save_df_to_csv
import functools
import torch
import psutil

def measure_resources(func):
    """Decorator to measure execution time, peak host RAM (RSS),

    peak VRAM allocated, and total execution FLOPs during a PyTorch training run.
    Appends these metrics directly to the returned summary_df.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Detect Device & Extract Inputs from function arguments
        model = kwargs.get("model", args[0] if len(args) > 0 else None)
        train_data = kwargs.get("train_data", args[1] if len(args) > 1 else None)
        config = kwargs.get("hyperparam_config", args[3] if len(args) > 3 else None)

        device_str = (
            str(config["train"].get("device", "cpu")).lower()
            if config
            else "cpu"
        )
        is_cuda = "cuda" in device_str and torch.cuda.is_available()
        device = torch.device(device_str if is_cuda else "cpu")

        # 2. Reset VRAM Peak Tracking
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        # 3. Setup CPU & CUDA Timers
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        start_wall = time.perf_counter()
        process = psutil.Process(os.getpid())

        # 4. Execute Function
        result = func(*args, **kwargs)

        # 5. Measure Elapsed Time
        wall_sec = time.perf_counter() - start_wall
        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            gpu_time_min = (start_event.elapsed_time(end_event) / 1000.0) / 60.0
            time_metric_name = "Execution Time (GPU min)"
            time_val = gpu_time_min
        else:
            time_metric_name = "Execution Time (Wall min)"
            time_val = wall_sec / 60.0

        # 6. Measure Peak System RAM (RSS)
        peak_ram_gb = process.memory_info().rss / (1024**3)

        # 7. Measure Peak VRAM Allocated
        if is_cuda:
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        else:
            peak_vram_gb = 0.0

        # 8. Estimate Theoretical FLOPs (Single Batch Pass * Total Steps)
        total_gflops = 0.0
        if model is not None and train_data is not None and config is not None:
            try:
                total_gflops = _estimate_training_gflops(
                    model, train_data, config, device
                )
            except Exception as e:
                print(f"⚠️ Could not calculate FLOPs: {e}")

        # 9. Console Output Summary
        print("\n" + "=" * 50)
        print("📊 RESOURCE CONSUMPTION SUMMARY")
        print("=" * 50)
        print(f"⏱️  {time_metric_name:<28}: {time_val:.3f}")
        print(f"💾 Peak VRAM Allocated       : {peak_vram_gb:.3f} GB")
        print(f"🖥️  Peak System RAM (RSS)     : {peak_ram_gb:.3f} GB")
        print(f"🧮 Total Theoretical Compute  : {total_gflops:.3f} GFLOPs")
        print("=" * 50 + "\n")

        # 10. Append to returned summary_df if present
        if (
            isinstance(result, tuple)
            and len(result) == 3
            and isinstance(result[2], pd.DataFrame)
        ):
            model_out, history, summary_df = result
            resource_rows = pd.DataFrame(
                {
                    "Metric": [
                        time_metric_name,
                        "Peak VRAM Allocated (GB)",
                        "Peak System RAM (GB)",
                        "Total Compute Workload (GFLOPs)",
                    ],
                    "Value": [time_val, peak_vram_gb, peak_ram_gb, total_gflops],
                }
            )
            updated_summary = pd.concat(
                [summary_df, resource_rows], ignore_index=True
            )
            return model_out, history, updated_summary

        return result

    return wrapper


#=== DECORATOR TO TRACK GPU RESOURCES ===#
def track_resources(func):
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

#=== DECORATOR TO LOG EXECUTION TIME ===#
def log_execution_time(func):
    """
    A decorator that measures and logs the execution time of a function.

    The execution time is recorded in seconds and logged using `log_message()`. 
    The log is saved in a file named `'execution_log.txt'`, providing insights 
    into function performance.

    Parameters:
        func (callable): The function being decorated. It can take any arguments and return any value.

    Returns:
        callable: A wrapped version of `func` that logs its execution time.

    Raises:
        None

    Additional Details:
        - Uses `time.time()` to capture start and end times.
        - Formats timestamps using `time.strftime("%Y-%m-%d %H:%M:%S")` for readability.
        - Logs execution time in a structured message format: 
          `"function_name executed in X.XXXXX seconds"`.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timing
        log_message(f"STARTED {func.__name__}")
        result = func(*args, **kwargs)  # Execute the original function
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        
        # Get the current time in a readable format
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        
        # Log the execution time to a file with the time format
        log_message(f"ENDED {func.__name__} executed in {execution_time:.5f} seconds")
        
        return result  # Return the original function's result
    return wrapper
