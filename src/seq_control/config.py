"""
Sequence Control Logging and Environment Setup
==============================================

This module initializes the execution environment for sequence control scripts. 
It manages timezone-aware timestamping, automated log directory creation, Matplotlib 
plot styling, and logging setup.

Key Features
------------
* **Timezone Configuration:** Uses ``Europe/Berlin`` to generate date-stamped execution IDs.
* **Directory Management:** Automatically creates log folders at ``src/seq_control/logs/YYYY-MM-DD/``.
* **Matplotlib Styling:** Applies the custom style file ``src/seq_control/style.mplstyle``.
* **Execution Logging:** Configures file-based logging for execution history and run descriptions.

Functions
---------
* :func:`log_message`: Writes an informational message to the current execution log file.
* :func:`get_run_description`: Interactively prompts the user for a run description via CLI and records it.

Usage Example
-------------
.. code-block:: python

    from src.seq_control.logging_setup import log_message, get_run_description

    # Prompt user for run context at start
    get_run_description()

    # Log custom status messages
    log_message("Sequence initialized successfully.")
"""

# Import standard libraries
from datetime import datetime
import os
import logging
import matplotlib.pyplot as plt
import pytz

# === Load or initialize run_id === #
tz = pytz.timezone('Europe/Berlin')
date = datetime.now(tz).strftime("%Y-%m-%d")
date_and_time = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")

# Use the custom style for plots
plt.style.use("src/seq_control/style.mplstyle")

path_name = "src/seq_control/logs/" + date + "/"

if not os.path.exists(path_name):
    os.makedirs(path_name)

exlog_path_name = path_name + date_and_time + "_execution_log.txt"

# Configure logging to display ERROR and above
logging.basicConfig(filename=exlog_path_name, 
                    level=logging.INFO, 
                    format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

def log_message(message):
    """
    Log a message to the log file.

    :param message: The text string to record in the execution log.
    :type message: str
    """
    logging.info(message)

def get_run_description():
    """
    Prompts for a description only when executed directly.

    :return: The user-entered description string.
    :rtype: str
    """
    print("\n" + "="*50)
    description = input("Describe this run: ")
    print("="*50 + "\n")
    log_message(description)
    return description



