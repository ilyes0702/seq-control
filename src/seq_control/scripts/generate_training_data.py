"""
Script to generate and save training data for plant models.

This script selects benchmark plant model and generates and saves data for training and validation.
"""

# Import utilities
from seq_control.utils.data_generation_utils import *
from seq_control.utils.plotting_utils import *

# Import dictionaries
from seq_control.dicts.plant_dict import *

def main() -> None:
	"""Generate training data.
	"""

	log_message("Beginning of data generation script")
	get_run_description()

	log_message("Starting training data generation...")

	# 2. Generate data for the plants defined in plant_list. The data for plant pl can be then found in the folder results/YYYY-MM-DD/YYYY-MM-DD/<pl.__class__.__name__>
	for plant_name, plant_data in plant_dict.items():
		pl = plant_data["plant"]
		dirname = pl.__class__.__name__
		hyperparam_config = pl.hyperparam_config
		save_to_json(hyperparam_config, dirname, "training_data_hyperparam_config")
    
		generate_io_dataset(
			pl,
			hyperparam_config["training_data_cfg"],
			dirname=dirname,
			save_sample_plot=True,
			save_all_plots=False,  
			save_overlay_plot=True,
			save_sequence_data=True
			)
	
	log_message("Finished training data generation.")

if __name__ == "__main__":
	main()