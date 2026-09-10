"""
Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.
"""

# Import utilities
from seq_control.utils.saving_and_loading_utils import *
from seq_control.utils.data_generation_utils import *
from seq_control.utils.plotting_utils import *

# Import plants
from seq_control.classes.plants.ChemostatPlant import *
from seq_control.classes.plants.IdiophasePlant import *
from seq_control.classes.plants.TrophophasePlant import *
from seq_control.classes.plants.CocultivationPlant import *
from seq_control.classes.plants.IndForProteinProductionPlant import *

def main() -> None:
	"""Create a plant instance and generate training data. Change which plant is instantiated by uncommenting the other	option below. The dataset folder is named after the plant class.
	"""

	# Define which plants to study and the respective configuration dictionaries
	plant_list = [
		ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
		TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
		IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
		CoCultivationPlant(hyperparam_config=hyperparam_config_CoCultivationPlant),
		IndForProteinProductionPlant(hyperparam_config=hyperparam_config_IndForProteinProductionPlant)
	]

	# Generate data for the plants defined in plant_list. The data for plant pl can be then found in the folder results/YYYY-MM-DD/YYYY-MM-DD/<pl.__class__.__name__>
	for pl in plant_list:
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

		

if __name__ == "__main__":
	main()