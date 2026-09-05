"""
Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.

Usage: run this script from the project root (or ensure PYTHONPATH
includes the project) so imports resolve correctly.
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


import matplotlib.pyplot as plt



# Apply project style for plots (if plotting is enabled)
plt.style.use("src/seq_control/style.mplstyle")


def main() -> None:
	"""Create a plant instance and generate training data.

	Change which plant is instantiated by uncommenting the other
	option below. The dataset folder is named after the plant class.
	"""

	plant_list = [
		#ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant),
		#TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant),
		#IdiophasePlant(hyperparam_config=hyperparam_config_IdiophasePlant),
		#CoCultivationPlant(hyperparam_config=hyperparam_config_CoCultivationPlant),
		IndForProteinProductionPlant(hyperparam_config=hyperparam_config_IndForProteinProductionPlant)
	]


	for pl in plant_list:

		dirname = pl.__class__.__name__
		hyperparam_config = pl.hyperparam_config
		save_to_json(hyperparam_config, dirname, "training_data_hyperparam_config")
    
		data = generate_and_save_dataset(
			pl,
			hyperparam_config["training_data_cfg"],
			dirname=dirname,
			show_plots=False,  
			show_overlay_plot=True,
			save_logs=True
			)
	# stats = {
	# 	"shape": tuple(data.shape),
	# 	"dtype": str(data.dtype),
	# 	"mean": data.mean(dim=0),
	# 	"std": data.std(dim=0),
	# 	"min": data.amin(dim=0),
	# 	"max": data.amax(dim=0),
	# 	"quantiles": torch.quantile(data, torch.tensor([0.25, 0.5, 0.75]), dim=0),
	# }

		#print(data.keys)

if __name__ == "__main__":
	main()