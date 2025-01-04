#!/bin/bash

#SBATCH --job-name=mp2
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 64
#SBATCH --mem-per-cpu=3G


#SBATCH --output=x_mp.out




#srun python3 eval.py
srun python3 get_best.py



#srun --output=x_percents_unet_prune_IPAD.txt  python3 v_find_closest_pruned_percents.py -p unet_prune_IPAD
#srun --output=x_percents_unet_prune_random.txt  python3 v_find_closest_pruned_percents.py -p unet_prune_random
#srun --output=x_percents_unet_prune_uniform.txt  python3 v_find_closest_pruned_percents.py -p unet_prune_uniform

#srun python3 v_graphs_and_ts_and_da.py

#srun python3 v_error_spreadsheet.py

#srun python3 v_info_about_dataset.py