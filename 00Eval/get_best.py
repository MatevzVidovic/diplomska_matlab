

import os
import yaml


# 
# folders = ['bcosfire', 'coye2']
# bin_threshs = [0.02 * i for i in range(1, 51)]

# results = {name: {bin_thresh: {k: None for k in ["IoU"]} for bin_thresh in bin_threshs} for name in folders}


with open('results.yaml', 'r') as file:
    results = yaml.load(file, Loader=yaml.FullLoader)


best = {}
for folder in results:
    best_metrics = {}
    for bin_thresh in results[folder]:
        for metric in results[folder][bin_thresh]:
            if results[folder][bin_thresh][metric] is None:
                results[folder][bin_thresh][metric] = (0.0, bin_thresh)
            else:
                results[folder][bin_thresh][metric] = (float(results[folder][bin_thresh][metric]), bin_thresh)

            if metric not in best_metrics:
                best_metrics[metric] = results[folder][bin_thresh][metric]
            else:
                if results[folder][bin_thresh][metric][0] > best_metrics[metric][0]:
                    best_metrics[metric] = results[folder][bin_thresh][metric]
    
    best[folder] = best_metrics

with open('best.yaml', 'w') as file:
    yaml.dump(best, file)
