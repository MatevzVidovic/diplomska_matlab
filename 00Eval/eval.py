





import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)



python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024, var_blacklist=["tree_ix_2_module", "mask_path"])
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path)







import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from timeit import default_timer as timer
import gc

# from helper_img_and_fig_tools import show_image, save_plt_fig_quick_figs, save_img_quick_figs


import yaml





"""
get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU are adapted from:
from train_with_knowledge_distillation import get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU
"""

@py_log.autolog(passed_logger=MY_LOGGER)
def get_conf_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes=2):
    """
    predictions and targets can be matrixes or tensors.
    
    In both cases we only get a single confusion matrix
    - in the tensor case it is simply agreggated over all examples in the batch.
    """

    try:

        predictions_np = predictions.astype(np.uint64)
        targets_np = targets.astype(np.uint64)
        # for batch of predictions
        # if len(np.unique(targets)) != 2:
        #    print(len(np.unique(targets)))
        
        
        try:
            assert (predictions.shape == targets.shape)
        except:
            print("predictions.shape: ", predictions.shape)
            print("targets.shape: ", targets.shape)
            raise AssertionError





        """
        c = get_conf_matrix(np.array([0,1,2,3,3]), np.array([0,2,2,3,0]))
        print(c)

        PREDICTIONS
        0, 1, 2, 3
        [[1 0 0 1]   0 |
        [0 0 0 0]   1 |
        [0 1 1 0]   2  TARGETS
        [0 0 0 1]]  3 |
        """


        # The mask is here mostly to make this a 1D array.
        mask = (targets_np >= 0) & (targets_np < num_classes)




        """
        Example for 4 classes:
        Possible target values are [0, 1, 2, 3].
        
        Possible label values are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
        
        Label values [0, 1, 2, 3] are those that are 0 in the target.
        0 is for those who are also 0 in the prediction, 1 for those which are 1 in the prediction, etc.
        Label values [4, 5, 6, 7] are those that are 1 in the target, etc.
        Then this gets reshaped into a confusion matrix.
        np.reshape fills the matrix row by row.
        

        So the leftmost column will be the background.
        The top row will be the background.

        The diagonal will be the correct predictions.

        
        """

        if num_classes > 8:
            raise NotImplementedError("This function is not intended for more than 8 classes. Because np.uint8. Its easy to make it more general.")

        # print(mask) # 2d/3d tensor of true/false
        label = num_classes * targets_np[mask].astype(np.uint8) + predictions_np[mask].astype(np.uint8)
        # show_image([(predictions_np[mask], "Predictions"), (targets_np[mask], "Targets"))
        # gt_image[mask] vzame samo tiste vrednosti, kjer je mask==True
        # print(mask.shape)  # batch_size, 128, 128
        # print(label.shape) # batch_size * 128 * 128 (with batch_size==1:   = 16384)
        # print(label)  # vector composed of 0, 1, 2, 3 (in the multilabel case)
        count = np.bincount(label, minlength=num_classes ** 2)  # number of repetitions of each unique value
        # print(count) # [14359   475    98  1452]
        # so [predBGisBG, predBGisFG, predFGisBG, predFGisFG]
        confusion_matrix = count.reshape(num_classes, num_classes)


        return confusion_matrix


    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e









@py_log.autolog(passed_logger=MY_LOGGER)
def get_IoU_from_predictions(predictions, targets, num_classes=2):
    """
    Returns vector of IoU for each class.
    IoU[0] is the IoU for the background, for example.
    """

    try:

        confusion_matrix = get_conf_matrix(predictions, targets, num_classes)
        IoU, where_is_union_zero = conf_matrix_to_IoU(confusion_matrix, num_classes)

        return IoU, where_is_union_zero
    

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e

@py_log.autolog(passed_logger=MY_LOGGER)
def conf_matrix_to_IoU(confusion_matrix, n_classes):
    """
    c = get_conf_matrix(np.array([0,1,2,3,3]), np.array([0,2,2,3,3]))
    print(c)
    [[1 0 0 0]
     [0 0 0 0]
     [0 1 1 0]
     [0 0 0 2]]
    miou = conf_matrix_to_mIoU(c)  # for each class: [1.  0.  0.5 1. ]
    print(miou) # 0.625
    """

    try:
        if confusion_matrix.shape != (n_classes, n_classes):
            print(confusion_matrix.shape)
            raise NotImplementedError()

        unions = (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        
        where_is_union_zero = unions == 0
        unions[where_is_union_zero] = 1  # to make the division not fail

        IoU = np.diag(confusion_matrix) / unions

        IoU[where_is_union_zero] = np.nan  # if union is 0, then IoU is undefined
        
        # print("Conf matrix:", confusion_matrix)
        # print("IoU diag:", IoU)

        return IoU, where_is_union_zero

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e






def get_F1_from_predictions(predictions, targets):
    confusion_matrix = get_conf_matrix(predictions, targets)
    IoU = conf_matrix_to_F1(confusion_matrix)

    return IoU

def conf_matrix_to_F1(confusion_matrix):


    try:

        TP = confusion_matrix[0][0] # this is actually the background
        FN = confusion_matrix[0][1]
        FP = confusion_matrix[1][0]
        TN = confusion_matrix[1][1] # this is the target.

        # We could switch them, but it doesn't matter computationally.


        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        F1 = 2 * (precision * recall) / (precision + recall)

        return F1
    

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e





class MultiClassDiceLoss():
    def __init__(self, smooth=1):
        self.smooth = smooth

    @py_log.autolog(passed_logger=MY_LOGGER)
    def forward(self, inputs:np.ndarray, targets:np.ndarray):

        try:

            # Initialize Dice Loss
            dice_loss = 0.0

            c = 1 # skip background class

            # input_to_be_flat = inputs[:,c,:,:].squeeze(1)
            # input_flat = input_to_be_flat.reshape(-1)
            input_flat = inputs.reshape(-1)
            
            target_to_be_flat = (targets == c).astype(np.float32)
            target_flat = target_to_be_flat.reshape(-1)




            # Compute intersection
            intersection = (input_flat * target_flat)
            intersection = intersection.sum()
            
            # Compute Dice Coefficient for this class
            dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
            
            # Accumulate Dice Loss
            dice_loss += 1 - dice
            
            # Average over all classes
            dice_loss =  dice_loss / (inputs.shape[1] - 1) # -1 because we skip the background class

            return dice_loss

        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER)
            raise e

mcdl = MultiClassDiceLoss()





import pickle



import os
import os.path as osp
import sys

import cv2
import numpy as np

folders = ['bcosfire', 'coye2']
bin_threshs = [0.02 * i for i in range(1, 51)]


# folders = ['bcosfire']
# bin_threshs = [0.2 * i for i in range(0, 6)]





results = {name: {bin_thresh: {k: None for k in ["IoU"]} for bin_thresh in bin_threshs} for name in folders}


for folder in folders:
    folder_path = osp.join(folder)
    
    all_files = os.listdir(folder_path)

    pred_files_names_stripped = []

    for file in all_files:
        stripped = file.removesuffix(".png")
        if not (stripped.endswith("_bin") or stripped.endswith("_gt")):
            pred_files_names_stripped.append(stripped)
    
    # print(f"pred_files_names_stripped: {pred_files_names_stripped}")
    
    ground_truth_files_names = [f"{name}_gt.png" for name in pred_files_names_stripped]
    pred_files_names = [f"{name}.png" for name in pred_files_names_stripped]

    # print(f"pred_files_names: {pred_files_names}")
    # print(f"ground_truth_files_names: {ground_truth_files_names}")






    # MCDL doesnt really make sense in this situation.
    # What the alg is predicting is never supposed to mimic the target.
    # Only after the binarization it is supposed to mimic the target. So it makes no sense to do it that way.

    # # this is irrelevant to the threashold so only needs to be done once
    
    # MCDL = 0

    # for ix2, (pred_file_name, gt_file_name) in enumerate(zip(pred_files_names, ground_truth_files_names)):
        
        
    #     pred_file = cv2.imread(osp.join(folder_path, pred_file_name), cv2.IMREAD_GRAYSCALE)
    #     gt_file = cv2.imread(osp.join(folder_path, gt_file_name), cv2.IMREAD_GRAYSCALE)

    #     pred_file
    #     gt_binary = gt_file > 0
    #     py_log.log_manual(MY_LOGGER, pred_file=pred_file, gt_binary=gt_binary)
    #     MCDL += mcdl.forward(pred_file, gt_binary)

    #     break

    # MCDL /= len(pred_files_names)



    for ix, bin_thresh in enumerate(bin_threshs):
        IoU = 0
        # F1 = 0
        for ix2, (pred_file_name, gt_file_name) in enumerate(zip(pred_files_names, ground_truth_files_names)):
            
            
            pred_file = cv2.imread(osp.join(folder_path, pred_file_name), cv2.IMREAD_GRAYSCALE)
            gt_file = cv2.imread(osp.join(folder_path, gt_file_name), cv2.IMREAD_GRAYSCALE)

            pred_file = pred_file / 255
            gt_file = gt_file / 255

            # print(f"pred_file_name: {pred_file_name}")
            # print(f"gt_file_name: {gt_file_name}")
            # print(f"pred_file: {pred_file}")
            # print(f"gt_file: {gt_file}")

            # py_log.log_manual(MY_LOGGER, pred_file=pred_file, gt_file=gt_file)
            
            # cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("gt", cv2.WINDOW_NORMAL)

            # cv2.resizeWindow("pred", 200, 200)
            # cv2.resizeWindow("gt", 200, 200)

            # cv2.imshow("pred", pred_file)
            # cv2.imshow("gt", gt_file)


            # cv2.waitKey(0)

            # input("Press Enter to continue...")
            # cv2.destroyAllWindows()


            pred_binary = pred_file >= bin_thresh
            gt_binary = gt_file > 0

            py_log.log_manual(MY_LOGGER, pred_binary=pred_binary, gt_binary=gt_binary)

            curr_IoU, where_is_union_zero = get_IoU_from_predictions(pred_binary, gt_binary)
            if not where_is_union_zero[1]:
                IoU += curr_IoU.item(1) # only IoU for sclera (not background)

            # F1 += get_F1_from_predictions(pred_binary, gt_binary)

            print(f"pred_file_name: {pred_file_name}, IoU: {IoU}")
        
        IoU /= len(pred_files_names)
        # F1 /= len(pred_files_names)
        

        results[folder][bin_thresh]["IoU"] = IoU
        # results[folder][bin_thresh]["F1"] = F1
        # results[folder][bin_thresh]["MCDL"] = MCDL

        print(f"{ix}, folder: {folder}, bin_thresh: {bin_thresh}, IoU: {IoU}")

if True:
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    # make yaml
    with open("results.yaml", "w") as f:
        yaml.dump(results, f)

print(results)





#     ground_truths = [pred_file.split("_")[0] for pred_file in pred_files_stripped]


#     for pred_file in pred_files:

#         # strip of extension, with rstrip

#         goal_imgs = osp.join("combined_data", "Images")
#         pred_imgs = osp.join("full_vein_sclera_data", "Images")








# approx_IoU_size = 0
# IoU_size = 0
# num_batches = len(dataloader)

# self.model.eval()
# test_loss, approx_IoU, F1, IoU = 0, 0, 0, 0
# with torch.no_grad():
#     for X, y in dataloader:
#             X, y = X.to(self.device), y.to(self.device)
#             pred = self.model(X)


#             # loss_fn computes the mean loss for the entire batch.
#             # We cold also get the loss for each image, but we don't need to.
#             # https://discuss.pytorch.org/t/loss-for-each-sample-in-batch/36200

#             # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#             # The fact the shape of pred and y are diferent seems to be correct regarding loss_fn.
#             test_loss += self.loss_fn(pred, y).item()



#             pred_binary = pred[:, 1] > pred[:, 0]

#             F1 += get_F1_from_predictions(pred_binary, y)
#             approx_IoUs, where_is_union_zero = get_IoU_from_predictions(pred_binary, y)
#             if where_is_union_zero[1] == False:
#                 approx_IoU += approx_IoUs.item(1) # only IoU for sclera (not background)
#                 approx_IoU_size += 1


#             # X and y are tensors of a batch, so we have to go over them all
#             for i in range(X.shape[0]):

#                 pred_binary = pred[i][1] > pred[i][0]


#                 curr_IoU, where_is_union_zero = get_IoU_from_predictions(pred_binary, y[i])
#                 if where_is_union_zero[1] == False:
#                     IoU += curr_IoU.item(1) # only IoU for sclera (not background)
#                     IoU_size += 1
#                 # print(f"This image's IoU: {curr_IoU:>.6f}%")




# test_loss /= num_batches # not (num_batches * batch_size), because we are already adding batch means
# approx_IoU /= approx_IoU_size
# F1 /= num_batches
# IoU /= IoU_size # should be same or even more accurate as (num_batches * batch_size)






