


import os
import os.path as osp
import sys


vein_masks = osp.join("full_vein_sclera_data", "Masks")
sclera_masks = osp.join("full_sclera_data", "Masks")

goal_masks = osp.join("combined_data", "Masks")

os.makedirs("combined_data", exist_ok=True)
os.makedirs(goal_masks, exist_ok=True)

vein_files = os.listdir(vein_masks)

for vein_file in vein_files:
    print(vein_file)

    # strip of extension, with rstrip
    vein_file_stripped = vein_file.rstrip(".png")

    os.system(f"cp {osp.join(sclera_masks, vein_file)} {osp.join(goal_masks, vein_file)}")
    os.system(f"mv {osp.join(goal_masks, vein_file)} {osp.join(goal_masks, f"{vein_file_stripped}_sclera.png")}")

    os.system(f"cp {osp.join(vein_masks, vein_file)} {osp.join(goal_masks, vein_file)}")
    os.system(f"mv {osp.join(goal_masks, vein_file)} {osp.join(goal_masks, f"{vein_file_stripped}_vessels.png")}")


goal_imgs = osp.join("combined_data", "Images")
vein_imgs = osp.join("full_vein_sclera_data", "Images")

os.system(f"cp -r {vein_imgs} {goal_imgs}")