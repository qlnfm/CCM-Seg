import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from glob import glob
from tqdm import tqdm


def cl_skeleton(img):
    binary = img > 0
    return skeletonize(binary)


def clDice(v_p, v_l):
    s_p = cl_skeleton(v_p)
    s_l = cl_skeleton(v_l)
    t_prec = np.sum(s_p * v_l) / (np.sum(s_p) + 1e-6)
    t_sens = np.sum(s_l * v_p) / (np.sum(s_l) + 1e-6)
    cldice = 2 * (t_prec * t_sens) / (t_prec + t_sens + 1e-6)
    return cldice


def calculate_folder_cldice(dir1, dir2):
    files = glob(os.path.join(dir1, "*.png"))
    cldice_list = []

    print(f"{len(files)} images...")

    for path1 in tqdm(files):
        filename = os.path.basename(path1)
        path2 = os.path.join(dir2, filename)

        if not os.path.exists(path2):
            print(f"WARNING：No {filename}")
            continue

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        img1 = (img1 / 255).astype(np.float32)
        img2 = (img2 / 255).astype(np.float32)

        score = clDice(img1, img2)
        cldice_list.append(score)

    if not cldice_list:
        return 0

    return np.mean(cldice_list)


if __name__ == '__main__':
    dir_gt = "Dataset/annotations"
    dir_root = 'Output/predictions_set2'
    scores = []
    for dir_pred in os.listdir(dir_root):
        score = calculate_folder_cldice(os.path.join(dir_root, dir_pred), dir_gt)
        scores.append(score)
    print(f"\nclDice: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
