#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric CoHPuter Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

from __future__ import print_function, division
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

import os
import glob
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import random
import pickle
import numpy as np


# Add any python libraries here

def generate_synthetic_pairs(img_a, WP, HP, rho):

    # Obtain the shape of the image
    W, H, C = img_a.shape

    # Step 1:
    # Generate random numbers for the x and y coordinates of the top-left corner of the patch (PA)
    x_PA = np.random.randint(0, W-WP)
    y_PA = np.random.randint(0, H-HP)

    # Generate random numbers for the x and y coordinates of the corners of the patch (CA)
    x_CA_top_left = x_PA
    y_CA_top_left = y_PA
    x_CA_top_right = x_PA
    y_CA_top_right = y_PA + HP
    x_CA_bottom_left = x_PA + WP
    y_CA_bottom_left = y_PA
    x_CA_bottom_right = x_PA + WP
    y_CA_bottom_right = y_PA + HP

    # Generate random numbers for the x and y coordinates of the perturbed corners of the patch (CB)
    x_CB_top_left = x_CA_top_left + np.random.randint(-rho, rho)
    y_CB_top_left = y_CA_top_left + np.random.randint(-rho, rho)
    x_CB_top_right = x_CA_top_right + np.random.randint(-rho, rho)
    y_CB_top_right = y_CA_top_right + np.random.randint(-rho, rho)
    x_CB_bottom_left = x_CA_bottom_left + np.random.randint(-rho, rho)
    y_CB_bottom_left = y_CA_bottom_left + np.random.randint(-rho, rho)
    x_CB_bottom_right = x_CA_bottom_right + np.random.randint(-rho, rho)
    y_CB_bottom_right = y_CA_bottom_right + np.random.randint(-rho, rho)

    # Step 2:
    # Compute the homography matrix (HAB) between the original corner points of the patch (PA) and the perturbed corner points (CB)
    CA = np.float32([[x_CA_top_left, y_CA_top_left], [x_CA_top_right, y_CA_top_right], [x_CA_bottom_left, y_CA_bottom_left], [x_CA_bottom_right, y_CA_bottom_right]])
    CB = np.float32([[x_CB_top_left, y_CB_top_left], [x_CB_top_right, y_CB_top_right], [x_CB_bottom_left, y_CB_bottom_left], [x_CB_bottom_right, y_CB_bottom_right]])
    HAB = cv2.getPerspectiveTransform(CA, CB)

    # Compute the inverse of the homography matrix (HBA)

    HBA = np.linalg.inv(HAB)

    # Step 3:
    # Warp the image img_a using the computed inverse homography matrix (HBA)
    img_b = cv2.warpPerspective(img_a, HBA, (H, W))

    # Extract the patch PB from the warped image img_b using the original corner points (CA) of the patch (PA)
    PB = img_b[x_CA_top_left:x_CA_top_left+WP, y_CA_top_left:y_CA_top_left+HP, :]

    # Compute the homography matrix (H4Pt) between the original corner points (CA) and the corner points (CB) of the patch PB
    H4Pt = CB - CA

    # Stack the image patches PA and PB depthwise to obtain an input of size MPxNPx2K
    PA = img_a[x_PA:x_PA+WP, y_PA:y_PA+HP, :]

    # # Normalize PA
    mean_PA = np.mean(PA)
    PA = (PA - mean_PA) / 255

    # Normalize PB
    mean_PB = np.mean(PB)
    PB = (PB - mean_PB) / 255

    # Stack the normalized image patches PA and PB depthwise to obtain an input of size WPxHPx2C
    input_patch = np.dstack((PA, PB))

    return input_patch, PA, PB, CA, CB, H4Pt

class HomographyDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs_files = sorted(os.listdir(root))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs_files[idx])

        img_a = cv2.imread(img_path)

        input_patch, PA, PB, CA, CB, H4Pt = generate_synthetic_pairs(img_a, 128, 128, 32) 

        input_patch = torch.from_numpy(np.swapaxes(input_patch, 2, 0))
        H4Pt = torch.tensor(np.reshape(H4Pt, (8)))

        return input_patch, PA, PB, CA, CB, H4Pt

    
    def __len__(self):
        return len(self.imgs_files)



def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    # # Define the path of the dataset
    # data_path = "/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/Train/"

    # # Define the path to save the synthetic data
    # save_path = "/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/Synth_Pairs/"

    # train_test_val_path = "/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/train_test_val/"

    # # Define the size of the patch and the maximum perturbation value (rho)
    # WP = 128
    # HP = 128
    # rho = 20

    # # Loop through all the .jpg images in the dataset
    # for file in os.listdir(data_path):
    #     if file.endswith(".jpg"):
    #         # Read the image
    #         img_a = cv2.imread(os.path.join(data_path, file))
    #         # Generate synthetic data
    #         input_patch, H4Pt = generate_synthetic_pairs(img_a, WP, HP, rho)
    #         vis_input = input_patch[:,:,:3]
    #         # Save the synthetic data
    #         np.save(os.path.join(save_path, file), input_patch)
    #         np.save(os.path.join(save_path, file+'_H4Pt'), H4Pt)
    #         # cv2.imwrite("/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/Synth_Imgs/" + file + ".jpg", vis_input)

    # if os.listdir(save_path):
    #     split_data(save_path, train_test_val_path)


    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()


