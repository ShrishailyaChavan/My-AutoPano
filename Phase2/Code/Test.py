#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import HomographySupModel, HomographyUnsupModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
from Wrapper import generate_synthetic_pairs, HomographyDataset
from torch.utils.data import Dataset, DataLoader

# Don't generate pyc codes
sys.dont_write_bytecode = True

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")


def SetupAll():
	"""
	Outputs:
	ImageSize - Size of the Image
	"""
	# Image Input Shape
	ImageSize = [32, 32, 3]

	return ImageSize


def StandardizeInputs(Img):
	##########################################################################
	# Add any standardization or cropping/resizing if used in Training here!
	##########################################################################
	return Img


def ReadImages(Img):
	"""
	Outputs:
	I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
	I1 - Original I1 image for visualization purposes only
	"""
	I1 = Img

	if I1 is None:
		# OpenCV returns empty list if image is not read!
		print("ERROR: Image I1 cannot be read")
		sys.exit()

	I1S = StandardizeInputs(np.float32(I1))

	I1Combined = np.expand_dims(I1S, axis=0)

	return I1Combined, I1


def TestOperation(ModelPath, ModelType, DirNamesTest, MiniBatchSize):
	"""
	Inputs:
	ImageSize is the size of the image
	ModelPath - Path to load trained model from
	TestSet - The test dataset
	LabelsPathPred - Path to save predictions
	Outputs:
	Predictions written to /content/data/TxtFiles/PredOut.txt
	"""
	# Predict output with forward pass, MiniBatchSize for Test is 1
	if ModelType == 'Sup':
		model = HomographySupModel()
	elif ModelType == 'Unsup':
		model = HomographyUnsupModel()
	
	lossFunc = nn.MSELoss()

	model.load_state_dict(torch.load(ModelPath))

	testDataset = HomographyDataset(DirNamesTest, transform=None)
	testDataloader = DataLoader(testDataset, batch_size=MiniBatchSize)

	with torch.no_grad():
			model.eval()
			valLoss = 0.
			for iterations, (input_patch, PA, PB, CA, CB, H4Pt) in enumerate(testDataloader):
				input = input_patch.to(device)
				Ht = Ht.to(device)
				if ModelType == 'Sup':
					H_pred = model(input)
					Ht = Ht.view(Ht.shape[0],-1)
				if ModelType == 'Unsup':
					CA = CA
					PA, PB = torch.chunk(input, dim=1, chunks=2)
					_, H_pred = model(input, CA, PA)
				
				loss = lossFunc(H_pred, Ht)
				valLoss += loss.item()

	test_img = random.choice(sorted(os.listdir(DirNamesTest)))
	test_img = test_img.permute(1,2,0).numpy()
	test_img = test_img - test_img.min()
	test_img = test_img / test_img.max()
	test_img = (test_img*255).astype(np.uint8)
	base_pts= (CA).numpy().reshape(-1,1,2).astype(np.int32)
	gt_pts= (Ht*32).numpy().reshape(-1,1,2).astype(np.int32)
	pred_pts= (H_pred*32).numpy().reshape(-1,1,2).astype(np.int32)
	gt_pts = gt_pts + base_pts
	pred_pts = pred_pts + base_pts
	test_img = cv2.polylines(test_img.copy(), [gt_pts], True, (0,255,0), 2)
	test_img= cv2.polylines(test_img, [pred_pts], True, (255,0,0), 2)




def Accuracy(Pred, GT):
	"""
	Inputs:
	Pred are the predicted labels
	GT are the ground truth labels
	Outputs:
	Accuracy in percentage
	"""
	return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
	if not (os.path.isfile(LabelsPathTest)):
		print("ERROR: Test Labels do not exist in " + LabelsPathTest)
		sys.exit()
	else:
		LabelTest = open(LabelsPathTest, "r")
		LabelTest = LabelTest.read()
		LabelTest = map(float, LabelTest.split())

	if not (os.path.isfile(LabelsPathPred)):
		print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
		sys.exit()
	else:
		LabelPred = open(LabelsPathPred, "r")
		LabelPred = LabelPred.read()
		LabelPred = map(float, LabelPred.split())

	return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
	"""
	LabelsTrue - True labels
	LabelsPred - Predicted labels
	"""

	# Get the confusion matrix using sklearn.
	LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
	cm = confusion_matrix(
		y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
	)  # Predicted class.

	# Print the confusion matrix as text.
	for i in range(10):
		print(str(cm[i, :]) + " ({0})".format(i))

	# Print the class-numbers for easy reference.
	class_numbers = [" ({0})".format(i) for i in range(10)]
	print("".join(class_numbers))

	print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
	"""
	Inputs:
	None
	Outputs:
	Prints out the confusion matrix with accuracy
	"""
	Parser = argparse.ArgumentParser()

	Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )

	Args = Parser.parse_args()

	ModelType = Args.ModelType

	ModelPath = '/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Checkpoints/ckpt29.pt'
	DirNamesTest = '/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/Test/'

	MiniBatchSize = 32

	TestOperation(ModelPath, ModelType, DirNamesTest, MiniBatchSize)



if __name__ == "__main__":
	main()
