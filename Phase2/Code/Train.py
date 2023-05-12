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
# termcolor, do (pip install termcolor)

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from Wrapper import generate_synthetic_pairs, HomographyDataset
from Network.Network import HomographySupModel, HomographyUnsupModel, LossFn, Loss
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

print("Which device?", device)
LossThisBatch = None

def GenerateBatch(BasePath, DirNamesTrain, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
    print ("A new epoch started")
    print("Length of the training set", len(DirNamesTrain))
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        WP = 128
        HP = 128
        rho = 32

        img_a = cv2.imread(RandImageName)

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        I1 = generate_synthetic_pairs(img_a, WP, HP, rho)[0]    #np.float32(cv2.imread(RandImageName))
        I1 = np.swapaxes(I1, 2, 0)
        Coordinates = generate_synthetic_pairs(img_a, WP, HP, rho)[1] 

        Coordinates = np.reshape(Coordinates, (1,8))
        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(I1))
        CoordinatesBatch.append(torch.tensor(Coordinates))


    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesVal,
    DirNamesTrain,
    NumEpochs,
    MiniBatchSize,
    CheckPointPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    trainDataset = HomographyDataset(DirNamesTrain, transform=None)
    valDataset = HomographyDataset(DirNamesVal, transform=None)

    trainDataloader = DataLoader(trainDataset, batch_size=MiniBatchSize)
    valDataloader = DataLoader(valDataset, batch_size=MiniBatchSize)
    if ModelType == 'Sup':
        model = HomographySupModel().to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9)
        torch.autograd.set_detect_anomaly(True)
        # torch.autograd.set_detect_anomaly(True)

        loss_plot = np.zeros(NumEpochs)
        acc_plot = np.zeros(NumEpochs)
        epochs_plot = np.zeros(NumEpochs)

        for Epochs in range(NumEpochs):
            model.train()
            epochLoss = 0.
            unsupLoss = 0.
            counter = 0

            for iteration, (input_patch, PA, PB, CA, CB, H4Pt) in enumerate(trainDataloader):
                I1, Ht = input_patch.to(device), H4Pt.to(device)
                counter = counter + 1
                H_pred = model(I1)
                # print(Ht)
                # print(H_pred)
                loss = Loss(H_pred, Ht)

                epochLoss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration % 50 == 0:
                    epochLoss_avg = epochLoss / (iteration+1)
                    with torch.no_grad():
                        model.eval()
                        valLoss = 0.
                        valLoss2 = 0.
                        for iteration_, (input_patch, PA, PB, CA, CB, H4Pt) in enumerate(valDataloader):
                            input = input_patch.to(device)
                            Ht = H4Pt.to(device)
                            H_pred = model(input)
                            # H_gt = H_gt.view(H_gt.shape[0],-1)
                            loss = Loss(H_pred, Ht)
                            valLoss += loss.item()

                        valLoss_avg = valLoss / (iteration +1)
                        valLoss_avg2 = valLoss2 / (iteration_ +1)

                        print(f"Epoch : {Epochs}, Iter : {iteration}, Train Loss : {epochLoss_avg}, Val Loss : {valLoss_avg}, Val Loss 2 : {valLoss_avg2}")

                    model.train()

            loss_plot[Epochs] = epochLoss/counter
            epochs_plot[Epochs] = Epochs   
            print('Loss plot:=', loss_plot)
            print('Epoch plot:= ', epochs_plot)
            print('Epoch [{}/{}], Loss: {:.4f}'.format(Epochs+1, NumEpochs, epochLoss/counter))
     
            torch.save(model.state_dict(), os.path.join(CheckPointPath, f"ckpt{Epochs}.pt"))
        
        plt.subplot(1,1,1)
        plt.xlim(0,NumEpochs)
        plt.ylim(0, 100)
        plt.title('Training loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.plot(epochs_plot, loss_plot)
        plt.savefig('/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Results/EpochLoss.png')

    elif ModelType == 'Unsup':
        model = HomographySupModel().to(device)
        lossFunc = nn.L1Loss()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9)
        torch.autograd.set_detect_anomaly(True)
        # torch.autograd.set_detect_anomaly(True)

        loss_plot = np.zeros(NumEpochs)
        acc_plot = np.zeros(NumEpochs)
        epochs_plot = np.zeros(NumEpochs)

        for Epochs in range(NumEpochs):
            model.train()
            epochLoss = 0.
            unsupLoss = 0.
            counter = 0

            for iteration, (input_patch, PA, PB, CA, CB, H4Pt) in enumerate(trainDataloader):
                PA, PB = torch.chunk(input_patch, dim=1, chunks=2)				
                PB_pred , error = model(input_patch.to(device), CA.to(device), PA.to(device))
                loss = lossFunc(PB_pred, PB)
                loss2 = lossFunc(error, Ht)
                unsupLoss += loss2.item()

                epochLoss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration % 50 == 0:
                    epochLoss_avg = epochLoss / (iteration+1)
                    with torch.no_grad():
                        model.eval()
                        valLoss = 0.
                        valLoss2 = 0.
                        for iteration_, (input_patch, PA, PB, CA, CB, H4Pt) in enumerate(valDataloader):
                            input = input_patch.to(device)
                            Ht = H4Pt.to(device)
                            CA = CA
                            PA, PB = torch.chunk(input_patch, dim=1, chunks=2)
                            PB_pred, error = model(input, CA, PA)
                            loss = lossFunc(PB_pred, PB)
                            loss2 = lossFunc(error, Ht)
                            valLoss2 += loss2.item()

                        valLoss_avg = valLoss / (iteration +1)
                        valLoss_avg2 = valLoss2 / (iteration_ +1)

                        print(f"Epoch : {Epochs}, Iter : {iteration}, Train Loss : {epochLoss_avg}, Val Loss : {valLoss_avg}, Val Loss 2 : {valLoss_avg2}")

                    model.train()

            loss_plot[Epochs] = epochLoss/counter
            epochs_plot[Epochs] = Epochs   
            print('Loss plot:=', loss_plot)
            print('Epoch plot:= ', epochs_plot)
            print('Epoch [{}/{}], Loss: {:.4f}'.format(Epochs+1, NumEpochs, epochLoss/counter))
     
            torch.save(model.state_dict(), os.path.join(CheckPointPath, f"ckpt{Epochs}.pt"))
        
        plt.subplot(1,1,1)
        plt.xlim(0,NumEpochs)
        plt.ylim(0, 100)
        plt.title('Training loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.plot(epochs_plot, loss_plot)
        plt.savefig('/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Results/EpochLoss.png')

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=30,
        help="Number of Epochs to Train for, Default:30",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=32,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # # Setup all needed parameters including file reading
    # (   
    #     DirNamesVal,
    #     DirNamesTrain,        
    #     SaveCheckPoint,
    #     ImageSize,
    #     NumTrainSamples,
    #     TrainCoordinates,
    #     NumClasses,
    # ) = SetupAll(BasePath, CheckPointPath)

    DirNamesTrain = "/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/Train/"
    DirNamesVal = "/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase2/Data/Val/"

    TrainOperation(
        DirNamesVal,
        DirNamesTrain,
        NumEpochs,
        MiniBatchSize,
        CheckPointPath,
        ModelType,
    )


if __name__ == "__main__":
    main()