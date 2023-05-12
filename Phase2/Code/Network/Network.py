"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""
import pytorch_lightning as pl
import torch.nn as nn
import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F# Don't generate pyc codes
import pytorch_lightning as pl
import torch.nn as nn
import sys
import torch
import numpy as np
from TensorDLT import TensorDLT
import cv2
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
# import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

print("Which device?", device)

# def LossFn(delta, img_a, patch_b, corners):
#     ###############################################
#     # Fill your loss function of choice here!
#     ###############################################

#     ###############################################
#     # You can use kornia to get the transform and warp in this project
#     # Bonus if you implement it yourself
#     ###############################################
#     loss = ...
#     return loss

def Loss(Ht, Gt):
    criterion = nn.MSELoss()

    Gt = Gt.float()  # Ground truth
    loss = torch.sqrt(criterion(Ht, Gt))
    return loss


def LossFn(delta, img_a, patch_b, corners):
    # reshape delta to 2x3 matrix
    delta = delta.view(-1, 2, 3)
    
    # create identity homography
    identity = torch.eye(3).to(delta.device).unsqueeze(0)
    # concatenate delta with identity
    homography = torch.cat((delta, identity[:, :2, :]), dim=1)
    
    # warp patch_b using homography
    warped_patch = cv2.warp_perspective(patch_b, homography, dsize=(img_a.shape[2], img_a.shape[3]))
    
    # crop patch from img_a using corners
    img_a_patch = cv2.crop(img_a, corners[:, 0], corners[:, 1], patch_b.shape[2], patch_b.shape[3])
    
    # calculate L2 loss
    loss = nn.MSELoss(warped_patch, img_a_patch)
    return loss



# class HomographySupModel(pl.LightningModule):
#     def __init__(self):
#         super(HomographySupModel, self).__init__()
#         self.model = SupervisedNet()

#     def forward(self, a):
#         return self.model(a)

#     def training_step(self, batch, homography):
#         img_a, patch_a, patch_b, corners, gt = batch
#         delta = self.model(patch_a, patch_b)
#         loss = LossFn(delta, img_a, patch_b, corners)
#         logs = {"loss": loss}
#         return {"loss": loss, "log": logs}

#     def validation_step(self, batch, homography):
#         # img_a, patch_a, patch_b, corners, gt = batch
#         delta = self.model(batch)
#         loss = Loss(delta, homography)
#         return {"val_loss": loss}

#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         logs = {"val_loss": avg_loss}
#         return {"avg_val_loss": avg_loss, "log": logs}


class HomographySupModel(nn.Module):
    def __init__(self):
        super(HomographySupModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.regress = torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 1024),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(1024,8)
        )
        
        self.dropout = nn.Dropout(0.1) 
        self.fc1 = nn.Linear(16*16*128, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.conv1(x.float().to(device))
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.conv8(x)

        x = x.view(-1, 16*16*128)
        # x1 = self.fc1(x)
        # x1 = self.fc2(x1)
        x = self.fc1(x)
        x = self.fc2(x)

        # x = x.view(-1, 128, 16, 16)

        # x2 = self.regress(x)

        # x = x1 + x2

        return x

class HomographyUnsupModel(HomographySupModel):
    def __init__(self):
        super(HomographyUnsupModel, self).__init__()
        self.dlt = TensorDLT()

    def forward(self, x, CA, PA): 
        out = self.model(x)
        error = self.regress(out)
        error = error*32
        error = error.view(error.shape[0],-1,2)
        H = self.dlt(CA, CA+error)
        H_inv = torch.inverse(H)
        patchB = warp_perspective(PA, H_inv, (128,128))
        return patchB, error / 32.


