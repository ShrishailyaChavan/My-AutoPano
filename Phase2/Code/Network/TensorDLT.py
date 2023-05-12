import torch
import torch.nn as nn

class TensorDLT(nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()
    
    def forward(self, delta, corners):
        # convert delta to H4Pt
        H4Pt = delta.view(delta.shape[0], 2, 2)
        
        # create matrices for TensorDLT calculation
        ones = torch.ones((delta.shape[0], 1, 1), dtype=torch.float32, device=delta.device)
        zeros = torch.zeros((delta.shape[0], 1, 1), dtype=torch.float32, device=delta.device)
        A1 = torch.cat((H4Pt, zeros, corners[:, 0:1, :]), dim=2)
        A2 = torch.cat((zeros, H4Pt, corners[:, 1:2, :]), dim=2)
        A3 = torch.cat((zeros, zeros, ones), dim=2)
        
        # calculate estimated homography matrix
        A = torch.cat((A1, A2, A3), dim=1)
        _, _, V = torch.svd(A)
        H = V[:, -1].view(delta.shape[0], 3, 3)
        
        return H