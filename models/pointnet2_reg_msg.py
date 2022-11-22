"""
This model is for regression on QM7, QM9 datasets using the PointNet2 architecture
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, 
                    n_centroids_1: int,
                    msg_radii_1: List[float],
                    msg_nsample_1: List[int],
                    n_centroids_2: int,
                    msg_radii_2: List[float],
                    msg_nsample_2: List[int],
                    in_channel: int):
        super(get_model, self).__init__()
        # in_channel = 3 if normal_channel else 0
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(n_centroids_1, 
                                                msg_radii_1, 
                                                msg_nsample_1, 
                                                in_channel,
                                                [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(n_centroids_2, 
                                                msg_radii_2, 
                                                msg_nsample_2, 
                                                320,
                                                [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, 
                                            None, 
                                            None, 
                                            640 + 3, 
                                            [256, 512, 1024], 
                                            True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Forward pass of the regression network

        Args:
            xyz (torch.Tensor): has shape (batch_size, max_n_atoms, 3 + feature_dim)

        Returns:
            torch.Tensor: size [batch_size,]
        """
        B, _, _ = xyz.shape
        in_xyz = xyz[:, :, :3] # Norm are the features in the 4th and on columns
        in_features = xyz[:, :, 3:] # XYZ are the cartesian coordinates

        l1_xyz, l1_features = self.sa1(in_xyz, in_features)
        if torch.any(torch.isnan(l1_xyz)):
            raise ValueError("l1_xyz contains NaNs")
        
        if torch.any(torch.isnan(l1_features)):
            raise ValueError("l1_features contains NaNs")
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        x = l3_features.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        return torch.sum(torch.square(pred - target))



