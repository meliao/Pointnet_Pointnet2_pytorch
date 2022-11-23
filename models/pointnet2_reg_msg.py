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
                    in_channels: int,
                    out_channels: int):

        super(get_model, self).__init__()
        self.out_channels = out_channels
        self.pointnet = PointNet2MSGModel(n_centroids_1=n_centroids_1,
                                            msg_radii_1=msg_radii_1,
                                            msg_nsample_1=msg_nsample_1,
                                            n_centroids_2=n_centroids_2,
                                            msg_radii_2=msg_radii_2,
                                            msg_nsample_2=msg_nsample_2,
                                            in_channels=in_channels,
                                            out_channels=out_channels)

        self.linear_layer_1 = nn.Linear(out_channels, 64)
        self.linear_layer_2 = nn.Linear(64, 1)
        self.sign_flip_list = [torch.Tensor([1, 1, 1]),
                                torch.Tensor([1, -1, -1]),
                                torch.Tensor([-1, 1, -1]),
                                torch.Tensor([-1, -1, 1])]
    def _flip_signs_U_matrix(self, 
                                U_matrix: torch.Tensor, 
                                sign_flips: torch.Tensor) -> torch.Tensor:
        # Want to be super explicit about broadcasting here; I want to 
        # flip the signs of the columns of U
        sign_flip_mat = sign_flips.view([-1, 1, 3]).repeat([1, 3, 1])
        return torch.mul(U_matrix, sign_flip_mat)
        

    def _align_coords(self, U_matrices: torch.Tensor, 
                                    points_and_features: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication for each element in the batch

        Args:
            U_matrix (torch.Tensor): Have shape (batch_size, 3, 3)
            data_coords (torch.Tensor): Have shape (batch_size, 3, max_n_atoms)

        Returns:
            torch.Tensor: Have shape (batch_size, max_n_atoms, 3)
        """

        # If P is the point cloud (n_points, 3), we want to do U^T @ P^T
        out = torch.clone(points_and_features)
        P = points_and_features[:, :, :3]
        out[:, :, :3] = torch.bmm(U_matrices.permute([0, 2, 1]), P.permute([0, 2, 1])).permute([0, 2, 1])
        return out


    def forward(self, points_and_features: torch.Tensor, U_matrices: torch.Tensor) -> torch.Tensor:
        """For each possible sign flip of the U_matrices, this function
        computes the re-orientation of points_and_features, and then makes a 
        forward pass through the network.
        

        Args:
            points_and_features (torch.Tensor): _description_
            U_matrices (torch.Tensor): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            torch.Tensor: _description_
        """
        n_batch, n_points, _ = points_and_features.shape
        out_features = torch.empty(size=(n_batch, 4, self.out_channels))
        for i, sign_flip_arr in enumerate(self.sign_flip_list):
            U_matrices_flipped = self._flip_signs_U_matrix(U_matrices, sign_flip_arr)
            aligned_coords = self._align_coords(U_matrices_flipped, points_and_features)

            out_features[:, i] = self.pointnet(aligned_coords)

        # LogSumExp pooling
        features = torch.logsumexp(out_features, dim=1) # shape [B, out_channels]

        features_1 = F.relu(self.linear_layer_1(features))

        out = self.linear_layer_2(features_1)

        return out



class PointNet2MSGModel(nn.Module):
    def __init__(self, 
                    n_centroids_1: int,
                    msg_radii_1: List[float],
                    msg_nsample_1: List[int],
                    n_centroids_2: int,
                    msg_radii_2: List[float],
                    msg_nsample_2: List[int],
                    in_channels: int,
                    out_channels: int):
        super(PointNet2MSGModel, self).__init__()
        # in_channel = 3 if normal_channel else 0
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(n_centroids_1, 
                                                msg_radii_1, 
                                                msg_nsample_1, 
                                                in_channels,
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
        self.fc3 = nn.Linear(256, out_channels)

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
        return torch.mean(torch.square(pred - target))



