import torch
import numpy as np
import pytest
import importlib
import os
import sys

# from models.pointnet2_reg_msg import get_model, get_loss
from data_utils.MoleculeDataSet import load_and_align_QM7, PointCloudMoleculeDataSet

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# model = importlib.import_module('pointnet2_reg_msg')

class TestPointNet2RegMSG:
    def _start(self) -> None:
        base_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
        ROOT_DIR = base_dir
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        self.model = importlib.import_module('pointnet2_reg_msg')

    def _build_default_get_model(self) -> None:
        self._start()
        n_centroids_1 = 16
        msg_radii_1 = [2., 4., 6.]
        msg_nsample_1 = [2, 4, 8]

        in_channels = 5
        out_channels = 5

        x = self.model.get_model(n_centroids_1=n_centroids_1,
                        msg_radii_1=msg_radii_1,
                        msg_nsample_1=msg_nsample_1,
                        n_centroids_2=n_centroids_1,
                        msg_radii_2=msg_radii_1,
                        msg_nsample_2=msg_nsample_1,
                        in_channels=in_channels,
                        out_channels=out_channels)
        return x

    def _build_default_PointNet2(self) -> None:
        n_centroids_1 = 16
        msg_radii_1 = [2., 4., 6.]
        msg_nsample_1 = [2, 4, 8]

        in_channels = 5
        out_channels = 5

        x = self.model.PointNet2MSGModel(n_centroids_1=n_centroids_1,
                        msg_radii_1=msg_radii_1,
                        msg_nsample_1=msg_nsample_1,
                        n_centroids_2=n_centroids_1,
                        msg_radii_2=msg_radii_1,
                        msg_nsample_2=msg_nsample_1,
                        in_channels=in_channels,
                        out_channels=out_channels)
    
    def test_build_get_model(self) -> None:
        """
        Tests that the model builds
        """
        self._start()
        x = self._build_default_get_model()

    def test_build_PointNet2(self) -> None:
        self._start()
        x = self._build_default_PointNet2()
        
    def test_loss(self) -> None:
        self._start()
        loss = self.model.get_loss()

        a = np.random.normal(size=10)
        b = np.random.normal(size=10)

        expected_val = np.mean(np.square(a - b))

        out = loss(torch.Tensor(a), torch.Tensor(b))

        assert np.allclose(expected_val, out.numpy()), f"{expected_val} vs {out}"

    def test_sign_flip_0(self) -> None:
        self._start()
        x = self._build_default_get_model()

        U_matrices_in = torch.Tensor([[[2, 2, 2],
                                        [3, 3, 3],
                                        [1, 1, 1]],
                                        
                                        [[5, 6, 7],
                                        [3, 3, 3],
                                        [1, 1, 1]]])

        expected_out = torch.Tensor([[[-2, 2, -2],
                                        [-3, 3, -3],
                                        [-1, 1, -1]],
                                        
                                        [[-5, 6, -7],
                                        [-3, 3, -3],
                                        [-1, 1, -1]]])

        sign_flip_arr = torch.Tensor([-1, 1, -1])

        out = x._flip_signs_U_matrix(U_matrices_in, sign_flip_arr)

        assert torch.allclose(out, expected_out)

    def test_align_coords_0(self) -> None:
        """
        Make sure the point cloud with NaNs and identity U matrix doesn't get
        messed up
        """
        self._start()
        x = self._build_default_get_model()

        U_matrix = torch.Tensor([   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]],
                                    
                                    [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]]
                                ])
        points_and_features = torch.full((2, 10, 8), torch.nan)

        points_and_features[:, :6, :] = torch.normal(torch.zeros((2, 6, 8)), 1.)

        out = x._align_coords(U_matrix, points_and_features)

        assert torch.allclose(out, points_and_features, equal_nan=True)

    def test_align_coords_1(self) -> None:
        """
        Make sure the point cloud with NaNs and identity U matrix doesn't get
        messed up
        """
        self._start()
        x = self._build_default_get_model()

        U_matrix = torch.Tensor([   [[0, 0, 1],
                                    [1,  0, 0],
                                    [0,  1, 0]],
                                    
                                    [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]]
                                ])

        points_and_features = torch.Tensor([
            [[1, 2, 3, 1, 0, 0, 0, 0],
             [4, 5, 6, 0, 1, 0, 0, 0],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan]],

             [[1, 2, 3, 1, 0, 0, 0, 0],
             [4, 5, 6, 0, 1, 0, 0, 0],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan]]
        ])

        expected_out = torch.Tensor([
            [[2, 3, 1, 1, 0, 0, 0, 0],
             [5, 6, 4, 0, 1, 0, 0, 0],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan]],

             [[1, 2, 3, 1, 0, 0, 0, 0],
             [4, 5, 6, 0, 1, 0, 0, 0],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
             [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan]]
        ])
        out = x._align_coords(U_matrix, points_and_features)

        assert torch.allclose(out, expected_out, equal_nan=True)


class TestIntegrationQM7Data:
    def test_integration_0(self) -> None:
        DATA_FP = '/Users/owen/projects/invariant-random-features-code/data/qm7/qm7.mat'
        n_train = 100
        n_test = 100
        train_dset, val_dset, test_dset = load_and_align_QM7(fp=DATA_FP,
                                                        n_train=n_train,
                                                        n_test=n_test,
                                                        validation_set_fraction=0.1)
        batch_size = 12
        loader = torch.utils.data.DataLoader(train_dset, 
                                                batch_size=batch_size, 
                                                shuffle=True)

        # Imports for building model
        base_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
        ROOT_DIR = base_dir
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        model = importlib.import_module('pointnet2_reg_msg')

        # Build the model
        n_centroids_1 = 16
        msg_radii_1 = [2., 4., 6.]
        msg_nsample_1 = [2, 4, 8]
        in_channels = 5
        out_channels = 1
        x = model.PointNet2MSGModel(n_centroids_1=n_centroids_1,
                        msg_radii_1=msg_radii_1,
                        msg_nsample_1=msg_nsample_1,
                        n_centroids_2=n_centroids_1,
                        msg_radii_2=msg_radii_1,
                        msg_nsample_2=msg_nsample_1,
                        in_channels=in_channels,
                        out_channels=out_channels)

        loss_fn = model.get_loss()

        for points_and_features, U_matrices, targets in loader:
            
            pred = x.forward(points_and_features).flatten()

            assert torch.logical_not(torch.any(torch.isnan(pred)))

            assert list(pred.shape) == list(targets.shape)

            loss = loss_fn(pred, targets)

            loss.backward()

    def test_integration_1(self) -> None:
        DATA_FP = '/Users/owen/projects/invariant-random-features-code/data/qm7/qm7.mat'
        n_train = 100
        n_test = 100
        train_dset, val_dset, test_dset = load_and_align_QM7(fp=DATA_FP,
                                                        n_train=n_train,
                                                        n_test=n_test,
                                                        validation_set_fraction=0.1)
        batch_size = 12
        loader = torch.utils.data.DataLoader(train_dset, 
                                                batch_size=batch_size, 
                                                shuffle=True)

        # Imports for building model
        base_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
        ROOT_DIR = base_dir
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        model = importlib.import_module('pointnet2_reg_msg')

        # Build the model
        n_centroids_1 = 16
        msg_radii_1 = [2., 4., 6.]
        msg_nsample_1 = [2, 4, 8]
        in_channels = 5
        out_channels = 64
        x = model.get_model(n_centroids_1=n_centroids_1,
                        msg_radii_1=msg_radii_1,
                        msg_nsample_1=msg_nsample_1,
                        n_centroids_2=n_centroids_1,
                        msg_radii_2=msg_radii_1,
                        msg_nsample_2=msg_nsample_1,
                        in_channels=in_channels,
                        out_channels=out_channels)

        loss_fn = model.get_loss()

        for points_and_features, U_matrices, targets in loader:
            
            pred = x.forward(points_and_features, U_matrices).flatten()

            assert torch.logical_not(torch.any(torch.isnan(pred)))

            assert list(pred.shape) == list(targets.shape)

            loss = loss_fn(pred, targets)

            loss.backward()
