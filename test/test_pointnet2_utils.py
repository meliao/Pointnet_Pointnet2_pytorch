import torch
import numpy as np
import pytest

from models.pointnet2_utils import (PointNetSetAbstractionMsg, 
                                    PointNetSetAbstraction,
                                    farthest_point_sample,
                                    query_ball_point, 
                                    sample_and_group_all, 
                                    sample_and_group)

class TestFarthestPointSample:
    def test_FPS_simple_input(self) -> None:
        in_arr = np.array([[[0, 0, 1], # 0
                        [0, 0, 1.1], # 1
                        [1, 0, 0], # 2
                        [1.1, 0, 0]]]) # 3
        in_tensor = torch.Tensor(in_arr)
        out = farthest_point_sample(in_tensor, 2)
        
        if out[0, 0] in [2, 3]:
            assert out[0, 1] == 1
        else:
            assert out[0, 1] == 3


    def test_FPS_simple_input_with_nans(self) -> None:
        in_arr = np.array([[[0, 0, 1], # 0
                        [0, 0, 1.1], # 1
                        [1, 0, 0], # 2
                        [np.nan, np.nan, np.nan]]]) # 3
        in_tensor = torch.Tensor(in_arr)
        print(in_tensor.shape)
        out = farthest_point_sample(in_tensor, 2)
        print("OUT", out)
        if out[0, 0] in [0, 1]:
            assert out[0, 1] == 2
        else:
            assert out[0, 1] == 1

    def test_FPS_random_input_NaNs(self) -> None:
        """
        Tests that inputting NaNs responds in well-defined behavior for
        the furthest_point_sampling
        """
        
        
        in_xyz_arr = np.full((2, 7, 3), np.nan)
        in_xyz_arr[0, :6] = np.random.normal(size=(6, 3))
        in_xyz_arr[1, :5] = np.random.normal(size=(5, 3))
        
        in_tensor = torch.Tensor(in_xyz_arr)
        
        out = farthest_point_sample(in_tensor, 3)

        assert out[0, 0] != 6
        assert out[0, 1] != 6
        assert out[0, 2] != 6
        assert out[1, 0] < 5
        assert out[1, 1] < 5
        assert out[1, 2] < 5

    def test_FPS_small_point_cloud(self) -> None:
        """
        Tests that the behavior is well-defined when the furthest_point_sampling
        function is asked for more points than the point cloud contains
        """
        in_xyz_arr = np.full((2, 7, 3), np.nan)
        in_xyz_arr[0, :6] = np.random.normal(size=(6, 3))
        in_xyz_arr[1, :5] = np.random.normal(size=(5, 3))
        in_tensor = torch.Tensor(in_xyz_arr)

        out = farthest_point_sample(in_tensor, 20)
        out_arr = out.numpy()

        out_0 = out_arr[0]
        out_0_sorted_distinct = np.sort(np.unique(out_0))
        assert np.allclose(out_0_sorted_distinct, np.arange(6))


        out_1 = out_arr[1]
        out_1_sorted_distinct = np.sort(np.unique(out_1))
        assert np.allclose(out_1_sorted_distinct, np.arange(5))


class TestSampleAndGroupAll:
    def test_0(self) -> None:
        n_batch = 12
        max_n_points = 20
        input_xyz = torch.full((n_batch, max_n_points, 3), torch.nan)
        input_xyz[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 3)), std=1.)
        input_features = torch.full((n_batch, max_n_points, 10), torch.nan)
        input_features[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 10)), std=1.)

        out_xyz, out_features = sample_and_group_all(input_xyz, input_features)

        assert list(out_xyz.shape) == [n_batch, 1, 3]
        assert list(out_features.shape) == [n_batch, 1, max_n_points, 3+10]


class TestSampleAndGroup:
    def test_0(self) -> None:
        """
        Tests that sample_and_group returns output of the expected size
        """
        n_batch = 12
        max_n_points = 20
        input_xyz = torch.full((n_batch, max_n_points, 3), torch.nan)
        input_xyz[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 3)), std=1.)
        input_features = torch.full((n_batch, max_n_points, 10), torch.nan)
        input_features[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 10)), std=1.)

        radius = 1.
        nsample = 4
        npoint = 5

        out_xyz, out_features = sample_and_group(npoint=npoint,
                                                    radius=radius,
                                                    nsample=nsample,
                                                    xyz=input_xyz,
                                                    features=input_features,
                                                    returnfps=False)
        assert list(out_xyz.shape) == [n_batch, npoint, 3]
        assert list(out_features.shape) == [n_batch, npoint, nsample, 3+10]

    def test_1(self) -> None:
        """
        Tests that sample_and_group returns output of the expected size without 
        NaNs 
        """
        n_batch = 12
        max_n_points = 20
        input_xyz = torch.full((n_batch, max_n_points, 3), torch.nan)
        input_xyz[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 3)), std=1.)
        input_features = torch.full((n_batch, max_n_points, 10), torch.nan)
        input_features[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 10)), std=1.)

        radius = 1.
        nsample = 4
        npoint = 5

        out_xyz, out_features = sample_and_group(npoint=npoint,
                                                    radius=radius,
                                                    nsample=nsample,
                                                    xyz=input_xyz,
                                                    features=input_features,
                                                    returnfps=False)
        assert list(out_xyz.shape) == [n_batch, npoint, 3]
        assert list(out_features.shape) == [n_batch, npoint, nsample, 3+10]
        
        assert torch.logical_not(torch.any(torch.isnan(out_xyz))), out_xyz
        assert torch.logical_not(torch.any(torch.isnan(out_features))), out_features


class TestQueryBallPoint:

    def test_query_ball_point_0(self) -> None:
        """
        Tests that the query_ball_point function runs without error
        """
        in_xyz = torch.Tensor([[[0, 0, 0],
                                [1, 0, 0],
                                [2, 0, 0],
                                [3, 0, 0],
                                [4, 0, 0]]])
        in_query_points = torch.Tensor([[[0, 0, 0],
                                        [4, 0, 0]]])
        in_query_idxes = torch.Tensor([[0, 4]])
        
        out = query_ball_point(1., 2, in_xyz, in_query_points, in_query_idxes)

    def test_query_ball_point_1(self) -> None:
        """
        Tests that the query_ball_point function returns the correct number of 
        points when there are more than enough points for subsampling
        """
        in_xyz = torch.Tensor([[[0, 0, 0],
                                [1, 0, 0],
                                [2, 0, 0],
                                [3, 0, 0],
                                [4, 0, 0]]])
        in_query_points = torch.Tensor([[[0, 0, 0]]])
        in_query_idxes = torch.Tensor([[0]])
        
        out = query_ball_point(3., 2, in_xyz, in_query_points, in_query_idxes)
        expected_out = torch.Tensor([[0, 1]]).type(torch.long)
        assert torch.allclose(out, expected_out)    


    def test_query_ball_point_2(self) -> None:
        """
        Tests that the query_ball_point function returns the correct number of 
        points when there are not enough points for subsampling
        """
        in_xyz = torch.Tensor([[[0, 0, 0],
                                [1, 0, 0],
                                [2, 0, 0],
                                [3, 0, 0],
                                [4, 0, 0]]])
        in_query_points = torch.Tensor([[[0, 0, 0]]])
        in_query_idxes = torch.Tensor([[0]])
        
        out = query_ball_point(1., 3, in_xyz, in_query_points, in_query_idxes)
        expected_out = torch.Tensor([[0, 1, 0]]).type(torch.long)
        assert torch.allclose(out, expected_out)    


class TestPointNetSetAbstractionMsg:
    def test_0(self) -> None:
        """
        Asserts the module can do a forward and backward pass 
        when given the proper input.
        """

        npoint = 3
        radius_list = [1.0, 2.0, 3.0]
        nsample_list = [2, 4, 8]
        in_channel = 0
        mlp_list = [[32, 32, 64], [64, 64, 128], [64, 96, 128]]

        x = PointNetSetAbstractionMsg(npoint=npoint, 
                                        radius_list=radius_list,
                                        nsample_list=nsample_list, 
                                        in_channel=in_channel, 
                                        mlp_list=mlp_list)

        n_batch = 12
        max_n_points = 20
        input_xyz = torch.full((n_batch, max_n_points, 3), torch.nan)
        input_xyz[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 3)), std=1.)

        input_xyz.requires_grad = True

        assert list(input_xyz.shape) == [n_batch, max_n_points, 3]


        out = x(input_xyz, None)
        assert torch.logical_not(torch.any(torch.isnan(out[0]))), out[0]
        assert torch.logical_not(torch.any(torch.isnan(out[1]))), out[1]


        assert list(out[0].shape) == [n_batch, npoint, 3]

        loss_0 = torch.sum(out[0])
        loss_1 = torch.sum(out[1])

        loss_0.backward()

        # loss_1.backward()


    def test_1(self) -> None:
        """
        Forward and backward pass through the module when there are 
        point features
        """

        npoint = 4
        radius_list = [1.0, 2.0, 3.0]
        nsample_list = [2, 4, 8]
        in_channel = 10
        mlp_list = [[32, 32, 64], [64, 64, 128], [64, 96, 128]]

        x = PointNetSetAbstractionMsg(npoint=npoint, 
                                        radius_list=radius_list,
                                        nsample_list=nsample_list, 
                                        in_channel=in_channel, 
                                        mlp_list=mlp_list)

        n_batch = 12
        max_n_points = 20
        input_xyz = torch.full((n_batch, max_n_points, 3), torch.nan)
        input_xyz[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 3)), std=1.)

        input_xyz.requires_grad = True

        input_features = torch.full((n_batch, max_n_points, 10), torch.nan)
        input_features[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 10)), std=1.)
        input_features.requires_grad = True

        out_xyz, out_features = x(input_xyz, input_features)

        assert torch.logical_not(torch.any(torch.isnan(out_xyz))), out_xyz
        assert torch.logical_not(torch.any(torch.isnan(out_features))), out_features

        assert list(out_xyz.shape) == [n_batch, npoint, 3]
        assert list(out_features.shape) == [n_batch, npoint, 320]

        loss = torch.sum(out_features)

        loss.backward()


class TestPointNetSetAbstraction:
    def test_0(self) -> None:
        """
        Tests forward and backward pass, using the settings as used inside 
        pointnet2_reg_msg
        """
        x = PointNetSetAbstraction(npoint=None, 
                                    radius=None, 
                                    nsample=None, 
                                    in_channel=10 + 3, 
                                    mlp=[256, 512, 1024], 
                                    group_all=True)
        n_batch = 12
        max_n_points = 20
        input_xyz = torch.full((n_batch, max_n_points, 3), torch.nan)
        input_xyz[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 3)), std=1.)

        input_xyz.requires_grad = True

        input_features = torch.full((n_batch, max_n_points, 10), torch.nan)
        input_features[:, :15] = torch.normal(mean=torch.zeros((n_batch, 15, 10)), std=1.)
        input_features.requires_grad = True

        out_xyz, out_features = x.forward(input_xyz, input_features)

        # Not testing for lack of NaNs because by this point inside
        # pointnet2_reg_msg, there should not be any NaNs

        # assert torch.logical_not(torch.any(torch.isnan(out_xyz))), out_xyz
        # assert torch.logical_not(torch.any(torch.isnan(out_features))), out_features

        assert list(out_xyz.shape) == [n_batch, 1, 3]
        assert list(out_features.shape) == [n_batch, 1, 1024]

        loss = torch.sum(out_features)

        loss.backward()