import torch
import numpy as np
import pytest

from models.pointnet2_utils import (farthest_point_sample,)

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
        
        out = farthest_point_sample(in_tensor, 2)

        assert out[0, 0] != 6
        assert out[0, 1] != 6
        assert out[1, 0] < 5
        assert out[1, 1] < 5