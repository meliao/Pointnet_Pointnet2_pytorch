# import unittest
import numpy as np
from data_utils.MoleculeDataSet import PointCloudMoleculeDataSet, CHARGES_LIST_QM7


class TestMoleculeDataSet:


    def test_charges_to_one_hot(self) -> None:
    
        coords = np.full((2, 5, 3), np.nan)
        coords[0, :4] = np.random.normal(size=(4, 3))
        coords[1, :3] = np.random.normal(size=(3, 3))
        charges = np.array([[1, 1, 6, 6, 0],
                            [7, 7, 1, 0, 0]])
        
        energies = np.random.normal(size=2)
        
        x = PointCloudMoleculeDataSet(coords, charges, energies)
        
        x.align_coords_cart()
        x.charges_to_one_hot_QM7()
        
        expected_one_hot_encoding = np.full((charges.shape[0], charges.shape[1], len(CHARGES_LIST_QM7)), np.nan)
        expected_one_hot_encoding[0, :4] = np.array([[1, 0, 0, 0, 0],
                                                    [1, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0],
                                                    [0, 1, 0, 0, 0]])
        expected_one_hot_encoding[1, :3] = np.array([[0, 0, 1, 0, 0],
                                                    [0, 0, 1, 0, 0],
                                                    [1, 0, 0, 0, 0]])
        assert np.allclose(x.one_hot_point_features, 
                        expected_one_hot_encoding, equal_nan=True)
    

# if __name__ == "__main__":
#     unittest.main()