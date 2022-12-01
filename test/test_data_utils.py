# import unittest
import os
import numpy as np
from data_utils.MoleculeDataSet import PointCloudMoleculeDataSet, CHARGES_LIST_QM7, load_and_align_QM7


home_dir = os.path.expanduser('~')
QM7_FP = os.path.join(home_dir, 'projects/invariant-random-features-code/data/qm7/qm7.mat')

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
    

class TestIntegrationQM7Data:
    def test_data_loading(self) -> None:
        # DATA_FP = '/Users/owen/projects/invariant-random-features-code/data/qm7/qm7.mat'
        n_train = 100
        n_test = 100
        train_dset, val_dset, test_dset = load_and_align_QM7(fp=QM7_FP,
                                                        n_train=n_train,
                                                        n_test=n_test,
                                                        validation_set_fraction=0.1)

        assert len(train_dset) == 90
        assert len(val_dset) == 10
        assert len(test_dset) == 100

        coords_and_features, U_matrix, energy = train_dset[23]

        assert list(coords_and_features.shape) == [train_dset.max_n_atoms, 5 + 3]
        assert list(U_matrix.shape) == [3, 3]




# if __name__ == "__main__":
#     unittest.main()