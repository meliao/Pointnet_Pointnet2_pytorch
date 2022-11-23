from typing import Tuple
import numpy as np
from scipy import linalg, io
import torch
from torch.utils.data import Dataset



CHARGES_LIST_QM9 = [1, 6, 7, 8, 9]
CHARGES_LIST_QM7 = [1, 6, 7, 8, 16]

class PointCloudMoleculeDataSet(Dataset):
    def __init__(self, coords_cart: np.ndarray, charges: np.ndarray, energies: np.ndarray) -> None:
        """
        coords_cart has shape (n_samples, max_n_atoms, 3)
        charges has shape (n_samples, max_n_atoms)
        energies has shape (n_samples,)
        """
        # print(charges.shape)
        self._coords_cart = coords_cart
        self._charges = charges
        self.n_samples, self.max_n_atoms, _ = self._coords_cart.shape
        self.n_atoms = np.sum(charges != 0, axis=1)
        # print(self.n_atoms.shape)
        self.energies = energies
        self.coords_centered = None
        self.one_hot_point_features = None
        self.U_matrices = None

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returms a tuple of torch.Tensors.

        points_and_features has shape [self.max_n_atoms, 3 + 5]
        points_and_features[:, :3] is xyz coordinates of the atoms.
        points_and_features[:, 3:] is one-hot element-type encoding.

        U_matrix is the 3x3 left singular values of points_and_features[:, 3:].T

        energies_out is the label
        """
        coords_out = self.coords_centered[index]
        charge_features_out = self.one_hot_point_features[index]
        points_and_features_out = torch.cat([coords_out, charge_features_out], dim=-1)
        U_matrix = self.U_matrices[index]
        energies_out = self.energies[index]
        return (points_and_features_out, U_matrix, energies_out)

    def align_coords_cart(self) -> None:
        out = np.full_like(self._coords_cart, np.nan)
        out_U_mats = np.zeros((self.n_samples, 3, 3))

        for i in range(self.n_samples):
            n_atoms_i = self.n_atoms[i]
            coords_i = self._coords_cart[i, :n_atoms_i]
            coords_i_centered = coords_i - np.mean(coords_i, axis=0)
            U, _, _ = linalg.svd(coords_i_centered.transpose(), full_matrices=False)
            # coords_aligned = np.matmul(U.transpose(), coords_i.transpose()).transpose()
            out[i, :n_atoms_i] = coords_i_centered
            out_U_mats[i] = U

        self.coords_centered = torch.Tensor(out)
        self.U_matrices = torch.Tensor(out_U_mats)
        

    def charges_to_one_hot_QM7(self) -> None:
        out = np.full((self.n_samples, 
                        self._charges.shape[1], 
                        len(CHARGES_LIST_QM7)), np.nan)
        charges_lst = CHARGES_LIST_QM7
        charges_lst_arr = np.array(CHARGES_LIST_QM7)
        for i in range(self.n_samples):
            n_atoms_i = self.n_atoms[i]
            out[i, :n_atoms_i] = np.zeros_like(out[i, :n_atoms_i])
            charges_i = self._charges[i, :n_atoms_i]
            col_idxes = [charges_lst.index(x) for x in charges_i]
            for atom_idx, charge_col_idx in enumerate(col_idxes):
                out[i, atom_idx, charge_col_idx] = 1.
        self.one_hot_point_features = torch.Tensor(out)

    def charges_to_one_hot_QM9(self) -> None:
        pass


def load_and_align_QM7(fp: str,
                        n_train: int,
                        n_test: int, 
                        validation_set_fraction: float) -> Tuple[PointCloudMoleculeDataSet, 
                                                                    PointCloudMoleculeDataSet, 
                                                                    PointCloudMoleculeDataSet]:
    """Loads the QM7 dataset and prepares 3 PointCloudMoleculeDataSet objects

    Args:
        fp (str): filepath of QM7
        n_train (int): Number of samples to be split among the train and val datasets
        n_test (int): Number of samples in the test dataset
        validation_set_fraction (float): Fraction of data from n_train to put in the val dataset

    Returns:
        Tuple[PointCloudMoleculeDataSet, PointCloudMoleculeDataSet, PointCloudMoleculeDataSet]: train, val, test datasets
    """
    q = io.loadmat(fp)

    energies_all = q['T'].flatten()
    n_validation = int(np.floor(n_train * validation_set_fraction))
    n_train_eff = n_train - n_validation
    assert n_train + n_test <= energies_all.shape[0]

    # logging.info("Releasing train, validation, test datasets of size %i, %i, %i", n_train_eff, n_validation, n_test)

    perm = np.random.permutation(energies_all.shape[0])

    train_idxes = perm[:n_train_eff]
    val_idxes = perm[n_train_eff: n_train]
    test_idxes = perm[n_train: n_train + n_test]

    assert np.intersect1d(train_idxes, val_idxes).shape[0] == 0
    assert np.intersect1d(train_idxes, test_idxes).shape[0] == 0
    assert np.intersect1d(test_idxes, val_idxes).shape[0] == 0

    train_dset = PointCloudMoleculeDataSet(q['R'][train_idxes], q['Z'][train_idxes], energies_all[train_idxes])
    val_dset = PointCloudMoleculeDataSet(q['R'][val_idxes], q['Z'][val_idxes], energies_all[val_idxes])
    test_dset = PointCloudMoleculeDataSet(q['R'][test_idxes], q['Z'][test_idxes], energies_all[test_idxes])


    train_dset.align_coords_cart()
    val_dset.align_coords_cart()    
    test_dset.align_coords_cart() 

    train_dset.charges_to_one_hot_QM7()   
    val_dset.charges_to_one_hot_QM7()   
    test_dset.charges_to_one_hot_QM7()   
    
    return train_dset, val_dset, test_dset