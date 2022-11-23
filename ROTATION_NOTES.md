# Rotation Notes

## Dataset Object Implementations
The class `PointCloudMoleculeDataSet` has the following attributes:
 * `self._coords_cart` an array of size (n_samples, max_n_atoms, 3). Indexing the first column gives one a matrix of size (n_atoms, 3)
 * `self.U_matrices` has size (n_samples, 3, 3) and is constructed via the following:
    * For each point cloud in `self._coords_cart`, the data matrix of size (n_atoms, 3) is transposed to get a data matrix of size (3, n_atoms). The SVD of this transposed data matrix is computed. The U matrix (left singular vectors are in the columns of U) is saved in `self.U_matrices`

## Alignment

Given a point cloud $P \in \mathbb{R}^{3 \times N}$, take its singular value decomposition $P = U \Sigma V^T$. Then consider the aligned coordinates $U^T P$. These aligned coordinates are rotation-invariant. Consider some rotation matrix Q and the rotated matrix $QP$. 
If $u_i, v_i$ are matching left and right singular vectors of $P$, i.e. $P v_i = \sigma_i u_i$, then $Q u_i, v_i$ are left and right singular vectors of $QP$. 
That is, $QPv_i = \sigma_i Qu_i$. So then we can write the SVD of $QP$ in terms of $U, \Sigma, V$, the singular value decomposition of $P$.
$$ 
SVD(QP) = Q \ SVD(P) = Q U \Sigma V^T
$$
Now, we will align the rotated point cloud via $(Q U)^T QP = U^T Q^T Q P = U^T P$

## Sign Flips 

The singular vectors $u_i, v_i$ are determined up to a sign flip. That is, $-u_i, -v_i$ also satisfy the definition of singular vectors. So we need to flip the signs of the columns of $U$. 