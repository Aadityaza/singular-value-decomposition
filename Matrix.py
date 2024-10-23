import numpy as np

def SVD(mat,display_details=False):
        """
        Perform Singular Value Decomposition (SVD) on the input matrix.
        
        Parameters:
        matrix (numpy.ndarray): Input matrix to decompose
        
        Returns:
        tuple: (U, S, V)
            U (numpy.ndarray): Left singular vectors
            S (numpy.ndarray): Singular values
            V (numpy.ndarray): Right singular vectors (transposed)
        """
        # SVD using numpy
        U, s, Vt = np.linalg.svd(mat)
        
        # Convert singular values to diagonal matrix
        S = np.zeros(mat.shape)
        np.fill_diagonal(S, s)
        if display_details:
            print("The original matrix A is decomposed into A = U * S * V^T")
            print("U: Left singular vectors (orthogonal matrix)")
            print("S: Diagonal matrix of singular values (in descending order)")
            print("V^T: Transpose of right singular vectors (orthogonal matrix)")
            
            print("Singular Value Decomposition Results:")
            print("-------------------------------------")
            print(f"Original matrix shape: {mat.shape}")
            print(f"U matrix shape: {U.shape}")
            print(f"S matrix shape: {S.shape}")
            print(f"V^T matrix shape: {Vt.shape}")
            
        return U, S, Vt

