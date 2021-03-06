# drazin.py
"""Volume 1: The Drazin Inverse.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import csgraph as csgraph

# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """

    Commute = np.allclose(A @ Ad, Ad @ A)
    Power = np.allclose(np.linalg.matrix_power(A, k + 1) @ Ad, np.linalg.matrix_power(A,k))
    Remove = np.allclose(Ad @ A @ Ad, Ad)

    return Commute and Power and Remove


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """

    T1, Q1, k1 = la.schur(A, sort = lambda x: (abs(x) > tol))
    T2, Q2, k2 = la.schur(A, sort = lambda x: (abs(x) <= tol))

    n = A.shape[0]

    U = np.hstack((Q1[:, :k1], Q2[:, :n-k1]))

    Uinv = la.inv(U)

    V = Uinv @ A @ U
    Z = np.zeros_like(A, dtype=float)

    if (k1 != 0):
        Minv = la.inv(V[:k1, :k1])
        Z[:k1, :k1] = Minv

    return U @ Z @ Uinv

def laplacian(A):
    D = A.sum(axis = 1)
    L =  np.diag(D) - A
    assert np.allclose(L, csgraph.laplacian(A))
    return L

# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """

    assert np.allclose(A,A.T), "Adjacency matrix is not symmetric"

    n = A.shape[0]

    I = np.identity(n)

    L = laplacian(A)
    assert np.allclose(L,L.T), "Laplacian is not symmetric?"

    R = np.zeros_like(A, dtype=float)

    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                Lj = L.copy()
                Lj[j] = I[j]
                LjD = drazin_inverse(Lj)
                assert is_drazin(Lj, LjD, index(Lj)), "Drazin inverse incorrect?"
                R[i,j] = LjD[i,i]

    assert np.allclose(R, R.T), "R is not symmetric"
    return R


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        raise NotImplementedError("Problem 4 Incomplete")


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")
