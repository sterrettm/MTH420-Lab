# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    
    Q, R = linalg.qr(A, mode="economic")

    A2 = R
    b2 = np.matmul(np.transpose(Q), b)

    x = linalg.solve_triangular(A2, b2)
    
    return x

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    data = np.load("housing.npy")

    Xraw = data[:,0][np.newaxis].T
    Y = data[:,1][np.newaxis].T
    X = np.hstack((Xraw, np.ones_like(Xraw)))
    
    params = least_squares(X,Y) 

    Xmax = np.amax(Xraw)
    
    X_line = np.arange(0,Xmax + 0.1,0.5)
    Y_line = X_line * params[0] + params[1]

    plt.scatter(Xraw.flatten(), Y.flatten())
    plt.plot(X_line, Y_line)
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    
    data=np.load("housing.npy")
    Xraw = data[:,0][np.newaxis].T
    Y = data[:,1][np.newaxis].T

    # Build up A for
    A3 = np.ones_like(Xraw)

    for i in range(1, 4):
       A3 = np.hstack((np.power(Xraw, i), A3))

    A6 = np.ones_like(Xraw)
    for i in range(1, 7):
       A6 = np.hstack((np.power(Xraw, i), A6))

    A9 = np.ones_like(Xraw)
    for i in range(1, 10):
       A9 = np.hstack((np.power(Xraw, i), A9))

    A12 = np.ones_like(Xraw)
    for i in range(1, 13):
       A12 = np.hstack((np.power(Xraw, i), A12))

    X3 = least_squares(A3, Y)
    X6 = least_squares(A6, Y)
    X9 = least_squares(A9, Y)
    X12 = least_squares(A12, Y)

    Reg3 = A3 @ X3
    Reg6 = A6 @ X6
    Reg9 = A9 @ X9
    Reg12 = A12 @ X12

    plt.plot(Xraw.flatten(), Reg3)
    plt.plot(Xraw.flatten(), Reg6)
    plt.plot(Xraw.flatten(), Reg9)
    plt.plot(Xraw.flatten(), Reg12)

    # Compare polyfit results to our results
    poly3 = np.polyfit(Xraw.flatten(), Y.flatten(), 3)[np.newaxis].T
    print(poly3)
    print(X3)

    plt.scatter(Xraw.flatten(), Y.flatten())
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")
