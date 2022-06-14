# newtons_method.py
"""Volume 1: Newton's Method.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    
    x = x0
    i = 0
    while (i < maxiter):
        if (np.isscalar(x)):
            x -= alpha * f(x) / Df(x)
            if (abs(f(x)) <= tol):
                break
            
        else:
            x = x - alpha * np.linalg.inv(Df(x)) @ f(x)
            if (np.linalg.norm(x) <= tol):
                break
        i += 1
    
    return x, False, maxiter


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    
    f =  lambda r : (P1 * ((1 + r) ** N1 - 1) - P2 * (1 - (1+r) ** (-N2)))
    Df = lambda r : (P1 * N1 * (1 + r) ** (N1 - 1) - P2 * N2 * (1 + r) ** (-N2 - 1)) 
    
    r = newton(f,0.1,Df)[0]
    
    return r


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    
    index = 0
    results = np.zeros((100))
    X = np.arange(0.1,1.1,0.01)
    
    for alpha in X:
        x = x0
        
        iters = 0
        while (iters < maxiter):
            x -= alpha * f(x) / Df(x)
            if (abs(f(x)) <= tol):
                break
            iters += 1
        
        results[index] = iters
        index += 1
    
    plt.plot(X, results)
    plt.show()
    return alpha

# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    
    gamma = 5
    delta = 1
    
    tolerance = 1e-3
    maxiters = 15
    alpha = 0.5
    
    f  = lambda x : np.array([(gamma - 1) * x[0] * x[1] - x[0], delta + (delta - 1) * x[1] - x[0] * x[1] - x[1] * x[1]])
    Df = lambda x : np.array([[(gamma - 1) * x[1] - 1, (gamma  - 1) * x[0]], [-x[1], -x[0] + delta - 1 - 2*x[1]]])
    
    while True:
        x0 = np.random.uniform(low=-0.25, high=0)
        y0 = np.random.uniform(low=0, high=0.25)
        X0 = np.array([x0, y0])
        
        X1 = newton(f, X0, Df, tolerance, maxiters, 1)
        X2 = newton(f, X0, Df, tolerance, maxiters, 0.55)
        
        if (np.allclose(X1, np.array([0,1])) or np.allclose(X1, np.array([0,-1]))) and np.allclose(X2, np.array([3.75, 0.25])):
            return X0
   

# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    
    x_real = np.linspace(domain[0], domain[1], res) # Real parts.
    x_imag = np.linspace(domain[2], domain[3], res) # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag 
    
    X = X_0
    
    for iter in range(0, iters):
        X = X - f(X) / Df(X)
    
    Y = np.zeros_like(X)
    Y = np.argmin(np.abs(np.repeat(X[:,:,np.newaxis], len(zeros), axis=2) - zeros), axis=2)
    
    plt.pcolormesh(np.real(X_0), np.imag(X_0), Y, cmap="brg")
    plt.show()