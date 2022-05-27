# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name>
<Class>
<Date>
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    x = cp.Variable(3, nonneg = True)
    A = np.array([[2,1,3]])
    B = np.array([[-1,-2,0],[0,-2,4],[2,10,3]])
    C = np.array([-3,-1,12])

    objective = cp.Minimize(A @ x)
    constraints = [B @ x >= C]
    problem = cp.Problem(objective, constraints)

    result = problem.solve()
    return x.value, result


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    print(A.shape)
    x = cp.Variable(A.shape[1])

    objective = cp.Minimize(cp.norm(x, 1))
    constraint = [A @ x == b]

    problem = cp.Problem(objective, constraint)
    result = problem.solve()
    return x.value, result


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    p = cp.Variable(6, nonneg=True)
    A = np.array([[4,7,6,8,8,9]])
    EQA   = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1],[1,0,1,0,1,0],[0,1,0,1,0,1]])
    EQb   = np.array([7,2,4,5,8])
    #INEQ = np.array([]) 

    objective = cp.Minimize(A @ p)
    constraint = [EQA @ p == EQb]

    problem = cp.Problem(objective, constraint)

    result = problem.solve()
    return p.value, result


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q = np.array([[3,2,1],[2,4,2],[1,2,3]])
    r = np.array([[3],[0],[1]])

    x = cp.Variable(3)
    objective = cp.Minimize(1/2 * cp.quad_form(x, Q) + r.T @ x)

    problem = cp.Problem(objective)

    result = problem.solve()
    return x.value, result


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    x = cp.Variable(A.shape[1], nonneg = True)

    objective = cp.Minimize(cp.norm(A @ x - b))
    constraint = [cp.sum(x) == 1]

    problem = cp.Problem(objective, constraint)

    result = problem.solve()
    return x.value, result

    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")
