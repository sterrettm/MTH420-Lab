# python_intro.py
"""
Matthew Sterrett
sterretm@oregonstate.edu
"""

import numpy as np

def sphere_volume(radius):
    """Takes radius as input and outputs the volume of a sphere with that radius"""
    return 4/3 * 3.14159 * radius ** 3

def prob4():
    A = np.array([[3,-1,4], [1,5,-9]])
    B = np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])

    return np.dot(A,B)

def tax_liability(income):
    """ Calcultes tax liability using a simplified tax bracket system """
    bracket1 = min(9875, income)
    bracket2 = min(40125-9875, max(income-9875, 0))
    bracket3 = max(income-40125, 0)

    return bracket1 * 0.1 + bracket2 * 0.12 + bracket3 * 0.22

def prob6a():
    A = [i for i in range(1,8)]
    B = [5 for i in range(1,8)]

    sum =  [0 for i in range(1,8)]
    prod = [0 for i in range(1,8)]
    five = [0 for i in range(1,8)]

    for i in range(0,7):
        sum[i] = A[i] + B[i]
        prod[i] = A[i] * B[i]
        five[i] = 5 * A[i]
    
    return sum, prod, five

def prob6b():
    A = np.array([i for i in range(1,8)])
    B = np.array([5 for i in range(1,8)])

    sum = A + B
    prod = A * B
    five = 5 * A

    return sum, prod, five

if __name__ == "__main__":
    print("Hello, world!")
    print("Volume:",sphere_volume(10))
    print("prob4:",prob4())
    print("Tax:",tax_liability(63000))

    print(prob6a(), prob6b())
