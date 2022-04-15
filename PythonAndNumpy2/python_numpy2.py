# python_intro.py
"""Python Essentials: Introduction to Python.
<Name>
<Class>
<Date>
"""

import numpy as np

#Problem 1
def isolate(a, b, c, d, e):
    print(a,b,c,sep = '     ', end=' ')
    print(d,e)

#Problem 2
def first_half(string):
    return string[0:len(string) // 2]


def backward(first_string):
    return first_string[::-1]

#Problem 3
def list_ops():
    animals = ["bear", "ant", "cat", "dog"]
    animals.append("eagle")
    animals[2] = "fox"
    animals.pop(1)
    animals[animals.index("eagle")] = "hawk"
    animals[-1] = animals[-1] + "hunter"
    return animals

#Problem 4
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """

    return sum([(1/(i+1) * (-1) ** i) for i in range(0,n)])

    raise NotImplementedError("Problem 4 Incomplete")



def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """

    return A.copy() + (-1 * A) * (A < 0)

def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    
    A = np.array([[0,2,4],[1,3,5]])
    B = np.array([[3,0,0],[3,3,0],[3,3,3]])
    C = np.array([[-2,0,0], [0, -2, 0], [0, 0, -2]])

    row1 = np.hstack((np.zeros((3,3)), np.transpose(A), np.identity(3)))
    row2 = np.hstack((A, np.zeros((2,2)), np.zeros((2,3))))
    row3 = np.hstack((B, np.zeros((3,2)), C))

    result = np.vstack((row1, row2, row3))

    return result

def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    raise NotImplementedError("Problem 7 Incomplete")


def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    raise NotImplementedError("Problem 8 Incomplete")


