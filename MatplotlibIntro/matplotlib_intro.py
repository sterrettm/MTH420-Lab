# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    values = np.random.normal(size=(n,n))
    means = np.mean(values, axis=1)
    var = np.var(means)
    return var

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    X = np.arange(100, 1001, 100)
    results = []

    for x in X:
        res = var_of_means(x)
        results.append(res)
    
    plt.plot(X, results)
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    X = np.arange(-2*np.pi, 2*np.pi, 0.01)
    S = np.sin(X)
    C = np.cos(X)
    A = np.arctan(X)

    plt.plot(X, S)
    plt.plot(X, C)
    plt.plot(X, A)
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    X1 = np.arange(-2, 1, 0.01)
    X2 = np.arange(1.01, 6.01, 0.01)
    Y1 = 1/(X1 - 1)
    Y2 = 1/(X2 - 1)

    plt.plot(X1, Y1, 'm--', linewidth = 4)
    plt.plot(X2, Y2, 'm--', linewidth = 4)

    plt.xlim(-2,6)
    plt.ylim(-6,6)

    plt.show()


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    X = np.arange(0,2*np.pi+0.01, 0.01)
    
    plt.axis([0,2*np.pi,-2,2])

    ax1.plot(X, np.sin(X), 'g-')
    ax1.set_title("sin(x)")

    ax2.plot(X, np.sin(2*X), 'r--')
    ax2.set_title("sin(2x)")

    ax3.plot(X, 2*np.sin(X), 'b--')
    ax3.set_title("2 sin(x)")

    ax4.plot(X, 2*np.sin(2*X), 'm:')
    ax4.set_title("2 sin(2x)")

    plt.suptitle("Variants of sin(x)")

    plt.show()

# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """

    x = np.linspace(-2*np.pi, 2*np.pi, 200)
    y = x.copy()
    X, Y = np.meshgrid(x,y)
    Z = (np.sin(X) * np.sin(Y)) / (x * y)

    print(X)
    print(Y)

    #fig, (ax1, ax2) = plt.subplots(2)
    plt.axis([-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])

    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="plasma")
    plt.colorbar()
    
    plt.subplot(122)
    plt.contourf(X,Y,Z, 10, cmap='magma')
    plt.colorbar()

    plt.show()

    raise NotImplementedError("Problem 6 Incomplete")
