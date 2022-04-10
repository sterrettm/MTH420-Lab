# standard_library.py
"""Python Essentials: The Standard Library.
<Name>
<Class>
<Date>
"""

import calculator
import itertools
import box
import random
import sys
import time

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), sum(L) / len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    print("Numbers are immutable")
    print("Strings are immutable")
    print("Lists are mutable")
    print("Tuples are immutable")
    print("Sets are mutable")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    aSq = calculator.product(a, a)
    bSq = calculator.product(b, b)
    sqSum = calculator.sum(aSq, bSq)
    return calculator.sqrt(sqSum)


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    results = []
    
    for i in range(len(A) + 1):
        tuples = list(itertools.combinations(A, i))
        for elem in tuples:
            results.append(set(elem))
    
    return results


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    
    numbers = list(range(1,10))
    startTime = time.time() 
    roll = -1
    
    timeRemaining = timelimit
    
    while True:
    
        if (sum(numbers) > 6):
            roll = random.randrange(1,7) + random.randrange(1,7)
        else:
            roll = random.randrange(1,7)
    
        print("\nNumbers left:",numbers,"\nRoll:",roll)
        
        if box.isvalid(roll, numbers):
            while True:
                timeRemaining = timelimit - (time.time() - startTime)
                if timeRemaining < 0:
                    break
            
                print("Seconds left:",round(timeRemaining, 2))
                
                inputData = input("Numbers to eliminate: ")
                
                inputNumbers = box.parse_input(inputData, numbers)
                
                if inputNumbers == [] or sum(inputNumbers) != roll:
                    print("Invalid input\n")
                else:
                    # This is a correct input
                    for num in inputNumbers:
                        numbers.remove(num)
                    break
            
            # Check if we have ran out of time
            if timeRemaining < 0:
                print("Game over!\n")
                break
        else:
            print("Game over!\n")
            break
    
    score = sum(numbers)
    
    timePlayed = time.time() - startTime
    
    print("Score for player "+player+":",score,"points\nTime played:",round(timePlayed, 2),"seconds")
    if (score == 0):
        print("Congratulations!! You shut the box!")
    else:
        print("Better luck next time >:)")
    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1], int(sys.argv[2]))