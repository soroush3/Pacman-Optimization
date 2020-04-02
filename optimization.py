# optimization.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import itertools
import math

import pacmanPlot
import graphicsUtils
import util

# You may add any helper functions you would like here:
# def somethingUseful():
#     return True

best = ((), math.inf)

def findIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b)
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.
    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).
        If none of the constraint boundaries intersect with each other, return [].

    An intersection point is an N-dimensional point that satisfies the
    strict equality of N of the input constraints.
    This method must return the intersection points for all possible
    combinations of N constraints.

    """
    "*** YOUR CODE HERE ***"
    A = []
    b = []

    for item in constraints:
        A.append(list(item[0]))
        b.append(item[1])

    if len(b) == 0:
        return []

    A = np.array(A)
    b = np.array(b)

    # A is a square matrix
    if len(A) == len(A[0]):
        sol = np.linalg.solve(A, b)
        sol = [tuple(x) for x in sol]
    # not square matrix, must find all combinations that formulate
    # square matrix
    else:
        sol = []
        var = len(A[0])
        indices = [i for i in range(len(A))]
        # get combinations
        combinations = itertools.combinations(indices, var)
        for pair in combinations:
            tA = A[pair, :]
            tB = b[list(pair)]
            # make sure the tA matrix is not singular
            try:
                tempSol = (np.linalg.solve(tA, tB))
                sol.append(tuple(tempSol))
            except:
                pass

    return sol

def findFeasibleIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    feasible intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).

        If none of the lines intersect with each other, return [].
        If none of the intersections are feasible, return [].

    You will want to take advantage of your findIntersections function.

    """
    "*** YOUR CODE HERE ***"
    # range an intersection can be within to still be considered feasible
    wiggle = (1 * (10**-12))

    # find every intersection from the given constraints
    intersections = findIntersections(constraints)

    # if there aren't any intersections at all, return []
    if not intersections:
        return []

    feasible_intersections = []
    # go through intersections and determine if they satisfy all constraints    
    for intersection in intersections:
        toAdd = False
        for con in constraints:
            sum = np.dot(intersection, con[0])
            if (sum <= con[1] + wiggle) or (sum <= con[1] - wiggle):
                toAdd = True
            else:
                toAdd = False
                break
        if toAdd:
            feasible_intersections.append(intersection)

    return feasible_intersections

def solveLP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    find a feasible point that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your findFeasibleIntersections function.

    """
    "*** YOUR CODE HERE ***"
    # feasible intersections given the constraints
    feasible_intersections = findFeasibleIntersections(constraints)

    minim = math.inf
    point = None
    # get dot product of each (intersection, cost) and determine which is
    # optimal
    for inter in feasible_intersections:
        c = np.dot(inter, cost)
        if c < minim:
            minim = c
            point = inter
    
    if not point:
        return None
    
    return (point, minim)

def wordProblemLP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveLP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
            ((sunscreen_amount, tantrum_amount), maximal_utility)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"

    """
    20 ounces <= sunscreen
    15.5 ounces <= tantrum

    total ounces <= 250, sunscreen + tantrum <= 250

    weight of bag <= 50 pounds

    ounce of sunscreen = 0.5 pounds
    ounce of tantrum = .25 pounds

    ounce of sunscreen gives utility of 7
    ounce of tantrum gives utility of 4
    maximize utility
    """
    #(sunscreen, tantrum), max
    constraints = [((-1, 0), -20),((0, -1), -15.5),((2.5, 2.5), 100),\
                                            ((0.5, .25), 50)]
    # since the SolveLP is looking to minimize, multiply objective by -1
    # to allow SolveLP to maximize
    utility = (-7, -4)
    sol = list(solveLP(constraints, utility))
    sol[1] *= -1
    return tuple(sol)

def getConstraints(constraints, ind, xVars):
    lowerConstraint = constraints[:]
    upperConstraint = constraints[:]
    low, up = [0] * len(xVars), [0] * len(xVars)
    low2, up2 = [0] * len(xVars), [0] * len(xVars)
    low[ind], up[ind] = 1, -1
    low2[ind], up2[ind] = -1, 1
    low, up = tuple(low), tuple(up)
    low2, up2 = tuple(low2), tuple(up2)
    low, up = (low, math.floor(xVars[ind])), (up, -math.ceil(xVars[ind]))
    low2, up2 = (low2, -math.floor(xVars[ind])), (up2, math.ceil(xVars[ind]))
    lowerConstraint.append(low)
    upperConstraint.append(up)
    lowerConstraint.append(low2)
    upperConstraint.append(up2)
    return lowerConstraint, upperConstraint

def branchAndBoundIP(constraints, cost):
    """
    Branch and bounding with use of recursion to determine a feasible Integer
    solution from the given constraints and cost
    """

    lpSol = solveLP(constraints, cost)
    # base case, no feasible LP solution
    if lpSol == None:
        return
    # found new IP best, done with this part of branch
    allInt = allIntegers(lpSol)
    global best
    if allInt and lpSol[1] < best[1]:
        best = lpSol
        return
    # branch is fathomed as IP solution found but not better than current best
    if allInt:
        return

    for i in range(len(lpSol[0])):
        if not withinIntegerRange(lpSol[0][i]):
            lowerConstraint, upperConstraint = getConstraints(constraints, i, lpSol[0])
            branchAndBoundIP(lowerConstraint, cost)
            branchAndBoundIP(upperConstraint, cost)
    
    return

def solveIP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    use the branch and bound algorithm to find a feasible point with
    interger values that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your solveLP function.

    """
    "*** YOUR CODE HERE ***"
    global best
    best = ((), math.inf)
    # grab relaxed version, LP
    lpSol = solveLP(constraints, cost)
    # if no solution, done, return None
    if lpSol == None:
        return None
    # if the solution satisfies
    if allIntegers(lpSol):
        return lpSol
    # if not, branch and bound
    branchAndBoundIP(constraints, cost)
    return best


def wordProblemIP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveIP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
        ((f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS), minimal_cost)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    """
    Order of variables
    Dunkin_Gates, Dunkin_Sorrells, Eatunique_Gates, Eatunique_Sorrells, ...
    """
    constraints = [((1.2, 0, 0, 0, 0, 0), 30), ((0, 1.2, 0, 0, 0, 0), 30),
            ((0, 0, 1.3, 0, 0, 0), 30), ((0, 0, 0, 1.3, 0, 0), 30),
            ((0, 0, 0, 0, 1.1, 0), 30), ((0, 0, 0, 0, 0, 1.1), 30),
            ((-1, 0, 0, 0, 0, 0), 0), ((0, -1, 0, 0, 0, 0), 0),
            ((0, 0, -1, 0, 0, 0), 0), ((0, 0, 0, -1, 0, 0), 0),
            ((0, 0, 0, 0, -1, 0), 0), ((0, 0, 0, 0, 0, -1), 0),
                ((-1, 0, -1, 0, -1, 0), -15),((0, -1, 0, -1, 0, -1), -30)]
    cost = (12, 20, 4, 5, 2, 1)

    sol = solveIP(constraints, cost)
    return sol

def foodDistribution(truck_limit, W, C, T):
    """
    Given M food providers and N communities, return the integer
    number of units that each provider should send to each community
    to satisfy the constraints and minimize transportation cost.

    Input:
        truck_limit: Scalar value representing the weight limit for each truck
        W: A tuple of M values representing the weight of food per unit for each 
            provider, (w1, w2, ..., wM)
        C: A tuple of N values representing the minimal amount of food units each
            community needs, (c1, c2, ..., cN)
        T: A list of M tuples, where each tuple has N values, representing the 
            transportation cost to move each unit of food from provider m to
            community n:
            [ (t1,1, t1,2, ..., t1,n, ..., t1N),
              (t2,1, t2,2, ..., t2,n, ..., t2N),
              ...
              (tm,1, tm,2, ..., tm,n, ..., tmN),
              ...
              (tM,1, tM,2, ..., tM,n, ..., tMN) ]

    Output: A length-2 tuple of the optimal food amounts and the corresponding
            objective value at that point: (optimial_food, minimal_cost)
            The optimal food amounts should be a single (M*N)-dimensional tuple
            ordered as follows:
            (f1,1, f1,2, ..., f1,n, ..., f1N,
             f2,1, f2,2, ..., f2,n, ..., f2N,
             ...
             fm,1, fm,2, ..., fm,n, ..., fmN,
             ...
             fM,1, fM,2, ..., fM,n, ..., fMN)

            Return None if there is no feasible solution.
            You may assume that if a solution exists, it will be bounded,
            i.e. not infinity.

    You can take advantage of your solveIP function.

    """
    M = len(W)
    N = len(C)
    "*** YOUR CODE HERE ***"
    constraints = []
    # adding constraints for x(i) >= 0
    for i in range(M * N):
        temp = [0] * (M*N)
        temp[i] = -1
        temp = tuple(temp)
        constraints.append((temp,0))
    # adding weight limit constraints
    for i in range(M):
        providerWeight = W[i]
        for j in range(N):
            temp = [0] * (M*N)
            temp[i*N + j] = providerWeight
            temp = tuple(temp)
            constraints.append((temp, truck_limit))
    # adding minimal amount of food each community needs constraints
    for i in range(N):
        temp = [0] * (M*N)
        j = i
        while j < len(temp):
            temp[j] = -1
            j += N
        constraints.append((temp, -C[i]))
        
    # constructing cost vector
    cost = []
    for item in T:
        cost.extend(list(item))
    cost = tuple(cost)
    
    return solveIP(constraints, cost)
    

def allIntegers(solution):
    """
    Determines if the given solution is a complete integer solution
    """
    for n in solution[0]:
        if not withinIntegerRange(n):
            return False
    return True

def withinIntegerRange(n:int):
    """
    Determines if the given number is within 1e-12 of the closest integer
    """

    wiggle = (1 * (10**-12))

    if (math.ceil(n) - wiggle) <= n <= (math.ceil(n) + wiggle):
        return True

    if (math.floor(n) - wiggle) <= n <= (math.floor(n) + wiggle):
        return True

    return False

