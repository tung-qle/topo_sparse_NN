from z3 import *
import numpy as np
import time

def closedness(support2, support1):
    """
    Input: Receive support of 2 weight matrices
    support2: binary mask (sparsity pattern) for the second matrix
    support1: binary mask (sparsity pattern) for the first matrix
    Output:
    True -> L_I is closed
    False -> L_I is not closed
    None -> Does not terminate
    """
    # Set support1 equal to the support of the transpose of X1
    support1 = support1.T

    # Size of the factors
    m = support2.shape[0]
    r = support2.shape[1]
    n = support1.shape[0]

    # Define variables
    A = Reals(['a.' + str(i) + '.' + str(j) for i in range(m) for j in range(n)])
    X1 = Reals(['x1.' + str(i) + '.' + str(j) for i in range(m) for j in range(r)])
    Y1 = Reals(['y1.' + str(i) + '.' + str(j) for i in range(n) for j in range(r)])
    eps = Reals('eps')[0]
    X2 = Reals(['x2.' + str(i) + '.' + str(j) for i in range(m) for j in range(r)])
    Y2 = Reals(['y2.' + str(i) + '.' + str(j) for i in range(n) for j in range(r)])

    # Support constraints
    constraint1 = And([X1[i * r + j] == 0 for i in range(m) for j in range(r) if support2[i][j] == 0] 
                        + [Y1[i * r + j] == 0 for i in range(n) for j in range(r) if support1[i][j] == 0])    
    constraint2 = And([X2[i * r + j] == 0 for i in range(m) for j in range(r) if support2[i][j] == 0] 
                        + [Y2[i * r + j] == 0 for i in range(n) for j in range(r) if support1[i][j] == 0])

    XY1 = [Sum([Product(X1[i * r + k], Y1[j * r + k]) for k in range(r)]) for i in range(m) for j in range(n)]
    XY2 = [Sum([Product(X2[i * r + k], Y2[j * r + k]) for k in range(r)]) for i in range(m) for j in range(n)]
    equal1 = Or([XY1[i * n + j] != A[i * n + j] for j in range(n) for i in range(m)])
    limit2 = And([XY2[i * n + j] - A[i * n + j] < eps for j in range(n) for i in range(m)]
                    + [A[i * n + j] - XY2[i * n + j] < eps for j in range(n) for i in range(m)])

    # Run the quantifier elimination algorithm
    s = SolverFor("NRA")
    s.add(ForAll(X1 + Y1, Implies(constraint1, equal1)))
    s.add(ForAll(eps, Exists(X2 + Y2, And(constraint2, Or(eps <= 0, limit2)))))
    begin = time.time()
    result = s.check()
    end = time.time()
    print("Running time:", end - begin)
    if result == unsat:
        return True
    elif result == unknown:
        return None
    return False
    
if __name__ == "__main__":
    support1 = np.array([[1, 1], [1, 1]])
    support2 = np.array([[1, 1], [1, 1]]) 
    print(closedness(support1, support2))