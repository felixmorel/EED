import numpy as np
from gurobipy import *
# Author: Loïc Van Hoorebeeck
# Date: 2020-04-01
def get_approx_planes(P0, B, D, p_min, p_max, Relaxed=False):
    # Return the approximation plane  
    # with the constraint pt' B pt + b' pt + c = 0
    #
    # P0 should be feasible
   
    # Gurobipy is imported via *
    mod = Model()

    eps_p = pow(10, -8)

    # up to GN :)
    n_gen = len(B)
    N=n_gen
    
    #Linear terms corresponds to the sum of the production
    b=-np.ones(n_gen)  

    # Problem is non convex because of the quadratic equality constraint-> solve the convex relaxation
    mod.setParam( 'OutputFlag', False )
    mod.Params.timeLimit = 100
    p = mod.addVars(range(n_gen), lb=p_min, ub=p_max, name='p')

    if (Relaxed==False):
        mod.Params.NonConvex = 2

        mod.addConstr(0 ==
            quicksum(
             B[i, j]*p[i]*p[j] for i in range(n_gen) for j in range(n_gen))
            + quicksum( b[i]*p[i] for i in range(n_gen))
            + D, name='Demand')
    else:
        mod.addConstr(0 >=
            quicksum(
             B[i, j]*p[i]*p[j] for i in range(n_gen) for j in range(n_gen))
            + quicksum( b[i]*p[i] for i in range(n_gen))
            + D, name='Demand')


    n = -B @ P0 - b*0.5
    n_normed = n / np.linalg.norm(n)
    mod.setObjective(quicksum(n_normed[g]*(p[g]-P0[g]) for g in range(n_gen)), GRB.MAXIMIZE)
    mod.Params.mipgap = 0.00001
    mod.update()

    mod.optimize()
    x = mod.getAttr('x', p)
    x = np.array(list(x.values()))
    scalar_product = mod.getAttr('ObjVal')

    k_lower = -n @ P0
    k_upper = k_lower - n.T @ (n_normed * scalar_product)
    # Could be simplified since n.T @ n_normed = 1
    assert abs(np.dot(n, x) + k_upper) < eps_p

    return (n, k_lower, k_upper)
