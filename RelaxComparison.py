"""
This file analyzes and compares the relaxations 
Author: Felix Morel
Date : 10/06/2020

"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

import sys
import matplotlib.pyplot as plt
from Params import load,loss,zones
from Relaxed import LinUpperB,LinearRelaxation, RelaxErrors,RelaxTime
from LinApproxLVH import get_approx_planes
from Static_model import SimplePriceFun 
from NLSolverStat import Solve,SQP
import time

""" This function computes the total number of extreme points for 3 different demands"""
def testNEdges(N):
    for M in range(3): 
        (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR)=load(N)
        D=(M+1)*D/3
        B=loss(N)
        
        n=2**(N-1)
        elements=np.arange(0,N) 
        Limits=np.vstack((Pmin,Pmax))
        N_edges=0
        for k in range(N):   
            index=np.append(elements[:k],elements[k+1:])
            for i in range(n):
                binary=bin(i)
                coord=list(binary)[2:] #get rid of '0b'
                coord=np.array([int(i) for i in coord])
                seq=np.zeros(N-1,dtype=int)
                seq[N-1-len(coord):]=coord
                V_sup=np.zeros(N)
                V_sup[index]=Limits[seq,index]
                V_sup[k]=Pmax[k]
                D_sup=sum(V_sup)-V_sup@B@V_sup
                
                V_inf=np.zeros(N)
                V_inf[index]=Limits[seq,index]
                V_inf[k]=Pmin[k]
                D_inf=sum(V_inf)-V_inf@B@V_sup
                if (D_inf<=D and D_sup>=D):
                    N_edges=N_edges+1
                
        print(N_edges, 'edges for problem of size ', N , ' and ',D, 'demand' )

""" 
This function:
    Displays the POF's of 3 relaxations for 3 different demands
    Displays the combined errors of the 2 linear relaxations
"""
def ErrorRelaxation():
    plt.close("all")
    N=6
    Demand=np.array([400,800,1200])
    nProblems=4
    Error1=np.zeros([2,nProblems,3])
    Error2=np.zeros([2,nProblems,3])
    Error3=np.zeros([2,nProblems,3])
    for i in range(3):
        (E1,E2,E3)=RelaxErrors(N,Demand[i]);
        Error1[:,:,i]=100*E1.T
        Error2[:,:,i]=100*E2.T
        Error3[:,:,i]=100*E3[1:,:].T
    
    bar_width = 0.4
    opacity = 0.8
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
     
    fig, ax = plt.subplots()
    plt.subplot(131)
    Emiss1 = plt.bar(np.arange(3), Error1[0,0,:]/2, bar_width, alpha=opacity,
                     color='b',edgecolor='white', label='E: N points')
    bars1=Error1[0,0,:]/2
    Emiss2 = plt.bar(np.arange(3)+bar_width , Error1[0,2,:]/2, bar_width, alpha=opacity,
                     color=colors[1], edgecolor='white', label='E: 1 point')
    bars2=Error1[0,2,:]/2
         
    Cost1 = plt.bar(np.arange(3), Error1[1,0,:]/2, bar_width, alpha=opacity,
                    bottom=bars1, color=colors[0],edgecolor='white', label='F: N points')
    Cost2 = plt.bar(np.arange(3)+bar_width , Error1[1,2,:]/2, bar_width, alpha=opacity,
                    bottom=bars2, color='orange',edgecolor='white', label='F: 1 point')
    plt.xlabel('Demand')
    plt.ylabel('Mean error of the 2 objectives [%]')
    plt.title('axes')
    plt.xticks(np.arange(3) + 0.5*bar_width, ('D=400MW', 'D=800MW', 'D=1200MW'))
    plt.legend()
    plt.tight_layout()  
    plt.show()
    bottom, top = plt.ylim()  
    
    plt.subplot(132)
    Emiss1 = plt.bar(np.arange(3), Error2[0,0,:]/2, bar_width, alpha=opacity,
                     color='b',edgecolor='white', label='E: N points')
    bars1=Error2[0,0,:]/2
    Emiss2 = plt.bar(np.arange(3)+bar_width , Error2[0,2,:]/2, bar_width, alpha=opacity,
                     color=colors[1],edgecolor='white', label='E: 1 point')
    bars2=Error2[0,2,:]/2
         
    Cost1 = plt.bar(np.arange(3), Error2[1,0,:]/2, bar_width, alpha=opacity,
                    bottom=bars1, color=colors[0],edgecolor='white', label='F: N points')
    Cost2 = plt.bar(np.arange(3)+bar_width , Error2[1,2,:]/2, bar_width, alpha=opacity,
                    bottom=bars2, color='orange',edgecolor='white', label='F: 1 point')
    plt.xlabel('Demand')
    plt.ylabel('Mean error of the 2 objectives [%]')
    plt.title('distance')
    plt.xticks(np.arange(3) + 0.5*bar_width, ('D=400MW', 'D=800MW', 'D=1200MW'))
    plt.legend()
    plt.ylim((bottom, top))
    plt.tight_layout()
    plt.show()
    
    plt.subplot(133)
    Emiss1 = plt.bar(np.arange(3), Error3[0,0,:]/2, bar_width, alpha=opacity,
                     color='b',edgecolor='white', label='E: N points')
    bars1=Error3[0,0,:]/2
    Emiss2 = plt.bar(np.arange(3)+bar_width , Error3[0,2,:]/2, bar_width, alpha=opacity,
                     color=colors[1],edgecolor='white', label='E: 1 point')
    bars2=Error3[0,2,:]/2
         
    Cost1 = plt.bar(np.arange(3), Error3[1,0,:]/2, bar_width, alpha=opacity,
                    bottom=bars1, color=colors[0],edgecolor='white', label='F: N points')
    Cost2 = plt.bar(np.arange(3)+bar_width , Error3[1,2,:]/2, bar_width, alpha=opacity,
                    bottom=bars2, color='orange',edgecolor='white', label='F: 1 point')  
    plt.xlabel('Demand')
    plt.ylabel('Mean error of the 2 objectives [%]')
    plt.title('w-objective')
    plt.xticks(np.arange(3) + 0.5*bar_width, ('D=400MW', 'D=800MW', 'D=1200MW'))
    plt.legend()
    plt.ylim((bottom, top))
    plt.tight_layout()
    plt.show()
        
""" 
This function displays :
    the computational time for the 2 relaxations (Gurobi and Scipy)
    the computational time for solving 1 Linear, Quadratic and NonConvex (Scipy) problem
"""    
def TimeRelaxation():
    Number=[100]#[2,3,6,10,40,100]
    size=len(Number)
    nRelax=3
    Computime=np.zeros([size,nRelax])
    # Comparison of the 2 Linear Relaxations
    for nPb in range(size):
        N=Number[nPb]
        (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR)=load(N)
        if(N==40):
            D=7500
        B=loss(N)

        t1=time.time()
        """ Linear relaxation using N points """
        LinearRelaxation(N,D)
        
        t2=time.time()
        """ Linear relaxation using 1 feasible point and Gurobi"""    
        model = gp.Model('Find P0')
        model.setParam( 'OutputFlag', False )    
        model.setParam('NonConvex', 2)
        Pow = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
        PLoss = model.addVar()
        model.addQConstr(PLoss== sum( sum(Pow[i]*Pow[j]*B[i,j] for j in range(N))for i in range(N)))
        model.addConstr(Pow.sum()-PLoss == D, name='Demand')
        model.setObjective(0)
        model.optimize()
        P0=np.zeros(N)
        for i in range(N):
            P0[i]=Pow[i].x
        get_approx_planes(P0, B, D, Pmin, Pmax, True)
        
        t3=time.time()   
        """ Linear relaxation using 1 feasible point and Scipy"""    
        bnds=np.transpose(np.vstack((Pmin,Pmax)))
        P0=Pmin.copy()    
        def objective(P):
            return (0)
        def Gradient(P):
            return(np.zeros(N))
        def Hessian(P):
            return(np.zeros([N,N]))        
        def cons_f(P):
            PL=sum(sum(P[i]*P[j]*B[i,j] for j in range(N)) for i in range(N))
            sum_eq=sum(P)-PL-D
            return (sum_eq)         
        def cons_J(P):
            Jac=np.ones(N)-2*P@B
            return(Jac)
        def cons_H(P,v):
            return(-2*v*B)      
        if N<=10:         
            const=[{'type': 'eq', 'fun': cons_f}]    
            solution = minimize(objective,P0, method='SLSQP',jac=Gradient, bounds=bnds,constraints=const)      
        else:    
            NL_const = NonlinearConstraint(cons_f, 0, 0, jac=cons_J, hess=cons_H)
            solution = minimize(objective ,P0, method='trust-constr',jac=Gradient,
                                hess=Hessian,constraints=NL_const, bounds=bnds)
        P0 = solution.x
        (n, k_lower, k_upper) = get_approx_planes(P0, B, D, Pmin, Pmax, True)
        t4=time.time()

        Computime[nPb]=np.array([[t2-t1, t3-t2, t4-t3]])

    bw = 0.25
    opacity = 0.8
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    plt.bar(np.arange(size),Computime[:,0], bw, alpha=opacity, color=colors[0], label='Relaxation: N points')        
    plt.bar(np.arange(size)+bw,Computime[:,1], bw, alpha=opacity, color=colors[1], label='Relaxation: 1 point and Gurobi')
    plt.bar(np.arange(size)+2*bw,Computime[:,2], bw, alpha=opacity, color=colors[2], label='Relaxation: 1 point and Scipy')
            
    plt.xlabel('Problem size')
    plt.ylabel('Time [s]')
    plt.title('Computational time for the relaxations for different problem sizes')
    plt.xticks(np.arange(size) + bw, ('N=2', 'N=3','N=6','N=10','N=40','N=100'))
    plt.legend()
    plt.tight_layout()
    plt.show()


    #Comparison of Time for one Linear, quadratic, nonconvex problem
    Computime2=np.zeros([size,nRelax])
    for nPb in range(size):
        N=Number[nPb]
        (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR)=load(N)
        if(N==40):
            D=7500
        B=loss(N)

        t=RelaxTime(N,D)

        Computime2[nPb,:-1]=t
        t0=time.time()
        Solve(N,1,1,D)
        t1=time.time()
        Computime2[nPb,-1]= t1-t0


    bw = 0.25
    opacity = 0.8
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    plt.bar(np.arange(size),Computime2[:,0], bw, alpha=opacity, color=colors[0], label='Convex Relaxation')   
    plt.bar(np.arange(size)+bw,Computime2[:,1], bw, alpha=opacity, color=colors[1], label='Linear Relaxation')
    plt.bar(np.arange(size)+2*bw,Computime2[:,2], bw, alpha=opacity, color=colors[2], label='Scipy on non-convex')
    
    plt.xlabel('Problem size')
    plt.ylabel('Time [s]')
    plt.title('Computational time for solving one optimization problem for different problem sizes')  
    plt.xticks(np.arange(size) + bw, ('N=2', 'N=3','N=6','N=10','N=40','N=100'))
    plt.legend()
    plt.tight_layout()
    plt.show()

