"""
This file implements the Scipy solver and second order SQP

Author: Felix Morel
Date : 11/06/2020

"""
import numpy as np
import math
import time
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

import matplotlib.pyplot as plt
from Params import load,loss
from Static_model import SimplePriceFun 
from Simple_Hermite import Hermite
PI=math.pi

"""
Solves with Scipy the scalarized problem associated to objective:
    min w_E*E(p)+w_C*C(p)
Choose method between:
    'trust-const'; 'SLSQP'
"""
def Solve(N,w_E,w_C,D, method='trust-const'):
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    bnds=np.transpose(np.vstack((Pmin,Pmax)))
    P0=Pmin.copy()    
    def objective(P):
        Cost = sum(a[i]+b[i]*P[i]+c[i]*P[i]*P[i] for i in range(N))
        Emission = sum(alpha[i]+beta[i]*P[i]+gamma[i]*P[i]*P[i] +eta[i]*np.exp(P[i]*delta[i]) for i in range(N))
        return (w_E*Emission+w_C*Cost)
    
    def Gradient(P):
        GradC=b+2*c*P
        GradE= beta+2*gamma*P+delta*eta*np.exp(delta*P)
        Grad=w_C*GradC+w_E*GradE
        return(Grad)
        
    def Hessian(P):
        Hess= 2*w_C*c+w_E*(2*gamma+delta*delta*eta*np.exp(delta*P))
        H=Hess*np.eye(N)
        return(H)        
        
    def cons_f(P):
        PL=sum(sum(P[i]*P[j]*B[i,j] for j in range(N)) for i in range(N))
        sum_eq=sum(P)-PL-D
        return (sum_eq)
    
    def cons_J(P):
        Jac=np.ones(N)-2*P@B
        return(Jac)
    def cons_H(P,v):
        return(-2*v*B)
    
    if (method=='SLSQP'):
        const=[{'type': 'eq', 'fun': cons_f}]    
        solution = minimize(objective,P0, method='SLSQP',jac=Gradient, bounds=bnds,constraints=const)
    else:
        NL_const = NonlinearConstraint(cons_f, 0, 0, jac=cons_J, hess=cons_H)
        solution = minimize(objective,P0, method='trust-constr',jac=Gradient,
                            hess=Hessian,constraints=NL_const, bounds=bnds)
    P = solution.x
    return(objective(P),P)


"""
Solves the convex relaxation of the static EED
If Pl and Pu are defined: vectors of size N representing the specific operating zone for POZ problem
If figures is True: Displays the convergence of the method

"""    
def SQP(N,w_E,w_C,D,Pl=0,Pu=0, figures='False'): 
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    if (type(Pl)!=int):  
            Pmin=Pl.copy()
            Pmax=Pu.copy()
    t0=time.time()
    
    """Computing P0"""    
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

    if (N<=10):
        const=[{'type': 'eq', 'fun': cons_f}]
        solution = minimize(objective ,P0, method='SLSQP',jac=Gradient, bounds=bnds,constraints=const)
    else: 
        def cons_J(P):
            Jac=np.ones(N)-2*P@B
            return(Jac)
        def cons_H(P,v):
            return(-2*v*B)
        NL_const = NonlinearConstraint(cons_f, 0, 0, jac=cons_J, hess=cons_H)
        solution = minimize(objective ,P0, method='trust-constr',jac=Gradient,
                            hess=Hessian,constraints=NL_const, bounds=bnds)
    P0 = solution.x
    tol=1e-2
    Maxiter=25
    Obj=np.zeros(Maxiter)
    C = sum(a[k]+b[k]*P0[k]+c[k]*P0[k]*P0[k] for k in range(N))
    E = sum(alpha[k]+beta[k]*P0[k]+gamma[k]*P0[k]*P0[k]+eta[k]*np.exp(delta[k]*P0[k]) for k in range(N))
    
    Obj[0]=w_C*C+w_E*E
    Pk=P0.copy()
    it=1    
    stepsize=1
    while (it<Maxiter  and stepsize>tol): #and tol<Obj[it-1]-opt
        
        model=gp.Model('SQP Step')
        model.setParam( 'OutputFlag', False )
        DeltaP = model.addVars(range(N),lb=Pmin-Pk,ub=Pmax-Pk)

        Surplus=sum(Pk)-Pk@B@Pk-D
        model.addConstr(Surplus+sum(DeltaP[k]*(1-2*Pk@B[k]) for k in range(N))==0)          
              
        GradC=b+c*Pk*2
        GradE= beta+gamma*Pk*2+delta*eta*np.exp(delta*Pk)
        Grad=w_C*GradC+w_E*GradE
        Hessian= w_C*2*c+w_E*(2*gamma+delta*delta*eta*np.exp(delta*Pk))
        Lagr=sum(DeltaP[k]*DeltaP[k]*Hessian[k] for k in range(N))
        objective = sum(Grad[k]*DeltaP[k] for k in range(N)) + 0.5*Lagr
        model.setObjective(objective)
        model.optimize()

        Prev=Pk.copy()    
        for i in range(N):
            Pk[i] = Pk[i] + DeltaP[i].x        
        
        stepsize=np.linalg.norm(Prev-Pk)
        C = sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
        E = sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
        Obj[it]=w_C*C+w_E*E
        
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1
        
    if (figures==True):
        t1=time.time()
        [opt,P_opt]=Solve(N,w_E,w_C,D) 
        t2=time.time()
        plt.figure()
    
        Pos=Obj[:it]-np.ones(it)*opt
        Neg=-Pos.copy()
        
        Pos=(Obj[:it]-np.ones(it)*opt>0)*Pos
        Neg=(Obj[:it]-np.ones(it)*opt<0)*Neg
        plt.plot(range(it),Pos, label='Positive Part ')
        plt.plot(range(it),Neg, label='Negative Part ')
    
        plt.xlabel('Iterations')
        plt.ylabel('$f_k-f*$')
        plt.title("Rate of convergence of SQP method ")
        plt.legend()
        plt.grid()  
        print(t1-t0, "sec for SQP ")
        print(t2-t1, "sec for Scipy ")
        print('\007')        
    return(E,C,Pk)
    
   
#This function tests the SQP solver
def testSQP(N):   
    (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N) 
    if (N==40): #Demand for 40-unit test case not suited for the transmission losses
        D=7500     
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,D)
    (E,C,Pk)=SQP(N,price,1,D, figures=True)
