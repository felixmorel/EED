"""
This file is used for computing the linear relaxation bounds using N points

Author: Felix Morel
Date : 09/06/2020

"""

import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from Params import load,loss
from Static_model import SimplePriceFun 
from LinApproxLVH import get_approx_planes
from Simple_Hermite import Hermite, HermiteError,SmartHermiteError
from NLSolverStat import Solve


PI=math.pi

#This function computes N extreme points and returns the coefficients of the upper bound
def LinUpperB(Pmin,Pmax,B,D):
    N=len(Pmin)
    p=np.zeros([N,N])
    elements=np.arange(0,N) 
    for i in range(N): 
        model = gp.Model('Extreme point')
        x = model.addVars(range(N),vtype=GRB.BINARY)
        Px = model.addVars(range(N))
        Py = model.addVars(range(N))
        for j in range(N):
            if j==i:
                model.addConstr(Px[j]==Pmin[j])
                model.addConstr(x[i]==1)
                model.addConstr(Py[j]==Pmax[j])
            else:
                model.addConstr(Px[j]==x[j]*Pmin[j]+(1-x[j])*Pmax[j])
                model.addConstr(Py[j]== Px[j])
    
        model.addConstr(D-sum(Px[j] for j in range(N)) + sum( sum( Px[k]*Px[j]*B[k,j] for j in range(N)) for k in range(N))>=0)
        model.addConstr(D-sum(Py[j] for j in range(N)) + sum( sum( Py[k]*Py[j]*B[k,j] for j in range(N))for k in range(N))<=0)
    
        model.setObjective(0)
        model.setParam( 'OutputFlag', False )
        model.optimize()
        Limits=np.zeros(N)
        for j in range(N):
            if model.getVars()[j].x==0:
                Limits[j]=Pmax[j]
            else:
                Limits[j]=Pmin[j]   
        a=B[i,i]
        index=np.append(elements[:i],elements[i+1:])
        b=2* B[index,i].dot(Limits[index])-1
        c=D+ Limits[index].dot(B[index][:,index].dot(Limits[index])) - np.ones(N-1).dot(Limits[index])
        
        r= b**2-4*a*c
        if r<0:
            print("Error in equation")
        z = (-b+np.sqrt(r))/(2*a)
        if (z<Pmin[i] or z>Pmax[i]):   
            z = (-b-np.sqrt(r))/(2*a) 
        else:
            print((-b-np.sqrt(r))/(2*a), i , Pmin[i], Pmax[i], "Check LinUpperB")
       
        p[i,i]=z
        p[i,index]=Limits[index]

    A=np.zeros([N+1,N+1])
    A[:-1,:-1]=p
    A[:-1, -1]=np.ones(N)
    eye=np.zeros(N+1)
    eye[0]=1
    A[-1,:]=eye
    s=np.zeros(N+1)
    s[-1]=1
    nk=np.linalg.solve(A, s)
    n=nk[:-1]
    k=nk[-1]
    return(n,k,p)

#Find the linear constraints that bounds the demand
def LinearRelaxation(N,D):
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    #Upper bound
    (n,k_upper, p)= LinUpperB(Pmin,Pmax,B,D) 

    #Lower bound
    model = gp.Model('Lower bound of demand')
    P = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    PL = model.addVar()
    model.addQConstr(PL>= sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
    model.addConstr(P.sum()-PL == D, name='Demand')
    obj= sum(P[i]*n[i]  for i in range(N) ) 
    model.setObjective(obj)
    model.setParam( 'OutputFlag', False )
    model.optimize() 
    k_lower=-obj.getValue()
    return(n,k_upper,k_lower)
    
    
"""
This function compares the possible relaxations and their accuracy
Given a demand, it returns the 3 errors on the Costs and Emissions.
""" 
def RelaxErrors(N,Demand):   
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    it=10
    C= np.zeros(it)
    E= np.zeros(it)
    
    problems=[ "Convex relaxation" , "Linear lower bound" , "Linear upper bound" ,  "Linear lower bound LVH" , "Linear upper bound LVH"]
    n_meth=0
    wErrror=np.zeros([len(problems),2])
    plt.figure()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for method in problems:
        m = gp.Model(method)
        m.setParam( 'OutputFlag', False )
        P = m.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
        PL = m.addVar()
        x = m.addVars(N)
        y = m.addVars(N) 
        for i in range(N):
            m.addConstr(x[i]==delta[i]*P[i])
            m.addGenConstrExp(x[i], y[i])    
        
        if (method=="Non-convex problem"):
            m.setParam('NonConvex', 2)
            m.addQConstr(PL== sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
            m.addConstr(P.sum() == Demand+PL)
            methname=method
            line='-'
        
        elif (method=="Convex relaxation"):            
            m.addQConstr(PL>= sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
            m.addConstr(P.sum()-PL == Demand, name='Demand')
            methname=method
            line='-'
             
        elif (method=="Linear lower bound"):
            E_truth=E.copy()
            C_truth=C.copy()      
            (n,k_upper, k_lower) = LinearRelaxation(N,Demand)
            m.addConstr( sum(P[i]*n[i] for i in range(N))+k_lower>=0) #Lowerbound
            n_meth=n_meth+1
            methname= "Linear lower and upper bound"
            line='--'
                
        elif( method=="Linear upper bound"):
            E_linL=E.copy()
            C_linL=C.copy()
            m.addConstr( sum(P[i]*n[i] for i in range(N))+k_upper>=0) #Upperbound
            methname=''
            
        
        elif (method=="Linear lower bound LVH"):
            E_linU=E.copy()
            C_linU=C.copy()
            
            model = gp.Model('Find P0')
            model.setParam( 'OutputFlag', False )
    
            model.setParam('NonConvex', 2)
            Pow = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
            PLoss = model.addVar()
            model.addQConstr(PLoss== sum( sum(Pow[i]*Pow[j]*B[i,j] for j in range(N))for i in range(N)))
            model.addConstr(Pow.sum()-PLoss == Demand, name='Demand')
            model.setObjective(0)
            model.optimize()
            P0=np.zeros(N)
            for i in range(N):
                P0[i]=Pow[i].x
            
            (n, k_lower, k_upper) = get_approx_planes(P0, B, Demand, Pmin, Pmax)
            m.addConstr( sum(P[i]*n[i] for i in range(N))+k_lower>=0) #Lowerbound
            n_meth=n_meth+1
            methname= "Linear lower and upper bound LVH"
            line=':'              
            
        elif(method=="Linear upper bound LVH"):
            E_LVHL=E.copy()
            C_LVHL=C.copy()
            m.addConstr( sum(P[i]*n[i] for i in range(N))+k_upper>=0) #Upperbound
            methname=''
        
        index=0
        while(method!=problems[index]):
            index=index+1
            
        Cost = sum(a[k]+b[k]*P[k]+c[k]*P[k]*P[k] for k in range(N))
        Emission = sum(alpha[k]+beta[k]*P[k]+gamma[k]*P[k]*P[k]+eta[k]*y[k] for k in range(N))
        
        w= np.linspace(0.001, 0.999 , num=it)
        for i in range(it) : 
            obj= price*w[i]*Emission+ (1-w[i])*Cost 
            m.setObjective(obj)
            m.optimize()
    
            C[i]=Cost.getValue()
            E[i]=Emission.getValue()           
    
            if (n_meth>0):
                E_error= abs(E[i]-E_truth[i])
                C_error = abs(C[i]-C_truth[i])
                wErrror[index,0]=max(wErrror[index,0], round(E_error/E_truth[i],5))
                wErrror[index,1]=max(wErrror[index,1], round(C_error/C_truth[i],5))  
                
        plt.plot(E,C,'.', color=colors[n_meth], label=methname)
        [contE,s]=Hermite(E,C, price*w, 1-w )
        plt.plot(contE,s , color=colors[n_meth], linestyle=line) 
    
    
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Comparison of the relaxations of '+str(N)+' generators problem and demand = '+ str(int(Demand))+ ' MW')
    plt.grid()
    plt.legend()
    
    Error1=np.zeros([len(problems)-1,2])    
    [E_error,C_error]=HermiteError(E_truth,C_truth,E_linL,C_linL, price*w, 1-w)
    Error1[0,0]=E_error
    Error1[0,1]=C_error

    [E_error,C_error]=HermiteError(E_truth,C_truth,E_LVHL,C_LVHL, price*w, 1-w)
    Error1[2,0]=E_error
    Error1[2,1]=C_error 

    Error2=np.zeros([len(problems)-1,2])    
    [E_error,C_error]=SmartHermiteError(E_truth,C_truth,E_linL,C_linL, price*w, 1-w)
    Error2[0,0]=E_error
    Error2[0,1]=C_error

   
    [E_error,C_error]=SmartHermiteError(E_truth,C_truth,E_LVHL,C_LVHL, price*w, 1-w)
    Error2[2,0]=E_error
    Error2[2,1]=C_error  
    return(Error1,Error2,wErrror)

    
"""This function compares the computational time for the linear and convex relaxation with Gurobi""" 
def RelaxTime(N,Demand):   
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)

    Time=np.zeros(2)
    problems=["Convex relaxation" , "Linear Relax"]
    n_meth=0

    for method in problems:
        m = gp.Model(method)
        m.setParam( 'OutputFlag', False )
        P = m.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
        PL = m.addVar()
        x = m.addVars(N)
        y = m.addVars(N) 
        for i in range(N):
            m.addConstr(x[i]==delta[i]*P[i])
            m.addGenConstrExp(x[i], y[i])    
        
        if (method=="Convex relaxation"):            
            m.addQConstr(PL>= sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
            m.addConstr(P.sum()-PL == Demand, name='Demand')
        
        elif (method== "Linear Relax"): 
            
            [opt,P0]=Solve(N,price,1,Demand)
            n=np.ones(N)-2*B@P0
            k=- np.dot(n,P0)
            m.addConstr( sum(P[i]*n[i] for i in range(N))+k>=0) 
         
        t0=time.time()
            
        Cost = sum(a[k]+b[k]*P[k]+c[k]*P[k]*P[k] for k in range(N))
        Emission = sum(alpha[k]+beta[k]*P[k]+gamma[k]*P[k]*P[k]+eta[k]*y[k] for k in range(N))
        
        obj= price*Emission+ Cost 
        m.setObjective(obj)
        m.optimize()           
        t1=time.time()
        Time[n_meth]=t1-t0
        n_meth=n_meth+1
    return(Time)
    