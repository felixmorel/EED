"""
This file is used for Dynamic EED with nonlinear Emission
Author: Felix Morel
Date : 09/06/2020

"""


import numpy as np
import math
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt
from Params import load
from Static_model import SimplePriceFun 
from Simple_Hermite import Hermite
from DynamicD import DEED_QP

PI=math.pi
"""
This function computes 10 PO values for the Dynamic EED with nonlinear emission function
compareto= "QuadraticDEED"; "NonConvexLoss"

"""
def DEED_NL(N, compareto="QuadraticDEED"):
    (StaticD,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    T=24

    t1=int(T/3)
    theta= np.linspace(-PI/2, PI/2, num=t1)
    theta2= np.linspace(-PI, 3*PI/2, num=T-t1)
    phi= 0.15
    
    Demand=np.zeros(T)
    Demand[:t1]= 0.75*StaticD*(1+phi*np.sin(theta))
    for i in np.arange(0,T-t1):
        if compareto=="QuadraticDEED":
             #Compare with DynamicD
            Demand[i+t1]= max(min(Demand[i+t1-1]+0.95*sum(UR)*np.sin(theta2[i]), 0.95* sum(Pmax) ), sum(Pmin))

        else:
            #Compare with NonConvexLoss
            Demand[i+t1]= max(min(Demand[i+t1-1]+0.9*sum(UR)*np.sin(theta2[i]), 0.8* sum(Pmax) ), sum(Pmin)) 

    Demand[-1]=Demand[0]

    
    SR=0.05*Demand
    SR2=SR/3      
    it=10
    C= np.zeros(it)
    E= np.zeros(it)

    w= np.linspace(0.001, .999 , num=it)
    LB= np.repeat([Pmin],T, axis=0)
    UB= np.repeat([Pmax],T, axis=0)
    
    model = gp.Model('Non-Quadratic Dynamic Model')
    model.setParam( 'OutputFlag', False )
    
    P = model.addVars(T,N,lb=LB, ub=UB, name='P')
    x = model.addVars(T,N)
    y = model.addVars(T,N)
    z = model.addVars(T,N)
    m = model.addVars(T,N)
    m2 = model.addVars(T,N)

    for t in range(T):
        model.addConstr(sum(P[t,k] for k in range(N)) == Demand[t], name='Demand at '+ str(t))       
        for n in range(0,N):
            if t>0:
                """ Ramp rate limits """
                model.addConstr(P[t,n] <= P[t-1,n] + UR[n], name='Maxramp'+ str(t) + str(n))
                model.addConstr(P[t,n] >= P[t-1,n] - DR[n], name='Minramp'+ str(t) + str(n))
            
            """ Reserve constraints"""
            model.addConstr(z[t,n]==Pmax[n]-P[t,n])
            model.addConstr(m[t,n]==min_([z[t,n],UR[n]]))
            model.addConstr(m2[t,n]==min_([z[t,n],UR[n]/6]))
            
            """ Exponential term of Emission objective """
            model.addConstr(x[t,n]==delta[n]*P[t,n])
            model.addGenConstrExp(x[t,n], y[t,n])
            
        model.addConstr(sum(m[t,i]for i in range(N))>=SR[t])
        model.addConstr(sum(m2[t,i]for i in range(N))>=SR2[t])
            
        
    Cost = sum(sum(a[k]+b[k]*P[t,k]+c[k]*P[t,k]*P[t,k] for k in range(N)) for t in range(T))  
    Emission = sum(sum(alpha[k]+beta[k]*P[t,k]+gamma[k]*P[t,k]*P[t,k]+eta[k]*y[t,k] for k in range(N))for t in range(T))
    
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma, np.mean(Demand))
    
    for i in range(it) :  
        obj= price*w[i]*Emission+ (1-w[i])*Cost 
        model.setObjective(obj)
        model.optimize()

        Pow=np.zeros([T,N])
        for t in range(T):
            for n in range(N):
                Pow[t,n] = P[t,n].x 
        
        if i==0:
            EconomicSol=Pow
        elif i ==it-1:
            EmissionSol=Pow
            
        C[i]=Cost.getValue()
        E[i]=Emission.getValue()
    return(E,C,EconomicSol,EmissionSol,price,w,T)
    
def figures(N): 
    #Compares the POF of the Quadratic and Nonlinear DEED 
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    [E_qp,C_qp,Eco_qp,Emis_qp,price,w,T]=DEED_QP(N)
    [E,C,EconomicSol,EmissionSol,price,w,T]=DEED_NL(N)
    
    plt.figure()
    
    plt.plot(E_qp,C_qp,'ro')
    [contE,s]=Hermite(E_qp,C_qp, price*w, 1-w )
    plt.plot(contE,s , label='Quadratic Problem')
    
    
    plt.plot(E,C,'ro')
    [contE1,s1]=Hermite(E,C, price*w, 1-w )
    plt.plot(contE1,s1 , label='Non-quadratic Problem')
    
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Pareto-optimal front for a 24h operation')
    plt.legend()


    time=np.linspace(0, 24, num=T)
    plt.figure()
    plt.subplot(121) 
    bars=0
    for n in range(N):
        plt.bar(time, EconomicSol[:,n], bottom=bars, color=colors[n])
        bars+=EconomicSol[:,n]
    
    plt.xlabel('Time [Hr]')
    plt.ylabel('Power [MW]')
    plt.title('One day power production with the Economic Dispatch')
    
    plt.subplot(122)
    bars=0
    for n in range(N):
        plt.bar(time, EmissionSol[:,n], bottom=bars, color=colors[n])
        bars+=EmissionSol[:,n]
       
    plt.xlabel('Time [Hr]')
    plt.ylabel('Power [MW]')
    plt.title('One day power production with the Emission Dispatch')
    
