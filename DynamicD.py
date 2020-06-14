"""
This file is used for Dynamic EED
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

PI=math.pi
"""
This function computes 10 PO values for the Dynamic EED QP
If reserves=false, the program does not take the spinning reserves into consideration

"""
def DEED_QP(N, reserves=True):
    (StaticD,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    T=24
    
    t1=int(T/3)
    theta= np.linspace(-PI/2, PI/2, num=t1)
    theta2= np.linspace(-PI, 3*PI/2, num=T-t1)
    phi= 0.15
    
    Demand=np.zeros(T)
    Demand[:t1]= 0.75*StaticD*(1+phi*np.sin(theta))
    for i in np.arange(0,T-t1):
        Demand[i+t1]= max(min(Demand[i+t1-1]+0.95*sum(UR)*np.sin(theta2[i]), 0.95* sum(Pmax) ), sum(Pmin))
    Demand[-1]=Demand[0]
    
    SR=0.05*Demand
    SR2=SR/3 
        
    it=10
    C= np.zeros(it)
    E= np.zeros(it)
    w= np.linspace(0.001, 0.999 , num=it)
    LB= np.repeat([Pmin],T, axis=0)
    UB= np.repeat([Pmax],T, axis=0)
    
    model = gp.Model('Dynamic EED')
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
            
            if reserves:   
                """ Reserve constraints"""
                model.addConstr(z[t,n]==Pmax[n]-P[t,n])
                model.addConstr(m[t,n]==min_([z[t,n],UR[n]]))
                model.addConstr(m2[t,n]==min_([z[t,n],UR[n]/6]))
            
        if reserves:   
            model.addConstr(sum(m[t,i]for i in range(N))>=SR[t])
            model.addConstr(sum(m2[t,i]for i in range(N))>=SR2[t])
            

    Cost = sum(sum(a[k]+b[k]*P[t,k]+c[k]*P[t,k]*P[t,k] for k in range(N)) for t in range(T))  
    Emission = sum(sum(alpha[k]+beta[k]*P[t,k]+gamma[k]*P[t,k]*P[t,k] for k in range(N))for t in range(T))
    
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
    
""" Displays the obtained POF and the Power dispatch of the Dynamic EED"""
def figures(N): 
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    [E1,C1,EcoSol,EmisSol,price,w,T]=DEED_QP(N)
    [E,C,EconomicSol,EmissionSol,price,w,T]=DEED_QP(N,False)
    
    plt.figure()
    plt.plot(E1,C1,'ro')
    [contE1,s1]=Hermite(E1,C1, price*w, 1-w )
    plt.plot(contE1,s1 , label='With reserves') 
    
    plt.plot(E,C,'ro')
    [contE,s]=Hermite(E,C, price*w, 1-w )
    plt.plot(contE,s , label='Without reserves') 
    
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
    
