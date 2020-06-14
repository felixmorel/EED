"""
This file is used for Dynamic EED with nonlinear Emission and transmission losses
Author: Felix Morel
Date : 09/06/2020

"""


import numpy as np
import math
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt
import time
from Params import load,loss
from Static_model import SimplePriceFun 
from Simple_Hermite import Hermite
from NonQuadratic import DEED_NL

PI=math.pi
"""
This function computes 10 PO values for the Dynamic EED with Transmission Losses
The method for computing these points depends on the following parameter:
method= "scal"; "LimE" ; "LimF"

"""
def DEED_PL(N,method):
    (StaticD,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    T=24    
    t1=int(T/3)
    theta= np.linspace(-PI/2, PI/2, num=t1)
    theta2= np.linspace(-PI, 3*PI/2, num=T-t1)
    Demand=np.zeros(T)
    phi= 0.15
    Demand[:t1]= 0.75*StaticD*(1+phi*np.sin(theta))
    for i in np.arange(0,T-t1):
        #Modified demand otherwise the problem is not feasible anymore...
        Demand[i+t1]= max(min(Demand[i+t1-1]+0.9*sum(UR)*np.sin(theta2[i]), 0.8*sum(Pmax)), sum(Pmin))
    Demand[-1]=Demand[0]
      
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma, np.mean(Demand))
    SR=0.05*Demand
    SR2=SR/3
         
    it=10
    C= np.zeros(it)
    E= np.zeros(it)
    w= np.linspace(0.001, 0.999 , num=it)
    
    LB= np.repeat([Pmin],T, axis=0)
    UB= np.repeat([Pmax],T, axis=0)
    
    model = gp.Model('Non-Convex Problem')
    P = model.addVars(T,N,lb=LB, ub=UB, name='P')
    PL = model.addVars(T)

    x = model.addVars(T,N)
    y = model.addVars(T,N)
    z = model.addVars(T,N)
    m = model.addVars(T,N)
    m2 = model.addVars(T,N)
    model.Params.NonConvex = 2
    
    for t in range(T):
        """Losses on the transfers """

        model.addQConstr(PL[t]<= sum( sum(P[t,i]*P[t,j]*B[i,j] for j in range(N))for i in range(N)))
        model.addQConstr(PL[t]>= sum( sum(P[t,i]*P[t,j]*B[i,j] for j in range(N))for i in range(N)))

        model.addConstr(sum(P[t,k] for k in range(N))-PL[t] == Demand[t], name='Demand at '+ str(t))        
        
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
    
    if N==6:
        LimE=np.linspace(14700,17000,num=it)
        LimF=np.linspace(1010000, 1050000,num=it) 
        
    elif N==10:
        LimE=np.linspace(57000,63000,num=it)
        LimF=np.linspace(1795000,1875000,num=it)
    
    
    for i in range(it) :     
        if (method== "scal"):
            obj= price*w[i]*Emission+ (1-w[i])*Cost 
            
        elif (method=="LimE"):
            obj= Cost 
            if i>0:  
                model.remove(const)
            const=model.addQConstr(Emission<=LimE[i])

        elif (method=="LimF"):
            obj=Emission
            if i>0:  
                model.remove(const)
            const=model.addQConstr(Cost<=LimF[it-i-1])
        
        model.setObjective(obj)
        model.setParam( 'OutputFlag', False )
        model.Params.timeLimit = 500
        model.update()
        tm=1
        model.optimize()
        
        # In case the problem takes too much time
        while (model.status!=2):
            model.update()
            model.optimize()
            tm=tm+1
            print(model.SolCount)    
        print(tm, 'Time periods')
        print(model.status)

        C[i]=Cost.getValue()
        E[i]=Emission.getValue()
    return(E,C,price,w,T)
    
"""
Displays the POF for the DEED using tranmission losses
If method=='scal: scalarization of the objective+ comparison with the DEEd without losses
Else method== 'LimE' or 'LimF' : e-constraint method

 """   
def figures(N, method="scal"): 
    t0=time.time()
    [E,C,price,w,T]=DEED_PL(N, method)
    t1=time.time()
    print("Time for solving the nonconvex problem: ", np.round(t1-t0,3))
    
    plt.figure(2)
    if (method=="scal"):
        plt.plot(E,C,'ro',label='Linear combination of objectives')
        [contE,s]=Hermite(E,C, price*w, 1-w )  
        plt.plot(contE,s, '--', label='Hermite interpolation of the linearization')
        
        
        [E_nl,C_nl,EconomicSol,EmissionSol,price,w,T]=DEED_NL(N)#, "NonConvexDEED")
        plt.plot(E_nl,C_nl,'ro')
        [contE_nl,s_nl]=Hermite(E_nl,C_nl, price*w, 1-w )  
        plt.plot(contE_nl,s_nl , label='Convex problem')
        
        print("\007")
        
        
    
    elif (method=="LimE"):
        plt.plot(E,C,'gx',label='$\epsilon$ constraint on Emissions') 
    elif (method=="LimF"):
        plt.plot(E,C,'mx',label='$\epsilon$ constraint on Cost') 
    else:
        print("Choose adequate method name between: 'scal', 'LimE' and 'LimF'")
        
        
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Pareto-optimal front for a 24h operation')
    plt.legend()
    plt.grid(True)
