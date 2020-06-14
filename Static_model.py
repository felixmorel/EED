"""
This file is  used for the static EED problem as QP
Author: Felix Morel
Date : 09/06/2020

"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from Params import load,loss

# This function computes the price penalty factor on the static QP
def PricePenaltyFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand):
    Num=a+np.multiply(b,Pmax)+np.multiply(c,np.multiply(Pmax,Pmax))
    Denom = alpha+np.multiply(beta,Pmax)+np.multiply(gamma,np.multiply(Pmax,Pmax))
    h=Num/Denom
    index=sorted(range(len(h)), key=lambda k: h[k])
    h=h[index]

    m=[Pmax[i] for i in index]
    Sum=np.cumsum(m)
    i = np.argmax(Sum>Demand)
    coeff=(Sum[i]-Demand)/ m[i]                               
    hm=(1-coeff)*h[i]+coeff*h[i-1]
    return(hm)    

# This function computes the average price on the emissions on the static QP
def SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand):
    N=len(a)   
    model = gp.Model('Quadratic Problem')
    Pow = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    model.addConstr(Pow.sum() == Demand, name='Demand')
 
    Cost = sum(a[k]+b[k]*Pow[k]+c[k]*Pow[k]*Pow[k] for k in range(N))
    Emission = sum(alpha[k]+beta[k]*Pow[k]+gamma[k]*Pow[k]*Pow[k] for k in range(N))

    model.setObjective(Cost)
    model.setParam( 'OutputFlag', False )
    model.optimize()
    
    Cmin=Cost.getValue()
    Emax=Emission.getValue()
    
    model.setObjective(Emission)
    model.setParam( 'OutputFlag', False )
    model.optimize()
    
    Cmax = Cost.getValue()
    Emin = Emission.getValue()
    
    p=(Cmax-Cmin)/(Emax-Emin)
    return(p)
  
"""
This function obtains the POF for the simple problem
Optional Parameters:
spacing: 1,2,3 for different spacements
tan= 'yes','no' : plots the tangent at a given Pareto value

"""   
def Simple(N, Spacing=1, tan='no'):        
    (Demand,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    if Spacing==0:
        price=1
    elif Spacing==1:
        price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    else:   
        price = PricePenaltyFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    it=100
    C= np.zeros(it)
    E= np.zeros(it)
    
    model = gp.Model('Static Problem')
    Pow = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    model.addConstr(Pow.sum() == Demand, name='Demand')
    
    Cost = sum(a[k]+b[k]*Pow[k]+c[k]*Pow[k]*Pow[k] for k in range(N))
    Emission = sum(alpha[k]+beta[k]*Pow[k]+gamma[k]*Pow[k]*Pow[k] for k in range(N)) 
    w= np.linspace(0.001, 0.999 , num=it)
    
    for i in range(it) : 
        
        obj= price*w[i]*Emission+ (1-w[i])*Cost  
        model.setObjective(obj)
        model.setParam( 'OutputFlag', False )
        model.optimize()         
        
        C[i]=Cost.getValue()
        E[i]=Emission.getValue()
    
    plt.figure()
    
    if (tan=='yes'):
        plt.plot(E,C, label='Continious Pareto-optimal front')

        index= 93
        grad=w[index]/(w[index]-1)
        obj=  -grad*Emission + Cost
        model.setObjective(obj)
        model.setParam( 'OutputFlag', False)
        model.optimize()

        y=Cost.getValue()
        x=Emission.getValue()
        
        vecy=np.array([grad,-grad])
        vecx=np.array([-1,1])
        
        plt.plot(x+10*vecx,y-10*vecy, 'g',linewidth=5.0)
        plt.plot(x,y,'ro')
    
    else:
        plt.plot(E,C,'ro', label='Continious Pareto-optimal front')

    plt.xlabel('Emissions')
    plt.ylabel('Cost')
    plt.title('Pareto-optimal points')
    
    return(E,C)

    