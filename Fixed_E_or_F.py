"""
This file is  used for the static EED problem as QP and e-constraint
Author: Felix Morel
Date : 09/06/2020

"""
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from Params import load
from Static_model import SimplePriceFun

#Computes the POF and for a given limit
# Limit= 'E' or 'C'
def eConst(N=6, Limit="E"):
       
    (Demand,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)    
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    it=100
    C= np.zeros(it)
    E= np.zeros(it)
    model = gp.Model('Quadratic Problem')
    Pow = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    model.addConstr(Pow.sum() == Demand, name='Demand')
    
    Cost = sum(a[k]+b[k]*Pow[k]+c[k]*Pow[k]*Pow[k] for k in range(N))
    Emission = sum(alpha[k]+beta[k]*Pow[k]+gamma[k]*Pow[k]*Pow[k] for k in range(N))
    
    w= np.linspace(0, 1 , num=it)
    for i in range(it) : 
        obj= price*w[i]*Emission+ (1-w[i])*Cost 

        model.setObjective(obj)
        model.setParam( 'OutputFlag', False )
        model.optimize()
    
        C[i]=Cost.getValue()
        E[i]=Emission.getValue()
      
    plt.figure()
    plt.xlabel('Emission')
    plt.ylabel('Cost')
    
    
    if Limit=="E": 
        # Limitations in the Emissions 
        Emax= np.round(np.mean(E)) #1206.0
        index=np.argmax(Emax>E)
        plt.plot(E[index:],C[index:])
        plt.plot(E[:index],C[:index],'--', color=plt.gca().lines[-1].get_color())
    
            
        QC= model.addConstr(Emission<=Emax, name='MaxEmission')
        model.setObjective(Cost)
        model.setParam('QCPDual', 1) # In order to compute the dual too
        model.optimize()
        
        y=Cost.getValue()
        x=Emission.getValue()
        
        grad= QC.QCPi #Get the dual variable of QC
        
        vecy=np.array([grad,-grad])
        vecx=np.array([-1,1])
        
        plt.plot(x+10*vecx,y-10*vecy, 'g',linewidth=5.0)
        plt.vlines(x=Emax, ymin=min(C), ymax=max(C), linewidth=2.0)
        plt.plot(x,y,'ro')
        plt.title('Optimal solution for fixed Emission')
    
    
    elif Limit=="C":
       # Limitations in the costs
        Cmax= np.round(np.mean(C)) #62000
        index=np.argmax(Cmax<C)
        plt.plot(E[:index],C[:index])
        plt.plot(E[index:],C[index:],'--', color=plt.gca().lines[-1].get_color())
        
            
        QC= model.addConstr(Cost<=Cmax, name='MaxCost')
        model.setObjective(Emission)
        model.setParam('QCPDual', 1) # In order to compute the dual too
        model.optimize()
        
        y=Cost.getValue()
        x=Emission.getValue()
        
        gradinv= QC.QCPi #Get the dual variable of QC
        grad=1/gradinv
        vecy=np.array([grad,-grad])
        vecx=np.array([-1,1])
          
        plt.plot(x+10*vecx,y-10*vecy, 'g',linewidth=5.0)
        plt.hlines(y=Cmax, xmin=min(E), xmax=max(E), linewidth=2.0)
        plt.plot(x,y,'ro') 
        plt.title('Optimal solution for fixed Cost')
    
    
    
    
