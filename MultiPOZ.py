"""

This File solves  the pareto optimal front for the problem considering Transmission Losses
and different allowed operating Zones.
Author: Felix Morel
Date : 11/06/2020

"""

import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from Params import load,loss,zones
from Static_model import SimplePriceFun 
from Simple_Hermite import Hermite
from NLSolverStat import Solve, SQP
from POZ import InZone, eConstE_SQP, eConstF_SQP, SQP_MINLP
   
    
#This function computes the minimal distance between point (E,C) 
#and the POF defined by E_opt and C_opt
# Treats the particular case of 6-unit system, with D=800
def mindist(E_opt,C_opt, E, C):
    price=(max(C_opt)-min(C_opt))/(max(E_opt)-min(E_opt))
    dist= price**2*(E_opt-E)*(E_opt-E) + (C_opt-C)*(C_opt-C) 
    i= np.argmin(dist)

    if i>=0 : #or E<E_opt[i]:
        if E-E_opt[i]>0: 
            j=i-1
        else:
            j=i+1
        
        #Numerical exception of this example
        if (i==189 and E>E_opt[i]) or (i==190 and E<E_opt[i]):
             ## meaning i=-10 and j=-11 or inversely: hole in POF 
            Esub=644.2573628221216 #Objective not Pareto optimal but on the edge of the atteinable domain
            Csub=41919.93302538373
            m=(C_opt[-10]-Csub)/(price*(E_opt[-10]-Esub))
            e=m/(price*(1+m**2))*(C-C_opt[-10]+price*E/m+price*E_opt[-10]*m)
            c= price*m*(e-E_opt[-10])+C_opt[-10]
            d= price**2*(e-E)*(e-E) + (c-C)*(c-C) 
            Error_E=abs(e-E)
            Error_C=abs(c-C)
            if dist[-11]<d:
                Error_E=abs(E_opt[-11]-E)
                Error_C=abs(C_opt[-11]-C)
        else: 
            m=(C_opt[i]-C_opt[j])/(price*(E_opt[i]-E_opt[j]))
            e=m/(price*(1+m**2))*(C-C_opt[i]+price*E/m+price*E_opt[i]*m)
            c= price*m*(e-E_opt[i])+C_opt[i]
            Error_E=abs(e-E)
            Error_C=abs(c-C) 

    else:      
        Error_E=abs(E_opt[i]-E)
        Error_C=abs(C_opt[i]-C)
        
    return (Error_E,Error_C)
    
""" 
This function computes and plots the POF of the POZ problem with e-constraint method
It computes also the maximal error with minimal distance
"""    
def ErrorPOZ():
    N=6
    it=100
    (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    (Zones,Units)=zones(N)    
    
    D=2*D/3
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma, D)
    
    (Emax, C, Pk)=SQP_MINLP(N,0,1,D)
    (Emin, Cmin, Pk)=SQP_MINLP(N,price,1,D)
    (E, Cmax, Pk)=SQP_MINLP(N,1,0,D)
    t0=time.time()
    LimE=np.linspace(Emin,Emax, num=it)
    E1=np.zeros(it)
    C1=np.zeros(it)
    for i in range(it) : 
        (E_i,C_i, Pk)= eConstE_SQP(N,LimE[i],D)
        C1[i]=C_i
        E1[i]=E_i
    
    print("OK LimE")
    print(time.time()-t0)
    
    LimF=np.linspace(Cmin,Cmax ,num=it)
    LimC1=np.zeros(it)
    LimE1=np.zeros(it)   
    for i in range(it) : 
        (E_i,C_i, Pk)= eConstF_SQP(N,LimF[i],D)
        LimC1[i]=C_i
        LimE1[i]=E_i 
    
    C=np.hstack((LimC1[::-1],C1) ) 
    E=np.hstack((LimE1[::-1], E1) )
    
    plt.figure()
    plt.plot(E[:-10],C[:-10],'g')
    plt.plot(E[-10:],C[-10:],'g')
    
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Pareto-optimal front')
    plt.grid()

    
    """Error analysis"""
    w= np.linspace(0.001, 0.999 , num=it)
    CnoPOZ= np.zeros(it)
    EnoPOZ= np.zeros(it)
    for i in range(it) : # POF of problem without POZ
        w_E=price*w[i]
        w_C=1-w[i]
        (Ek, Ck, Pk)=SQP(N,w_E,w_C,D)
        CnoPOZ[i]=Ck.copy()
        EnoPOZ[i]=Ek.copy()
        
    [e,spline]=Hermite(EnoPOZ,CnoPOZ, price*w, 1-w )
    
    plt.figure()
    plt.plot(E,C,'g.')
    plt.plot(e,spline , 'm', label='Convex front') 
    
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Pareto-optimal front')
    plt.grid()
    plt.legend()
    

    # Error computation following normalized minimum distance
    ErrorEmiss=0
    ErrorCost=0       
    E_emax=0
    C_emax=0
   
    E_cmax=0
    C_cmax=0
    for  i in range(it):
        (E_e,E_c)=mindist(E,C, EnoPOZ[i],CnoPOZ[i])
    
        if E_e>ErrorEmiss: # Maximal error on the emissions
            E_emax=EnoPOZ[i]
            C_emax=CnoPOZ[i]
        if E_c>ErrorCost:
            E_cmax=EnoPOZ[i]
            C_cmax=CnoPOZ[i]
            
        ErrorEmiss=max(ErrorEmiss, E_e)
        ErrorCost=max(ErrorCost, E_c)
    
    print("Maximal difference of POZ : ")   
    print("Emissions: ", ErrorEmiss)   
    print("Costs: ", ErrorCost)   
    plt.plot(E_emax,C_emax, 'bo', label='Max error on E')
    plt.plot(E_cmax, C_cmax, 'ko', label='Max error on C')

    
""" 
This function computes and plots the sub-POF associated to all operating sets at the global POF
"""   
def multiPOZ():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    N=6
    it=100
    w= np.linspace(0.001, 0.999 , num=it)
    (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    (Zones,Units)=zones(N)     
    D=2*D/3
    
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma, D)
    plt.figure() 
    CnoPOZ= np.zeros(it)
    EnoPOZ= np.zeros(it)
    for i in range(it) : 
        w_E=price*w[i]
        w_C=1-w[i]
        (Ek, Ck, Pk)=SQP(N,w_E,w_C,D)
        CnoPOZ[i] = Ck.copy()
        EnoPOZ[i] = Ek.copy()
        
        
    [e,spline]=Hermite(EnoPOZ,CnoPOZ, price*w, 1-w )
    plt.plot(e,spline , 'k', label='Convex front') 
     
    (E,C,Pk) = SQP_MINLP(N,1,0,D)
    (LowerB,UpperB)= InZone(Pk,Pmin,Pmax,Zones, Units)
    
    (E,C,Pk) = SQP_MINLP(N,0,1,D)
    (ZoneL,ZoneU)= InZone(Pk,Pmin,Pmax,Zones, Units)
    
    status=0
    count = 0
    nZones=N
    while(count<=nZones and status==0):
        C= np.zeros(it)
        E= np.zeros(it)
        p = SimplePriceFun(ZoneL,ZoneU,a,b,c,alpha,beta,gamma, D)
        for i in range(it) :
            w_E=p*w[i]
            w_C=1-w[i]
            (E_i,C_i, Pk)=SQP(N,w_E,w_C,D,ZoneL, ZoneU)
            C[i]=C_i
            E[i]=E_i
        plt.plot(E,C,color=colors[count], label="Front for zone"+str(count+1))   
        if (np.prod(ZoneL==LowerB)==1):
            status=1
        if (p<price):   
            (emission,cost,P)=eConstE_SQP(N,min(E),D)
        else:
            (emission,cost,P)=eConstF_SQP(N,max(C),D)
            
        prec=ZoneL   
        (ZoneL,ZoneU)= InZone(P,Pmin,Pmax,Zones, Units)
        if (sum(prec==ZoneL)<N-1):
            print("hidden zone")
            print(prec)
            print(ZoneL)
        count=count+1
    
    if status==0:
        print("Not covered all zones")
    
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Pareto-optimal front')
    plt.grid()
    plt.legend()
