"""

This File solves the static EED problem with Transmission Losses and Prohibited operating Zones.
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
from NLSolverStat import Solve,SQP

PI=math.pi

""" Solves one problem with POZ given the scalarized objective: f(p)=w_E*E(p)+w_C*C(p) and demand D"""
def SQP_MINLP(N,w_E,w_C,D):
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    (Zones,Units)=zones(N)    
    """Computing P0 without POZ"""    
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
    
    if (N<=10):
        const=[{'type': 'eq', 'fun': cons_f, 'jac':cons_J}]
        solution = minimize(objective ,P0, method='SLSQP',jac=Gradient, bounds=bnds,constraints=const)
    else: 
        
        def cons_H(P,v):
            return(-2*v*B)
        NL_const = NonlinearConstraint(cons_f, 0, 0, jac=cons_J, hess=cons_H)
        solution = minimize(objective ,P0, method='trust-constr',jac=Gradient,
                            hess=Hessian,constraints=NL_const, bounds=bnds)
    P0 = solution.x
    tol=1e-2
    Maxiter=100
    Obj=np.zeros(Maxiter)
    C = sum(a[k]+b[k]*P0[k]+c[k]*P0[k]*P0[k] for k in range(N))
    E = sum(alpha[k]+beta[k]*P0[k]+gamma[k]*P0[k]*P0[k]+eta[k]*np.exp(delta[k]*P0[k]) for k in range(N))
    Obj[0]=w_C*C+w_E*E
    Pk=P0.copy()
    Prev=P0.copy()
    it=1    
    stepsize=1
    while (it<Maxiter  and stepsize>tol):      
        model=gp.Model('SQP Step, MIQP')
        model.setParam( 'OutputFlag', False )
        DeltaP = model.addVars(range(N),lb=Pmin-Pk,ub=Pmax-Pk)
        
        Surplus=sum(Pk)-Pk@B@Pk-D
        model.addConstr(Surplus+sum(DeltaP[k]*(1-2*Pk@B[k]) for k in range(N))>=0)          
        for i in range(N): # POZ
            Zon_i=Zones[Units==i]
            n_i=len(Zon_i)
            if n_i>=1:
                bb=model.addVars(range(n_i+1), vtype=GRB.BINARY)
                model.addConstr(DeltaP[i]<=Zon_i[0,0]*bb[0]+(1-bb[0])*Pmax[i]-Pk[i])
                for j in np.arange(1,n_i):
                    model.addConstr(DeltaP[i]>=Zon_i[j-1,1]*bb[j]-Pk[i])
                    model.addConstr(DeltaP[i]<=Zon_i[j,0]*bb[j]+(1-bb[j])*Pmax[i]-Pk[i])        
                model.addConstr(DeltaP[i]>=Zon_i[-1,1]*bb[n_i]-Pk[i])
                model.addConstr(bb.sum()==1)     
          
        GradC=b+c*Pk*2
        GradE= beta+gamma*Pk*2+delta*eta*np.exp(delta*Pk)
        Grad=w_C*GradC+w_E*GradE
        Hessian= w_C*2*c+w_E*(2*gamma+delta*delta*eta*np.exp(delta*Pk))
        Lagr=sum(DeltaP[k]*DeltaP[k]*Hessian[k] for k in range(N))
        objec = sum(Grad[k]*DeltaP[k] for k in range(N)) + 0.5*Lagr
        model.setObjective(objec)
        model.optimize()
        PPrev=Prev.copy()
        Prev=Pk.copy()    
        for i in range(N):
            Pk[i] = Pk[i] + DeltaP[i].x  
        
        stepsize=np.linalg.norm(Prev-Pk)
   
        if np.linalg.norm(PPrev-Pk)<tol:  # The algorithm cycles between 2 zones
            res=sum(Pk)-Pk@B@Pk-D
            resPrev=sum(Prev)-Prev@B@Prev-D    
            if (resPrev==max(res,resPrev) and res<=0):
                Pk=Prev.copy()

            (opt,p)=Solve(N,w_E,w_C,D)
            stepsize=-1          
            
        C = sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
        E = sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
        Obj[it]=w_C*C+w_E*E
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1
    return(E,C,Pk)
     
# This function returns the active operating zone for a given production dispatch Pk
def InZone(Pk,Pmin,Pmax,Zones, Units):
    N=len(Pk)
    LowerB=np.copy(Pmin)
    UpperB=np.copy(Pmax)
    for i in range(N):
        Zon_i=Zones[Units==i]
        ni=len(Zon_i)
        for j in range(ni):
            if Pk[i]<=Zon_i[j,0]:
                UpperB[i]=Zon_i[j,0]
                break
            LowerB[i]=Zon_i[j,1]
    return(LowerB,UpperB)
    
#e-constraint on the fuel costs with limit=LimF 
def eConstF_SQP(N,LimF,D):
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    (Zones,Units)=zones(N)    
    """Computing P0 without POZ"""    
    bnds=np.transpose(np.vstack((Pmin,Pmax)))
    def P0_Obj(P):
        return (0)
    def P0_Grad(P):
        return(np.zeros(N))  
        
    def Objective(P):
        Obj = sum(alpha[i]+beta[i]*P[i]+gamma[i]*P[i]*P[i] +eta[i]*np.exp(P[i]*delta[i]) for i in range(N))
        return (Obj)
    def Gradient(P):
        Grad= beta+2*gamma*P+delta*eta*np.exp(delta*P)
        return(Grad)   
        
    def cons_f(P):
        PL=sum(sum(P[i]*P[j]*B[i,j] for j in range(N)) for i in range(N))
        sum_eq=sum(P)-PL-D
        return (sum_eq)
    def cons_J(P):
        Jac=np.ones(N)-2*P@B
        return(Jac)
    
    def cons_C(P):
        constraint= LimF-sum(a[k]+b[k]*P[k]+c[k]*P[k]*P[k] for k in range(N))
        return (constraint) 
    def cons_GradC(P):
        Grad=-b-2*c*P
        return(Grad)
        
    const=[{'type': 'eq', 'fun': cons_f}, {'type': 'ineq', 'fun': cons_C}]      
    solution = minimize(P0_Obj ,Pmin, method='SLSQP',jac=P0_Grad, bounds=bnds,constraints=const)
    P0 = solution.x
    tol=1e-2
    Maxiter=100
    Obj=np.zeros(Maxiter)
    Obj[0]=sum(alpha[k]+beta[k]*P0[k]+gamma[k]*P0[k]*P0[k]+eta[k]*np.exp(delta[k]*P0[k]) for k in range(N))
    
    Pk=P0.copy()
    Prev=P0.copy()
    it=1    
    stepsize=1
    while (it<Maxiter  and stepsize>tol):      
        model=gp.Model('SQP Step, MILP')
        model.setParam( 'OutputFlag', False )
        DeltaP = model.addVars(range(N),lb=Pmin-Pk,ub=Pmax-Pk)
        Surplus=sum(Pk)-Pk@B@Pk-D
        model.addConstr(Surplus+sum(DeltaP[k]*(1-2*Pk@B[k]) for k in range(N))>=0)

        model.addQConstr(LimF- sum(a[k]+b[k]*(Pk[k]+DeltaP[k])+c[k]*(Pk[k]+DeltaP[k])*(Pk[k]+DeltaP[k]) for k in range(N))>=0 )

        for i in range(N): # POZ
            Zon_i=Zones[Units==i]
            n_i=len(Zon_i)
            if n_i>=1:
                bb=model.addVars(range(n_i+1), vtype=GRB.BINARY)
                model.addConstr(DeltaP[i]<=Zon_i[0,0]*bb[0]+(1-bb[0])*Pmax[i]-Pk[i])
                for j in np.arange(1,n_i):
                    model.addConstr(DeltaP[i]>=Zon_i[j-1,1]*bb[j]-Pk[i])
                    model.addConstr(DeltaP[i]<=Zon_i[j,0]*bb[j]+(1-bb[j])*Pmax[i]-Pk[i])        
                model.addConstr(DeltaP[i]>=Zon_i[-1,1]*bb[n_i]-Pk[i])
                model.addConstr(bb.sum()==1)     
          
        Grad= beta+gamma*Pk*2+delta*eta*np.exp(delta*Pk)
        Hessian= 2*gamma+delta*delta*eta*np.exp(delta*Pk)
        Lagr=sum(DeltaP[k]*DeltaP[k]*Hessian[k] for k in range(N))
        objec = sum(Grad[k]*DeltaP[k] for k in range(N)) + 0.5*Lagr
        model.setObjective(objec)
        model.optimize()
        PPrev=Prev.copy()
        Prev=Pk.copy()           
        
        for i in range(N):
            Pk[i] = Pk[i] + DeltaP[i].x  
        stepsize=np.linalg.norm(Prev-Pk)
        
        if np.linalg.norm(PPrev-Pk)<tol: # The algorithm cycles between 2 zones
            res=sum(Pk)-Pk@B@Pk-D
            resPrev=sum(Prev)-Prev@B@Prev-D 
            if (abs(res)<=1e-4 or abs(resPrev)<=1e-4):
                if (resPrev==max(res,resPrev) and res<=0):
                    Pk=Prev.copy()
                break 
            # Compares the best solution of the 2 zones
            (LowerB,UpperB)=InZone(Pk,Pmin,Pmax,Zones,Units)  #Current zone
            Limits=np.transpose(np.vstack((LowerB,UpperB)))
            sol= minimize(Objective ,Pk, method='SLSQP',jac=Gradient, bounds=Limits,constraints=const)
            Pk=sol.x
            obj= sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))

            (LowerB1,UpperB1)=InZone(Prev,Pmin,Pmax,Zones,Units)  #Previous zone
            obj1=Obj[it-1]
            if np.prod(LowerB==LowerB1)==0: #Different zone
                Limits=np.transpose(np.vstack((LowerB1,UpperB1)))
                sol1= minimize(Objective ,Prev, method='SLSQP',jac=Gradient, bounds=Limits,constraints=const)
                Prev=sol1.x
                obj1= sum(alpha[k]+beta[k]*Prev[k]+gamma[k]*Prev[k]*Prev[k]+eta[k]*np.exp(delta[k]*Prev[k]) for k in range(N))
                if obj1<obj:
                    Pk=Prev.copy()    
            stepsize=-1
            
        Obj[it]= sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1
    return(Obj[it-1],LimF-cons_C(Pk),Pk)

#e-constraint on the emssions with limit=LimE 

def eConstE_SQP(N,LimE,D):
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    (Zones,Units)=zones(N)    
    """Computing P0 without POZ"""    
    bnds=np.transpose(np.vstack((Pmin,Pmax)))
    def P0_Obj(P):
        return (0)
    def P0_Grad(P):
        return(np.zeros(N))  
        
    def Objective(P):
         Obj = sum(a[i]+b[i]*P[i]+c[i]*P[i]*P[i] for i in range(N))
         return (Obj)
    def Gradient(P):
        Grad=b+2*c*P
        return(Grad)   
          
    def cons_f(P):
        PL=sum(sum(P[i]*P[j]*B[i,j] for j in range(N)) for i in range(N))
        sum_eq=sum(P)-PL-D
        return (sum_eq)
    def cons_J(P):
        Jac=np.ones(N)-2*P@B
        return(Jac)
    
    def cons_C(P):
        constraint= LimE-sum(alpha[k]+beta[k]*P[k]+gamma[k]*P[k]*P[k]+eta[k]*np.exp(delta[k]*P[k]) for k in range(N))
        return (constraint) 
    def cons_GradC(P):
        Grad=-beta-2*gamma*P -delta*eta*np.exp(delta*P)
        return(Grad)
        
    const=[{'type': 'eq', 'fun': cons_f}, {'type': 'ineq', 'fun': cons_C}]      
    solution = minimize(P0_Obj ,Pmin, method='SLSQP',jac=P0_Grad, bounds=bnds,constraints=const)
    P0 = solution.x
    tol=1e-2
    Maxiter=100
    Obj=np.zeros(Maxiter)
    Obj[0]= sum(a[k]+b[k]*P0[k]+c[k]*P0[k]*P0[k] for k in range(N))
    
    Pk=P0.copy()
    Prev=P0.copy()
    it=1    
    stepsize=1
    while (it<Maxiter  and stepsize>tol):      
        model=gp.Model('SQP Step, MIQP')
        model.setParam( 'OutputFlag', False )
        DeltaP = model.addVars(range(N),lb=Pmin-Pk,ub=Pmax-Pk)
        x = model.addVars(N)
        y = model.addVars(N)     
        for i in range(N):
            model.addConstr(x[i]==delta[i]*DeltaP[i])
            model.addGenConstrExp(x[i], y[i]) 
            
        Surplus=sum(Pk)-Pk@B@Pk-D
        model.addConstr(Surplus+sum(DeltaP[k]*(1-2*Pk@B[k]) for k in range(N))==0)

        model.addQConstr(LimE- sum(alpha[k]+beta[k]*(Pk[k]+DeltaP[k])+gamma[k]*(Pk[k]+DeltaP[k])*(Pk[k]+DeltaP[k])
                        +eta[k]*np.exp(delta[k]*Pk[k])*y[k] for k in range(N))>=0 )
    
        for i in range(N): # POZ
            Zon_i=Zones[Units==i]
            n_i=len(Zon_i)
            if n_i>=1:
                bb=model.addVars(range(n_i+1), vtype=GRB.BINARY)
                model.addConstr(DeltaP[i]<=Zon_i[0,0]*bb[0]+(1-bb[0])*Pmax[i]-Pk[i])
                for j in np.arange(1,n_i):
                    model.addConstr(DeltaP[i]>=Zon_i[j-1,1]*bb[j]-Pk[i])
                    model.addConstr(DeltaP[i]<=Zon_i[j,0]*bb[j]+(1-bb[j])*Pmax[i]-Pk[i])        
                model.addConstr(DeltaP[i]>=Zon_i[-1,1]*bb[n_i]-Pk[i])
                model.addConstr(bb.sum()==1)     
         
        Grad=b+c*Pk*2
        Hessian=2*c
        Lagr=sum(DeltaP[k]*DeltaP[k]*Hessian[k] for k in range(N))
        objec = sum(Grad[k]*DeltaP[k] for k in range(N)) + 0.5*Lagr
        model.setObjective(objec)
        model.optimize()
        PPrev=Prev.copy()
        Prev=Pk.copy()               
        for i in range(N):
            Pk[i] = Pk[i] + DeltaP[i].x  
        stepsize=np.linalg.norm(Prev-Pk)
        
        if np.linalg.norm(PPrev-Pk)<tol:
            res=sum(Pk)-Pk@B@Pk-D
            resPrev=sum(Prev)-Prev@B@Prev-D 
            if (abs(res)<=1e-4 or abs(resPrev)<=1e-4):
                if (resPrev==max(res,resPrev) and res<=0):
                    Pk=Prev.copy()
                break
            
            (LowerB,UpperB)=InZone(Pk,Pmin,Pmax,Zones,Units) 
            Limits=np.transpose(np.vstack((LowerB,UpperB)))
            sol= minimize(Objective ,Pk, method='SLSQP',jac=Gradient, bounds=Limits,constraints=const)
            Pk=sol.x
            obj= sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))

            (LowerB1,UpperB1)=InZone(Prev,Pmin,Pmax,Zones,Units) 
            obj1=Obj[it-1]
            if np.prod(LowerB==LowerB1)==0:
                Limits=np.transpose(np.vstack((LowerB1,UpperB1)))
                sol1= minimize(Objective ,Prev, method='SLSQP',jac=Gradient, bounds=Limits,constraints=const)
                Prev=sol1.x
                obj1= sum(a[k]+b[k]*Prev[k]+c[k]*Prev[k]*Prev[k] for k in range(N))
                if obj1<obj:
                    Pk=Prev.copy()
            stepsize=-1
            
            
        Obj[it]= sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1

    """Figures"""
    opt=Obj[it-1]+1e-6
    plt.figure()

    Pos=Obj[:it]-np.ones(it)*opt
    Neg=-Pos.copy()
    
    Pos=(Obj[:it]-np.ones(it)*opt>0)*Pos
    Neg=(Obj[:it]-np.ones(it)*opt<0)*Neg
    plt.plot(range(it),Pos, label='Positive Part ')
    plt.plot(range(it),Neg, label='Negative Part ')

    plt.xlabel('Iterations')
    plt.ylabel('$f_k-f*$')
    plt.title("Rate of convergence of eConstE ")
    plt.legend()
    plt.grid()
    
    return(LimE-cons_C(Pk),Obj[it-1],Pk)    
    
    
# Function displays the Front obtained with scalalarization method for the 6-unit problem 
# It displays also the outputs related to each problem and highlights the active POZ
def figures(): 
    N=6
    it=100
    (D,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    D=2*D/3
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma, D)
   
    C= np.zeros(it)
    E= np.zeros(it)
    CnoPOZ= np.zeros(it)
    EnoPOZ= np.zeros(it)
    w= np.linspace(0.001, 0.999 , num=it)
    
    Power=np.zeros([it,N])
    for i in range(it) : 
        w_E=price*w[i]
        w_C=1-w[i]
        #POZ
        (E_i,C_i, Pk)=SQP_MINLP(N,w_E,w_C,D)
        Power[i]=Pk.copy()
        C[i]=C_i
        E[i]=E_i
        #NoPOZ
        (Ek,Ck, Pk)=SQP(N,w_E,w_C,D)
        CnoPOZ[i] = Ck.copy()
        EnoPOZ[i] = Ek.copy()
     
    plt.figure()
    plt.plot(E,C,'b.',label='Front with POZ') 
    
    [e,spline]=Hermite(EnoPOZ,CnoPOZ, price*w, 1-w )
    plt.plot(e,spline , 'm', label='Convex front') 
  
    plt.xlabel('Emission [lb]')
    plt.ylabel('Cost [$]')
    plt.title('Pareto-optimal front')
    plt.grid()
    plt.legend()
    
    #Plot the outputs and the corresponding POZ    
    [Zones,Units]=zones(N)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    for i in range(N):   
        plt.plot(range(it),Power[:,i], label='P'+str(i+1), color=colors[i]) 
        Zon_i=Zones[Units==i]
        ni=len(Zon_i)
        for j in range(ni):
            cond1= abs(Power[:,i]-Zon_i[j,0])<1e-2
            cond1=np.append(cond1,False)
            xMin=np.argmax(cond1)
            xMax=np.argmax(1-cond1[xMin:])+xMin
            if xMin>0 or cond1[0]:
                xMin=xMin-1
                xMax=xMax+1
                plt.hlines(y=Zon_i[j,0], xmin=0, xmax=it,color=colors[i], linestyle='--', linewidth=2.0)
                plt.hlines(y=Zon_i[j,1], xmin=0, xmax=it,color=colors[i], linestyle='--',  linewidth=2.0)
    
            cond2= abs(Power[:,i]-Zon_i[j,1])<1e-2
            cond2=np.append(cond2,False)
            xMin=np.argmax(cond2)
            xMax=np.argmax(1-cond2[xMin:])+xMin
            if xMin>0 or cond2[0]:
                xMin=xMin-1
                xMax=xMax+1
                plt.hlines(y=Zon_i[j,0], xmin=0, xmax=it,color=colors[i], linestyle='--', linewidth=2.0)
                plt.hlines(y=Zon_i[j,1], xmin=0, xmax=it,color=colors[i], linestyle='--',  linewidth=2.0)
    
    plt.xlabel('min $f= \pi wE(P)+(1-w)F(P)$')
    plt.ylabel('Power [MW]')
    plt.title('Output generation with different objectives and active POZ')
    plt.legend()
