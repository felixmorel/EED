"""

This file is dedicated to the static nonconvex problem taking into consideration the transmission losses B
It concerns the First order solvers

Author: Felix Morel
Date : 09/06/2020

"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt
from Params import load,loss
from Static_model import SimplePriceFun 
from Relaxed import LinUpperB,LinearRelaxation
from NLSolverStat import Solve
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

def SolveGurobi(N,w_E,w_C, Demand, method="ConvexRelax"):
    (Unused,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    
    m = gp.Model('Static Nonlinear EED with transmission losses')
    Pow = m.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    PLoss = m.addVar()
    x = m.addVars(N)
    y = m.addVars(N) 
    for n in range(N):
        m.addConstr(x[n]==delta[n]*Pow[n])
        m.addGenConstrExp(x[n], y[n])  
        
    if (method=="NonConvex"): 
        m.setParam('NonConvex', 2)
        m.addQConstr(PLoss== sum( sum(Pow[i]*Pow[j]*B[i][j] for j in range(N))for i in range(N)))
        m.addConstr(Pow.sum() == Demand+PLoss)
        
    elif (method=="ConvexRelax"):
        m.addQConstr(PLoss>= sum( sum(Pow[i]*Pow[j]*B[i][j] for j in range(N))for i in range(N)))
        m.addConstr(Pow.sum() == Demand+PLoss)
        
    elif (method=="LinRelax"):
        t=time.time()
        (n,k_upper, k_lower) = LinearRelaxation(N,Demand)
        print(time.time()-t,' sec to add for computing the extreme points')
        m.addConstr( sum(Pow[i]*n[i] for i in range(N))+k_upper<=0)
        m.addConstr( sum(Pow[i]*n[i] for i in range(N))+k_lower>=0)     
    
    Cost = sum(a[k]+b[k]*Pow[k]+c[k]*Pow[k]*Pow[k] for k in range(N))
    Emission = sum(alpha[k]+beta[k]*Pow[k]+gamma[k]*Pow[k]*Pow[k]+eta[k]*y[k] for k in range(N))
    
    obj= w_E*Emission + w_C*Cost 
    m.setObjective(obj)
    m.setParam( 'OutputFlag', False )
    m.optimize()
    
    opt=obj.getValue()
    P=np.zeros(N)
    for i in range(N):
        P[i] = Pow[i].x     
    return(opt,P)
    


"""
Choose between:
    N=2,3,6,10(,40,100)
    method='NonConvex', 'ConvexRelax'
    solver='Gurobi', 'Scipy'
"""
def GradMethod(N=10, method='ConvexRelax', solver='Gurobi'): 
    plt.close("all")

    (Demand,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    
    model=gp.Model('Projection Model')
    model.setParam( 'OutputFlag', False )
    P = model.addVars(range(N),lb=Pmin,ub=Pmax)
    PL = model.addVar()
    if (method=="NonConvex"): 
            model.setParam('NonConvex', 2)
            model.addQConstr(PL== sum( sum(P[i]*P[j]*B[i][j] for j in range(N))for i in range(N)))   
            model.addConstr(P.sum() == Demand+PL)  
            
    else:
        model.addQConstr(PL>= sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
        model.addConstr(P.sum()-PL == Demand, name='Demand')
   
    if (solver=='Gurobi'):
        t0=time.time() 
        [opt,P_opt]=SolveGurobi(N,price,1,Demand, method)
        t1=time.time()
        print(t1-t0 ,'sec for Gurobi')
        
        """Computing P0"""                    
        model.setObjective(0)
        model.optimize()
        t2=time.time()
        print(t2-t1 ,'P0')

        P0=np.zeros(N)
        for i in range(N):
            P0[i] = P[i].x 
    
    else:
        t0=time.time() 
        [opt,P_opt]=Solve(N,price,1,Demand) 
        t1=time.time()
        print(t1-t0 ,'sec for Scipy')
        
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
            sum_eq=sum(P)-PL-Demand
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
        t2=time.time()
        print(t2-t1 ,'P0')
        
    
    print()
    print("Gradient Method")
    tol=1e-2
    L=max(2*c+price*(2*gamma+delta*delta*eta*np.exp(delta*Pmax)))
    mu=min(2*c+price*(2*gamma+delta*delta*eta*np.exp(delta*Pmin)))

    Maxiter=int(0.25*(1+L/mu)*np.log(L*np.linalg.norm(P0-P_opt)**2/(2*tol)))+1
    Maxiter=min(Maxiter,50) #Otherwise too large vector of iterates

    Obj=np.zeros(Maxiter)
    C = sum(a[k]+b[k]*P0[k]+c[k]*P0[k]*P0[k] for k in range(N))
    E = sum(alpha[k]+beta[k]*P0[k]+gamma[k]*P0[k]*P0[k]+eta[k]*np.exp(delta[k]*P0[k]) for k in range(N))
    Obj[0]=C+price*E
    
    #Used if method=ConvexRelax
    GradRate=np.zeros(Maxiter)
    normP0=np.linalg.norm(P0-P_opt)**2
    GradRate[0]=L/2*normP0 

    Pk=P0.copy()
    it=1
    if (method=='NonConvex'):
        print(L,mu)
        h=1/L
    else:
        h=2/(mu+L)
    
    while (it<Maxiter and tol<Obj[it-1]-opt):
        
        GradC=b+c*Pk*2
        GradE= beta+gamma*Pk*2+delta*eta*np.exp(delta*Pk)
        Grad=GradC+price*GradE    
        Pk=Pk-h*Grad
 
        projection= sum((P[i]-Pk[i])*(P[i]-Pk[i]) for i in range(N))
        model.setObjective(projection)
        model.optimize()
        
        if model.Status!= GRB.OPTIMAL:
            print('Optimization was stopped with status ' + str(model.Status))
            
        for i in range(N):
            Pk[i] = P[i].x 
   
        C = sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
        E = sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
        
        Obj[it]=C+price*E    
        GradRate[it]=L/2*((L-mu)/(L+mu))**(2*it)*normP0
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1    
    
    plt.figure()
    if (method=='ConvexRelax'):
        plt.plot(range(it),GradRate[:it],'b--', label='Gradient theoretical rate')
        
    plt.plot(range(it),Obj[:it]-np.ones(it)*opt,'b', label='Gradient Method ')
    plt.xlabel('Iterations')
    plt.ylabel('$f_k-f*$')
    plt.title('Rate of convergence of the Gradient method ')
    plt.legend()
    plt.grid(True)   
    t3=time.time()    
    print(t3-t1, "for gradient")
    
 
"""
Choose between:
    N=2,3,6,10(,40,100)
    method='NonConvex', 'ConvexRelax'
    solver='Gurobi', 'Scipy'
"""
def AccMethod(N=10, method='ConvexRelax', solver='Gurobi'): 
    (Demand,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    
    model=gp.Model('Projection Model')
    model.setParam( 'OutputFlag', False )
    P = model.addVars(range(N),lb=Pmin,ub=Pmax)
    PL = model.addVar()
    if (method=="NonConvex"): 
            model.setParam('NonConvex', 2)
            model.addQConstr(PL== sum( sum(P[i]*P[j]*B[i][j] for j in range(N))for i in range(N)))   
            model.addConstr(P.sum() == Demand+PL)  
            
    elif (method=="ConvexRelax"):
        model.addQConstr(PL>= sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
        model.addConstr(P.sum()-PL == Demand, name='Demand')
    
    if (solver=='Gurobi'):
        t0=time.time() 
        [opt,P_opt]=SolveGurobi(N,price,1,Demand, method)
        t1=time.time()
        print(t1-t0 ,'sec for Gurobi')
        
        """Computing P0"""            
        model.setObjective(0)
        model.optimize()
        t2=time.time()     
        print(t2-t1 ,'P0')

        P0=np.zeros(N)
        for i in range(N):
            P0[i] = P[i].x   
    
    else:
        t0=time.time() 
        [opt,P_opt]=Solve(N,price,1,Demand) 
        t1=time.time()
        print(t1-t0 ,'sec for Scipy')
        
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
            sum_eq=sum(P)-PL-Demand
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
        t2=time.time()
        print(t2-t1 ,'P0')
    
    tol=1e-2
    L=max(2*c+price*(2*gamma+delta*delta*eta*np.exp(delta*Pmax)))
    mu=min(2*c+price*(2*gamma+delta*delta*eta*np.exp(delta*Pmin)))
    C = sum(a[k]+b[k]*P0[k]+c[k]*P0[k]*P0[k] for k in range(N))
    E = sum(alpha[k]+beta[k]*P0[k]+gamma[k]*P0[k]*P0[k]+eta[k]*np.exp(delta[k]*P0[k]) for k in range(N))
    f0= C+price*E
    Maxiter= int(np.sqrt(L/mu)*np.log(2*(f0-opt)/tol))+1
    Maxiter=min(Maxiter,50)
    Obj=np.zeros(Maxiter)
    Obj[0]=f0
    AccRate=np.zeros(Maxiter)
    AccRate[0]=2*(Obj[0]-opt)    
    
    print()
    print("Accelerated Gradient")

    it=1
    Pk=P0.copy()
    yk=Pk.copy()
    stepsize=(np.sqrt(L)-np.sqrt(mu))/(np.sqrt(L)+np.sqrt(mu))
    while (it<Maxiter and tol<Obj[it-1]-opt):      
        GradC=b+c*yk*2
        GradE= beta+gamma*yk*2+delta*eta*np.exp(delta*yk)
        Grad=GradC+price*GradE        
        Prev=Pk.copy()
        Pk=yk-Grad/L
        projection= sum((P[i]-Pk[i])*(P[i]-Pk[i]) for i in range(N))
        
        model.setObjective(projection)
        model.optimize()

        for i in range(N):
            Pk[i] = P[i].x              
        yk=Pk+stepsize*(Pk-Prev)       
        
        C = sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
        E = sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
        Obj[it]=C+price*E
        AccRate[it]=2*(1-np.sqrt(mu/L))**it*(Obj[0]-opt)
    
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1
    t3=time.time()
    print(t3-t1, "sec for Accelerated")
    
    plt.figure(1)
    if (method=='ConvexRelax'):        
        plt.plot(range(it),AccRate[:it], '--', color='orange', label= 'Accelerated gradient theoretical rate')       
    plt.plot(range(it),Obj[:it]-np.ones(it)*opt, color='orange', label='Accelerated Gradient Method')
    
    
    plt.title('Rate of convergence of the two methods')
    plt.xlabel('Iterations')
    plt.ylabel('$f_k-f*$')
    plt.legend()
    plt.grid(True)


"""
Choose between:
    N=2,3,6,10,40(,100)
    solver='Gurobi', 'Scipy'
"""
def SPG(N=10, solver="Gurobi"):
    (Demand,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    
    t0=time.time() 
    if (solver=='Gurobi'):
        [opt,P_opt]=SolveGurobi(N,price,1,Demand, 'ConvexRelax')
        t1=time.time()
        print(t1-t0 ,' sec for Gurobi')
        print(opt)
        #Gurobi does not provide an accurate solution
        [opt,P_opt]=Solve(N,price,1,Demand) 
        print(opt)
        
    else:   
        [opt,P_opt]=Solve(N,price,1,Demand) 
        t1=time.time()
        print(t1-t0 ,' sec for scipy')
    
    bnds=np.transpose(np.vstack((Pmin,Pmax)))
    P0=Pmin.copy()    
    def objective(P): return (0)
    def Gradient(P): return(np.zeros(N))
    def Hessian(P): return(np.zeros([N,N]))        
    def cons_f(P):
        PL=sum(sum(P[i]*P[j]*B[i,j] for j in range(N)) for i in range(N))
        sum_eq=sum(P)-PL-Demand
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
    t2=time.time()
    print(t2-t1 ,'P0')

    tol=1e-2
    L=max(2*c+price*(2*gamma+delta*delta*eta*np.exp(delta*Pmax)))
    C = sum(a[k]+b[k]*P0[k]+c[k]*P0[k]*P0[k] for k in range(N))
    E = sum(alpha[k]+beta[k]*P0[k]+gamma[k]*P0[k]*P0[k]+eta[k]*np.exp(delta[k]*P0[k]) for k in range(N))

    Maxiter=50
    Obj=np.zeros(Maxiter)
    Obj[0]=C+price*E
    
    print()
    print("Spectral Projected Gradient")
    model=gp.Model('Projection Model')
    P = model.addVars(range(N),lb=Pmin,ub=Pmax)
    PL = model.addVar()    
    model.addQConstr(PL>= sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
    model.addConstr(P.sum()-PL == Demand, name='Demand')   
    model.setParam( 'OutputFlag', False )

    it=1
    Pk=P0.copy()
    dk=np.zeros([N])
    stepmin=1e-10
    stepmax=1e10
    stepsize=1/L
    sigma1=0.1
    sigma2=0.9
    g=1e-4
    M=10
    while (it<Maxiter and tol<Obj[it-1]-opt):     
        GradC=b+c*Pk*2
        GradE= beta+gamma*Pk*2+delta*eta*np.exp(delta*Pk)
        Grad=GradC+price*GradE        
        Prev=Pk.copy()
        Pk=Pk-stepsize*Grad
        projection= sum((P[i]-Pk[i])*(P[i]-Pk[i]) for i in range(N)) 
        model.setObjective(projection)
        model.optimize()
        for i in range(N):
            dk[i] = P[i].x - Prev[i]     
        
        coeff=1
        index=max(0,it-M)
        fmax=max(Obj[index:it])
        Pk=Prev+coeff*dk
        C = sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
        E = sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
        fk=C+price*E
        while(fk>fmax+g*coeff*Grad@dk):
            
            Num=-0.5*coeff**2*Grad@dk
            Denom= fk-Obj[it-1]-coeff*Grad@dk
            temp=Num/Denom
            if (temp>=sigma1 and temp<=coeff*sigma2):
                coeff=temp         
            else:
                coeff=coeff/2
            
            Pk=Prev+coeff*dk
            C = sum(a[k]+b[k]*Pk[k]+c[k]*Pk[k]*Pk[k] for k in range(N))
            E = sum(alpha[k]+beta[k]*Pk[k]+gamma[k]*Pk[k]*Pk[k]+eta[k]*np.exp(delta[k]*Pk[k]) for k in range(N))
            fk=C+price*E
        
        sk=Pk-Prev
        yk=2*c*(Pk-Prev) + price*(2*gamma*(Pk-Prev)+ delta*eta*(np.exp(delta*Pk)-np.exp(delta*Prev)))
        if sk@yk<=0: stepsize=stepmax
        else: stepsize=max(stepmin,min(sk@sk/(sk@yk),stepmax))
        Obj[it]=fk
        
        if( (it % 10)==0):
            print(it, " of ", Maxiter)
        it=it+1
        
    t3=time.time()
    print(t3-t1, "sec for SPG")
    
    plt.figure()
    plt.plot(range(it),Obj[:it]-np.ones(it)*opt, label='SPG')
    plt.title('Rate of convergence of the Spectral Projected Gradient')
    plt.xlabel('Iterations')
    plt.ylabel('$f_k-f*$')
    plt.legend()
    plt.grid(True)

    