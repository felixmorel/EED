"""
This file is used for the Hermite approximation and error analysis with Hermite
Author: Felix Morel
Date : 09/06/2020

"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from Params import load
from Static_model import Simple, SimplePriceFun
      
    
def Piecewise_Hemite_approx(val,f, x, df) :
    h=x[1]-x[0]
    c0= f[0]
    c1= df[0]
    c2= 3*(f[1]-f[0])/h**2- (df[1]+2*df[0])/h
    c3= (df[1]+df[0])/h**2 - 2*(f[1]-f[0])/h**3
    y= c0 + c1*(val-x[0]) + c2*(val-x[0])**2 + c3*(val-x[0])**3
    return y

def InvHermite(y,f, x, df) :
    h=x[1]-x[0]
    c0= f[0]
    c1= df[0]
    c2= 3*(f[1]-f[0])/h**2- (df[1]+2*df[0])/h
    c3= (df[1]+df[0])/h**2 - 2*(f[1]-f[0])/h**3
    
    print(c3,c2,c1,c0)
    p = np.poly1d([c3,c2,c1,c0])
    print()
    vals=(p - y).roots
    
    val=h
    for v in vals: # elimnate negative and imaginary answers
        if (v>0 and v.imag<1e-5):
            val=min(val,v.real)
            
    return (val+x[0])

def Hermite(E,C, w1, w2 ):
    """ Hermite Approximation"""
    it=len(E)
    interval=1000
    contE=np.linspace(min(E), max(E) , num=interval)
    s= np.zeros(interval)
    for i in range(interval):
        index = [np.argmax(E<=contE[i]), np.argmax(E<=contE[i])-1]
        x=E[index]
        f=C[index]
        df = - w1[index]/w2[index]
        if index[0]==it-1:
            inf= (C[index[1]]-C[index[0]])/(E[index[1]]-E[index[0]])
            dy = inf - df[1]
            df[0]= inf + dy         
        s[i] = Piecewise_Hemite_approx(contE[i],f, x, df)         
    return(contE,s)
    

def HermiteError(e,c,E_H,C_H, w_E, w_C):
    C_error=0
    nPoints=len(E_H)
    for i in range(len(e)):
        if(e[i]>=max(E_H)):
            index=np.argmax(E_H)
            y= C_H[index]

        else: 
            n=np.argmax(E_H<=e[i])
            index=[n,n-1]
            df=- w_E[index]/w_C[index]
            if index[0]==nPoints-1:
                inf= (C_H[index[1]]-C_H[index[0]])/(E_H[index[1]]-E_H[index[0]])
                dy = inf - df[1]
                df[0]= inf + dy
                
            y=Piecewise_Hemite_approx(e[i],C_H[index], E_H[index],df )
        C_error=max(C_error,abs(c[i]-y)/c[i])
        
    E_error=0
    for i in range(len(c)):
        if(c[i]>=max(C_H)):
            index=np.argmax(C_H)
            x= E_H[index]
        else: 
            n=np.argmax(C_H>=c[i])
            index=[n,n-1]
            df=- w_E[index]/w_C[index]
            if index[0]==nPoints-1:
                inf= (C_H[index[1]]-C_H[index[0]])/(E_H[index[1]]-E_H[index[0]])
                dy = inf - df[1]
                df[0]= inf + dy
                
            x=InvHermite(c[i],C_H[index], E_H[index],df )
        E_error=max(E_error,abs(e[i]-x)/e[i])
    
    return(E_error,C_error)
    
def minDistance(P0,f, x, df):
    h=x[1]-x[0]
    p=np.zeros(4)
    p[3]= f[0]
    p[2]= df[0]
    p[1]= 3*(f[1]-f[0])/h**2- (df[1]+2*df[0])/h
    p[0]= (df[1]+df[0])/h**2 - 2*(f[1]-f[0])/h**3
    
    model = gp.Model('Min distance')
    E = model.addVar(lb=x[0],ub=x[1])
    Var = model.addVar()
    C = model.addVar()

    model.addConstr(Var==E-x[0])
    model.addGenConstrPoly(Var,C,p)
    dist=(E-P0[0])*(E-P0[0]) + (C-P0[1])*(C-P0[1])
    
    model.setObjective(dist)
    model.setParam( 'OutputFlag', False )
    model.optimize()
    
    return(E.x,C.x)    


def SmartHermiteError(e,c,E_H,C_H, w_E, w_C):
    C_error=0
    E_error=0
    nPoints=len(E_H)
    

    """
    plt.figure()
    plt.plot(e,c,'.', color='b', label='reference')
    [contE,s]=Hermite(e,c, w_E, w_C )
    plt.plot(contE,s , color='b') 
    plt.plot(E_H,C_H,'.', color='orange', label='reference')
    [contE,s]=Hermite(E_H,C_H, w_E, w_C )
    plt.plot(contE,s , color='orange')
    """
    
    price=(c[-1]-c[0])/(e[0]-e[-1])
    w_E=w_E/price
    e=e*price
    E_H=E_H*price  
    
    for i in range(len(e)):
        dfmax= (C_H[-1]-C_H[-2])/(E_H[-2]-E_H[-1])
        pk=  C_H[-1]-dfmax*E_H[-1]
        cond= c[i]- dfmax*e[i] - pk
    
    
        pk=  w_C[0]*E_H[0]-w_E[0]*C_H[0]
        cond2=w_C[0]*e[i]-w_E[0]*c[i]-pk
        if (cond>=0):
            E_e=abs(e[i]-E_H[-1])
            C_e=abs(c[i]-C_H[-1])
            #plt.plot(E_H[-1]/price,C_H[-1],'co')
    
    
        elif (cond2>=0):
            E_e=abs(e[i]-E_H[0])
            C_e=abs(c[i]-C_H[0])
            #plt.plot(E_H[0]/price,C_H[0],'mo')
            
        else:
            for k in range(len(E_H)-1):
                cond1=cond2                      
                pk=  w_C[k+1]*E_H[k+1]-w_E[k+1]*C_H[k+1]
                cond2=w_C[k+1]*e[i]-w_E[k+1]*c[i]-pk
                if np.sign(cond1)!=np.sign(cond2):
                    break
                
            index=[k+1,k]
            df=- w_E[index]/(w_C[index])
            if index[0]==nPoints-1:
                inf= (C_H[index[1]]-C_H[index[0]])/(E_H[index[1]]-E_H[index[0]])
                dy = inf - df[1]
                df[0]= inf + dy
            
            P0=np.array([e[i],c[i]])        
            (E_opt,C_opt)=minDistance(P0, C_H[index], E_H[index],df)
            #plt.plot(E_opt/price,C_opt,'ro')
            E_e=abs(e[i]-E_opt)
            C_e=abs(c[i]-C_opt)
            
        E_error=max(E_error,E_e/e[i])
        C_error=max(C_error, C_e/c[i])
    return(E_error,C_error)
    

def main(N, compare=False):
    (Demand,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
     
    it=5
    C= np.zeros(it)
    E= np.zeros(it)
    w= np.linspace(0.001, 0.999 , num=it)

    model = gp.Model('Quadratic Problem')
    Pow = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    
    model.addConstr(Pow.sum() == Demand, name='Demand')
    price = SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,Demand)
    Cost = sum(a[k]+b[k]*Pow[k]+c[k]*Pow[k]*Pow[k] for k in range(N))
    Emission = sum(alpha[k]+beta[k]*Pow[k]+gamma[k]*Pow[k]*Pow[k] for k in range(N))
    
    for i in range(it) : 
      
        obj= price*w[i]*Emission+ (1-w[i])*Cost 
        
        model.setObjective(obj)
        model.setParam( 'OutputFlag', False )
        model.optimize()
        
        C[i]=Cost.getValue()
        E[i]=Emission.getValue()
    
    
    [contE,s]=Hermite(E,C, price*w, 1-w )
    plt.figure()
    plt.plot(contE,s , label='Hermite approximation, using 5 points')
    
    if compare:
        [E1,C1]=Simple(N)
        plt.plot(E1,C1, label='Continious Pareto-optimal front')
        
        [E_e,C_e]=HermiteError(E1,C1,E,C, price*w, 1-w)
        print("Error on the costs:", C_e) 
        print("Error on the emissions:",E_e)
    
    plt.plot(E,C, 'ro',label='Pareto optimal points')
    plt.legend()
    plt.xlabel('Emission')
    plt.ylabel('Cost')
    plt.title('Piecewise Hermite approximation of the pareto-optimal front')