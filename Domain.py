"""
This file analyzes the nonconvex domain for a simplified model of 2 and 3 generators

Author: Felix Morel
Date : 09/06/2020

"""



import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Params import load,loss
from Static_model import SimplePriceFun 

from Relaxed import LinUpperB,LinearRelaxation
from LinApproxLVH import get_approx_planes
from NLSolverStat import Solve
from Grad import SolveGurobi

PI=math.pi

"""
The function displays all figures required for the domain analysis of the 2 or 3-unit test case
Choose between:
    N=2 or 3
"""
def figures(N):
    plt.close("all")
    
    (d,Pmax,Pmin,a,b,c,alpha,beta,gamma,delta,eta,UR,DR) = load(N)
    B=loss(N)
    M=np.zeros([N+1,N+1])
    M[:-1,:-1]=B
    M[-1,:-1]=-np.ones(N)*0.5
    M[:-1,-1] = -np.ones(N)*0.5
    M[-1,-1] = d
    [l,v]=np.linalg.eig(B)
    samples=10**(5-N)
    xc=-np.linalg.inv(B).dot(M[:-1,-1])
    center=np.transpose(np.repeat([xc],samples, axis=0))
    detB=np.linalg.det(B)
    detM=np.linalg.det(M)
    
    rad=np.zeros(N)
    for i in range(N):   
        v[i]=v[i]/np.linalg.norm(v[i])
        rad[i]=np.sqrt(-detM/(l[i]*detB))    
        
    if N==2:
        """Plot the entire ellipse """
        theta=np.linspace(0,2*PI,num=samples)
        ellipse=np.array([rad[0]*np.sin(theta), rad[1]*np.cos(theta)])
        P=v.dot(ellipse)+center 
        plt.figure()
        plt.plot(P[0],P[1])
        plt.xlabel('$P_1$')
        plt.ylabel('$P_2$')
        plt.title('Domain')
        plt.grid()
        """ end """
        
        theta=np.linspace(101*PI/80,51*PI/40,num=samples)
        ellipse=np.array([rad[0]*np.sin(theta), rad[1]*np.cos(theta)])
        P=v.dot(ellipse)+center   
        
        plt.figure()
        plt.plot(P[0],P[1], label='Ellipsoid')
        plt.vlines(x=Pmin[0], ymin=Pmin[1], ymax=Pmax[1], linewidth=2.0)
        plt.vlines(x=Pmax[0], ymin=Pmin[1], ymax=Pmax[1], linewidth=2.0)       
        plt.hlines(y=Pmin[1], xmin=Pmin[0], xmax=Pmax[0], linewidth=2.0)
        plt.hlines(y=Pmax[1], xmin=Pmin[0], xmax=Pmax[0], linewidth=2.0)  
        plt.xlabel('$P_1$')
        plt.ylabel('$P_2$')
        plt.title('Domain')
        plt.grid()
        
        plt.figure(3)
        plt.xlabel('Emission [lb]')
        plt.ylabel('Cost [$]')
        plt.title('Achievable domain of the objective')
        plt.grid()

        box=np.zeros(N)
        index=0
        for i in range(samples):
            const=np.zeros(2*N)
            const[:N]=P[:,i]-Pmin
            const[N:]=Pmax-P[:,i]         
        
            if all(const>=0):                    
                Cost = sum(a[k]+b[k]*P[k,i]+c[k]*P[k,i]*P[k,i] for k in range(N))
                Emission = sum(alpha[k]+beta[k]*P[k,i]+gamma[k]*P[k,i]*P[k,i] for k in range(N))
                plt.figure(3)
                plt.plot(Emission,Cost,'b.',label=str(i))
                
                if index==0:
                    box= P[:,i-1]
                    index=i                 
                box=np.vstack((box,P[:,i]))
        box=np.vstack((box,P[:,index+len(box)]))
        box=box.T
       
        (n,k,p)=LinUpperB(Pmin,Pmax,B,d)
        
        x=np.linspace(min(p[:,0]),Pmax[0])
        y=-(n[0]*x+k)/n[1]
    
        it=10
        P_relax=np.zeros([N,it])
        price= SimplePriceFun(Pmin,Pmax,a,b,c,alpha,beta,gamma,d)
        w= np.linspace(0, 1 , num=it)
        for i in range(it) : 
            w_E=price*w[i]
            w_C=1-w[i]
            [opt,P_opt]=SolveGurobi(N,w_E,w_C, d, method="ConvexRelax")
            
            P_relax[:,i]=P_opt.copy()
        
        plt.figure()
        plt.plot(box[0],box[1],label="Demand constraint")
        plt.plot(x,y, label = "Linear adjustment")    
        plt.plot(P_relax[0],P_relax[1],'ro', label='Relaxed Solutions')

        
        plt.xlabel('$P_1$')
        plt.ylabel('$P_2$')
        plt.title('Relaxed convex set')
        plt.grid()
        plt.legend()
        
    elif N==3:
        
        theta=np.linspace(2.32,2.34,num=samples)
        phi=np.linspace(4.92,4.935,num=samples)
        
        ellipse=np.zeros([N,samples,samples])
        ellipse[0]= rad[0]*np.outer(np.sin(theta),np.sin(phi))
        ellipse[1]= rad[1]*np.outer(np.cos(theta),np.sin(phi))
        ellipse[2]= rad[2]*np.outer(np.ones(samples),np.cos(phi))
        
        P=np.zeros([N,samples,samples])
        for i in range(samples):  
            P[:,:,i]=v.dot(ellipse[:,:,i])+center

        plt.figure(1)
        plt.xlabel('Emission [lb]')
        plt.ylabel('Cost [$]')
        plt.title('Achievable domain of the objective')
        plt.grid()
        box=np.zeros([N,samples,samples])
        for i in range(samples):
            for j in range(samples):
                const=np.zeros(2*N)
                const[:N]=P[:,i,j]-Pmin
                const[N:]=Pmax-P[:,i,j]
                     
                if all(const>=0):
                    Cost = sum(a[k]+b[k]*P[k,i,j]+c[k]*P[k,i,j]*P[k,i,j] for k in range(N))
                    Emission = sum(alpha[k]+beta[k]*P[k,i,j]+gamma[k]*P[k,i,j]*P[k,i,j] for k in range(N))
                    plt.figure(1)
                    plt.plot(Emission,Cost,'b.')
                    box[:,i,j]= P[:,i,j]
                    P[:,i,j]=np.zeros(N) #In order to display in red the box points
                    
        
    
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x=np.linspace(Pmin[0],Pmax[0])
        y=np.linspace(Pmin[1],Pmax[1])
        z=np.linspace(Pmin[2],Pmax[2])
        
        [Yz,Zy]=np.meshgrid(y,z)
        [Xz,Zx]=np.meshgrid(x,z)
        [Xy,Yx]=np.meshgrid(x,y)
        m=50
        Xmin=Pmin[0]*np.ones([m,m])
        Xmax=Pmax[0]*np.ones([m,m])     
        Ymin=Pmin[1]*np.ones([m,m])
        Ymax=Pmax[1]*np.ones([m,m])   
        Zmin=Pmin[2]*np.ones([m,m])
        Zmax=Pmax[2]*np.ones([m,m])    
        ax.plot_wireframe(Xmax,Yz,Zy,rstride=5, cstride=5)
        ax.plot_wireframe(Xmin,Yz,Zy,rstride=5, cstride=5)
        ax.plot_wireframe(Xz,Ymax,Zx,rstride=5, cstride=5)
        ax.plot_wireframe(Xy,Yx,Zmax,rstride=5, cstride=5)
    
        box=box[np.nonzero(box)]
        box=box.reshape(N,int(len(box)/N))
        
        ax.scatter(P[0,:,:], P[1,:,:], P[2,:,:],c='b', marker='o', s=0.5)
        ax.scatter(box[0],box[1], box[2], c='r',s=2)
        ax.set_xlabel('$P_1$')
        ax.set_ylabel('$P_2$')
        ax.set_zlabel('$P_3$')
        plt.grid()
        plt.title('Domain')
        
        
        
        """ Linear approximation """
        (n,k,p)=LinUpperB(Pmin,Pmax,B,d)

        xx=np.linspace(min(p[:,0]),Pmax[0])
        yy=np.linspace(min(p[:,1]),Pmax[1])
        
        [Xx,Yy]=np.meshgrid(xx,yy)    
        Z=-(n[0]*Xx+n[1]*Yy+k)/n[2]
         
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xx,Yy, Z, label="Linear approximation")
        ax.scatter(box[0],box[1], box[2], c='r',s=2, label= 'Domain of definition')

        plt.title('Linear Approximation of the domain')
        ax.set_xlabel('$P_1$')
        ax.set_ylabel('$P_2$')
        ax.set_zlabel('$P_3$')
        plt.grid()
    
    
        Dist=np.zeros(len(box[0]))
        for i in range(len(box[0])):
            Dist[i]=abs(box[:,i].dot(n)+k)/np.linalg.norm(n)
                    
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(box[0],box[1], Dist, c='b') 
        plt.title('Distance of the Linear Approximation')
        ax.set_xlabel('$P_1$')
        ax.set_ylabel('$P_2$')
        ax.set_zlabel('$Error$')
    
    
    
    """ Get the maximum error """
    model = gp.Model('Max distance')
    
    P = model.addVars(range(N),lb=Pmin,ub=Pmax, name='P')
    model.setParam('NonConvex', 2)
    PL = model.addVar()
    model.addQConstr(PL== sum( sum(P[i]*P[j]*B[i,j] for j in range(N))for i in range(N)))
    
    model.addConstr(P.sum()-PL == d, name='Demand')
    dist=-(sum(P[i]*n[i] for i in range(N))+k)/np.linalg.norm(n)
    model.setObjective(dist, GRB.MAXIMIZE)
    model.setParam( 'OutputFlag', False )
    model.optimize()
    
    print(dist.getValue()) #0.15661626890427272
    
            