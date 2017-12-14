# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:12:08 2016

@author: ben
"""

""" This is Neumann's solution to the classic stefan problem.
This solution is what is called the "Level Set Method."
I have followed the implementation in Chen et al., (1997)"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf,erfc
from scipy.optimize import fsolve
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

### Physical Constants ###
spy  = 60.*60.*24.*365.24           #Seconds per year
### (Van der Veen 2013) ###
# pg 144 Van der Veen - from Yen CRREL 1981
Lf = 3.335e5                        #Latent heat of fusion (J/kg)
rho = 1000.                        #Bulk density of water (kg/m3), density changes are ignored
Ks = spy*2.1                          #Conductivity of ice (J/mKs)
cs = 2009.                        #Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks/(rho*cs)                #Cold ice diffusivity (m2/sec)
# Engineering Toolbox
Kl = spy*0.58                       #Conductivity of water (J/mKs)
cl = 4217.                      #Heat capacity of water (J/kgK)
kl = Kl/(rho*cl)

# Problem Constants
PTB = np.array([0.0])    # phase transition boundary
t0 = 0.                 # start time
Tm = 0.0                # melting point
T_ = -10.0              # left boundary condition
Tr = -16.33                         #Reference Tempearature, set so that we can use cs = 2009 (Aschwanden 2012 eq.75)

#################################################################################

### Level Set Solution ###

for N in [21,51,101,201]:
    # Problem Initialization
    #N = 201
    l = 2.
    dx=l/(N-1)
    xs = np.linspace(0,l,N)
    dt = .01/365.
    t = t0
    
    # initialize the signed distance function based of the starting PTB
    phi = xs - PTB[0]
    # initialize the temperature function to T_ on the left and Tm everywhere that is liquid
    T = np.ones(N)*T_
    T[phi>0.] = Tm
    
    # Write matrices for calculating the heat flux, distance gradient and diffusive terms
    heat_flux = sparse.lil_matrix((N, N))
    heat_flux.setdiag(-(Ks/(dx))*np.ones(N))              # The diagonal
    heat_flux.setdiag((Ks/(dx))*np.ones(N-1),k=1)       # The fist upward off-diagonal.    
    heat_flux = heat_flux.tocsr()
    
    VN = ks*dt/(dx**2) 
    center = sparse.lil_matrix((N, N))
    center.setdiag((1+2*VN)*np.ones(N))        
    center.setdiag(-VN*np.ones(N),k=1)       
    center.setdiag(-VN*np.ones(N),k=-1) 
    center[0,1:] = 0.
    center[0,0] = 1.
    center[-1,:-1] = 0.
    center[-1,-1] = 1.
    
    # Write another matrix for reinitialization of the signed distance function
    # timestep, k, and diffusive term, kd, here are somewhat arbitrary. I just need to get phi back into a distance function
    k = .01
    kd = .03
    VN = k*kd/(dx**2)
    #print VN
    Re = sparse.lil_matrix((N, N))
    Re.setdiag((1+2*VN)*np.ones(N))              # The diagonal
    Re.setdiag(-VN*np.ones(N-1),k=1)       # The fist upward off-diagonal.
    Re.setdiag(-VN*np.ones(N-1),k=-1)     
    Re[0,1] = -2.*VN   
    Re[-1,-2] = -2.*VN   
    
    def reinit(phi):
        Se = phi/(np.sqrt(phi**2))
        d_phi = np.ones_like(phi)    
        count = 0.
        #plt.plot(xs,phi,'k',label='$\phi (x,0)$')
        while max(abs(d_phi)) > .01:
            d_phi = k*Se*(1.-abs(np.gradient(phi,dx)))
            phi += d_phi
            phi[0] -= k*kd*2/dx
            phi[-1] -= -k*kd*2/dx
            phi = spsolve(Re,phi)
            count += 1
            #plt.plot(xs,phi,'r')
        #plt.plot(xs,phi,'r',label='reinitialized')
        #print count
        return phi      
    
    start_time = time.time()
    ts = np.array([t])
    PTB = np.array([0.0])
    while t < 0.1:
        # Define the speed function
        grad_phi = np.gradient(phi,dx)
        speed = 1./(rho*Lf)*heat_flux*T*(grad_phi/(abs(grad_phi)))
        # update distance function for new time step
        phi -= dt*speed*abs(grad_phi)
        # reinitialize distance function
        phi = reinit(phi)
        T = spsolve(center,T)
        T[np.logical_and(phi>0.,T<0.)] = 0.
        t += dt
        #print t      
        
        ts = np.append(ts,t)
        PTB = np.append(PTB,[np.min(xs[T>=(-1e-2)])])
    
    print 'time = ', time.time() - start_time    
    
    plt.plot(ts,PTB,'r-',lw=2,label='Level Set Method')  
        
        
"""
grad_phi = np.gradient(phi,dx)
speed = 1./(rho*Lf)*heat_flux*T*(grad_phi/(abs(grad_phi)))
# update distance function for new time step
phi -= dt*speed*abs(grad_phi)
# reinitialize distance function
phi = reinit(phi)

plt.ylabel('meters from $\phi$')
plt.xlabel('meters')
plt.ylim(-.8,.5)
plt.xlim(0.2,0.8)
plt.legend(loc=2)
plt.savefig('Reinitialize.png')

#################################################################################

### Problem Solution from Sarler (1995) ###
"""
s0 = PTB[0]

# location of the phase boundary
def MeltLoc(lam,t):
    return s0 + 2*lam*(t-t0)**.5
# Constant Bs
def Bs(lam):
    return (Tm-T_)/(erf(lam*ks**(-.5)))
# Temperature in the solid
def Ts(lam,x,t):
    return T_ + Bs(lam)*erf((x-s0)/((4*ks*(t-t0))**.5))
# equation to solve for lambda, set this == 0
def Transcendental(lam):
    lhs = rho*Lf*lam
    rhs = -Ks*Bs(lam)*np.pi**(-.5)*np.exp(-lam**2*ks**(-1))*ks**(-.5)
    return lhs-rhs

# first I need to determine the constant lambda
lam = fsolve(Transcendental,1.)[0]

#ax2.plot(xs,Ts(lam,xs,ts[-1]),'k')
plt.plot(ts,MeltLoc(lam,ts),'k',lw=2,label='Analytic')

plt.ylabel('meters')
plt.xlabel('years')
plt.ylim(0,0.7)
plt.xlim(0,.1)
plt.legend(loc=2)
#plt.savefig('LevelSet.png')#"""