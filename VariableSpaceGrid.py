# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:00:45 2016

@author: ben
"""

""" This is Neumann's solution to the classic stefan problem.
This solution is what is called the "variable space grid.
I have followed the implementation in Kutluay et al., (1997)"""

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
s0 = 1.
t0 = 0.
Tm = 0.0
T_ = -10.0
Tr = -16.33                         #Reference Tempearature, set so that we can use cs = 2009 (Aschwanden 2012 eq.75)

#################################################################################

### Variable Space Grid Solution ###
ax3 = plt.subplot(111)
for N in [21,51,101,201]:
    #N = 20
    l = 2.
    
    dt = .05/365.#1000/spy
    t = 0.
    dx=s0/(N-1)
    xs = np.linspace(0,s0,N)
    r = ks*dt/(dx**2) 
    
    T = np.ones(N)*T_
    T[-1] = Tm
    
    s = s0
    
    ts = np.array([t])
    PTB = np.array([s0])
    
    start_time = time.time()
    while t < .1:    
        s_dot = Ks/(Lf*rho*2*dx)*(3*T[-1]-4*T[-2]+T[-3])
        s += s_dot*dt
    
        dx=s/(N-1)
        xs = np.linspace(0,s,N) 
        Nfix = len(xs[xs<s0])
    
        mu = dt*s_dot/(2*dx*s)*np.linspace(0,s,N)
        nu = ks*dt/(dx**2)  
        
        A = sparse.lil_matrix((N, N))
        A.setdiag((1+2*nu)*np.ones(N))              # The diagonal
        A.setdiag((-mu[:-1]-nu),k=1)       # The fist upward off-diagonal.
        A.setdiag((mu[1:]-nu),k=-1) 
        #Boundary Conditions
        for i in range(Nfix):
            A[i,:] = np.zeros(N) # zero out the row
            A[i,i] = 1.0       # set diagonal in that row to 1.0
        A[-1,:] = np.zeros(N)
        A[-1,-1] = 1.0
        # For performance, other sparse matrix formats are preferred.
        # Convert to "Compressed Row Format" CR
        A = A.tocsr()
        T = spsolve(A,T)
        
        t += dt
        #print t
        
        ts = np.append(ts,t)
        PTB = np.append(PTB,s)
        plt.plot(ts,PTB-s0,'r',lw=2,label='Variable Space Grid')

    print 'time = ', time.time() - start_time    
        
        #if dt <= (2.*dx**2)/(4.+(dx*s_dot)**2):
        #    print 'stable'
        #else:
        #    print 'unstable'
"""
ax2 = plt.subplot(121)
xs = np.linspace(0,s,N)
plt.plot(xs,T,'b')
plt.ylim(-11.,1.)

ax3 = plt.subplot(111)
plt.plot(ts,PTB-s0,'r',lw=2,label='Variable Space Grid')

#################################################################################

### Problem Solution from Sarler (1995) ###

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

ax2.plot(xs,Ts(lam,xs,ts[-1]),'k')
plt.plot(ts,MeltLoc(lam,ts)-s0,'k',lw=2,label='Analytic')

plt.ylabel('meters')
plt.xlabel('years')
#plt.ylim(0,0.7)
plt.xlim(0,.1)
plt.legend(loc=4)
plt.savefig('UnstableVSG.png')#"""