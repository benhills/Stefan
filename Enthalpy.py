# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:18:02 2016

@author: ben
"""

"""This solution is what is called the "enthalpy method"
see Voller and Cross 1981"""

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
s0 = 0.
t0 = 0.
Tm = 0.0
T_ = -10.0
Tr = -16.33                         #Reference Tempearature, set so that we can use cs = 2009 (Aschwanden 2012 eq.75)

#################################################################################

### Enthalpy Solution ###

# Test over several time step sizes
ax3 = plt.subplot(111)
for dt in [.001,.01,.05,.1,.2,.5]:
    N = 201
    l = 2.

    dt = dt/365.#1000/spy
    t = 0.
    dx=l/N
    xs = np.arange(0,l,dx)
    r = ks*dt/(dx**2)

    H = ((Tm-Tr)*cs+Lf)*np.ones(N)
    H[np.where(xs<=s0)] = (T_-Tr)*cs
    T = H/cs + Tr
    T[T>Tm]=Tm

    A = sparse.lil_matrix((N, N))
    A.setdiag((1+2*r)*np.ones(N))              # The diagonal
    A.setdiag(-r*np.ones(N-1),k=1)       # The fist upward off-diagonal.
    A.setdiag(-r*np.ones(N-1),k=-1)
    #Boundary Conditions
    A[0,:] = np.zeros(N) # zero out the first row
    A[0,0] = 1.0       # set diagonal in that row to 1.0
    #A[-1,:] = np.zeros(N)
    A[-1,-1] = -2.*r
    A[-1,-2] = 2.*r
    # For performance, other sparse matrix formats are preferred.
    # Convert to "Compressed Row Format" CR
    A = A.tocsr()

    ts = np.array([t])
    PTB = np.array([np.min(xs[H>(0.0-Tr)*cs])])
    Mushy = [np.min(xs[H>(((Tm-Tr)*cs+Lf)+Tr*cs)*.95])]

    while t < .1:
        dT = spsolve(A,T)-T
        H += dT*cs
        T = H/cs + Tr
        T[T>Tm]=Tm
        t += dt
        ts = np.append(ts,[t])
        PTB = np.append(PTB,[np.min(xs[H>(0.0-Tr)*cs])])
        Mushy = np.append(Mushy,[np.min(xs[H>(((Tm-Tr)*cs+Lf)+Tr*cs)*.95])])

    p1, = plt.plot(ts,PTB,'r',lw=2)
    #plt.fill_between(ts,PTB,Mushy,color='r',alpha=0.2)

# Test over several spatial step sizes
for N in [21,51,101,201]:
    #N = 201
    l = 2.

    dt = .01/365.#1000/spy
    t = 0.
    dx=l/N
    xs = np.arange(0,l,dx)
    r = ks*dt/(dx**2)

    H = ((Tm-Tr)*cs+Lf)*np.ones(N)
    H[np.where(xs<=s0)] = (T_-Tr)*cs
    T = H/cs + Tr
    T[T>Tm]=Tm

    A = sparse.lil_matrix((N, N))
    A.setdiag((1+2*r)*np.ones(N))              # The diagonal
    A.setdiag(-r*np.ones(N-1),k=1)       # The fist upward off-diagonal.
    A.setdiag(-r*np.ones(N-1),k=-1)
    #Boundary Conditions
    A[0,:] = np.zeros(N) # zero out the first row
    A[0,0] = 1.0       # set diagonal in that row to 1.0
    #A[-1,:] = np.zeros(N)
    A[-1,-1] = -2.*r
    A[-1,-2] = 2.*r
    # For performance, other sparse matrix formats are preferred.
    # Convert to "Compressed Row Format" CR
    A = A.tocsr()

    ts = np.array([t])
    PTB = np.array(xs[np.min(np.where(H>(0.0-Tr)*cs))])
    #Mushy = [np.min(xs[H>(((Tm-Tr)*cs+Lf)+Tr*cs)*.95])]
    start_time = time.time()
    while t < .1:
        dT = spsolve(A,T)-T
        H += dT*cs
        T = H/cs + Tr
        T[T>Tm]=Tm
        t += dt
        ts = np.append(ts,[t])
        PTB = np.append(PTB,xs[np.min(np.where(H>(0.0-Tr)*cs))])
        #Mushy = np.append(Mushy,[np.min(xs[H>(((Tm-Tr)*cs+Lf)+Tr*cs)*.95])])
    print 'time = ', time.time()-start_time
    #plt.subplot(131)
    #plt.plot(xs,H)
    #ax2 = plt.subplot(121)
    #plt.plot(xs,T,'b')
    #plt.ylim(-11.,1.)
    #plt.ylabel('Temperature ($^\circ C$)')
    #plt.xlabel('meters')
    print dx
    p2, = plt.plot(ts,PTB,'b',lw=2)
    #plt.fill_between(ts,PTB,Mushy,color='r',alpha=0.2)
    #p2 = plt.Rectangle((-1, -1), 1, 1, fc="r",alpha=0.2)

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

#ax2.plot(xs,Ts(lam,xs,ts[-1]),'k')
#p3, = plt.plot(ts,MeltLoc(lam,ts),'k',lw=2)

plt.ylabel('meters')
plt.xlabel('years')
plt.ylim(0,0.7)
plt.xlim(0,.1)
plt.legend([p1,p2],['dt','dx'],loc=2)
plt.savefig('dts_Enthalpy.png')
