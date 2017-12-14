# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 9:12:00 2016

@author: benhills
"""

""" This is Neumann's solution to the classic stefan problem.
It is solved analytically following Carslaw and Jaeger."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf,erfc
from scipy.optimize import fsolve

### Physical Constants ###
spy  = 60.*60.*24.*365.24           #Seconds per year
### (Van der Veen 2013) ###
# pg 144 Van der Veen - from Yen CRREL 1981
Lf = 3.335e5                        #Latent heat of fusion (J/kg)
rho = 1000.                        #Bulk density of water (kg/m3), density changes are ignored
Ks = spy*2.1                          #Conductivity of ice (J/mKs)
cs = 2097.                        #Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
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

xs = np.arange(0,1.01,0.01)
ts = np.arange(0,0.101,0.001)
X,time = np.meshgrid(xs,ts)
T = Ts(lam,X,time)
T[X>MeltLoc(lam,time)] = 0.0

plt.contourf(time,X,T,levels=np.arange(-10,2,0.2),cmap='RdYlBu_r')
plt.colorbar(label='Temperature $^\circ C$')
plt.plot(ts,MeltLoc(lam,ts),'k',lw=2)

plt.fill_between(ts,MeltLoc(lam,ts),y2=100.,color='w')
plt.ylim(0,0.7)
plt.xlim(0,0.1)

plt.ylabel('meters')
plt.xlabel('years')
#plt.title('Freezing Water $T_0 = 1^\circ C$')
plt.savefig('Analytic')

#lams = np.arange(0,2,0.01)
#plt.plot(lams,Eq14(lams))