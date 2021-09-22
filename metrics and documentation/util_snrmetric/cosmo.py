# Reference: Hogg 1999 astro-ph/9905116v4
#
from numpy import *
from scipy.integrate import quad

def O_K():
    return 1-O_M-O_L

def D_H():                               #  Hubble Distance
    c=300000.
    return c/H          

def InvEz(z):
    Ez = sqrt ( O_M * (1+z)**3 + O_K() * (1+z)**2 + O_L)
    return 1/Ez

def D_C(z):
    integ,err = quad(InvEz,0,z)
    return D_H() * integ  

def D_M(z):
    if O_K() > 0:
        dm = D_H() * 1/sqrt(O_K()) * sinh( sqrt(O_K()) * D_C(z) / D_H() ) 
    elif O_K() == 0 :
        dm = D_C(z)
    elif O_K()<0:
        dm = D_H() * 1/sqrt(abs(O_K())) * sin( sqrt(abs(O_K())) * D_C(z) / D_H() )
    return dm

def D_L(z):
    return (1+z) * D_M(z)

def D_A(z):
    return D_M(z)/(1+z)

def dV_c(z):
    return D_H() * (1+z)**2 * D_A(z)**2 * InvEz(z) * dz
    
def Vc(z):
    if O_K() > 0:
        vc = (4*math.pi * D_H()**3)/(2.*O_K())*(D_M(z)/D_H()*
             sqrt(1+O_K()*(D_M(z)/D_H())**2)-1/sqrt(O_K())*
             arcsinh(sqrt(O_K())*D_M(z)/D_H()))
    elif O_K() == 0 :
        vc =  4/3.*math.pi* D_M(z)**3
    elif O_K()<0:
        vc = (4*math.pi * D_H()**3)/(2.*O_K())*(D_M(z)/D_H()*
             sqrt(1+O_K()*(D_M(z)/D_H())**2)-1/sqrt(abs(O_K()))*
             arcsin(sqrt(abs(O_K()))*D_M(z)/D_H()))
    return vc

def mu(z):
    return ( 5 * log10 (D_L(z))  + 25)

def Inv_zEz(z):
    Ez = sqrt ( O_M * (1+z)**3 + O_K() * (1+z)**2 + O_L)
    return 1/(Ez*(1+z))

def t_l(z):
    t_h = 9.78 * 100/H   # Gyr
    integ,err = quad(Inv_zEz,0,z) 
    return t_h*integ

###############

global H, O_M, O_L, c

H = 72.
O_M = 0.3
O_L= 0.7



