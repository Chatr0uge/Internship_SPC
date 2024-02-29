#!/usr/bin/env python
from __future__ import print_function
import os,sys

import numpy as np
#import autograd.numpy as np
from numpy.linalg import norm
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.utilities.autowrap import autowrap

import dill
dill.settings['recurse'] = True

import timeit
#from sympy.printing.theanocode import theano_function

import scipy.constants as sc
w_p2 = sc.e**2/sc.m_e/sc.epsilon_0
w_c0 = sc.e/sc.m_e
#############################################################
class Ray_Tracing_ODE:
    """
    A class to generate and store ray-tracing ODEs.
    """
    def __init__(self):
        if not os.path.exists('WKB_ODE_code.pkl'):
            Y = sp.symbols('x y z Nx Ny Nz',real=True)
            x, y, z, Nx, Ny, Nz = Y

            #Hamiltonian in Stix
            Npe     = sp.Function('Npe' ,is_real=True)(x,y,z,Nx,Ny,Nz)
            Npa     = sp.Function('Npa' ,is_real=True)(x,y,z,Nx,Ny,Nz)
            wc      = sp.Function("wc"  ,is_real=True)(x,y,z)
            wp2     = sp.Function("wp2" ,is_real=True)(x,y,z)
            vt      = sp.Function('vt'  ,is_real=True)(x,y,z)
            functions = (Npe, Npa, wc, wp2, vt)

            One        = sp.eye(3)
            N_cross    = sp.Matrix([[ 0  ,-Npa, 0  ],
                                    [ Npa, 0  ,-Npe],
                                    [ 0  , Npe, 0 ]])
            #cold tensor
            wc_cross   = sp.Matrix([[ 0  ,-wc , 0  ],
                                    [ wc , 0  , 0  ],    
                                    [ 0  , 0  , 0  ]])
            eps    = (One - wp2 * sp.Inverse(One-sp.I*wc_cross))
            D      = N_cross * N_cross + eps
            H_stix = ((D+D.H)/2).det()
            
            self.lambda_H = sp.lambdify(functions, H_stix)
            
            dDdw  = -(2 * wp2 * H_stix.diff(wp2) + wc * H_stix.diff(wc) + Npe * H_stix.diff(Npe) + Npa * H_stix.diff(Npa))      
            
            RHS   = sp.Matrix([-H_stix.diff(Nx),
                               -H_stix.diff(Ny),
                               -H_stix.diff(Nz),
                                H_stix.diff(x),
                                H_stix.diff(y),
                                H_stix.diff(z)]).T/(dDdw)
            ### Lambdify:
            derivatives_subs, dS = generateExtrnalDerivativeMap(functions,Y)
            RHS      = RHS.xreplace(derivatives_subs)
            var_list = Y + functions + dS
            
            self.y_dot = sp.lambdify(var_list, RHS)
            
            #Fs = sp.symbols('Npe Npa wc wp2 vt')
            #func_to_symbol = {Npe:Fs[0],
            #                  Npa:Fs[1],
            #                   wc:Fs[2],
            #                  wp2:Fs[3],
            #                   vt:Fs[4]}
            #RHS   =RHS.xreplace(func_to_symbol)
            #H_stix=H_stix.xreplace(func_to_symbol)
            #autowrap_var_list = Y + Fs + dS
            
            #self.lambda_H   = autowrap(H_stix, args = Fs, tempdir='codegen_ODE')
            #self.y_dot      = autowrap(RHS,    args = autowrap_var_list, tempdir='codegen_ODE')
            with open("WKB_ODE_code.pkl", 'wb') as pkl: 
                dill.dump([self.y_dot,self.lambda_H], pkl)
        else:
            with open("WKB_ODE_code.pkl", 'rb') as pkl: 
                self.y_dot,self.lambda_H = dill.load(pkl)
    
    def WKB(self,t,Y,w_0,eq,ne,dne_ds,te,dte_ds,extra_B=None):
        #x,y,z = Y[0:3]
        N     = np.array(Y[3:6])
        
    
        s,B,grad_B,grad_s=eq.grad_B_grad_s(Y[0:3])
        
        modB    = np.sqrt(B.dot(B))
        b       = B/modB
    
        wc      = -w_c0 * modB / w_0
        M_gradB = -w_c0 * grad_B / w_0
    
        N_z_mod = np.dot(N,b)
        N_x_mod = np.sqrt(N.dot(N) - N_z_mod**2)
    
        dN_zdN_x,dN_zdN_y,dN_zdN_z  = b
        dN_xdN_x,dN_xdN_y,dN_xdN_z  = (N - N_z_mod * b)/N_x_mod
        dwcdx,dwcdy,dwcdz           = np.dot(b,M_gradB)
        dN_zdx,dN_zdy,dN_zdz        = N_x_mod / wc * np.dot(np.array((dN_xdN_x,dN_xdN_y,dN_xdN_z)),M_gradB)
        dN_xdx,dN_xdy,dN_xdz        = -N_z_mod/N_x_mod * np.array((dN_zdx,dN_zdy,dN_zdz))
    
        X       = w_p2 * ne(s) / w_0**2
        dXds    = w_p2 / w_0**2 * dne_ds(s)
        dXdx,dXdy,dXdz = dXds*grad_s[0],dXds*grad_s[1],dXds*grad_s[2]
        #print(s,self.lambda_H(N_x_mod,N_z_mod,wc,X,0),X,Y[0],Y[1],Y[2])
        
        #print(s,dN_xdN_x,dN_xdN_y,dN_xdN_z)
        #print(grad_s)
        #if te is not None:
        #    v_t     = np.sqrt(2. * te(s)/511)
        #    dVds    = 0.5 * 2/511. * dte_ds(s) / v_t
        #    dVdx,dVdy,dVdz = dVds*grad_s[0],dVds*grad_s[1],dVds*grad_s[2]
        #else:
        v_t = 0
        dVdx,dVdy,dVdz = 0,0,0

        var_list = tuple(Y) + (N_x_mod,N_z_mod, wc, X, v_t) + (dN_xdx,dN_xdy,dN_xdz,dN_xdN_x,dN_xdN_y,dN_xdN_z) + (dN_zdx,dN_zdy,dN_zdz,dN_zdN_x,dN_zdN_y,dN_zdN_z) + (dwcdx,dwcdy,dwcdz,dXdx,dXdy,dXdz,dVdx,dVdy,dVdz)
        
        ydot = self.y_dot(*var_list)[0]        
        #ydot = ydot/np.sqrt(np.dot(ydot[:3],ydot[:3]))        
        return ydot
    
    def H(self,Y,w_0,eq,ne,te,extra_B=None):
        #x,y,z = Y[0:3]
        N     = np.array(Y[3:6])
        #s,B,dBdx,dBdy,dBdz,grad_s = eq.grad_B_grad_s(Y[0:3])
        s,B                       = eq.get_B(Y[0:3])

        if extra_B is not None:
            B = extra_B
            
        modB = np.sqrt(B.dot(B))
        b       = B/modB
        wc      = w_c0 * modB / w_0
        X       = w_p2 * ne(s) / w_0**2

        N_z     = np.dot(N,b)
        N_x     = np.sqrt(N.dot(N) - N_z**2)
        #print('{:.14E}'.format(self.lambda_H(N_x,N_z,wc,X,0)))
        return self.lambda_H(N_x,N_z,wc,X,0)#/np.fmax(1.,N_x**4)
    
    def dH(self,Y,w_0,eq,ne,te,extra_B=None):
        #if te is None:
        H = self.H   
        #else:
        #    H = self.H_hot

        N     = np.array(Y[3:6])
        NN    = 1.#np.sqrt(N.dot(N))
        k = 1e-8
        d     = np.array((k,k,k,k*NN,k*NN,k*NN))
        dww_0 = k
        dYwp  = np.array((1,1,1,1./(1+dww_0),1./(1+dww_0),1./(1+dww_0)))
        dYwm  = np.array((1,1,1,1./(1-dww_0),1./(1-dww_0),1./(1-dww_0)))
        dHdY  = np.zeros(6)
        for i in range(6):
            dY      = np.zeros(6)
            dY[i]  += d[i]
            dHdY[i] = (H(Y+dY,w_0,eq,ne,te,extra_B)-H(Y-dY,w_0,eq,ne,te,extra_B))/(2.*d[i])
        dHdY /= (H(Y*dYwp,w_0*(1.+dww_0),eq,ne,te,extra_B)-H(Y*dYwm,w_0*(1.-dww_0),eq,ne,te,extra_B))/(2.*dww_0)

        return dHdY
    
    def WKB_numerical(self,t,Y,w_0,eq,ne,dne_ds,te,dte_ds,extra_B=None):
        ydot = np.roll(self.dH(Y,w_0,eq,ne,te,extra_B),3)
        ydot[0:3] *= -1
        #ydot = ydot/np.sqrt(np.dot(ydot[:3],ydot[:3]))
        return ydot

def launch_wave(y0,w_0,eq,ne,dne_ds,te,dte_ds,ODEs = None, H_type = 'analytical',l=15., N_max=1e2, extra_B=None, hit_ground_s=1., atol_r=1e-5, atol_n=1e-7, rtol=1e-5, min_step=1e-11):
    ODEs = Ray_Tracing_ODE() if ODEs is None else ODEs

    def hit_ground(t, y): return hit_ground_s - eq.get_B(np.array(y[0:3]))[0]
    def hit_sky(t, y):    return N_max**2 - (y[3]**2+y[4]**2+y[5]**2)
    hit_ground.terminal  = True
    hit_ground.direction = -1
    hit_sky.terminal     = True
    hit_sky.direction    = -1

    #l = 150.19/10
    T0 = timeit.default_timer()
    start = timeit.default_timer()
    
    if H_type == 'analytical':
        RHS = ODEs.WKB
        # error = atol + rtol * abs(y)       
        #the best results with the analytic equilibrium is 
        #ar   = 1e-10; an = 1e-14
        #rtol = 3e-14;
        atol = [atol_r,atol_r,atol_r,atol_n,atol_n,atol_n];
        #min_step=1e-11#; max_step=1e-2
    else:
        RHS = ODEs.WKB_numerical
        #here is a typical accuracy for numerical Hamiltonian
        #atol = [1e-5,1e-5,1e-5,1e-7,1e-7,1e-7]; 
        #rtol = 1e-5; 
        atol = [atol_r,atol_r,atol_r,atol_n,atol_n,atol_n];
        #min_step=1e-9;# max_step=1e-1

    sol = solve_ivp(lambda t,y: RHS(t,y,w_0,eq,ne,dne_ds,te,dte_ds,extra_B), [0,l],  y0, 
                    events=(hit_ground,hit_sky),
                    method = 'LSODA', atol = atol, rtol = rtol, min_step = min_step)
    
    stop = timeit.default_timer()
    print('Computation time:',stop - start,'s')
    
    if not sol.success:
        print (sol.message)
    #sol = sol.y.T
    return sol#np.array(sol)

def launch_wave_NzXYZ(origin,k_lab,mod,w_0,eq,ne,N_z=None,dne_ds=None,te=None,dte_ds=None,LCFS_intersection=False,ODEs=None,H_type = 'analytical',l=15,N_max=1e4,extra_B=None,hit_ground_s=1.,atol_r=1e-5,atol_n=1e-7,rtol=1e-5,min_step=1e-11):
    """
    Setting up both k_lab and N_z simultaniously is redundunt. 
    Yet N_z is used to solve dispersion relation to get N_perp.
    Then k_lab vector is used to determin and asimuthal angle of N_perp. This works better for ODE init to exect solution.
    """
    if LCFS_intersection:
        traj_init, code   = eq.getRayIntersectionPoints(origin, k_lab)
    else:
        traj_init = origin
        
    s,B = eq.get_B(traj_init)
    if extra_B is not None:
        B = extra_B
        
    modB    = np.sqrt(B.dot(B))
    b       = B/modB
    if N_z is None:
        N_z     = np.dot(k_lab,b)
    wc      = -w_c0 * modB / w_0
    wp2     = w_p2 * ne(s) / w_0**2
    Nx1,Nx2 = N_perp(N_z,wp2,wc) #Roots of the cold-dispersion relation
    mod_key = float(mod == 'O') - 0.5
    N_x     = Nx1 if mod_key * (np.abs(wc) - 1) < 0 else Nx2

    if te:
        v_t     = np.sqrt(2. * te(s)/511)
        #print (N_z,N_x)
        from scipy.optimize import newton
        #
        N_x = newton(lambda N_x_mod,N_z_mod,X,wc,v_t: D_hot(N_x_mod,N_z_mod,X,wc,v_t).real, N_x,args=(N_z,wp2,wc,v_t))

    stix_Z  = b
    stix_X  = k_lab - stix_Z * np.dot(k_lab,stix_Z)
    M_lab_to_stix, M_stix_to_lab = Constract_Rotation_Matrix(stix_X,np.cross(stix_Z,stix_X))

    y0 = np.append(traj_init,np.dot(M_stix_to_lab,np.array((N_x,0.,N_z))))
    sol = launch_wave(y0,w_0,eq,ne,dne_ds,te,dte_ds,ODEs,H_type,l,N_max,extra_B,hit_ground_s,atol_r,atol_n,rtol,min_step)
    return sol

def search_along_line(traj_init,N_z,pm,w_0,eq,ne,mode='O',X_grater_than_1=False):
    s,B,grad_B,grad_s = eq.grad_B_grad_s(traj_init)
    L = np.linspace(1e-2/1000,1e-2,1000)
    search_line = traj_init[None,:] + pm * np.outer(L,grad_s/np.sqrt(grad_s.dot(grad_s)))
    s_and_B     =[np.hstack(eq.get_B(x))[0:4] for x in search_line]
    s           = np.array(s_and_B)[:,0]
    B           = np.array(s_and_B)[:,1:4]
    X           = w_p2 * ne(s) / w_0**2
    modB        = np.linalg.norm(B,axis=1)
    wc          = w_c0 * modB / w_0
    Nx1,Nx2     = N_perp(N_z,X,wc)
    mod_key = float(mode == 'O') - 0.5
    N_x     = Nx1 if mod_key * (np.abs(wc[0]) - 1) < 0 else Nx2
    if X_grater_than_1:
        sO_loc      = np.argmax(np.diff(N_x*(X>1.))>0)
    else:
        sO_loc      = np.argmax(np.diff(N_x)>0)
    traj_init   = search_line[sO_loc+1]
    return traj_init,search_line,Nx1,Nx2,X

def Mjolhus1984(ray,eq,ne,dne_ds,w_0):
    ray_coord     = ray[0:3,:].T
    N             = ray[3: ,:].T
    N_z,N_x,s,wc,X= ray_to_stix(ray,eq,ne,w_0)

    N_x_conversion     = np.min(N_x)
    loc_conversion     = ray_coord[np.argmin(N_x)]
    N_z_conversion     = N_z[np.argmin(N_x)]
    N_conversion       = N[np.argmin(N_x)]
    s_conversion       = s[np.argmin(N_x)]
    wc_conversion      = wc[np.argmin(N_x)]

    N_opt     = np.sqrt(wc_conversion/(1.+wc_conversion))
    s,B,grad_B,grad_s=eq.grad_B_grad_s(loc_conversion)
    Ln        = ne(s_conversion) / np.linalg.norm(dne_ds(s_conversion)*grad_s)
    lnT       = np.pi*w_0/299792458.0*Ln*np.sqrt(wc_conversion/2)*(2.*(1.+wc_conversion)*(N_opt-np.abs(N_z_conversion))**2+N_x_conversion**2)
    T         = np.exp(-lnT)
    return T, lnT, loc_conversion, N_conversion, N_z_conversion,Ln

def ray_to_stix(ray,eq,ne,w_0):
    ray_coord = ray[0:3,:].T
    N         = ray[3: ,:].T

    s_and_B   =[np.hstack(eq.get_B(x))[0:4] for x in ray_coord]
    s         = np.array(s_and_B)[:,0]
    B         = np.array(s_and_B)[:,1:4]
    X         = w_p2 * ne(s) / w_0**2
    modB      = np.linalg.norm(B,axis=1)
    wc        = w_c0 * modB / w_0
    b         = B[:,:]/modB[:,None]
    N_z       = np.sum(N*b,axis=1)
    modN      = np.linalg.norm(N,axis=1)
    N_x       = np.sqrt(modN**2-N_z**2)
    
    return N_z,N_x,s,wc,X

def path(ray):
    path      = np.hstack([0,np.cumsum(np.sqrt(np.sum((np.diff(ray[0:3,:], axis=-1)**2),axis=0)))])
    return path

def N_perp_XYZ(N_z,XYZ,eq,ne):
    s,B     = eq.get_B(np.array(XYZ))
    modB    = np.sqrt(B.dot(B))
    wc      = w_c0 * modB / w_0
    X       = w_p2 * ne(s) / w_0**2
    rNx1,rNx2 = N_perp(N_z,X,wc)
    return rNx1,rNx2

def N_perp(N_z,X,wc):
    eps   = 1.0-(X)/(1-wc**2)
    eta   = 1.0-(X)
    g     = wc*(X)/(1-wc**2)
    par_a = eps
    par_b = (g**2-(eps+eta)*(eps-N_z**2))
    par_c = -eta*g**2+eta*(eps-N_z**2)**2
    par_D = par_b**2-4.0*par_a*par_c

    rNx1 = np.sqrt((-par_b+np.sqrt(par_D))/(2.0*par_a))
    rNx2 = np.sqrt((-par_b-np.sqrt(par_D))/(2.0*par_a))
    return rNx1,rNx2

def Constract_Rotation_Matrix(X,Y):
    X = X/np.linalg.norm(X)
    Yn = Y# - X * np.dot(Y,X) #In case Y is not orthogonal to X
    Y = Yn/np.linalg.norm(Yn)
    Z = np.cross(X,Y)
    M_lab_to_box = np.vstack((X,Y,Z))
    M_box_to_lab = np.linalg.inv(M_lab_to_box)
    return M_lab_to_box, M_box_to_lab

def W7X_aiming_to_Cart(alpha, beta, origin):
    direction_cyl = np.array((-np.cos(alpha)*np.cos(beta),np.cos(alpha)*np.sin(beta),np.sin(alpha)))
    phy           = np.arctan2(origin[1],origin[0])
    X             = direction_cyl[0]*np.cos(phy) - direction_cyl[1]*np.sin(phy)
    Y             = direction_cyl[0]*np.sin(phy) + direction_cyl[1]*np.cos(phy)
    return np.array((X,Y,direction_cyl[2]))

def generateExtrnalDerivativeMap(functionsList, coordList):
    result = {}
    derivSymbols = []
    for function in functionsList:
        functionName = str(function)[:str(function).find('(')]
        for coord in coordList:
            coordName = str(coord)
            derivativeName = "d"+functionName+"_d"+coordName
            derivative = sp.symbols(derivativeName)
            df = function.diff(coord)
            if df != 0:
                result[df] = derivative
                derivSymbols.append(derivative)
    return result, tuple(derivSymbols)

def search_launches_mconf(d0,origin,eq,ne,dne_ds,w_0,ODEs,hit_ground_s=1.,atol_r=1e-5,atol_n=1e-7,rtol=1e-5,min_step=1e-11):
    alpha,beta = d0
    rays = []
    direction = W7X_aiming_to_Cart(alpha/180.*np.pi, beta/180.*np.pi, origin)
    entry,code   = eq.getRayIntersectionPoints(origin,direction)
    s,B   = eq.get_B(entry)
    b     = B/np.sqrt(B.dot(B))
    N_z   = np.dot(direction,b)
    #print(entry)
    #@print(B)
    #rint(N_z)
    rays.append(launch_wave_NzXYZ(entry,direction,'O',w_0,eq,ne,dne_ds=dne_ds,ODEs=ODEs,hit_ground_s=hit_ground_s,atol_r=atol_r,atol_n=atol_n,rtol=rtol,min_step=min_step).y)
    if True:
        ray_coord     = rays[-1][0:3,:].T
        N             = rays[-1][3: ,:].T
        N_z,N_x,s,wc,X= ray_to_stix(rays[-1],eq,ne,w_0)

        N_x_conversion     = np.min(N_x)
        loc_conversion     = ray_coord[np.argmin(N_x)]
        N_z_conversion     = N_z[np.argmin(N_x)]
        N_conversion       = N[np.argmin(N_x)]
        s_conversion       = s[np.argmin(N_x)]
        wc_conversion      = wc[np.argmin(N_x)]

        N_opt     = np.sqrt(wc_conversion/(1.+wc_conversion))
        s,B,grad_B,grad_s=eq.grad_B_grad_s(loc_conversion)
        Ln        = ne(s_conversion) / np.linalg.norm(dne_ds(s_conversion)*grad_s)
        lnT       =         np.pi*w_0/299792458.0*Ln*np.sqrt(wc_conversion/2)*(2.*(1.+wc_conversion)*(N_opt-np.abs(N_z_conversion))**2+N_x_conversion**2)
        T         = np.exp(-lnT)
    print(alpha,beta,lnT,T)
    return lnT

def search_launches(d0,origin,eq,ne,dne_ds,w_0,ODEs):
    alpha,beta = d0
    rays = []
    direction = W7X_aiming_to_Cart(alpha/180.*np.pi, beta/180.*np.pi, origin)
    entry =    origin #= eq.getRayIntersectionPoints(origin,direction)
    s,B   = eq.get_B(entry)
    b     = B/np.sqrt(B.dot(B))
    N_z   = np.dot(direction,b)
    rays.append(launch_wave_NzXYZ(entry,direction,'O',w_0,eq,ne,dne_ds=dne_ds,ODEs=ODEs).y)
    if True:
        ray_coord     = rays[-1][0:3,:].T
        N             = rays[-1][3: ,:].T
        N_z,N_x,s,wc,X= ray_to_stix(rays[-1],eq,ne,w_0)

        N_x_conversion     = np.min(N_x)
        loc_conversion     = ray_coord[np.argmin(N_x)]
        N_z_conversion     = N_z[np.argmin(N_x)]
        N_conversion       = N[np.argmin(N_x)]
        s_conversion       = s[np.argmin(N_x)]
        wc_conversion      = wc[np.argmin(N_x)]

        N_opt     = np.sqrt(wc_conversion/(1.+wc_conversion))
        s,B,grad_B,grad_s=eq.grad_B_grad_s(loc_conversion)
        Ln        = ne(s_conversion) / np.linalg.norm(dne_ds(s_conversion)*grad_s)
        lnT       =         np.pi*w_0/299792458.0*Ln*np.sqrt(wc_conversion/2)*(2.*(1.+wc_conversion)*(N_opt-np.abs(N_z_conversion))**2+N_x_conversion**2)
        T         = np.exp(-lnT)
    print(alpha,beta,lnT,T)
    return lnT

def fmin_Nopt_mconf(d0,origin,eq,w_0):
    alpha,beta = d0
    direction = W7X_aiming_to_Cart(alpha/180.*np.pi, beta/180.*np.pi, origin)
    entry,code   = eq.getRayIntersectionPoints(origin,direction)
    s,B,grad_B,grad_s=eq.grad_B_grad_s(entry)
    b       = B/np.sqrt(B.dot(B))
    N_z     = np.dot(direction,b)
    modB    = np.sqrt(B.dot(B))
    b       = B/modB
    wc      = w_c0 * modB / w_0
    N_opt   = np.sqrt(wc/(1.+wc))
    N_x     = np.abs(np.dot(direction,np.cross(b,grad_s)/np.linalg.norm(np.cross(b,grad_s))))
    #lnT       = np.pi*w_0/299792458.0*Ln*np.sqrt(wc/2)*(2.*(1.+wc)*(N_opt-np.abs(N_z))**2+N_x**2)
    lnT     = (2.*(1.+wc)*(N_opt-np.abs(N_z))**2+N_x**2)
    #print(alpha,beta,lnT)
    return lnT
