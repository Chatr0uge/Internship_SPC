import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import ufuncify
from sympy.utilities.lambdify import lambdify

class q_Equilibrium:
    def __init__(self, R0 = 1., a =0.75, B0 = 2.5, n0 = 1., T0 = 1., q0 = 1.1, qa = 2.,do_lambdify=False,backend='cython'):        
        X     = sp.symbols('x y z',real=True)
        x,y,z = X
        R = sp.sqrt(x**2 + y**2)   
        
        r = sp.sqrt(z**2 + (R - R0)**2)
        cos_theta = (R-R0)/r
        sin_theta = z/r
                
        q    = q0 + (qa - q0) * r**2/a**2
        B_phi= B0 * R0 / R
        
        psi   = -B0*R0*(R0-sp.sqrt(R0*R0-r**2))/q0
        B_the =(-B0*R0*r/sp.sqrt(R0*R0-r**2)/q0)/R
        #B_the= r * B_phi / (R0 * q) 
                
        sin_a     = y/R 
        cos_a     = x/R
        
        B_x  = -B_the * sin_theta * cos_a - B_phi * sin_a
        B_y  = -B_the * sin_theta * sin_a + B_phi * cos_a
        B_z  =  B_the * cos_theta       
        
        B        = sp.Matrix([B_x,B_y,B_z])
        b        = B.normalized()
        mag_B    = sp.sqrt(B_phi**2 + B_the**2)
        s        = psi/(-B0*R0*(R0-sp.sqrt(R0*R0-a**2))/q0)#(r/a)**2

        gradB = B.jacobian(X)
        gradb = b.jacobian(X)
        
        grad_mag_B = sp.Matrix([mag_B]).jacobian(X)
        grads = sp.Matrix([s]).jacobian(X)
        
        
        XYZ_i             = sp.symbols('s_i theta_i phi_i',real=True)
        s_i,t_i,p_i     = XYZ_i
        psi_search = s_i * (-B0*R0*(R0-sp.sqrt(R0*R0-a**2))/q0)
        r_search   = sp.sqrt(R0*R0-(R0 + psi_search * q0/(B0*R0))**2)
        R_i = R0 + r_search * sp.cos(t_i)
        X_i = R_i * sp.cos(p_i)
        Y_i = R_i * sp.sin(p_i)
        Z_i = r_search * sp.sin(t_i)
        
        #compiling fast binary functions:
        if do_lambdify:
            self.b          = lambdify(X,b.T,"numpy")
            self.gradb      = lambdify(X,gradb,"numpy")    
            self.B          = lambdify(X,B.T,"numpy")
            self.gradB      = lambdify(X,gradB,"numpy")
            self.mag_B      = lambdify(X,mag_B,"numpy")
            self.B_g        = lambdify(X,B.T,"numpy")
            
            self.grad_mag_B = lambdify(X,grad_mag_B,"numpy")
            self.s          = lambdify(X,s,"numpy")
            self.grads      = lambdify(X,grads,"numpy")
            
            self.X_lab      = lambdify(XYZ_i, X_i,"numpy")
            self.Y_lab      = lambdify(XYZ_i, Y_i,"numpy")
            self.Z_lab      = lambdify(XYZ_i, Z_i,"numpy")

        else: 
            self.b          = autowrap(b.T,args=X,backend=backend,tempdir='codegen_eq')
            self.gradb      = autowrap(gradb,args=X,backend=backend,tempdir='codegen_eq')       
            self.B          = autowrap(B.T,args=X,backend=backend,tempdir='codegen_eq')
            self.gradB      = autowrap(gradB,args=X,backend=backend,tempdir='codegen_eq')
            self.mag_B      = ufuncify(X,mag_B)
            #self.B_g        = ufuncify(X,B.T) #not working
            self.B_g        = lambdify(X,B.T,"numpy") #not working


            #self.mag_B      = autowrap(mag_B,args=X,backend='f2py',tempdir='codegen_eq')
            self.grad_mag_B = autowrap(grad_mag_B,args=X,backend=backend,tempdir='codegen_eq')
            self.s          = ufuncify(X,s)
            #self.s1         = autowrap(s,args=X,backend=backend,tempdir='codegen_eq')
            self.grads      = autowrap(grads,args=X,backend=backend,tempdir='codegen_eq')
            
            self.X_lab      = ufuncify(XYZ_i,X_i)#,args=,backend=backend,tempdir='codegen_eq')
            self.Y_lab      = ufuncify(XYZ_i,Y_i)#,args=XYZ_i,backend=backend,tempdir='codegen_eq')
            self.Z_lab      = ufuncify(XYZ_i,Z_i)#,args=XYZ_i,backend=backend,tempdir='codegen_eq')

        
    def grad_B_grad_s(self,r):
        #gradB = self.gradB(*r)
        return self.s(*r),self.B(*r)[0],self.gradB(*r),self.grads(*r)[0]
    
    #def grad_b_grad_s(self,r):
    #    x,y,z = np.array(r)
    #    gradb = [f(x,y,z) for f in self.gradb]
    #    grads = np.array([f(x,y,z) for f in self.grads])
    #    return self.s(x,y,z),np.array([self.Bx(x,y,z),self.By(x,y,z),self.Bz(x,y,z)]),gradb[:3],gradb[3:6],gradb[6:],grads
    
    def get_B(self,r):
        return self.s(*r),self.B(*r)[0]
    
    def get_s_and_B(self,r):
        return self.s(*r),self.B(*r)[0]
    
    def mag2xyz(self,s,theta,phi):
        return  self.X_lab(s,theta,phi),self.Y_lab(s,theta,phi),self.Z_lab(s,theta,phi)
    
    def get_s_B(self,X,Y,Z):
        return  self.s(X,Y,Z),np.rollaxis(self.B_g(X,Y,Z)[0],axis=0,start=4)
    
    #def get_modB(self,r):
    #    x,y,z = np.array(r)
    #    return self.modB(x,y,z)
     
    #def get_gradB(self,r):
    #    x,y,z = np.array(r)
    #    grad_mod_B = [f(x,y,z) for f in self.grad_mod_B] 
    #    return np.array(grad_mod_B)
