from __future__ import print_function
import numpy as np
from numpy import float32 as f32
from numpy import int32 as i32

import sys,os

class CUWA:
    """The CUWA code core class 
    """
    def __init__(self, GPU_device_id=0, block_size    = 32):
        """Use GPU_device_id=None 
        to initialise CUWA on machine with no GPU. The object can later be copied to a GPU machine for the actual run.
        set block_size    = 16 if you encounter "Resources Unavailable" errors.
        """
        
        if GPU_device_id is not None:
            global cuda, gpuarray, SourceModule, ElementwiseKernel
            import pycuda.driver as cuda
            import pycuda.gpuarray as gpuarray
            from pycuda.compiler import SourceModule
            from pycuda.elementwise import ElementwiseKernel
        
            cuda.init()
            #cuda.Device.count()
            current_dev = cuda.Device(GPU_device_id)
            self.ctx = current_dev.make_context()
            #self.ctx.push()

            self._GPU_memory = cuda.mem_get_info()
            print(self.GPU_memory)

        #GPU mapping block size
        self.block_size    = block_size; self.block         = (self.block_size,self.block_size,1)
        
        return
    
    def __del__(self):
        self.ctx.pop()
    
    @property
    def GPU_memory(self):
        self._GPU_memory = cuda.mem_get_info()
        return '{0:1.1f} GB is free of total {1:1.1f} GB'.format(self._GPU_memory[0]/1024.**3,self._GPU_memory[1]/1024.**3)
    
    @property
    def GPU_mem_requred(self):
        #GPU memory estimate
        float32size = 4
        return '{:0.1f} Gb of memory is required excluding (minor) CPML and diagnostics.'.format(
    (13*self.cX_box.size * self.cY_box.size * max(1,self.cZ_box.size)) * float32size / 1024**3)
    
    def set_comp_grid(self, omega_ref = 140e9*2.*np.pi, vac_resolution = 12, low_resolution_zoom = 1, Courant = 0.5,
                      origin = np.array((0,0,0)), e_x = np.array((1,0,0)), e_y = np.array((0,1,0)), 
                      x_range = (0.,0.1), y_range = (0.,0.1), z_thick  = 0., nPML = 8):
        """
        This function generate computation grid basis vectors (e_x, e_y), given the (origin) and x_range, y_range, and z_thick. Z axis is always [-z_thick/2, z_thick/2]. This grid has the size devisable by the self.block_size, which is a requirment for the GPU computations (extra points are added to the end of the domain on all sides).
        `omega_ref` is used for the calculation of the vacuum resolution, which is then set by `vac_resolution`.
        `low_resolution_zoom` is the scaling factor for the reduced resultion on which the plasma data is actually set. The mesh at this resolution is returned by this function. Fine grid is obtained using `self.get_fine_grid()`.
        """
        #
        self.n_steps_per_lambda = vac_resolution
        self.B_mesh_zoom        = low_resolution_zoom
        self.low_resolution     = vac_resolution/self.B_mesh_zoom
        self.nPML               = nPML
        
        self.omega_ref          = omega_ref
        self.Courant            = Courant
        
        self.e_x                = e_x/np.linalg.norm(e_x)
        self.e_y                = e_y/np.linalg.norm(e_y)
        self.origin             = origin
        self.M_lab_to_box, self.M_box_to_lab = Constract_Rotation_Matrix(e_x,e_y)
        
        self.dx     = 2.0*np.pi*299792458.0/omega_ref/self.n_steps_per_lambda
        self.cX_box = linspace32(x_range[0], x_range[1], self.dx, self.block_size, self.nPML)#[::self.B_mesh_zoom]
        self.cY_box = linspace32(y_range[0], y_range[1], self.dx, self.block_size, self.nPML)#[::self.B_mesh_zoom]
        if z_thick == 0.:
            self.cZ_box = np.array((0.,))
        else:    
            self.cZ_box = linspace32(-z_thick/2., z_thick/2., self.dx, self.block_size, self.nPML)#[::self.B_mesh_zoom]
        
        self.cX_box_sl = self.cX_box[::self.B_mesh_zoom]
        self.cY_box_sl = self.cY_box[::self.B_mesh_zoom]
        self.cZ_box_sl = self.cZ_box[::self.B_mesh_zoom]
        
        #midplane coordinats
        self.x0y0 = self.origin + np.dot(self.M_box_to_lab,np.array((self.cX_box[0],self.cY_box[0],0)))
        self.x1y0 = self.origin + np.dot(self.M_box_to_lab,np.array((self.cX_box[-1],self.cY_box[0],0)))
        self.x0y1 = self.origin + np.dot(self.M_box_to_lab,np.array((self.cX_box[0],self.cY_box[-1],0)))
        self.x1y1 = self.origin + np.dot(self.M_box_to_lab,np.array((self.cX_box[-1],self.cY_box[-1],0)))
        
        self.Size_X = self.cX_box.size
        self.Size_Y = self.cY_box.size
        self.Size_Z = self.cZ_box.size
        
        self.x_lab_sl,self.y_lab_sl,self.z_lab_sl = faster_rotated_meshgrid(self.cX_box_sl, self.cY_box_sl, self.cZ_box_sl, self.M_box_to_lab, self.origin)
        
        self.dt                = self.Courant * 2.0 * np.pi / (self.omega_ref * self.n_steps_per_lambda)
        self.suggested_n_steps = int(4 * max(self.Size_Z,self.Size_Y,self.Size_X)/self.Courant)
        self.gpu_grid = (int(self.Size_X//self.block[0]),int(self.Size_Y//self.block[1]))
        
        self.receivers = []
        self.sources   = []

        self.Poynting  = False
        self.Surface_Poynting = False
                
        return self.x_lab_sl,self.y_lab_sl,self.z_lab_sl
    
    def get_fine_grid(self):
        self.x_lab,self.y_lab,self.z_lab = faster_rotated_meshgrid(self.cX_box, self.cY_box, self.cZ_box, self.M_box_to_lab, self.origin)
        return self.x_lab,self.y_lab,self.z_lab
    
    def import_grid_from_data(self, shape, dx, omega_ref = 140e9*2.*np.pi, Courant = 0.5, nPML = 10):
        
        self.omega_ref          = omega_ref
        self.Courant            = Courant
        self.dx                 = dx
        
        self.e_x                = np.array((1,0,0))
        self.e_y                = np.array((0,1,0))
        
        self.n_steps_per_lambda = 2.0*np.pi*299792458.0/omega_ref/self.dx
        self.low_resolution     = self.n_steps_per_lambda#low_resolution if low_resolution else vac_resolution
        self.B_mesh_zoom        = 1#int(self.n_steps_per_lambda / self.low_resolution)
        self.nPML               = nPML#int(max(8,self.n_steps_per_lambda//2))
        
        self.origin                          = np.array((0.,0.,0.))
        self.M_lab_to_box, self.M_box_to_lab = np.eye(3),np.eye(3)
        
        self.dt                 = self.Courant * 2.0 * np.pi / (self.omega_ref * self.n_steps_per_lambda)
        
        self.Size_Z,self.Size_Y,self.Size_X = np.clip((np.array(shape)//self.block_size)*self.block_size,1,None)
        
        self.x0y0 = np.array((-self.nPML*dx,              -self.nPML*dx,0))
        self.x1y0 = np.array((self.Size_X*dx-self.nPML*dx,-self.nPML*dx,0))
        self.x0y1 = np.array((-self.nPML*dx,self.Size_Y*dx-self.nPML*dx,0))
        self.x1y1 = np.array((self.Size_X*dx-self.nPML*dx,self.Size_Y*dx-self.nPML*dx,0))
        
        self.suggested_n_steps = int(4 * max(self.Size_Z,self.Size_Y,self.Size_X)/self.Courant)
        self.gpu_grid = (int(self.Size_X//self.block[0]),int(self.Size_Y//self.block[1]))
        
        self.receivers = []
        self.sources   = []

        self.Poynting  = False
        self.Surface_Poynting = False
        
        print("Grid size: {0} {1} {2}".format(self.Size_Z,self.Size_Y,self.Size_X))
        print("nPML: {0}".format(self.nPML))
        print("Resolution (ref): {0}".format(self.n_steps_per_lambda))
        return
    
    def set_plasma_data(self, w_pi, w_c_si, bx_si, by_si, bz_si, Te_nu = 1e12, rampDown_width=0):
        """
        Te_nu - is the temepraure [eV] used for calculating collisional dumping.
        """
        #Making a copy is necessary because some python array operations make a "view",
        #which is not treated correctly by "to_gpu" command.
        self.w_p_s,self.w_c_s,self.bx_s,self.by_s,self.bz_s = [f[:,:,:].copy() for f in (w_pi,w_c_si,bx_si,by_si,bz_si)]
        
        if (self.B_mesh_zoom != 1) and (self.w_p_s.shape == self.x_lab_sl.shape):
            self.w_p = self.zoom_field_to_fine_grid(self.w_p_s)
        else:
            self.w_p = self.w_p_s
        
        #RampDown bezle in x_box and y_box for accurate Poynting vector calculaton.
        #The default is the bezle width of one wavelength plus nPML
        v_max = rampDown_width if rampDown_width is not None else self.n_steps_per_lambda+self.nPML
        self.rd_width = v_max
        if v_max > 0:
            Y,X  = np.mgrid[-self.Size_Y/2.:self.Size_Y/2.:self.Size_Y*1j, -self.Size_X/2.:self.Size_X/2.:self.Size_X*1j]
            x_rd = np.clip((self.Size_X/2.-np.abs(X)),0,v_max)*np.clip((self.Size_Y/2.-np.abs(Y)),0,v_max)/v_max**2                        
            rd   = (3-2.*x_rd)*x_rd**2 #* np.exp(-(1.0/(x_rd+1e-17)-1.)**2)
            self.w_p *=  rd[None,:,:]
        
        #Fill-in current update coefficients
        coef1             = self.w_p * self.w_p * self.dt * self.dt
        bx_w = self.bx_s*self.w_c_s*self.dt; by_w = self.by_s*self.w_c_s*self.dt; bz_w = self.bz_s*self.w_c_s*self.dt;        
        self.plasma_data = [gpuarray.to_gpu(np.float32(field)) for field in [bx_w,by_w,bz_w,coef1]]
        #self.nu   = nu
        import scipy.constants as sc
        self.nudt_coef1 = nu_ee(1., Te_nu) * self.dt / (sc.elementary_charge**2 * self.dt**2 / (sc.m_e * sc.epsilon_0))
        #bx_g,by_g,bz_g,coef1_g  = self.plasma_data
        
    def init_wave_fields(self):
        Size_Z,Size_Y,Size_X = self.Size_Z,self.Size_Y,self.Size_X
        nPML = self.nPML
        
        self.Wave_gpu = None
        self.Wave_gpu = [gpuarray.zeros((Size_Z,Size_Y,Size_X),dtype='float32') for _ in range(12)]
        #Ex_g,Ey_g,Ez_g,Hx_g,Hy_g,Hz_g,Jx_g,Jy_g,Jz_g,Jx0_g,Jy0_g,Jz0_g = self.Wave_gpu
        self.pH_x_g   = None
        self.pH_x_g   = [gpuarray.zeros((Size_Z,Size_Y,nPML*2),dtype='float32') for _ in range(4)]
        #pHyx_g,pHzx_g,pEyx_g,pEzx_g = self.pH_x_g
        self.pH_y_g   = None
        self.pH_y_g   = [gpuarray.zeros((Size_Z,nPML*2,Size_X),dtype='float32') for _ in range(4)]
        #pHxy_g,pHzy_g,pExy_g,pEzy_g = self.pH_y_g
        
    def init_diag_fields(self, Poynting = False, Surface_Poynting = False):
        self.Poynting = False
        self.Surface_Poynting = False
        
        for beam in self.receivers:
            if beam.side in [1,3]:
                beam.A_g = [gpuarray.zeros((self.Size_Z,1,self.Size_X),dtype='float32') for _ in range(6)]
                #reAx_g,reAy_g,reAz_g,imAx_g,imAy_g,imAz_g = self.A_g
            else:
                beam.A_g = [gpuarray.zeros((self.Size_Z,self.Size_Y,1),dtype='float32') for _ in range(6)]
                #reAx_g,reAy_g,reAz_g,imAx_g,imAy_g,imAz_g = self.A_g

        restart_save_size = 0
        output_config = []
        
        if restart_save_size > 0:
            self.restart = None
            self.restart = [gpuarray.zeros((restart_save_size,self.Size_Z,self.Size_X),dtype='float32') for _ in range(9)]
        
        if Poynting:
            self.Poynting  = True
            self.S__g = None
            self.S__g = [gpuarray.zeros((self.Size_Z,self.Size_Y,self.Size_X),dtype='float32') for _ in range(3)]
            #Sx_g,Sy_g,Sz_g = self.S__g
        if Surface_Poynting:
            self.Surface_Poynting = True
            self.Sx0N=None
            self.Sx0N=[gpuarray.zeros((self.Size_Z,self.Size_Y),dtype='float32') for _ in range(6)]
            #Sx0x_g,Sx0y_g,Sx0z_g,SxNx_g,SxNy_g,SxNz_g = self.Sx0N
            self.Sy0N=None
            self.Sy0N=[gpuarray.zeros((self.Size_Z,self.Size_X),dtype='float32') for _ in range(6)]
            #Sy0x_g,Sy0y_g,Sy0z_g,SyNx_g,SyNy_g,SyNz_g = self.Sy0N
            self.Sz0N=None
            self.Sz0N=[gpuarray.zeros((self.Size_Y,self.Size_X),dtype='float32') for _ in range(6)]
            #Sz0x_g,Sz0y_g,Sz0z_g,SzNx_g,SzNy_g,SzNz_g = self.Sz0N
            self.SzI = None
            self.SzI =[gpuarray.zeros((self.Size_Y,self.Size_X),dtype='float32') for _ in range(3)]
            #SzIx_g,SzIy_g,SzIz_g = self.SzI
            self.Surface_Poynting_g = self.Sx0N + self.Sy0N + self.Sz0N + self.SzI
            #Sx0x,Sx0y,Sx0z,SxNx,SxNy,SxNz,Sy0x,Sy0y,Sy0z,SyNx,SyNy,SyNz,Sz0x,Sz0y,Sz0z,SzNx,SzNy,SzNz,SzIx,SzIy,SzIz = self.Surface_Poynting_g

    def import_cu_source(self, J_intrpolate = 2, Jdt = 1, PEC = [], cpml_kappa = 5., cpml_alpha = 0.5, cpml_m1 = 3, cpml_ma = 1, cpml_sigma = 1.):
        """
        Importing and compiling computation kernel
        The kernels are jit-comiled with domaine Size hardcoded (for GPU performance).
        Therefore they need to be recompiled each time the domain size changes.
        
        PEC = ['x0','xN','y0','yN'] - list of PEC boundaries
        
        J_intrpolate = 0:
        Jx colocated with Ex and etc. No interpolation of other components (Jy and Jz) to location of Jx is done when vector operations are performed on J. This scheme is not very stable but somehow has good conservation properties. The solution crips to the right with time. Requires generous rampdown.
        
        J_intrpolate = 1:
        Jx colocated with Ex and etc. All the vector components are properly interpolated for vector operations. Very stable, poor conservation properties. Does not like rampdown.
        
        J_intrpolate = 2:
        Jx,Jy,Jz are colocated in between Ex,Ey,Ez (See Chapter 11.3 in Inan book Page 279). Good conservation properties. Requires generous rampdown.
        
        Jdt = 0 and Jdt = 1:
        Linerized J(t,E) with Jdt = 0 assuming collisional dumping  = 0.
        
        Jdt = 2:
        Exact solution of J(t) (See chapter 11.3 in Inan book Page 279).
        
        Jdt = 3:
        Fully implicit E-J stepping. Not implemented in 3D. 
        
        """
        
        if Jdt == 3 and self.Size_Z > 1:
            print('Jdt == 3 is not implemented for 3D. Choose another scheme. Exit.')
            return
        
        PECx0 = 0 if 'x0' not in PEC else 1
        PECxN = 0 if 'xN' not in PEC else 1
        PECy0 = 0 if 'y0' not in PEC else 1
        PECyN = 0 if 'yN' not in PEC else 1

        
        libname = os.path.join(os.path.dirname(__file__),"FDTD_Imp.cu")
        mod2 = SourceModule(open(libname, "r").read(),options=['-D_3D={0}'.format(int(self.Size_Z>1)),
                                                                   '-Dx_width={0}'.format(self.Size_X),
                                                                   '-Dy_width={0}'.format(self.Size_Y),
                                                                   '-Dz_width={0}'.format(self.Size_Z),
                                                                     '-D_nPML={0}'.format(self.nPML),
                                                                 '-D_Courant={0}f'.format(self.Courant),
                                                               '-D_grid_scale={0}'.format(self.B_mesh_zoom),
                                                                   '-D_KAPPA={0}f'.format(cpml_kappa),
                                                                   '-D_ALPHA={0}f'.format(cpml_alpha),
                                                                       '-D_M1={0}'.format(cpml_m1),
                                                                       '-D_MA={0}'.format(cpml_ma),
                                                               '-D_SIGMA_MAX={0}f'.format(cpml_sigma),
                                                             '-D_J_intrpolate={0}'.format(J_intrpolate),
                                                                      '-D_Jdt={0}'.format(Jdt),
                                                                    '-D_PECx0={0}'.format(PECx0),
                                                                    '-D_PECxN={0}'.format(PECxN),
                                                                    '-D_PECy0={0}'.format(PECy0),
                                                                    '-D_PECyN={0}'.format(PECyN),
                                                              #'--ptxas-options=-v',
                                                                       '-lineinfo'])

        self.Copy_EW         = ElementwiseKernel("float *a, float *b","b[i] = a[i];","gpu_copy")

        self.updateH         = mod2.get_function("update_3D_H_CPML")
        self.updateJ         = mod2.get_function("update_3D_J_CPML")
        self.updateE         = mod2.get_function("update_3D_E_CPML")
        self.sourceE         = mod2.get_function("E_source")
        self.Diagnostic      = mod2.get_function("Diagnostic_3D")
        self.Phasor_int      = mod2.get_function("Phasor_3D")
        self.S_Diagnostic    = mod2.get_function("S_Diagnostic_3D")
        self.Window_batch    = mod2.get_function("Window_batch")
        self.save_E_edge     = mod2.get_function("save_E_edge")
        self.E_restart       = mod2.get_function("E_restart")

    def set_antenna(self, beam, boundary='y_min', source = True, receiver = False):
        """
        boundary: 'x_min', 'x_max', 'y_min', 'y_max'
        """        
        if boundary == 'y_min':
            beam.side = 1
            Source_SIZE = self.Size_X
            pml_shift = self.dx*self.nPML*self.e_y
            A,B = self.x0y0+pml_shift,self.x1y0+pml_shift
            beam.k_i    = 0
            k_direcion = np.sign(np.dot(beam.focus_location-beam.origin,self.e_y))
        elif boundary == 'x_min':
            beam.side = 2
            Source_SIZE = self.Size_Y
            pml_shift = self.dx*self.nPML*self.e_x
            A,B = self.x0y0+pml_shift,self.x0y1+pml_shift
            beam.k_i    = 1
            k_direcion = np.sign(np.dot(beam.focus_location-beam.origin,self.e_x))
        elif boundary == 'y_max':
            beam.side = 3
            Source_SIZE = self.Size_X
            pml_shift = -self.dx*self.nPML*self.e_y
            A,B = self.x0y1+pml_shift,self.x1y1+pml_shift
            beam.k_i  = 0
            k_direcion = -np.sign(np.dot(beam.focus_location-beam.origin,self.e_y))
        elif boundary == 'x_max':
            beam.side = 4
            Source_SIZE = self.Size_Y
            pml_shift = -self.dx*self.nPML*self.e_x
            A,B = self.x1y0+pml_shift,self.x1y1+pml_shift
            beam.k_i    = 1
            k_direcion = -np.sign(np.dot(beam.focus_location-beam.origin,self.e_x))
            
        dis = disect(A,B,beam.origin,beam.focus_location)
        new_source_loc  = A + dis * (B - A)
        beam.X_beam_loc = int(round(dis * Source_SIZE))
        
        beam.new_source_loc = new_source_loc
        
        w_0 = beam.w_0 
        beam.T = int(round(self.n_steps_per_lambda / beam.w_0 * self.omega_ref / abs(self.Courant)))
        
        #Polarization in Lab frame
        if boundary == 'y_min':
            Source_Co_s = max(0,self.Size_Z//self.B_mesh_zoom//2),self.nPML//self.B_mesh_zoom,int(beam.X_beam_loc/self.B_mesh_zoom)
            Source_Co   = max(0,self.Size_Z//2),self.nPML,int(beam.X_beam_loc)
        elif boundary == 'x_min':
            Source_Co_s = max(0,self.Size_Z//self.B_mesh_zoom//2),int(beam.X_beam_loc/self.B_mesh_zoom),self.nPML//self.B_mesh_zoom
            Source_Co   = max(0,self.Size_Z//2),int(beam.X_beam_loc),self.nPML
        elif boundary == 'y_max':
            Source_Co_s = max(0,self.Size_Z//self.B_mesh_zoom//2),-self.nPML//self.B_mesh_zoom,int(beam.X_beam_loc/self.B_mesh_zoom)
            Source_Co   = max(0,self.Size_Z//2),-self.nPML,int(beam.X_beam_loc)
        elif boundary == 'x_max':
            Source_Co_s = max(0,self.Size_Z//self.B_mesh_zoom//2),int(beam.X_beam_loc/self.B_mesh_zoom),-self.nPML//self.B_mesh_zoom
            Source_Co   = max(0,self.Size_Z//2),int(beam.X_beam_loc),-self.nPML
        
        if beam.mode in ['O','X']:
            b_lab       = np.dot(self.M_box_to_lab,np.array((self.bx_s[Source_Co_s],self.by_s[Source_Co_s],self.bz_s[Source_Co_s])))

            beam.E_lab = polarization_wp_eq_0(b_lab,self.w_c_s[Source_Co_s]/w_0,self.w_p[Source_Co]/w_0,beam.k_lab,polarization=beam.mode) #returns complex polarization vector
        #else:
        #    beam.E_lab = beam.E_lab
                
        #k and E transforming lab to computation box
        beam.k_box     = np.dot(self.M_lab_to_box,beam.k_lab) * k_direcion
        beam.E_box     = np.dot(self.M_lab_to_box,beam.E_lab) 
        
        #Converting phasor to real amplitudes and phases
        beam.E0_box    = np.absolute(beam.E_box)
        beam.phy_box_p = np.angle(beam.E_box)/np.pi #in CUDA "sinp" function is used. sinp(x)=sin(pi*x)
        
        beam.z0        = - k_direcion * np.linalg.norm(beam.focus_location - new_source_loc)/self.dx
        beam.x_waist   = beam.Waist_radius/self.dx
        
        beam.block = (self.block[0],int(min(self.Size_Z,self.block[1])),1)
        beam.grid  = (int(Source_SIZE//self.block[0]),
                     int((self.Size_Z//min(self.Size_Z,self.block[1]))+(self.Size_Z % min(self.Size_Z,self.block[1]) > 0)))        
        if source:
            self.sources.append(beam) 
        if receiver:
            self.receivers.append(beam)        
        
    def run_source(self,t,beam, ramp_up = 10.):
    
        att=1 #attenuation of the beam to account for pulse operation
        if beam.pulse>0:
            att= np.exp(-(t*self.dt/beam.pulse-3)**2 )
            ramp_up=0.1
        self.sourceE(i32(t),i32(beam.side),f32(ramp_up),f32(self.n_steps_per_lambda / beam.w_0 * self.omega_ref),f32(beam.z0),f32(beam.x_waist),f32(beam.X_beam_loc),
                     f32(beam.k_box[beam.k_i]), f32(beam.k_box[2]),
                     f32(beam.E0_box[0]*att),f32(beam.phy_box_p[0]),
                     f32(beam.E0_box[1]*att),f32(beam.phy_box_p[1]),
                     f32(beam.E0_box[2]*att),f32(beam.phy_box_p[2]),
                     *self.Wave_gpu[:6], 
                     block = beam.block, grid = beam.grid)
    
    def run_phasor(self,t,beam):
        self.Phasor_int(i32(t),i32(beam.side),f32(self.n_steps_per_lambda / beam.w_0 * self.omega_ref),f32(beam.z0),f32(beam.x_waist),f32(beam.X_beam_loc),
                     f32(beam.k_box[beam.k_i]), f32(beam.k_box[2]),
                     f32(beam.E0_box[0]),f32(beam.phy_box_p[0]),
                     f32(beam.E0_box[1]),f32(beam.phy_box_p[1]),
                     f32(beam.E0_box[2]),f32(beam.phy_box_p[2]),
                     *(self.Wave_gpu[:3]+beam.A_g),
                     block = beam.block, grid = beam.grid)

    def run_step(self):
        #0    1    2    3    4    5    6    7    8    9     10    11
        #Ex_g,Ey_g,Ez_g,Hx_g,Hy_g,Hz_g,Jx_g,Jy_g,Jz_g,Jx0_g,Jy0_g,Jz0_g = self.Wave_gpu
        self.updateH(*(self.Wave_gpu[:6]+self.pH_x_g[:2]+self.pH_y_g[:2]), block = self.block, grid = self.gpu_grid)
        self.Copy_EW(self.Wave_gpu[6],self.Wave_gpu[9])
        self.Copy_EW(self.Wave_gpu[7],self.Wave_gpu[10])
        self.Copy_EW(self.Wave_gpu[8],self.Wave_gpu[11])
        self.updateJ(*(self.Wave_gpu+self.plasma_data+[f32(self.nudt_coef1)]), block = self.block, grid = self.gpu_grid)
        self.updateE(*(self.Wave_gpu+self.pH_x_g[2:]+self.pH_y_g[2:])  , block = self.block, grid = self.gpu_grid)

    def run_poynting(self):
        self.Diagnostic(*(self.Wave_gpu[:9]+self.S__g), block = self.block, grid = self.gpu_grid)
        
    def run_s_poynting(self):
        self.S_Diagnostic(*(self.Wave_gpu[:9]+self.Surface_Poynting_g), block = self.block, grid = self.gpu_grid)
        
    def integrate_S(self,t,steps):
        integration_start    = 10*self.sources[0].T
        self.integration_interval = 4*self.sources[0].T        
        if (integration_start < t <= integration_start+self.integration_interval) or (t >= steps - self.integration_interval):
            if self.Poynting:
                self.run_poynting()
            if self.Surface_Poynting:
                self.run_s_poynting()
            
            if t == integration_start+self.integration_interval:
                if self.Poynting:
                    self.Syi = self.S__g[1].get()/self.integration_interval
                    self.S__g[0].fill(0);self.S__g[1].fill(0);self.S__g[2].fill(0);
                if self.Surface_Poynting:
                    self.Sy0yi = self.Surface_Poynting_g[7].get()/self.integration_interval                
                    for field in self.Surface_Poynting_g:
                        field.fill(0)
    
    def get_magE(self):
        return np.sqrt(self.Wave_gpu[0].get()**2+self.Wave_gpu[1].get()**2+self.Wave_gpu[2].get()**2)

    def get_E(self, i):
        return self.Wave_gpu[i].get()
    
    def get_S(self):
        self.Sx = self.S__g[0].get(); self.Sy = self.S__g[1].get(); self.Sz = self.S__g[2].get()
        return self.Sx, self.Sy, self.Sz
    
    def intensities_from_S(self): 
        """Returns integrated Poynting vector intensities over each side of computation domain. In order: x0, xN, y0, yN, y0_start
        S_L, S_R, S_B, S_T, S_B0 
        """
        Sx, Sy, Sz = self.get_S()

        S_summ=np.concatenate((Sx[...,np.newaxis],Sy[...,np.newaxis],Sz[...,np.newaxis]),axis=-1)
        Poynting_vector = np.float32(S_summ/max(self.integration_interval,1))

        S_shift = self.nPML + 1
        S_B = np.trapz( Poynting_vector[:, S_shift         , S_shift:-S_shift,1], dx= self.dx)
        S_L = np.trapz( Poynting_vector[:, S_shift:-S_shift, S_shift         ,0], dx= self.dx)
        S_T = np.trapz(-Poynting_vector[:,-S_shift         , S_shift:-S_shift,1], dx= self.dx)
        S_R = np.trapz(-Poynting_vector[:, S_shift:-S_shift,-S_shift         ,0], dx= self.dx)
        S_B0= np.trapz(self.Syi[:, S_shift         , S_shift:-S_shift], dx= self.dx)

        if Sx.shape[0] > 1:
            S_L, S_R, S_B, S_T, S_B0 = [np.trapz(p, dx= self.dx) for p in [S_L, S_R, S_B, S_T, S_B0]]
        else:
            S_L, S_R, S_B, S_T, S_B0 = [p[0] for p in [S_L, S_R, S_B, S_T, S_B0]]
        return S_L, S_R, S_B, S_T, S_B0

    def intensities_from_S_surfaces(self): 
        """Returns integrated Poynting vector intensities over each side of computation domain. In order: x0, xN, y0, yN, y0_start
        S_L, S_R, S_B, S_T, S_B0 
        """
        S_Surface = [field.get()/max(self.integration_interval,1) for field in self.Surface_Poynting_g]
        Sx0x,Sx0y,Sx0z,SxNx,SxNy,SxNz,Sy0x,Sy0y,Sy0z,SyNx,SyNy,SyNz,Sz0x,Sz0y,Sz0z,SzNx,SzNy,SzNz,SzIx,SzIy,SzIz = S_Surface
        Sx0, SxN, Sy0, SyN, S_in = [np.trapz(np.trapz(p, dx= self.dx), dx= self.dx) for p in [Sx0x,SxNx,Sy0y,SyNy,self.Sy0yi]]
        return Sx0, -SxN, Sy0, -SyN, S_in
    
    def run_init(self, J_intrpolate = 2, Jdt = 1):
        self.import_cu_source(J_intrpolate = 2, Jdt = 1) # jit-compilation of cuda source code
        self.init_wave_fields() # this is where E,H,J0,J and PML fields are initialized as zeros on GPU 
        self.init_diag_fields() # this is where auxlilary fields are initialized 
    
    def ray_to_box(self,ray):
        rays_r_box = [];rays_N_box = []
    
        r = np.dot(self.M_lab_to_box,ray[0:3,:]-self.origin[:,None])
        N = np.dot(self.M_lab_to_box,ray[3:,:])
        return r, N
    
    def save_plasma_data_h5(self, save_dict = ['wp','wc','b'], folder = '', extra_fields = {}):
        """Saves rougth grid version of self plasma data.
        save_dict = ['wp','wc','s','b'] refers to C1.set_plasma_data(w_p, w_c, b_x, b_y, b_z)
        extra_fields = {'s': s} is a dict of other fields need to be stored
        w_p and w_c are normalised to self.omega_ref
        """
        import lib_CUWA_io
        import h5py

        w_norm    = self.omega_ref
        h5_output = {}
        if 'wp' in save_dict: h5_output.update({'wp':self.w_p_s/w_norm}) 
        if 'wc' in save_dict: h5_output.update({'wc':self.w_c_s/w_norm}) 
        h5_output.update(extra_fields)

        lib_CUWA_io.h5_out(folder+'plasma_data.h5', h5_output, self.dx*self.B_mesh_zoom, (-self.cZ_box[0],-self.cY_box[0],-self.cX_box[0]))

        if 'b' in save_dict:
            b_x,b_y,b_z = self.bx_s,self.by_s,self.bz_s
            with h5py.File(folder+'b.h5', "w") as hf:
                dset = [hf.create_dataset(bi, data=b_i) for (bi,b_i) in (('bx',b_x),('by',b_y),('bz',b_z))]
                lib_CUWA_io.Write_XMF(folder+'b.xmf',['b','b',0,b_x.shape,self.dx*self.B_mesh_zoom,-self.cZ_box[0],-self.cY_box[0],-self.cX_box[0]])

    def save_results_h5(self, t = 0, save_dict = ['E','S'], folder='', hdf5_save_scale_down=2):
        """Saves rresults of the computation to h5. The results come directly from GPU.
        save_dict = ['E','B','J','S'] refers to C1.set_plasma_data(w_p, w_c, b_x, b_y, b_z)
        """
        import lib_CUWA_io
        import h5py
        sd = hdf5_save_scale_down
        for field in save_dict:
            write = True
            if field in ['E']:
                Sx=self.Wave_gpu[0].get()[::sd,::sd,::sd];
                Sy=self.Wave_gpu[1].get()[::sd,::sd,::sd];
                Sz=self.Wave_gpu[2].get()[::sd,::sd,::sd];
            elif field in ['B']:
                Sx=self.Wave_gpu[3].get()[::sd,::sd,::sd];
                Sy=self.Wave_gpu[4].get()[::sd,::sd,::sd];
                Sz=self.Wave_gpu[5].get()[::sd,::sd,::sd];
            elif field in ['J']:
                Sx=self.Wave_gpu[6].get()[::sd,::sd,::sd];
                Sy=self.Wave_gpu[7].get()[::sd,::sd,::sd];
                Sz=self.Wave_gpu[8].get()[::sd,::sd,::sd];           
            elif field in ['S'] and self.Poynting:
                Sx=self.S__g[0].get()[::sd,::sd,::sd];
                Sy=self.S__g[1].get()[::sd,::sd,::sd];
                Sz=self.S__g[2].get()[::sd,::sd,::sd];
            else:
                write = False
            
            if write:
                with h5py.File(folder+field+'.h5', "w") as hf:
                    dset = [hf.create_dataset(bi, data=b_i.astype(np.float32, copy=False), dtype = np.float32) for (bi,b_i) in ((field+'x',Sx),(field+'y',Sy),(field+'z',Sz))]
                    lib_CUWA_io.Write_XMF(folder+field+'.xmf',[field,field,t,Sx.shape,self.dx*sd,-self.cZ_box[0],-self.cY_box[0],-self.cX_box[0]])

    def save_surface_poynting_h5(self, t=0, folder = ''):
        import lib_CUWA_io
        import h5py
        
        Size_Z,Size_Y,Size_X = self.Size_Z,self.Size_Y,self.Size_X
        dx = self.dx; nPML = self.nPML
        
        S_Surface = [field.get()/max(self.integration_interval,1) for field in self.Surface_Poynting_g]
        Sx0x,Sx0y,Sx0z,SxNx,SxNy,SxNz,Sy0x,Sy0y,Sy0z,SyNx,SyNy,SyNz,Sz0x,Sz0y,Sz0z,SzNx,SzNy,SzNz,SzIx,SzIy,SzIz = S_Surface

        S_list = [] 
        
        -self.cZ_box[0],-self.cZ_box[0],-self.cX_box[0]
        
        name = 'Sx0'
        with h5py.File(folder+name+".h5", "w") as hf:
            hf.create_dataset("Sx", data=Sx0x,dtype = np.float32)
            hf.create_dataset("Sy", data=Sx0y,dtype = np.float32)
            hf.create_dataset("Sz", data=Sx0z,dtype = np.float32)
            hf.close()
        S_list.append(['S',name,t,(Size_Z,Size_Y,1),dx,-self.cZ_box[0],-self.cY_box[0],-self.cX_box[0]-nPML*dx])

        name = 'SxN'
        with h5py.File(folder+name+".h5", "w") as hf:
            hf.create_dataset("Sx", data=SxNx,dtype = np.float32)
            hf.create_dataset("Sy", data=SxNy,dtype = np.float32)
            hf.create_dataset("Sz", data=SxNz,dtype = np.float32)
            hf.close()
        S_list.append(['S',name,t,(Size_Z,Size_Y,1),dx,-self.cZ_box[0],-self.cY_box[0],-self.cX_box[-1]+nPML*dx])

        name = 'Sy0'
        with h5py.File(folder+name+".h5", "w") as hf:
            hf.create_dataset("Sx", data=Sy0x,dtype = np.float32)
            hf.create_dataset("Sy", data=Sy0y,dtype = np.float32)
            hf.create_dataset("Sz", data=Sy0z,dtype = np.float32)
            hf.close()
        S_list.append(['S',name,t,(Size_Z,1,Size_X),dx,-self.cZ_box[0],-self.cY_box[0]-nPML*dx,-self.cX_box[0]])

        name = 'SyN'
        with h5py.File(folder+name+".h5", "w") as hf:
            hf.create_dataset("Sx", data=SyNx,dtype = np.float32)
            hf.create_dataset("Sy", data=SyNy,dtype = np.float32)
            hf.create_dataset("Sz", data=SyNz,dtype = np.float32)
            hf.close()
        S_list.append(['S',name,t,(Size_Z,1,Size_X),dx,-self.cZ_box[0],-self.cY_box[-1]+nPML*dx,-self.cX_box[0]])

        name = 'SzI'
        with h5py.File(folder+name+".h5", "w") as hf:
            hf.create_dataset("Sx", data=SzIx,dtype = np.float32)
            hf.create_dataset("Sy", data=SzIy,dtype = np.float32)
            hf.create_dataset("Sz", data=SzIz,dtype = np.float32)
            hf.close()
        S_list.append(['S',name,t,(1,Size_Y,Size_X),dx,0.0,-self.cY_box[0],-self.cX_box[0]])

        lib_CUWA_io.Write_XMF(folder+'S_surface.xmf',S_list,timed=False)
                    
    def zoom_field_to_fine_grid(self,field):
        if self.B_mesh_zoom != 1:
            from scipy.ndimage.interpolation import zoom
            if field.shape[0] == 1:
                field1 = zoom(field[0],self.B_mesh_zoom,order=1)
                field1 = field1[np.newaxis,:,:]
            else:
                field1 = zoom(field,self.B_mesh_zoom,order=1)
        else:
            field1 = field
        return field1    

def nu_ee(n_e, T_e, lnL = 15, Zeff = 1.):
    import scipy.constants as sc
    return 4 * (2 * np.pi)**0.5 * n_e * Zeff * sc.elementary_charge**4 * lnL / (3 * (sc.electron_mass)**0.5 * (4 * np.pi * sc.epsilon_0)**2 * (sc.elementary_charge*T_e)**1.5)
    
def faster_rotated_meshgrid(cX_box,cY_box,cZ_box,M_box_to_lab,origin):    
    
    if cZ_box.size > 1:
        Zbb = np.array([cZ_box[0],cZ_box[1]])
    else:
        Zbb = cZ_box  
    x_labi,y_labi,z_labi = np.einsum('ij,jabc->icba',
                               M_box_to_lab,np.meshgrid(cX_box, cY_box, Zbb, indexing='ij')
                                )+origin[:,None,None,None]
    if cZ_box.size > 1:
        x_lab1 = np.repeat(np.array([x_labi[1] - x_labi[0]]),len(cZ_box),axis=0)
        x_lab1[0] = x_labi[0]
        np.cumsum(x_lab1, axis = 0, out= x_lab1)

        y_lab1 = np.repeat(np.array([y_labi[1] - y_labi[0]]),len(cZ_box),axis=0)
        y_lab1[0] = y_labi[0]
        np.cumsum(y_lab1, axis = 0, out= y_lab1)

        z_lab1 = np.repeat(np.array([z_labi[1] - z_labi[0]]),len(cZ_box),axis=0)
        z_lab1[0] = z_labi[0]
        np.cumsum(z_lab1, axis = 0, out= z_lab1)
    else:
        x_lab1,y_lab1,z_lab1 = x_labi,y_labi,z_labi
    return x_lab1,y_lab1,z_lab1

def N_perp(N_z,X,wc):
    eps=    1.0-(X)/(1-wc**2)
    eta=    1.0-(X)
    g  = wc*(X)/(1-wc**2)
    par_a = eps
    par_b = (g**2-(eps+eta)*(eps-N_z**2))
    par_c = -eta*g**2+eta*(eps-N_z**2)**2
    par_D = par_b**2-4.0*par_a*par_c
 
    rNx1 = np.sqrt((-par_b+np.sqrt(par_D))/(2.0*par_a))
    rNx2 = np.sqrt((-par_b-np.sqrt(par_D))/(2.0*par_a))
    return rNx1,rNx2
        
def polarization_wp_eq_0(b_lab,wc_w,wp_w,k_lab,polarization='O'):
    N_stix_z  = np.dot(k_lab,b_lab)
    
    if wp_w > 0.0005:
        Nx1,Nx2 = N_perp(N_stix_z,wp_w**2,wc_w) #Roots of the cold-dispersion relation
        mod_key = float(polarization == 'O') - 0.5
        N_stix_xy = Nx1 if mod_key * (wc_w - 1) < 0 else Nx2
    else:
        N_stix_xy = np.sqrt(1.-N_stix_z**2)

    if polarization == 'O':
        E_stix_z = 1.0+0.j
        F = 0.5*wc_w*(N_stix_xy**2*wc_w+np.sqrt(wc_w**2*N_stix_xy**4+4.*N_stix_z**2))
        E_stix_x = -N_stix_z/N_stix_xy*(1.+wp_w**2*(1.+F-2.*wc_w**2/F)/(2.*N_stix_xy**2*(1.-wc_w**2)))*E_stix_z+0.j
        E_stix_y = 0.+1.j*wc_w/F*E_stix_x
        E_stix   = np.array((E_stix_x,E_stix_y,E_stix_z))

    if polarization == 'X':
        E_stix_y = 1.0+0.j
        F = 0.5*wc_w*(N_stix_xy**2*wc_w-np.sqrt(wc_w**2*N_stix_xy**4+4.*N_stix_z**2))
        fa = F-N_stix_xy**2*wc_w**2
        E_stix_x = 0.+1.j*wc_w*(-N_stix_z**2+wp_w**2*(F-wc_w**2)*(fa-N_stix_z**2)/((1.-wc_w**2)*fa))/fa
        E_stix_z = 0.+1.j*wc_w*N_stix_z*N_stix_xy/fa*(1.+wp_w**2/(N_stix_xy**2*(1.-wc_w**2))*(0.5*(1.+F)-N_stix_z**2*F/fa))
        E_stix   = np.array((E_stix_x,E_stix_y,E_stix_z))
    
    E_stix = E_stix/np.linalg.norm(E_stix)
    stix_Z = b_lab
    stix_X = k_lab - stix_Z * np.dot(k_lab,stix_Z)
                                                             #X                   #Y
    M_lab_to_stix, M_stix_to_lab = Constract_Rotation_Matrix(stix_X,np.cross(stix_Z,stix_X))
    return np.dot(M_stix_to_lab,E_stix)

def Constract_Rotation_Matrix(X,Y):
    Y = Y/np.linalg.norm(Y)
    #X = X - Y * np.dot(X,Y) #In case Y is not orthogonal to X
    X = X/np.linalg.norm(X)
    Z = np.cross(X,Y)
    M_lab_to_box = np.vstack((X,Y,Z))
    M_box_to_lab = np.linalg.inv(M_lab_to_box)
    return M_lab_to_box, M_box_to_lab

def linspace32(l_min,minl_max,dx,block,nPML):
    num = int(block * np.ceil(((minl_max - l_min)/dx + 2. * nPML)/block))
    return np.linspace(l_min-dx*nPML, l_min-dx*nPML + num*dx, num=num)

def disect(L1,L2,P1,P2):
    x1,y1,b = L1
    x2,y2,b = L2
    x3,y3,b = P1
    x4,y4,b = P2
    return ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        
#misc helpers
import timeit
def progress_bar(progress,block_status=0,start_timer=None):
    barLength = 80 # Modify this to change the length of the progress bar
    text   = ""
    status = ""
    if block_status == 0:
        start_timer = timeit.default_timer()
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    blocks = int(round(barLength*progress))
    if progress == 1:
        status = "\nComputation time: {0:.1f} s\r\n".format(timeit.default_timer() - start_timer)
    if blocks > block_status or progress >= 1:
        block_status = blocks
        text = "\r"+text+"[{0}] {1:.0f}% {2}".format( "#"*block_status + "-"*(barLength-block_status), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()
    return block_status, start_timer
