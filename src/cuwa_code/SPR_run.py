import numpy as np
import scipy as sp
import scipy.constants as sc 
import scipy.fftpack as fft
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import sys,os; sys.path.insert(0, os.path.expanduser('./libraries/lib_CUWA/')); 
import lib_CUWA_core as lib_CUWA #This library contains minimal code needed to run CUWA
import gaussian_beam as GB
import json


class SPR_Analysis: 
    
    def __init__(self, config_path : str, CUWA_instance : lib_CUWA.CUWA):
        
        self.runner = CUWA_instance
        self.CONSTANTS = json.load(open(config_path)) #loading the simulation constants
        
        self.CONSTANTS['physics_constant']['w'] = 2*np.pi*self.CONSTANTS['physics_constant']['fs'] * 1e9
        self.CONSTANTS['diff_shemes_constant']['origin'] = np.array(self.CONSTANTS['diff_shemes_constant']['origin'])
        self.CONSTANTS['diff_shemes_constant']['e_x'] = np.array(self.CONSTANTS['diff_shemes_constant']['e_x'])
        self.CONSTANTS['diff_shemes_constant']['e_y'] = np.array(self.CONSTANTS['diff_shemes_constant']['e_y'])
        
        ##TODO : resize the x-domain consistent with the beam size

        self.CONSTANTS['diff_shemes_constant']['x_range'] = [-3 * self.CONSTANTS['physics_constant']['ro'], 3 * self.CONSTANTS['physics_constant']['ro']]
        
    def generate_Gaussian_turb(self, seed = None) -> np.ndarray : 
        """
        generate_Gaussian_turb generate a 2D Gaussian turbulence field

        Returns
        -------
        np.ndarray
            2d array of the generated turbulence field
        """        
        
        self.runner.set_comp_grid(omega_ref = self.CONSTANTS['physics_constant']['w'], **self.CONSTANTS['diff_shemes_constant'])
        nx, ny, dx,  = self.runner.Size_Y,self.runner.Size_X,self.runner.dx
        lx,ly = self.CONSTANTS['physics_constant']['lcx'],self.CONSTANTS['physics_constant']['lcy']
        
        kx = 2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/nx,1/dx/nx)
        ky = 2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/ny,1/dx/ny)
        
        expx = np.exp(-kx**2*lx**2/8)
        expy = np.exp(-ky**2*ly**2/8)
        if seed is not None : 
            np.random.seed(seed)
        spectra = np.exp(np.random.rand(nx,ny)*2j*np.pi)
        spectra = spectra*expx[:,None]*expy[None,:]
        
        fluct = fft.ifftshift(fft.ifft2(fft.ifftshift(spectra)))
        self.turbulence_field = fluct.real * np.pi / dx**2 * lx * ly
        
        return self.turbulence_field
    
    def generate_global_density(self, seed = None) -> np.ndarray  : 
        
        """
        generate_global_density generate the final density field with gaussian turbulence with smooth edges

        Returns
        -------
        np.ndarray
            _description_
        """        
        
        self.runner.set_comp_grid(omega_ref = self.CONSTANTS['physics_constant']['w'], **self.CONSTANTS['diff_shemes_constant'])
        X,Y,W = self.runner.get_fine_grid()
        
        ldn, ymin, ycut = self.CONSTANTS['physics_constant']['ldn'], self.CONSTANTS['physics_constant']['ymin'], self.CONSTANTS['physics_constant']['ycut']
        L = self.CONSTANTS['physics_constant']['L']
        w = self.CONSTANTS['physics_constant']['w']
        amp = self.CONSTANTS['physics_constant']['amp']
        teta = self.CONSTANTS['physics_constant']['teta']
        
        nc=w**2/sc.e**2*(sc.m_e*sc.epsilon_0)
        n0=nc*np.fmax(Y-ymin,0)/L
        n0[n0<0]=0
            
        while n0[0,ycut,0]<nc*np.cos(teta)**2:
            ycut+=1
            
        dn= self.generate_Gaussian_turb(seed)
        dn=(dn-dn.mean())/ dn.std()
        
        band=np.exp( -(np.fmax(Y,0)-Y[0,ycut,0])**2 / ldn**2)
        dn=np.multiply(amp*dn*nc,band)
        dn[n0+dn<0]=0    
        dn[n0==0]=0
        
        
        n_total = n0 + dn * Y/L # for reducing the impact of the turbulence on the edges
        w_p = sc.e * np.sqrt((n_total) /(sc.m_e * sc.epsilon_0))
        w_p_extr2 = np.copy(w_p)
        w_p_extr2[w_p_extr2 > 0] = 1
        blure_image = gaussian_filter(w_p,10)
        blure_edge  = gaussian_filter(w_p_extr2,10)
        
        self.global_field = w_p * (blure_edge > 0.98) + blure_image * (blure_edge <= 0.98)
        return  self.global_field
    
    def run(self, global_density : np.ndarray, **kwargs) -> None :
        """
        run run the simulation with the given global density field

        Parameters
        ----------
        global_density : np.ndarray
            _description_
        """        
        self.runner.set_comp_grid(omega_ref = self.CONSTANTS['physics_constant']['w'], **self.CONSTANTS['diff_shemes_constant'])
        X,Y,W = self.runner.get_fine_grid()
        
        w_c = sc.e / sc.m_e * 2.4 + 0. * (X)
        ratio=self.runner.Size_X/self.runner.Size_Y
        b_x = b_y = np.zeros_like(X)
        b_z = np.ones_like(X)
        
        dt = self.runner.dt
        Cour = self.runner.Courant
        steps = self.runner.suggested_n_steps
        
        fs = self.CONSTANTS['physics_constant']['fs']
        Tfix=np.ceil(1/fs/1e9/dt)

        self.runner.Courant=fs/fs*self.runner.n_steps_per_lambda/Tfix
        self.runner.dt=dt*self.runner.Courant/Cour
        self.runner.suggested_n_steps=steps*Cour/self.runner.Courant

        
        beam = GB.GaussianBeam(frequency_GHz = fs,
                                        mode          = 'O',
                                        origin        = np.array((0.0,0,0)),
                                        gaussian_focus= np.array((0.003*np.tan(self.CONSTANTS['physics_constant']['teta']),0.003,0)),
                                        waist_radius  = self.CONSTANTS['physics_constant']['ro'],
                                        pulse = 0.9) #ns  

        self.runner.set_plasma_data(global_density, w_c, b_x, b_y, b_z) # TODO ask for the meaning of the parameters b_x, b_y, b_z
        self.runner.set_antenna(beam, source='True', receiver='True')
        #self.runner.set_antenna(beam2, source='False', receiver='True')
        self.runner.run_init()
        
        steps=round(steps*1.7)
        reAz_list=[];imAz_list=[];t_list = [];IQ=[] #preparing list for the signal
        record=1 #recognizing the arrival of the pulse
        
        for t in range(int(steps)):
            
            self.runner.run_source(t,beam)
            self.runner.run_phasor(t,beam)
            reAx_g,reAy_g,reAz_g,imAx_g,imAy_g,imAz_g = [field.get() for field in beam.A_g]
            self.runner.run_step()

            
        
            if t % (beam.T//8) == 0 and t> 1*beam.T:             
            
                if record == 1:   #recording the previous period signal
                    t_list.append(t*self.runner.dt)
                    IQ.append(np.sum(reAz_list)+1j*np.sum(imAz_list))
                    #reAz_list=[]  #resetting the recording
                    #imAz_list=[]
            
        
            if record==1:   #recording data for each period 
                reAz_list.append(np.trapz(reAz_g[0,0,:]))   
                imAz_list.append(np.trapz(imAz_g[0,0,:]))    
                if len(reAz_list)>beam.T:
                    reAz_list.pop(0)
                    imAz_list.pop(0)
        
        self.time_list = np.array(t_list)
        self.IQ = np.array(IQ)
