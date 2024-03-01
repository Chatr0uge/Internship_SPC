import numpy as np
import scipy as sp
import scipy.constants as sc 
import scipy.fftpack as fft
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import sys,os; sys.path.insert(0, os.path.expanduser('./libraries/lib_CUWA/')); 
import lib_CUWA_core as lib_CUWA #This library contains minimal code needed to run CUWA
import gaussian_beam as GB
from tqdm.auto import tqdm
import json

class SPR_Analysis(lib_CUWA.CUWA): 
    
    def init(self, config_path):
        
        lib_CUWA.CUWA.__init__(self)
        self.CONSTANTS = json.load(open(config_path))
        self.CONSTANTS['physics_constant']['w'] = 2*np.pi*self.CONSTANTS['f'] * 1e9
        
        return self
    
    def generate_Gaussian_turb(self) -> np.ndarray : 
        """
        generate_Gaussian_turb generates a Gaussian turbulence field in 2D.

        Parameters
        ----------
        lx : float
            _description_
        ly : float
            _description_
        nx : float
            _description_
        ny : int
            _description_
        dx : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """   
        nx, ny, dx,  = self.Size_Y,self.Size_X,self.dx
        lx,ly = self.CONSTANTS['physics_constant']['lcx'],self.CONSTANTS['physics_constant']['lcy']
        
        kx = 2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/nx,1/dx/nx)
        ky = 2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/ny,1/dx/ny)
        
        expx = np.exp(-kx**2*lx**2/8)
        expy = np.exp(-ky**2*ly**2/8)
        
        spectra = np.exp(np.random.rand(nx,ny)*2j*np.pi)
        spectra = spectra*expx[:,None]*expy[None,:]
        
        fluct = fft.ifftshift(fft.ifft2(fft.ifftshift(spectra)))
        
        return fluct.real * np.pi / dx**2 * lx * ly
    
    def generate_global_density(self) : 
        
        X,Y,W = self.get_fine_grid()
        
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
            
        dn= self.Gaussian_turb()
        dn=(dn-dn.mean())/ dn.std()
        
        band=np.exp( -(np.fmax(Y,0)-Y[0,ycut,0])**2 / ldn**2)
        dn=np.multiply(amp*dn*nc,band)
        dn[n0+dn<0]=0    
        dn[n0==0]=0
        
        
        
        w_p = sc.e * np.sqrt((n0 + dn) /(sc.m_e * sc.epsilon_0))
        w_p_extr2 = np.copy(w_p)
        w_p_extr2[w_p_extr2 > 0] = 1
        blure_image = gaussian_filter(w_p,10)
        blure_edge  = gaussian_filter(w_p_extr2,10)
        
        return  w_p * (blure_edge > 0.98) + blure_image * (blure_edge <= 0.98)
    
    def initialize_run(self) : 
        self.set_comp_grid()
        self.Gaussian_turb = self.generate_Gaussian_turb()
        self.global_density = self.generate_global_density()
        
        return self