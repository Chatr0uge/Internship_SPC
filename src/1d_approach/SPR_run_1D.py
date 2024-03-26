import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import os


class SPR_run_1d : 
    def __init__(self, lcx : float, L : float, amp : float, Nx : int, n_samples : int):
         
        self.lcx = lcx
        self.L = L
        self.amp = amp
        self.Nx = Nx
        self.x = np.linspace(0,2 * self.L, self.Nx)
        self.dx = self.x[1] - self.x[0]
        self.n_samples = n_samples
        self.n_c = 1
        
    def generate_gaussian_perturbation(self) -> np.ndarray:
        """
        generate_gaussian_perturbation generates a 1D Gaussian perturbation
        
        Returns
        -------
        np.ndarray
            1D array of the generated perturbation
        """
        
        
        phase = np.random.uniform(0, 2 * np.pi, self.Nx)
        rw = 2*np.pi*np.linspace(-0.5/self.dx,0.5/self.dx-1/self.dx/self.Nx,self.Nx)
        
        gaussian_spectrum =  np.exp(-rw**2 * self.lcx**2 / 8 + 1j * phase) / self.dx * self.lcx
        
        self.gaussian_distribution = np.fft.ifft(np.fft.ifftshift(gaussian_spectrum)).real
        return self.gaussian_distribution
    
    def generate_gaussian_samples(self) -> np.ndarray:
        """
        generate_gaussian_samples generate a set of gaussian perturbations of density field expected to be linear

        Returns
        -------
        np.ndarray
            gaussian perturbated field of density
        """   
        dn =  np.array([self.generate_gaussian_perturbation() for _ in range(self.n_samples)])
        dn = self.amp * (dn - dn.mean()) /  dn.std() 
       
        return dn

    def integrate_decay(self, dn) : 
        x = self.x
        L = self.L
        
        try : x_c = x[dn + x/L > self.n_c][0]
        except : print(dn.min(), self.lcx, self.L, self.amp)
        return (2 / 3e10)* np.trapz(1 / np.sqrt(1 - x[x < x_c]/L -  dn[x < x_c] / self.n_c), 
                                                dx=self.dx)
    
    def sampling_decay(self, dn : np.ndarray) :
    
        self.td = np.array(list(map(self.integrate_decay, dn)))
    
    def run(self) : 
        
        self.sampling_decay(self.generate_gaussian_samples())
            
class SPR_wrapper : 
    
    def __init__(self, params : dict):
        self.params_grids = list(ParameterGrid(params))
        self.results = {}
        self.td0 = {}
        self.std = {}
        
    def run_wrapper(self) -> None   :
        

        for i, params in enumerate(self.params_grids): 
            
            SPR = SPR_run_1d(**params)
            print(max(SPR.x / SPR.L), i, end = '\r')
            SPR.run()
            np.nan_to_num(SPR.td, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            mask = SPR.td > 0
            bins = np.linspace(1e-10, 1e-9,10)
            hist_td = np.histogram(SPR.td[mask], bins = bins, density = True)[0]
            self.results[i] = hist_td
            
            SPR = SPR_run_1d(**params)
            SPR.n_samples = 1
            SPR.amp = 0
            SPR.run()
            self.td0[i] = SPR.td[0] 
            
            self.std[i] = np.std(SPR.td)
                  
        
    