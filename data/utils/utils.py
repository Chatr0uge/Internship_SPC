import numpy as np
from DataBase import DataBase
from tqdm import tqdm
from scipy.stats import describe
import os 

def delay(time : np.ndarray, sig : np.ndarray) -> float : 
    """
    delay return the time delay of the signal

    Parameters
    ----------
    time : np.ndarray
        _description_
    sig : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
    """  
    sig = np.nan_to_num(sig, nan = 0)  
    return time[np.argmax(abs(sig))]

def delay_dispersion(time, signal) -> float: 
    """
    delay_dispersion return the dispersion of the signal given a threshold 1/e * max(signal)

    Parameters
    ----------
    time : np.ndarray
        .shape.shapeiption_
    signal : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
    """    
    signal = np.nan_to_num(signal, nan = 0)  

    mask = np.abs(signal) > np.exp(-1) * np.max(np.abs(signal))
    time_window = time[mask]

    return (time_window[-1] - time_window[0]) / 2 

def window_around_max(pulse, window_size = 2000) : 
    
    pulse = abs(pulse) 
    window = pulse[np.argmax(pulse) - window_size // 2 : np.argmax(pulse) + window_size //2]
    
    if window.shape[0] < window_size : 
        pad = window_size - window.shape[0]
        window =  pulse[np.argmax(pulse) - window_size // 2 - pad: np.argmax(pulse) + window_size //2]
        
    
    return window


def compute_area_left_right(signal) : 
    left= np.trapz(signal[:signal.shape[0]//2])
    right = np.trapz(signal[signal.shape[0]//2:])
    return right / left


def hysteresis_array(array, window_size = 1000) :
    window = window_around_max(array, window_size)
    return compute_area_left_right(window)



def convert_to_database(dir : str, db_path : str) -> None :
    
    """
    convert_to_database used to  convert the Data extracted from LEONARDO simulations

    Parameters
    ----------
    dir : str
        path of the parent folder of simulation results
    db_path : str
        path of the database
        
    """    
    
    
    pulse_list = []
    delay_list = []
    pulse_width = []
    params = np.load(dir  + '/params.npy', allow_pickle=True)

    #### INITIATING THE DATABASE ###
    
    db = DataBase(db_path)

    for i in range(len(params)): 
        print(i, end='\r')
        IQ_list = np.array([np.load(dir + f'/id_{i}/{file}') for file in os.listdir(dir + f'/id_{i}') if file[0] == 'I'])
        pl = np.nan_to_num(IQ_list[:,1], nan = 0)
        pulse_list.append(abs(np.array(pl)))
        delay_list.append(np.array([delay(*IQ) for IQ in IQ_list]))
        pulse_width.append(np.array([delay_dispersion(*IQ) for IQ in IQ_list]))
    np.save('pulse_list.npy', pulse_list)
    np.save('delay_list.npy', delay_list)
    np.save('pulse_width.npy', pulse_width)
    
    for i , sim_params in tqdm(enumerate(params)) : 
        
        dsc = describe(abs(np.array(pulse_list[i])), axis = 1)
        Keys = ['nobs', 'minmax', 'mean', 'variance', 'skewness', 
                'kurtosis']
        
        ds =dict(zip(Keys,dsc))
        d2 = {'id': i, 'amplitude': sim_params['amp'], 'lcx': sim_params['lcx'], 'lcy': sim_params['lcy'], 'bm': sim_params['bm'],
                'dt': 2.4509803921568632e-11, 'pulse_list': np.mean(abs(pulse_list[i]), axis = 0), 
                'delay_list': np.array(delay_list[i]), 'pulse_width': np.array(pulse_width[i]), 'skewness': ds['skewness'], 
                'kurtosis': ds['kurtosis'], 'mean_signal': ds['mean'], 'variance': ds['variance']
                }
        db.insert(d2)
