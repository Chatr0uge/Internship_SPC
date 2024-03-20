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
    sig = np.nan_to_num(sig, nan = -1000)  
    return time[np.argmax(abs(sig))]

def delay_dispersion(time, signal) -> float: 
    """
    delay_dispersion return the dispersion of the signal given a threshold 1/e * max(signal)

    Parameters
    ----------
    time : np.ndarray
        _description_
    signal : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
    """    
    signal = np.nan_to_num(signal, nan = -1000)  

    mask = np.abs(signal) > np.exp(-1) * np.max(np.abs(signal))
    time_window = time[mask]

    return (time_window[-1] - time_window[0]) / 2 

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
    
    
    amplitude_norm = []
    pulse_list = []
    delay_list = []

    params = np.load(dir  + '/params.npy', allow_pickle=True)

    #### INITIATING THE DATABASE ###
    
    db = DataBase(db_path)

    for i , params in enumerate(params) : 
        print(i)
        IQ_list = np.array([np.load(dir + f'/id_{i}/{file}') for file in os.listdir(dir + f'/id_{i}') if file[0] == 'I'])
        pulse_list.append(IQ_list[:,1])
        delay_list.append([delay(*IQ) for IQ in IQ_list])
        pulse_width = [delay_dispersion(*IQ) for IQ in IQ_list]

    for i , params in tqdm(enumerate(params)) : 
        
        sim_params = params[i]
        dsc = describe(abs(np.array(pulse_list[i])), axis = 1)
        Keys = ['nobs', 'minmax', 'mean', 'variance', 'skewness', 
                'kurtosis']
        
        ds =dict(zip(Keys,dsc))
        db.insert({'id': i, 'amplitude': sim_params['amp'], 'lcx': sim_params['lcx'], 'lcy': sim_params['lcy'], 'bm': sim_params['bm'],
                'dt': IQ_list[0,1] - IQ_list[0,0], 'pulse_list': np.mean(pulse_list[i], axis = 0), 
                'delay_list': np.array(delay_list[i]), 'pulse_width': np.array(pulse_width[i]), 'skewness': ds['skewness'], 
                'kurtosis': ds['kurtosis'], 'mean_signal': ds['mean'], 'variance': ds['variance']
                })