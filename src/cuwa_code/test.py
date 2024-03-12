from SPR_run import SPR_Analysis
import numpy as np
import sys,os; sys.path.insert(0, os.path.expanduser('./libraries/lib_CUWA/')); 
import lib_CUWA_core as lib_CUWA #This library contains minimal code needed to run CUWA
from mpi4py import MPI


batch_id=int(os.environ['SLURM_ARRAY_TASK_ID'])
rank=MPI.COMM_WORLD.Get_rank()
full_id = batch_id+rank
SPRA = SPR_Analysis(config_path='./config.json', CUWA_instance = lib_CUWA.CUWA(GPU_device_id=rank))
amplitude = np.logspace(-3,1, 50)

folder='./output' #output folder
if not os.path.exists(folder):
    os.makedirs(folder)
    
for amp in amplitude:    
    fname=folder+"/a_{:.4f}".format(amp)
    
    if not os.path.exists(fname):
        os.makedirs(fname)
        
_ = 0 * SPRA.generate_global_density()
SPRA.run(_)
np.save('./output/indicator.npy', [SPRA.time_list, SPRA.IQ])
IQ_0 = SPRA.IQ

seed_array = []
for amp in amplitude : 
    
    SPRA.CONSTANTS['physics_constant']['amp'] = amp
    seed = np.random.randint(low = 0, high = 2**31)  
    seed_array.append(seed)   
    gaussian_field = SPRA.generate_global_density(seed = seed)
        
    SPRA.run(gaussian_field)
    
    np.save(folder+"/a_{:.4f}/IQ_{}.npy".format(amp, full_id), [SPRA.time_list, SPRA.IQ - IQ_0])
    np.save(folder + "/a_{:.4f}/std_{}.npy".format(amp, full_id), [np.std(SPRA.turbulence_field), np.std(SPRA.turbulence_field, axis  =  0), np.std(SPRA.turbulence_field, axis = 1)])
    # E = SPRA.runner.get_magE() 
    # np.save(folder+"/a_{:.4f}/E_{}.npy6-=".format(amp, full_id),E)


np.save(folder+"/seed_{}.npy".format(full_id), seed_array)