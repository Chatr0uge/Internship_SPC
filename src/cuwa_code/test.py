from SPR_run import SPR_Analysis
import numpy as np
import sys,os; sys.path.insert(0, os.path.expanduser('./libraries/lib_CUWA/')); 
import lib_CUWA_core as lib_CUWA #This library contains minimal code needed to run CUWA
from mpi4py import MPI
from utils.utils import * 

### DEFINING THE RUNNER AND THE ID OF THE BATCH ###

batch_id=int(os.environ['SLURM_ARRAY_TASK_ID'])
rank=MPI.COMM_WORLD.Get_rank()
full_id = batch_id+rank
SPRA = SPR_Analysis(config_path='./config.json', CUWA_instance = lib_CUWA.CUWA(GPU_device_id=rank))

#### INITIATING THE DATABASE ###
#from DataBase import DataBase
#db = DataBase('/leonardo_work/FUAL8_SYNTGENE/SPR_run.db')


### DEFINING THE PARAMETERS GRID ###

from sklearn.model_selection import ParameterGrid
parameter = dict(amp = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,  
                        0.5, 0.7, 1, 2, 5] , 
                 lcx = [1, 0.5], 
                 lcy = [0.5, 2, 30],
                 bm = [1, 4, 10]
                )

params_grids = ParameterGrid(parameter)

### CREATING THE FOLDERS TO STORE THE DATA ###

folder='/leonardo_work/FUAL8_SYNTGENE/output_amp_9' #output folder
if not os.path.exists(folder): 
    os.makedirs(folder)
    
for i in range(len(params_grids)):    
    fname=folder+"/id_{}".format(i)
    
    if not os.path.exists(fname):
        os.makedirs(fname)
        
### RUNNING THE INDICATOR SIMULATION ###
        
_ = 0 * SPRA.generate_global_density()
SPRA.run(_)
np.save(folder + '/indicator.npy', [SPRA.time_list, SPRA.IQ])
IQ_0 = SPRA.IQ

### RUNNING THE SIMULATION FOR DIFFERENT PARAMETERS ###

seed_array = []

for i, params in enumerate([params_grids[1]]): 
    
    SPRA.CONSTANTS['physics_constant']['amp'], SPRA.CONSTANTS['physics_constant']['lcx'], SPRA.CONSTANTS['physics_constant']['lcy'],  SPRA.CONSTANTS['physics_constant']['ro'] = params['amp'], params['lcx'], params['lcy'], params['bm']
    
    seed = np.random.randint(low = 0, high = 2**31)
    seed_array.append(seed)
     
    gaussian_field = SPRA.generate_global_density(seed = seed)
        
    SPRA.run(gaussian_field)
    
    #db.c.execute("""SELECT EXISTS(SELECT id FROM data where id = ?)""", (i,))
    #exist = db.c.fetchall()[0][0]
    
    #if exist : # checking for multiple processes handling the same id
        
        #db.c.execute("SELECT  pulse_list, delay_list, pulse_width FROM data where id = ? ", (i, ))
        
        #pulse_list, delay_list, pulse_width = db.c.fetchall()[0]
        #print(pulse_list)
        
        #pulse_list = np.array(pulse_list.tolist().append(SPRA.IQ))
        #delay_list = np.array(delay_list.tolist().append(delay(SPRA.time_list, SPRA.IQ - IQ_0)))
        #pulse_width = np.array(pulse_width.tolist().append(delay_dispersion(SPRA.time_list, SPRA.IQ - IQ_0)))
         
        #db.c.execute("""UPDATE data SET pulse_list = ? , delay_list = ? , pulse_width = ? WHERE id = ?; """, (pulse_list, delay_list, pulse_width,i))
        #db.conn.commit()
        
    #else :

        #db.insert({'id': i, 'amplitude': params['amp'], 'lcx': params['lcx'], 'lcy': params['lcy'], 'bm': params['bm'], 'time_list': SPRA.time_list, 'pulse_list': np.array([SPRA.IQ]), 'delay_list': np.array([delay(SPRA.time_list,  SPRA.IQ - IQ_0)]), 'pulse_width': np.array([delay_dispersion(SPRA.time_list, SPRA.IQ - IQ_0)])})

    
    np.save(folder+"/id_{}/IQ_{}.npy".format(i, full_id), [SPRA.time_list, SPRA.IQ - IQ_0])
    
np.save(folder+"/seed_{}.npy".format(full_id), seed_array)
np.save(folder+"/params.npy",np.array(params_grids))