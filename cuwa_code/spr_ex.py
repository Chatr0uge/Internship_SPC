import numpy as np
import scipy as sp
import scipy.constants as sc 
import scipy.fftpack as fft

import sys,os; sys.path.insert(0, os.path.expanduser('../../lib_CUWA/')); 
import lib_CUWA_core as lib_CUWA #This library contains minimal code needed to run CUWA
import gaussian_beam as GB




def Delay(t,sig):
    s=np.abs(sig)

    tav=0;
    smax=0;
    for it in range(len(s)):
        if s[it]>smax:
            smax=s[it]
            tav=t[it]

    return tav

def Ddisp(t,sig,tav):
    s=np.abs(sig)
    it=0

    while t[it]<tav and it<len(s)-1:
        it+=1
    
    ut=it 
    while s[ut]>s[it]*np.exp(-1) and ut<len(s)-1:
        ut+=1

    lt=it 
    while s[lt]>s[it]*np.exp(-1) and lt<len(s)-1:
        lt-=1
    return (t[ut]-t[lt])/2


def Gaussian_turb(lx,ly,nx,ny,dx):
    kx=2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/nx,1/dx/nx)
    ky=2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/ny,1/dx/ny)
    expx=np.exp(-kx**2*lx**2/8)
    expy=np.exp(-ky**2*ly**2/8)
    spectra=np.exp(np.random.rand(nx,ny)*2j*np.pi)
    spectra=spectra*expx[:,None]*expy[None,:]
    fluct=fft.ifftshift(fft.ifft2(fft.ifftshift(spectra)))
    return np.real(fluct)*np.pi/dx**2*lx*ly
    #fluct=sp.fft.ifft2(sp.fft.ifftshift(spectra))

def Gaussian_turb1d(lc,nx,ny,dx):

    kx=2*np.pi*np.arange(-0.5/dx,0.5/dx-0.5/dx/nx,1/dx/nx)
    expx=np.exp(-kx**2*lc**2/8)
    spectra=np.exp(np.random.rand(nx,)*2j*np.pi)
    spectra=spectra[:,None]*expx[:,None]*np.ones((nx,ny))
    fluct=fft.ifftshift(fft.ifft2(fft.ifftshift(spectra,axes=0),axes=0),axes=0)
    return np.real(fluct)*np.sqrt(np.pi)/dx*lc
    #fluct=sp.fft.ifft2(sp.fft.ifftshift(spectra))

def run_CUWA(C1,w_p, w_c, b_x, b_y, b_z,numplots,path):
    C1.set_plasma_data(w_p, w_c, b_x, b_y, b_z)
    C1.set_antenna(beam, source='True', receiver='True')
    #C1.set_antenna(beam2, source='False', receiver='True')
    C1.run_init()
    progress_status = 0; st = 0;
    steps=round(C1.suggested_n_steps*1.7)
    reAz_list=[];imAz_list=[];t_list = [];IQ=[] #preparing list for the signal
    record=1 #recognizing the arrival of the pulse


    for t in range(int(steps)):
        progress_status, st = lib_CUWA.progress_bar(1.*t/(steps-1),progress_status, st)
        C1.run_source(t,beam)
        C1.run_phasor(t,beam)
        reAx_g,reAy_g,reAz_g,imAx_g,imAy_g,imAz_g = [field.get() for field in beam.A_g]
        C1.run_step()


        if numplots> 0 and (t+1) % np.floor(steps/numplots) == 0:
            fig   = plt.figure(figsize=(10*ratio,10))
            plt.contourf(X[0],Y[0],C1.w_p[0],30,cmap='cividis')
            plt.contour(X[0],Y[0],C1.get_magE()[0], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],cmap='plasma')
            fig.savefig(path+'E_{0}.png'.format(t+1),dpi=400)
            plt.close()
            

        
    
        if t % (beam.T//8) == 0 and t> 1*beam.T:             
        
            if record == 1:   #recording the previous period signal
                t_list.append(t*C1.dt)
                IQ.append(np.sum(reAz_list)+1j*np.sum(imAz_list))
                #reAz_list=[]  #resetting the recording
                #imAz_list=[]
        
    
        if record==1:   #recording data for each period 
            reAz_list.append(np.trapz(reAz_g[0,0,:]))   
            imAz_list.append(np.trapz(imAz_g[0,0,:]))    
            if len(reAz_list)>beam.T:
                reAz_list.pop(0)
                imAz_list.pop(0)
    
    return t_list, IQ



folder='output/' #output folder
if not os.path.exists(folder):
    os.makedirs(folder)

#batch_id=int(os.environ['SLURM_ARRAY_TASK_ID'])
#rank=MPI.COMM_WORLD.Get_rank()
full_id = 0



freqs=[51, 50.5, 50.3, 50.1, 50.03, 50, 49.97, 49.9, 49.7, 49.5, 49]
f0=np.max(freqs); #probing frequency


for fs in freqs:    
    fname=folder+"f_{:.4f}".format(fs)
    if not os.path.exists(fname):
        os.makedirs(fname)
    


C1 = lib_CUWA.CUWA() # Initialisation of the CUWA. When running on local GPU the GPU context is initialised 
L=0.1*f0**2/2500
lcx=0.01
lcy=0.02
teta=0*sc.pi/180
ro=0.05
amp=0.005*f0**2/2500


w=f0*sc.pi*2*1e9



X,Y,Z = C1.set_comp_grid(omega_ref = w, 
                         vac_resolution = 24,
                         origin = np.array((0,0,0)), 
                               e_x = np.array((1,0,0)), 
                               e_y = np.array((0,1,0)), 
                               x_range = (-0.15,0.15), 
                               y_range =  (0,0.21), 
                               z_thick  =  0.0,
                               nPML=24,
                               Courant = 0.5)




w_c = sc.e / sc.m_e * 2.4 + 0. * (X)
ratio=C1.Size_X/C1.Size_Y






b_x = b_y = np.zeros_like(X)
b_z = np.ones_like(X)


#preparing background density
ymin=0.05
nc=w**2/sc.e**2*(sc.m_e*sc.epsilon_0)
n0=nc*np.fmax(Y-ymin,0)/L
n0[n0<0]=0
x=np.fmax(Y-ymin,0)
ycut=0
while n0[0,ycut,0]<nc*np.cos(teta)**2:
    ycut+=1

w_p = sc.e * np.sqrt((n0) /(sc.m_e * sc.epsilon_0))
w_p_extr2 = np.copy(w_p)
w_p_extr2[w_p_extr2 > 0] = 1
blure_image = gaussian_filter(w_p,10)
blure_edge  = gaussian_filter(w_p_extr2,10)
w_p0 = w_p * (blure_edge > 0.98) + blure_image * (blure_edge <= 0.98)

#preparing density fluctuations
dn0=Gaussian_turb(lcx,lcy,C1.Size_Y,C1.Size_X,C1.dx)
av=np.sum(dn0[:,0])/C1.Size_X
rms=np.sum((dn0[:,0]-av)**2/C1.Size_X)**0.5
print('mean value of dn = '+str(av)+', rms = '+str(rms))
dn0=(dn0-av)/rms
ldn=0.6 #fluctuations are localised around cutoff



#saving default timestep to use for different probing frequencies
dt=C1.dt
Cour=C1.Courant
steps=C1.suggested_n_steps

for fs in freqs:



    Tfix=np.ceil(1/fs/1e9/dt)
    C1.Courant=f0/fs*C1.n_steps_per_lambda/Tfix
    C1.dt=dt*C1.Courant/Cour
    C1.suggested_n_steps=steps*Cour/C1.Courant
    #print('courant is '+str(C1.Courant)+', dt is '+str(C1.dt)+', steps= '+str(C1.suggested_n_steps) +', period is ' + str(1/fs/C1.dt/1e9) )
    
    beam = GB.GaussianBeam(frequency_GHz = fs,
                                 mode          = 'O',
                                 origin        = np.array((0.0,0,0)),
                                 gaussian_focus= np.array((0.003*np.tan(teta),0.003,0)),
                                 waist_radius  = ro,
                                 pulse = 0.9) #ns    
    
    if full_id==0:
        folname=folder+'f_{:.4f}/'.format(fs)
        t_list0, IQ0 = run_CUWA(C1,w_p0*0, w_c, b_x, b_y, b_z,0,folname)         
        delay0=Delay(t_list0,IQ0)
        fname=folder+'f_{:.4f}/time.dat'.format(fs)
        wfile = open(fname, "w")
        for element in t_list0:
            wfile.write(str(element) + "\n")
        wfile.close()    

        fname=folder+"f_{:.4f}/IQ0.dat".format(fs)
        wfile = open(fname, "w")
        for element in IQ0:
            wfile.write(str(element) + "\n")
        wfile.close()    



    
    band=np.exp( -(np.fmax(Y,0)-Y[0,ycut,0])**2 / ldn**2)
    dn=np.multiply(amp*dn0*nc,band)
    dn[n0+dn<0]=0    
    dn[n0==0]=0

    w_p = sc.e * np.sqrt((n0+dn) /(sc.m_e * sc.epsilon_0))
    w_p_extr2 = np.copy(w_p)
    w_p_extr2[w_p_extr2 > 0] = 1
    blure_image = gaussian_filter(w_p,10)
    blure_edge  = gaussian_filter(w_p_extr2,10)
    w_p = w_p * (blure_edge > 0.98) + blure_image * (blure_edge <= 0.98)

    t_list, IQ = run_CUWA(C1,w_p, w_c, b_x, b_y, b_z,10,folname)
    IQ = np.array(IQ)-np.array(IQ0)      
    
    fname=folder+"f_{:.4f}/IQ_{:04d}.dat".format(fs,full_id)
    wfile = open(fname, "w")
    for element in IQ:
        wfile.write(str(element) + "\n")
    wfile.close()    

    fig = plt.figure()
    plt.plot(t_list,np.abs(IQ))
    fig.savefig(folname+'signal.png',dpi=400)

    print(Delay(t_list,IQ))

