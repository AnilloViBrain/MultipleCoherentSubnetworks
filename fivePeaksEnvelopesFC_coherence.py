#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:28:29 2023

@author: Felipe
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../'))
import analysis.frequency as frequency
import analysis.synchronization as synchronization
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc
import tqdm
from multiprocessing import Lock
import concurrent.futures
import time

def RunFCD(seed):
    time.sleep(0.1)
    dt=1e-3
    fs=1000
    #t=np.arange(20,300,dt)
    N=90
    Nparts=20
    Nfb=5

    optimalK=4     
    directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/AAL90HalfHours/'
    #directory='/media/felipe/Elements/Kuramoto_HeatMap/'

    #Selected K
    K=4
    #Selected MD
    MD=0.021

    nperseg=5000
    noverlap=2500
    f=np.linspace(0,fs/2,nperseg//2+1)
    freq_peaks=f[[65,75,147,206,215]] #13,15,29.4,41.2,43
    freq_indexes=[65,75,147,206,215]
        
    # tt=np.arange(0,1780,dt)
    Twindow=89*fs
    fbands=[]
    f_lows=np.zeros_like(freq_peaks)
    f_highs=np.zeros_like(freq_peaks)
    for fp,freq in enumerate(freq_peaks):
        f_lows[fp]=freq-0.5
        f_highs[fp]=freq+0.5
        fbands.append('%.2f-%.2f Hz'%(f_lows[fp],f_highs[fp]))

    #pointsFC=[384,334,170,122,118] #5 cycles
    pointsFC=[231,200,102,73,70] #3 cycles
    
    print('Seed:',seed)
    filename=directory+'AAL90_Long_freqMean40HZ_freqSD0HZ_K%.3f_MD%.3f_seed%d'%(K*90,MD,seed)
    file_dict=sio.loadmat(filename)
    file_theta=file_dict['theta'][20000::,:]
    for fb,fband in enumerate(range(Nfb)):
        for n in tqdm.tqdm(range(Nparts)):
            sinTheta=np.sin(file_theta[n*Twindow:(n+1)*Twindow,:]).T
        
            ## Envelopes    
            # envelopes_low=synchronization.envelopesFrequencyBand(theta.T,f_low=f_lows[fb],f_high=f_highs[fb],fs=1000,applyLow=True)
            ## Total time correlation FC
            # FC=np.corrcoef(envelopes_low)
            
            ##FCD using coherence between band-pass envelopes
            FCD,corr_vectors,shift_amplitude=synchronization.extract_FCD(sinTheta,wwidth=pointsFC[fb]*2,wcoh=pointsFC[fb],maxNwindows=100000,olap=0.5,nfft=nperseg,freq_index=freq_indexes[fb],mode='ccoh')
            np.savez('/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence/FCenvelopes_Coherence2_Amplitude_fb%s_K=%d_MD=%d_seed%d_n%d.npz'%(fband,K,MD*1000,seed,n),FCD=np.abs(FCD),corr_vectors=np.abs(corr_vectors),shift=shift_amplitude)
            np.savez('/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence/FCenvelopes_Coherence2_Phase_fb%s_K=%d_MD=%d_seed%d_n%d.npz'%(fband,K,MD*1000,seed,n),FCD=np.imag(FCD),corr_vectors=np.imag(corr_vectors),shift=shift_amplitude)
            np.savez('/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence/FCenvelopes_Coherence2_Real_fb%s_K=%d_MD=%d_seed%d_n%d.npz'%(fband,K,MD*1000,seed,n),FCD=np.real(FCD),corr_vectors=np.real(corr_vectors),shift=shift_amplitude)
            
            del corr_vectors,FCD,shift_amplitude #don't delete if figures are required
            gc.collect()
        del sinTheta
        gc.collect()
    del file_theta

#Different seed 
#First ten seeds
#seeds=[3,5,8,13,21,34,55,89,144,233]
#Next ten seeds
seeds=np.arange(145,173,1,dtype=int)
for j in range(1):
    print('Starting FCD calculation')
    lock = Lock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        executor.map(RunFCD, seeds)
    
    
