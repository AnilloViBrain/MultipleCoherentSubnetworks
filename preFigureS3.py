#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:31:13 2023

@author: felipe
"""

#Load KuramotoNetworksPackage repository
import sys
import os
sys.path.append(os.path.abspath('../../'))
import analysis.frequency as frequency
import analysis.synchronization as synchronization
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#Parameters
N=90
Nfb=5
fbands=['12.50-13.50 Hz','14.50-15.50 Hz','28.90-29.90 Hz','40.70-41.70 Hz','42.50-43.50 Hz']
# file_labels=sio.loadmat('../../input_data/AAL_labels')
file_labels=sio.loadmat('../../input_data/AAL_short_labels')
labels=file_labels['label90']
Nseeds=10
th=0.232
occupancy_tensor=np.zeros((N,Nseeds,Nfb))
occupancy_orig_tensor=np.zeros((N,Nseeds,Nfb))

#Load data
for seed,SEED in enumerate([3,5,8,13,21,34,55,89,144,233]):
    for fb in range(Nfb):
        file=np.load('./coo/cooccurrences_fixed_%.3f_fb%s_K=4_MD=21_seed%d.npz'%(th,fbands[fb],SEED),allow_pickle=True)
        occupancy_tensor[:,seed,fb]=file['occupancy']
    occupancy_orig_tensor[:,seed,:]=occupancy_tensor[:,seed,:]
    occupancy_tensor[:,seed,:]=occupancy_tensor[:,seed,:]/np.sum(occupancy_tensor[:,seed,:],axis=1,keepdims=True)
    
#Plot
fig= plt.figure(figsize=(8.5,8.5))
gs=gridspec.GridSpec(5, 1)
letters=['A','B','C','D','E']
for fb in range(Nfb):
    occupancy_result=np.mean(occupancy_tensor[:,:,fb],axis=1)*100
    axSA=fig.add_subplot(gs[fb])
    axSA.bar(np.arange(0,90),occupancy_result,label=fbands[0],color=plt.cm.tab10(fb/10))
    axSA.set_xticks(np.arange(90))
    axSA.set_xticklabels('')
    axSA.set_ylabel('FO (%%) \n %s'%fbands[fb],fontsize=8)  
    axSA.set_xlim([-1,N])
    axSA.set_ylim([0,100])
    axSA.tick_params('both',labelsize=8)
    if fb==4:
        axSA.set_xlabel('Nodes',fontsize=8)          
        axSA.set_xticklabels(labels,rotation=90,fontsize=6)
    axSA.text(-0.1,1,letters[fb],transform=axSA.transAxes)

occupancy=np.mean(occupancy_tensor[:,:,:],axis=1)
occupancy_not_norm=np.mean(occupancy_orig_tensor[:,:,:],axis=1)
error=np.abs(occupancy-occupancy_not_norm)
print(np.sum(error))

sum_occupancy=np.sum(occupancy,axis=1)
std_occupancy=np.std(occupancy,axis=1)
np.savez('Occupancy_%.3f.npz'%th,occupancy=occupancy,std_occupancy=std_occupancy) #Save the occupany

# fig.savefig('FigS3.pdf',dpi=300,bbox_inches='tight')
# fig.savefig('FigS3.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
