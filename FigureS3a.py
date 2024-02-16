#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:52:48 2023

@author: felipe
"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../../'))
import analysis.frequency as frequency
import analysis.synchronization as synchronization
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc

N=90
Nfb=5
fbands=['12.50-13.50 Hz','14.50-15.50 Hz','28.90-29.90 Hz','40.70-41.70 Hz','42.50-43.50 Hz']
file_labels=sio.loadmat('../../../input_data/AAL_labels')
labels=file_labels['label90']
occupancy_tensor=np.load('Occupancy_0.232.npz')['occupancy']
# occupancy_tensor=np.load('occupancies_visualizationK=4MD=21.npz')['occupancy_tensor']

fig= plt.figure(figsize=(8.5,8.5))
gs=gridspec.GridSpec(5, 1)
for fb in range(Nfb):
    occupancy_result=occupancy_tensor[:,fb]*100
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
fig.savefig('FigS3.pdf',dpi=300,bbox_inches='tight')    
fig.savefig('FigS3.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
