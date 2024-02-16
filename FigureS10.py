#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 of paper Functional subnetworks 

@author: felipe
"""
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


def int_to_one_hot_expand_time(series,segment_duration=1):
    unique_values = np.unique(series)
    num_classes = len(unique_values)
    num_samples = len(series)
    one_hot_matrix = np.zeros((num_samples*segment_duration, num_classes),dtype=int)

    for i, value in enumerate(series):
        index = np.where(unique_values == value)[0][0]
        one_hot_matrix[i*segment_duration:(i+1)*segment_duration, index] = 1

    return one_hot_matrix

def expand_time(series,segment_duration=1):
    num_samples = len(series)
    expand_series = np.zeros((num_samples*segment_duration,),dtype=int)
    for i, value in enumerate(series):
        expand_series[i*segment_duration:(i+1)*segment_duration] = value+1

    return expand_series

def binary_to_int(binary_series):
    num_samples = np.shape(binary_series)[0]
    output=np.zeros((num_samples,))
    for i in range(num_samples):
        output[i]=int("".join(str(x) for x in binary_series[i,:]), 2)
    return output
    
#colors list
colors=[plt.cm.tab10(0),plt.cm.tab10(1),plt.cm.tab10(2),plt.cm.tab10(3),
        plt.cm.tab10(4),plt.cm.tab10(5),plt.cm.tab10(6),plt.cm.tab10(7),
        plt.cm.tab10(8),plt.cm.tab10(9),plt.cm.Dark2(0),plt.cm.Dark2(1),
        plt.cm.Dark2(2),plt.cm.Dark2(3),plt.cm.Dark2(4),plt.cm.Dark2(5),
        plt.cm.Dark2(6),plt.cm.Dark2(7),plt.cm.Set1(0),plt.cm.Set1(1),
        plt.cm.Set1(2),plt.cm.Set1(3),plt.cm.Set1(4),plt.cm.Set1(5),
        plt.cm.Set1(6),plt.cm.Set1(7),plt.cm.Set1(8),plt.cm.Set2(0),
        plt.cm.Set2(1),plt.cm.Set2(2),plt.cm.Set2(3),plt.cm.Set2(4),
        plt.cm.Set2(5),plt.cm.Set2(6),plt.cm.Set2(7)]

#Model parameter
MD=0.021
K=4
N=90
dt=1e-3
fs=1000
seed=3 #18000 seconds simulation
#clusters for each frequency
nnets_fb=[4,5,4,7,7]
maxfb=[3,3,3,6,6]
time_windows=[231*2,200*2,102*2,73*2,70*2]
sorted_indexes=np.load('index_sortFCs_Coherence2_corrected.npz',allow_pickle=True)['sorted_index'][()]
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence_clusters/'
init_time=44000
end_time=49000

init_time1=20000
end_time1=300000
len_time=end_time1-init_time1

fig=plt.figure(figsize=(8,6))
gs=gridspec.GridSpec(2, 2,wspace=0.3)
axA=fig.add_subplot(gs[0,0])
axB=fig.add_subplot(gs[1,0])
axC=fig.add_subplot(gs[:,1])
m=26
for fb in range(5):
    file=np.load(directory+'FC_clusters_Coherence2_Amplitude_fb%d_seed%d.npz'%(fb,seed),allow_pickle=True)
    labels=file['labels'][()]
    init_fb=int(2*(init_time/time_windows[fb]-0.5))
    end_fb=int(2*(end_time/time_windows[fb]-0.5))
    init_fb1=int(2*(init_time1/time_windows[fb]-0.5))
    end_fb1=int(2*(end_time1/time_windows[fb]-0.5))
    tt=np.linspace(0,5,end_fb-init_fb)
    for nnn in range(nnets_fb[fb]):
        if nnn==maxfb[fb]:
    	    marker='o'
        else:
    	    marker='o'
        axA.plot(tt,np.ma.masked_equal(labels[nnets_fb[fb]-2][init_fb:end_fb]==nnn,0)+m,marker=marker,color='C%d'%fb,markersize=2)
        m-=1
    if fb==0:
        binary_labels=expand_time(labels[nnets_fb[fb]-2][init_fb1:end_fb1],segment_duration=time_windows[fb]//2)[0:len_time-1000]
    else:
        binary_labels=np.vstack((binary_labels,expand_time(labels[nnets_fb[fb]-2][init_fb1:end_fb1],segment_duration=time_windows[fb]//2)[0:len_time-1000]))
    #plt.step(tt,labels[nnets_fb[fb]-2][init_fb:end_fb]==maxfb[fb]+6*fb,where='pre')
overlapped_subnets=np.cumprod(binary_labels,axis=0)[-1,:]
axB.step(np.arange(0,5,dt),overlapped_subnets[init_time-init_time1:end_time-init_time1],color='k')
duration=synchronization.durationfromLabels(overlapped_subnets,time_window=1,overlap=0)
list_duration=[]
for k in duration.keys():
    if len(duration[k])>0:
        for d in duration[k]:
            list_duration.append(d)
array_duration=np.array(list_duration)
axC.boxplot(array_duration/1000)

axA.set_ylabel('Subnetworks ocurrence',fontsize=8)
axA.set_xlabel('time (s)',fontsize=8)
axA.set_yticks([7,14,18,23,27])
axA.set_yticklabels(['43.0 Hz','41.2 Hz','29.4 Hz', '15.0 Hz','13.0 Hz'])
axA.tick_params('both',labelsize=8)
axA.text(-0.2,1,'A',transform=axA.transAxes)

axB.set_ylabel('Overlapped subnetworks',fontsize=8)
axB.set_xlabel('time (s)',fontsize=8)
axB.tick_params('both',labelsize=8)
axA.text(-0.2,1,'B',transform=axB.transAxes)

axC.set_ylabel('duration (s)',fontsize=8)
axC.set_xlabel('Overlapped subnetworks',fontsize=8)
axC.set_xticklabels([' '])
axC.tick_params('both',labelsize=8)
axA.text(-0.15,1,'C',transform=axC.transAxes)

fig.savefig('FigS10.pdf',dpi=300,bbox_inches='tight')    
fig.savefig('FigS10.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
