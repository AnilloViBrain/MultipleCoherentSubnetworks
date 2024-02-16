#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:06:18 2023

@author: felipe
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
from scipy.spatial import distance_matrix
Nfb=5
N=90
freq_labels=['12.5 - 13.5 Hz','14.5 - 15.5 Hz', '28.9 - 29.9 Hz', '40.7 - 41.7 Hz', '42.5 - 43.5 Hz']
nnets_fb=[4,5,4,7,7]
seeds=[3,5,8,13,21,34,55,89,144,145,146,147,148,149,150,151,152,153,154,233]
sorted_indexes=np.load('index_sortFCs_Coherence2_corrected.npz',allow_pickle=True)['sorted_index'][()]
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence_clusters/'

fig0=plt.figure()
gs0=gridspec.GridSpec(20, nnets_fb[0]+1)
fig1=plt.figure()
gs1=gridspec.GridSpec(20, nnets_fb[1]+1)
fig2=plt.figure()
gs2=gridspec.GridSpec(20, nnets_fb[2]+1)
fig3=plt.figure()
gs3=gridspec.GridSpec(20, nnets_fb[3]+1)
fig4=plt.figure()
gs4=gridspec.GridSpec(20, nnets_fb[4]+1)

figs=[fig0,fig1,fig2,fig3,fig4]
gss=[gs0,gs1,gs2,gs3,gs4]
counts={}

def relabel(labels,new_tags):
    N=np.max(labels)
    new_labels=np.zeros_like(labels)
    if N==np.max(new_tags):
        old_labels=np.copy(labels)
        for n in range(N+1):
            new_labels[old_labels==new_tags[n]]=n
    return new_labels
        
for nseed,seed in enumerate(seeds):
    counts[nseed]={}
    for fb in range(Nfb):
        file=np.load(directory+'FC_clusters_Coherence2_Amplitude_fb%d_seed%d.npz'%(fb,seed),allow_pickle=True)
        FC_clusters=file['FC_clusters'][()]
        #FC_clusters=file['FC_phase_clusters'][()]
        #FC_clusters=file['FC_real_clusters'][()]
        labels=file['labels'][()]
        unique, counts[nseed][fb] = np.unique(labels[nnets_fb[fb]-2], return_counts=True)
        for nnnets in range(nnets_fb[fb]):
            if nseed==0:
                FC_nnet=FC_clusters[nnets_fb[fb]][nnnets]
            else:
                index=sorted_indexes[fb][nnets_fb[fb]][nseed][nnnets]
                FC_nnet=FC_clusters[nnets_fb[fb]][index]
            FCt = np.zeros((N,N))
            FCt[np.tril_indices(N,k=-1)] = FC_nnet
            FCreconst = FCt+FCt.T
            FCmask=np.ma.masked_where(np.abs(FCreconst)<0.05, FCreconst)
            ax=figs[fb].add_subplot(gss[fb][nseed,nnnets])
            ax.imshow(FCmask,interpolation='None',cmap=plt.cm.RdBu_r,vmin=0,vmax=1)
            ax.set_axis_off()
        axhist=figs[fb].add_subplot(gss[fb][nseed,nnets_fb[fb]])
        if nseed==0:
            axhist.hist(labels[nnets_fb[fb]-2],bins=nnets_fb[fb])
        else:
            tags=sorted_indexes[fb][nnets_fb[fb]][nseed]
            labels_now=relabel(labels[nnets_fb[fb]-2],tags)
            axhist.hist(labels_now,bins=nnets_fb[fb])
        axhist.set_axis_off()
fig0.savefig('Networks_fb0_coherence6Cycles.png',dpi=300)
fig1.savefig('Networks_fb1_coherence6Cycles.png',dpi=300)
fig2.savefig('Networks_fb2_coherence6Cycles.png',dpi=300)
fig3.savefig('Networks_fb3_coherence6Cycles.png',dpi=300)
fig4.savefig('Networks_fb4_coherence6Cycles.png',dpi=300)
np.savez('count_Coherence2_corrected_networks_scaled.npz',counts=counts,allow_pickle=True)
