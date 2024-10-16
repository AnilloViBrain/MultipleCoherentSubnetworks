#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:39:05 2023

@author: felipe
"""
import sys
import os
directoryKNP='../../'
sys.path.append(os.path.abspath(directoryKNP))
import plot.networks as pltn 
import analysis.connectivityMatrices as connectivityMatrices
import numpy as np
import matplotlib.pyplot as plt


import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import gc
import tqdm
from scipy.spatial import distance_matrix
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

Nfb=5
N=90
freq_labels=['12.5 - 13.5 Hz','14.5 - 15.5 Hz', '28.9 - 29.9 Hz', '40.7 - 41.7 Hz', '42.5 - 43.5 Hz']
#clusters for each frequency
nnets_fb=[4,5,4,7,7]
sorted_indexes=np.load('index_sortFCs_Coherence2_corrected.npz',allow_pickle=True)['sorted_index'][()]
directory='./'

C=connectivityMatrices.loadConnectome(N,directoryKNP+'input_data/AAL_matrices')
file_indexes=sio.loadmat(directoryKNP+'input_data/indexes_hemispheres_front')
hemisphere_sorted=file_indexes['sorted_regions'][0,:]
frontal=file_indexes['frontal']
limbic=file_indexes['limbic']
parietal=file_indexes['parietal']
temporal=file_indexes['temporal']
occipital=file_indexes['occipital']

Nparts=20
file_labels=sio.loadmat(directoryKNP+'input_data/AAL_short_labels')
labels=file_labels['label90']
node_labels=labels[hemisphere_sorted]
C1=(C[np.tril_indices(N,k=-1)]+C[np.triu_indices(N,k=1)])/2
colormaps_fb=[plt.cm.Blues,plt.cm.Oranges,plt.cm.Greens,plt.cm.Reds,plt.cm.Purples]
colors_rsn=['C5','C6','C7','C8','C9','magenta','cyan']
nodes_order=circular_layout(node_labels, node_labels.tolist(),start_pos=90,
                              group_boundaries=[0, len(node_labels) // 2])
#Short label format, without hemisphere initial
for nn,nlabel in enumerate(node_labels):
    name=nlabel.split('.')[0]
    node_labels[nn]=name

seeds=[3,5,8,13,21,34,55,89,144,145,146,147,148,149,150,151,152,153,154,233]
Nseeds=len(seeds)
per_coh=np.zeros((Nseeds,5))
per_sd=np.zeros((Nseeds,5))


for nseed,seed in enumerate(seeds):
    for fb in range(Nfb):  
        nv=0
        corr_vectors=np.zeros((nnets_fb[fb],len(np.tril_indices(N,k=-1)[0]))) 
        corr_vectors_sd=np.zeros((nnets_fb[fb],len(np.tril_indices(N,k=-1)[0]))) 
        file=np.load(directory+'FC_clusters_Coherence2_Amplitude_fb%d_seed%d.npz'%(fb,seed),allow_pickle=True)
        FC_clusters=file['FC_clusters'][()]
        FC_sd_clusters=file['FC_sd_clusters'][()]
        for nnnets in range(nnets_fb[fb]):
            corr_vectors[nv,:]=FC_clusters[nnets_fb[fb]][nnnets]
            corr_vectors[nv,C1==0]=0
            corr_vectors_sd[nv,:]=FC_sd_clusters[nnets_fb[fb]][nnnets]
            corr_vectors[nv,C1==0]=0
            nv+=1
        per_coh[nseed,fb]=np.percentile(np.ravel(corr_vectors[corr_vectors>0]),20)
        per_sd[nseed,fb]=np.percentile(np.ravel(corr_vectors_sd[corr_vectors_sd>0]),10)
per_coh=np.mean(per_coh,axis=0)
per_sd=np.mean(per_sd,axis=0)
print(per_coh)

subnets=np.zeros((N,27))
FC_subnets=np.zeros((4005,27))
m=0
for fb in range(Nfb):
    FC_nnet=np.zeros((Nseeds,nnets_fb[fb],len(np.tril_indices(N,k=-1)[0])))
    FC_sd=np.zeros((Nseeds,nnets_fb[fb],len(np.tril_indices(N,k=-1)[0])))
    FC_phase_sd=np.zeros((Nseeds,nnets_fb[fb],len(np.tril_indices(N,k=-1)[0])))
    FC_phase_nnet=np.zeros((Nseeds,nnets_fb[fb],len(np.tril_indices(N,k=-1)[0])))
    FC_real_nnet=np.zeros((Nseeds,nnets_fb[fb],len(np.tril_indices(N,k=-1)[0])))
    for nnnets in range(nnets_fb[fb]):
        for nseed,seed in enumerate(seeds):    
            file=np.load(directory+'FC_clusters_Coherence2_Amplitude_fb%d_seed%d.npz'%(fb,seed),allow_pickle=True)
            FC_clusters=file['FC_clusters'][()]
            FC_phase_clusters=file['FC_phase_clusters'][()]
            FC_real_clusters=file['FC_real_clusters'][()]
            FC_sd_clusters=file['FC_sd_clusters'][()]
            FC_phase_sd_clusters=file['FC_phase_sd_clusters'][()]
            if nseed>0:
                index=sorted_indexes[fb][nnets_fb[fb]][nseed][nnnets]
                FC_nnet[nseed,nnnets,:]=FC_clusters[nnets_fb[fb]][index]
                FC_phase_nnet[nseed,nnnets,:]=FC_phase_clusters[nnets_fb[fb]][index]
                FC_real_nnet[nseed,nnnets,:]=FC_real_clusters[nnets_fb[fb]][index]
                FC_sd[nseed,nnnets,:]=FC_sd_clusters[nnets_fb[fb]][index]
                FC_phase_sd[nseed,nnnets,:]=FC_phase_sd_clusters[nnets_fb[fb]][index]
            else:
                FC_nnet[0,nnnets,:]=FC_clusters[nnets_fb[fb]][nnnets]
                FC_phase_nnet[0,nnnets,:]=FC_phase_clusters[nnets_fb[fb]][nnnets]
                FC_real_nnet[0,nnnets,:]=FC_real_clusters[nnets_fb[fb]][nnnets]
                FC_sd[0,nnnets,:]=FC_sd_clusters[nnets_fb[fb]][nnnets]
                FC_phase_sd[0,nnnets,:]=FC_phase_sd_clusters[nnets_fb[fb]][nnnets]
  
    for nnnets in range(nnets_fb[fb]):

        #significance_th=per90
        significance_phase_th=per_sd[fb]
        high_coh_th=per_coh[fb]
        print(fb,nnnets,significance_phase_th)
        
        
        non_freq_edges=np.argwhere(np.mean(np.abs(FC_real_nnet[:,nnnets,:]),axis=0)>0.9)
        
        non_similar_edges_phase=np.argwhere(np.mean(FC_sd[:,nnnets,:],axis=0)>significance_phase_th)
        
        FC_result=np.mean(FC_nnet[:,nnnets,:],axis=0)
        #FC_result[non_similar_edges]=0
        FC_result[non_similar_edges_phase]=0
        #FC_result[non_freq_edges]=0
        non_high_coherence=np.argwhere(FC_result<high_coh_th)
        FC_result[non_high_coherence]=0
        
        
        FCt = np.zeros((N,N))
        FC_final = np.zeros((N,N))
        FCt[np.tril_indices(N,k=-1)] = FC_result
        FCt = FCt+FCt.T
        FC_final=FCt
        FC_final[C==0]=0
        
        data90=np.zeros((N,))
        data90[np.sum(np.abs(FC_final),axis=0)>0]=1
        subnets[:,m]=data90
        
        FC_phase_result=np.mean(FC_phase_nnet[:,nnnets,:],axis=0)
        FC_phase_result[non_similar_edges_phase]=0
        FC_phase_result[non_high_coherence]=0
        FC_phase_final = np.zeros((N,N))
        FC_phase_final[np.tril_indices(N,k=-1)] = FC_phase_result
        FC_phase_final=FC_phase_final+FC_phase_final.T
        FC_subnets[:,m]=FC_phase_final[np.tril_indices(N,k=-1)]
        m+=1
        
#%%
file1=np.load('AAL90_resting_state_nets.npz')
RSnets_labels=file1['labels']
RSnets=file1['nets']
RSjoint=np.zeros((N,3))

RSjoint[(RSnets[:,0]+RSnets[:,6])>0,0]=1
RSjoint[(RSnets[:,1]+RSnets[:,3]+RSnets[:,4])>0,1]=1
RSjoint[(RSnets[:,2]+RSnets[:,5]+RSnets[:,4])>0,2]=1

total_nets=np.hstack((subnets,RSnets,RSjoint))

from sklearn.metrics.pairwise import pairwise_distances
cosine_distance=pairwise_distances(total_nets.T,total_nets.T,metric='cosine')
hamming_distance=pairwise_distances(total_nets.T,total_nets.T,metric='hamming')
euclidean_distance=pairwise_distances(total_nets.T,total_nets.T,metric='euclidean')
correlation=np.corrcoef(total_nets.T)

#%%
FC_cosine_distance=pairwise_distances(FC_subnets.T,FC_subnets.T,metric='cosine')
plt.figure(figsize=(4.5,4.5))
plt.imshow(1-FC_cosine_distance,cmap='afmhot_r')
plt.hlines([-0.45,3.5],xmin=-0.45,xmax=3.5,color='C0',linewidth=2)
plt.hlines([3.5,8.5],xmin=3.5,xmax=8.5,color='C1',linewidth=2)
plt.hlines([8.5,12.5],xmin=8.5,xmax=12.5,color='C2',linewidth=2)
plt.hlines([12.5,19.5],xmin=12.5,xmax=19.5,color='C3',linewidth=2)
plt.hlines([19.5,26.5],xmin=19.5,xmax=26.5,color='C4',linewidth=2)


plt.vlines([-0.45,3.5],ymin=-0.45,ymax=3.5,color='C0',linewidth=2)
plt.vlines([3.5,8.5],ymin=3.5,ymax=8.5,color='C1',linewidth=2)
plt.vlines([8.5,12.5],ymin=8.5,ymax=12.5,color='C2',linewidth=2)
plt.vlines([12.5,19.5],ymin=12.5,ymax=19.5,color='C3',linewidth=2)
plt.vlines([19.5,26.5],ymin=19.5,ymax=26.5,color='C4',linewidth=2)

plt.hlines([2.5,3.5],xmin=2.5,xmax=3.5,color='C0',linewidth=2)
plt.vlines([2.5,3.5],ymin=2.5,ymax=3.5,color='C0',linewidth=2)

plt.hlines([6.5,7.5],xmin=6.5,xmax=7.5,color='C1',linewidth=2)
plt.vlines([6.5,7.5],ymin=6.5,ymax=7.5,color='C1',linewidth=2)

plt.hlines([11.5,12.5],xmin=11.5,xmax=12.5,color='C2',linewidth=2)
plt.vlines([11.5,12.5],ymin=11.5,ymax=12.5,color='C2',linewidth=2)

plt.hlines([18.5,19.5],xmin=18.5,xmax=19.5,color='C3',linewidth=2)
plt.vlines([18.5,19.5],ymin=18.5,ymax=19.5,color='C3',linewidth=2)

plt.hlines([25.5,26.5],xmin=25.5,xmax=26.5,color='C4',linewidth=2)
plt.vlines([25.5,26.5],ymin=25.5,ymax=26.5,color='C4',linewidth=2)


plt.xticks([1.5,6.5,10.5,16,23],['12.5-13-5 Hz','14.5-15.5 Hz','28.9-29.9 Hz','40.7-41.7 Hz','42.5-43.5 Hz'],rotation=90,fontsize=8)
plt.yticks([1.5,6.5,10.5,16,23],['12.5-13-5 Hz','14.5-15.5 Hz','28.9-29.9 Hz','40.7-41.7 Hz','42.5-43.5 Hz'],fontsize=8)
cb=plt.colorbar(shrink=0.7,label='Similarity of \n coherent subnetworks')
# cb.ax.set_label('Similarity')
plt.tight_layout()
plt.savefig('FigS8b.pdf',dpi=600)

#%%
plt.figure(figsize=(3.1,5))
plt.imshow(1-cosine_distance[:,:][:,27::],cmap='afmhot_r',vmin=0,vmax=1)
# plt.imshow(correlation,cmap='seismic',vmin=-1,vmax=1)
# plt.imshow(euclidean_distance,cmap='afmhot_r')


plt.yticks([1.5,6.5,10.5,16,23,27,28,29,30,31,32,33,34,35,36],['12.5-13-5 Hz','14.5-15.5 Hz','28.9-29.9 Hz','40.7-41.7 Hz','42.5-43.5 Hz','VN','SMN','DAN','VAN','SubC','FPN','DMN','VN+DMN','SMN+SubC+VAN','FPN+SubC+DAN'],fontsize=8)
plt.xticks(np.arange(10),['VN','SMN','DAN','VAN','SubC','FPN','DMN','VN+DMN','SMN+SubC+VAN','FPN+SubC+DAN'],fontsize=8,rotation=90)

plt.hlines([-0.4,3.4],xmin=-.4,xmax=9.4,color='C0',linewidth=2)
plt.vlines([-0.4,6.5,7.5,9.4],ymin=-0.4,ymax=3.5,color='C0',linewidth=2)

plt.hlines([3.5,8.5],xmin=-.4,xmax=9.4,color='C1',linewidth=2)
plt.vlines([-0.4,6.5,7.5,9.5],ymin=3.5,ymax=8.5,color='C1',linewidth=2)

plt.hlines([8.5,12.5],xmin=-.4,xmax=9.4,color='C2',linewidth=2)
plt.vlines([-0.4,7.5,8.5,9.5],ymin=8.5,ymax=12.5,color='C2',linewidth=2)

plt.hlines([12.5,19.5],xmin=-.4,xmax=9.4,color='C3',linewidth=2)
plt.vlines([-0.4,8.5,9.5,9.5],ymin=12.5,ymax=19.5,color='C3',linewidth=2)

plt.hlines([19.5,26.5],xmin=-.4,xmax=9.4,color='C4',linewidth=2)
plt.vlines([-0.4,8.5,9.5,9.5],ymin=19.5,ymax=26.5,color='C4',linewidth=2)


plt.colorbar(label='Similarity of \n participating regions',shrink=1)
plt.tight_layout()
plt.savefig('FigS9b.pdf',dpi=600,bbox_inches='tight')


    
