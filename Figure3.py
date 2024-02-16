#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:39:05 2023

@author: felipe
"""
import sys
import os
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../'))
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
import plot.networks as pltn
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

Nfb=5
N=90
freq_labels=['12.5 - 13.5 Hz','14.5 - 15.5 Hz', '28.9 - 29.9 Hz', '40.7 - 41.7 Hz', '42.5 - 43.5 Hz']
#clusters for each frequency
nnets_fb=[4,5,4,7,7]
sorted_indexes=np.load('index_sortFCs_Coherence2_corrected.npz',allow_pickle=True)['sorted_index'][()]
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence_clusters/'
C=connectivityMatrices.loadConnectome(N,'../../input_data/AAL_matrices')
file_indexes=sio.loadmat('../../input_data/indexes_hemispheres_front')
hemisphere_sorted=file_indexes['sorted_regions'][0,:]
frontal=file_indexes['frontal']
limbic=file_indexes['limbic']
parietal=file_indexes['parietal']
temporal=file_indexes['temporal']
occipital=file_indexes['occipital']

Nparts=20
file_labels=sio.loadmat('../../input_data/AAL_short_labels')
labels=file_labels['label90']
node_labels=labels[hemisphere_sorted]
C1=(C[np.tril_indices(N,k=-1)]+C[np.triu_indices(N,k=1)])/2
pointsFC=[231,200,102,73,70]
colors_region=[0.26,0.8,0.44,0.84,0.06] #absolute
colormaps_fb=[plt.cm.Blues,plt.cm.Oranges,plt.cm.Greens,plt.cm.Reds,plt.cm.Purples]
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
        
        fig=plt.figure(figsize=(1.2,1.2))
        gss=gridspec.GridSpec(1,1,wspace=0.01,left=0.01,right=0.99,top=0.99,bottom=0.15)
        gss.tight_layout(figure=fig,pad=0.02)
        
        non_freq_edges=np.argwhere(np.mean(np.abs(FC_real_nnet[:,nnnets,:]),axis=0)>0.9)
        
        non_similar_edges_phase=np.argwhere(np.mean(FC_sd[:,nnnets,:],axis=0)>significance_phase_th)
        
        FC_result=np.mean(FC_nnet[:,nnnets,:],axis=0)
        #FC_result[non_similar_edges]=0
        FC_result[non_similar_edges_phase]=0
        #FC_result[non_freq_edges]=0
        non_high_coherence=np.argwhere(FC_result<high_coh_th)
        FC_result[non_high_coherence]=0
        
        print(np.shape(np.argwhere(FC_result>0)))
        
        FC_phase_result=np.mean(FC_phase_nnet[:,nnnets,:],axis=0)
        #FC_phase_result[non_similar_edges]=0
        FC_phase_result[non_similar_edges_phase]=0
        FC_phase_result[non_high_coherence]=0
        # FC_phase_result[non_freq_edges]=0
        FC_real_result=np.mean(FC_real_nnet[:,nnnets,:],axis=0)
        #FC_real_result[non_similar_edges]=0
        FC_real_result[non_similar_edges_phase]=0
        FC_real_result[non_high_coherence]=0
        # FC_real_result[non_freq_edges]=0
        
        FCt = np.zeros((N,N))
        FC_final = np.zeros((N,N))
        FCt[np.tril_indices(N,k=-1)] = FC_result
        FCt = FCt+FCt.T
        FC_phase_final = np.zeros((N,N))
        FC_phase_final[np.tril_indices(N,k=-1)] = FC_phase_result
        FC_phase_final=FC_phase_final+FC_phase_final.T
        FC_final=FCt
        FC_final[C==0]=0
        FC_G=np.copy(FC_final[hemisphere_sorted,:][:,hemisphere_sorted])
        FC_phase_final=FC_phase_final[hemisphere_sorted,:][:,hemisphere_sorted]
        
        #FC_phase_final[FC_G>0]=FC_phase_final[FC_G>0]
        
        FC_real_final = np.zeros((N,N))
        FC_real_final[np.tril_indices(N,k=-1)] = FC_real_result
        FC_real_final=FC_real_final+FC_real_final.T
        FC_real_final=FC_real_final[hemisphere_sorted,:][:,hemisphere_sorted]
        
        axnet=fig.add_subplot(gss[0],projection='3d',frame_on=False,position=[0,0,1,1])
        data90=np.zeros((N,))
        binaryFC=np.zeros_like(FC_G)
        binaryFC[FC_final!=0]=1
        data90[np.sum(binaryFC,axis=0)>0]=colors_region[fb]
        data90[np.sum(binaryFC,axis=1)>0]=colors_region[fb]
        colors_node=[]
        labels_node=[]
        for node in range(N):
            if data90[hemisphere_sorted[node]]!=0:
                # labels_node.append(node_labels[node])
                labels_node.append('')
                # if hemisphere_sorted[node] in frontal:
                #     colors_node.append(colormaps_fb[fb](0.3))
                # elif hemisphere_sorted[node] in limbic:
                #     colors_node.append(colormaps_fb[fb](0.4))
                # elif hemisphere_sorted[node] in temporal:
                #     colors_node.append(colormaps_fb[fb](0.5))
                # elif hemisphere_sorted[node] in parietal:
                #     colors_node.append(colormaps_fb[fb](0.6))
                # elif hemisphere_sorted[node] in occipital:
                colors_node.append(colormaps_fb[fb](0.7))
            else:
                labels_node.append('')
                colors_node.append(colormaps_fb[fb](0.0))
                
            
            
        pltn.plotAAL90Brain(data90,orientation=[90,270],interpolation='none',alpha=0.4,ax=axnet,cmap_name='nipy_spectral')
        axnet.set_axis_off()
        fig.savefig('./Subnets/Fig3_fb%d_subnet%d.png'%(fb,nnnets),dpi=600)
        plt.close(fig)
        
        arrayFC=FC_G[np.argwhere(FC_G>0)[:,0],np.argwhere(FC_G>0)[:,1]]
        arrayFC_phase=FC_phase_final[np.argwhere(FC_G>0)[:,0],np.argwhere(FC_G>0)[:,1]]
        arrayFC_real=FC_real_final[np.argwhere(FC_G>0)[:,0],np.argwhere(FC_G>0)[:,1]]
        indexes_i=np.argwhere(FC_G>0)[:,0]
        indexes_j=np.argwhere(FC_G>0)[:,1]
        fig, axcircle = plt.subplots(figsize=(2, 1.8), facecolor='white',
                                subplot_kw=dict(polar=True,frame_on=False))
        plot_connectivity_circle(arrayFC,labels_node,indices=(indexes_i,indexes_j),
                                  node_angles=nodes_order,
                                  facecolor='white', textcolor='black',
                                  node_colors=colors_node, colormap='gist_heat_r',
                                  node_linewidth=0.1, show=False,
                                  vmin=0,vmax=1, colorbar_pos=(-0.5,0.02), colorbar_size=.3,
                                  ax=axcircle,fontsize_names=8,fontsize_colorbar=8,padding=0.0)
        fig.tight_layout()
        fig.savefig('./Subnets/Fig3_fb%d_subnet%d_circular.png'%(fb,nnnets),dpi=600)
        plt.close(fig)
        
        fig, axcircle = plt.subplots(figsize=(2, 1.8), facecolor='white',
                                subplot_kw=dict(polar=True,frame_on=False))
        plot_connectivity_circle(arrayFC_phase,labels_node,indices=(indexes_i,indexes_j),
                                  node_angles=nodes_order,
                                  facecolor='white', textcolor='black',
                                  node_colors=colors_node, colormap='RdBu_r',
                                  node_linewidth=0.1, show=False,
                                  vmin=-1.0,vmax=1.0, colorbar_pos=(-0.5,0.02), colorbar_size=0.3,
                                  ax=axcircle,fontsize_names=8,fontsize_colorbar=8,padding=0.0)
        fig.tight_layout()
        fig.savefig('./Subnets/Fig3_imag_fb%d_subnet%d_circular.png'%(fb,nnnets),dpi=600)
        plt.close(fig)
        
        fig, axcircle = plt.subplots(figsize=(2, 1.8), facecolor='white',
                                subplot_kw=dict(polar=True,frame_on=False))
        plot_connectivity_circle(arrayFC_real,labels_node,indices=(indexes_i,indexes_j),
                                  node_angles=nodes_order,
                                  facecolor='white', textcolor='black',
                                  node_colors=colors_node, colormap='RdBu_r',
                                  node_linewidth=0.1, show=False,
                                  vmin=-1.0,vmax=1.0, colorbar_pos=(-0.5,0.02), colorbar_size=0.3,
                                  ax=axcircle,fontsize_names=8,fontsize_colorbar=8,padding=0.0)
        fig.tight_layout()
        fig.savefig('./Subnets/Fig3_real_fb%d_subnet%d_circular.png'%(fb,nnnets),dpi=600)
        plt.close(fig)
        
