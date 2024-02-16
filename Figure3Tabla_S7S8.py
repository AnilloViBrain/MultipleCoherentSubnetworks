#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:38:48 2023

@author: felipe
"""

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
Nseeds=10
Nparts=20
file_labels=sio.loadmat('../../input_data/AAL_short_labels')
labels=file_labels['label90']
node_labels=labels[hemisphere_sorted]
C1=(C[np.tril_indices(N,k=-1)]+C[np.triu_indices(N,k=1)])/2

pointsFC=[231,200,102,73,70]
colors_region=[0.26,0.8,0.44,0.84,0.06]
colormaps_fb=[plt.cm.Blues,plt.cm.Oranges,plt.cm.Greens,plt.cm.Reds,plt.cm.Purples]
nodes_order=circular_layout(node_labels, node_labels.tolist(),start_pos=90,
                              group_boundaries=[0, len(node_labels) // 2])
#Short label format, without hemisphere initial
for nn,nlabel in enumerate(node_labels):
    name=nlabel.split('.')[0]
    node_labels[nn]=name
    
per_coh=np.zeros((10,5))
per_sd=np.zeros((10,5))

seeds=[3,5,8,13,21,34,55,89,144,233]
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

nodes_clusters={}
for fb in range(Nfb):
    nodes_clusters[fb]={}
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
        
        
        non_freq_edges=np.argwhere(np.mean(np.abs(FC_real_nnet[:,nnnets,:]),axis=0)>0.99)
        
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
        

        # axnet=fig.add_subplot(gss[0],projection='3d')
        data90=np.zeros((N,))
        binaryFC=np.zeros_like(FC_final)
        binaryFC[FC_final!=0]=1
        data90[np.sum(binaryFC,axis=0)>0]=1
        data90[np.sum(binaryFC,axis=1)>0]=1
        nodes_clusters[fb][nnnets]=data90  
        # colors_node=[]
        # labels_node=[]
        # for node in range(N):
        #     if data90[hemisphere_sorted[node]]>0:
        #         labels_node.append(node_labels[node])
        #         if hemisphere_sorted[node] in frontal:
        #             colors_node.append(colormaps_fb[fb](0.3))
        #         elif hemisphere_sorted[node] in limbic:
        #             colors_node.append(colormaps_fb[fb](0.4))
        #         elif hemisphere_sorted[node] in temporal:
        #             colors_node.append(colormaps_fb[fb](0.5))
        #         elif hemisphere_sorted[node] in parietal:
        #             colors_node.append(colormaps_fb[fb](0.6))
        #         elif hemisphere_sorted[node] in occipital:
        #             colors_node.append(colormaps_fb[fb](0.7))
        #     else:
        #         labels_node.append('')
        #         colors_node.append(colormaps_fb[fb](0.0))
                
            
#%%          
plt.figure(figsize=(8,8))
width=0.6
total_subnets_by_node=np.zeros((90,))
for fb in range(Nfb):
    plt.subplot(5,1,fb+1)
    for nnnets in range(nnets_fb[fb]):
        total_subnets_by_node+=nodes_clusters[fb][nnnets]
        plt.bar(np.arange(0,90),nodes_clusters[fb][nnnets],label='%s-net %d'%(freq_labels[fb],nnnets),width=width,color=plt.cm.tab10(fb/10),bottom=1.1*nnnets)
    plt.ylabel('Subnets \n %s'%freq_labels[fb],fontsize=8)
    plt.xlim([-0.5,90.5])
    plt.xticks(np.arange(0,90),['']*90)     
plt.xlabel('Nodes',fontsize=8)          
plt.xticks(np.arange(0,90),labels,rotation=90,fontsize=6)
plt.tight_layout()     
plt.savefig('FigS7.pdf',dpi=300,bbox_inches='tight')
plt.savefig('FigS7.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')

#%%
fig=plt.figure(figsize=(8,6))
gs=gridspec.GridSpec(1,1)
axA=fig.add_subplot(gs[0])

sorted_nodes=np.argsort(total_subnets_by_node)
m=np.sum(nnets_fb)*1.25
for fb in range(Nfb):
    for nnnets in range(nnets_fb[fb]):
        axA.bar(np.arange(0,90),nodes_clusters[fb][nnnets][np.argsort(total_subnets_by_node)],width=width,color=plt.cm.tab10(fb/10),bottom=m)
        m-=1.25


#axA.set_yticks(np.arange(0.5,30,1.25))
#axA.set_yticklabels([1,2,3,4,1,2,3,4,1,2,3,4,5,1,2,3,4,5,6,1,2,3,4,5])
axA.tick_params('both',labelsize=8)
axA.set_ylabel('Subnetworks',fontsize=8)
axA.set_xlabel('Node',fontsize=8)
axA.set_xticks(np.arange(0,90))
axA.set_xticklabels(labels[np.argsort(total_subnets_by_node)],rotation=90,fontsize=6)
axA.set_yticks([7*1.25+0.5,14*1.25+0.5,18*1.25+0.5,23*1.25+0.5,27*1.25+0.5],['43 Hz','41.2 Hz','29.4 Hz', '15 Hz', '13 Hz'],fontsize=8)
axA.set_xlim([-0.5,89.5])
axA.set_ylim([1.25,34.75])
fig.savefig('FigS8.pdf',dpi=300,bbox_inches='tight')
fig.savefig('FigS8.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
#axA.text(-0.08,1,'A',transform=axA.transAxes)

np.savez('participation_order.npz',participation=np.argsort(total_subnets_by_node))

#%%
fig=plt.figure()
gss=gridspec.GridSpec(1,1)
axB=fig.add_subplot(gss[0])
axB.hist(total_subnets_by_node,bins=np.arange(-0.5,40.5),color='black')
axB.set_xticks(np.arange(20))
axB.set_ylabel('# Nodes',fontsize=8)
axB.set_xlabel('Participation in subnetworks',fontsize=8)
axB.tick_params('both',labelsize=8)
axB.set_ylim([-0.01,16])
axB.text(-0.08,1,'B',transform=axB.transAxes)
#fig.savefig('nodes_byNsubnets.png',dpi=600)

#%%
plt.figure()
std_occupancy=np.load('../FunctionalSubnetworks/Occupancy_0.232.npz')['std_occupancy']
plt.plot(total_subnets_by_node,std_occupancy,'o')
plt.show()
