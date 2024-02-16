#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:04:53 2023

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
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

N=90
C=connectivityMatrices.loadConnectome(N,'../../input_data/AAL_matrices')
D=connectivityMatrices.loadDelays(N,'../../input_data/AAL_matrices')
D=connectivityMatrices.applyMeanDelay(D,C,mean_delay=1)
file_indexes=sio.loadmat('../../input_data/indexes_hemispheres_front')
hemisphere_sorted=file_indexes['sorted_regions'][0,:]
file_labels=sio.loadmat('../../input_data/AAL_short_labels')
participation=np.load('participation_order.npz')['participation']
labels=file_labels['label90']
node_labels=labels[hemisphere_sorted]
nodes_order=circular_layout(node_labels, node_labels.tolist(),start_pos=90,
                              group_boundaries=[0, len(node_labels) // 2])
#Short label format, without hemisphere initial
for nn,nlabel in enumerate(node_labels):
    name=nlabel.split('.')[0]
    node_labels[nn]=name



CG=np.copy(C[hemisphere_sorted,:][:,hemisphere_sorted])
CG=(CG+CG.T)/2

CC=(C+C.T)/2
binaryC=np.zeros_like(C)

percentil90=np.percentile(np.ravel(C[C>0]),90)
print(percentil90)

indexes_i=np.argwhere(CG>percentil90)[:,0]
indexes_j=np.argwhere(CG>percentil90)[:,1]

nonZeroCG=CG[CG>0]


colors_node=[]
labels_node=[]
data90=np.ones((N,))
binaryC[CC>0]=1
degreeC=np.sum(binaryC,axis=1)
intensities=np.sum(CC,axis=1)

print(np.sort(degreeC))
medianC=np.median(intensities)
for node in range(N):
    if data90[hemisphere_sorted[node]]!=0:
        labels_node.append(node_labels[node])
        colors_node.append(plt.cm.Greys(degreeC[hemisphere_sorted[node]]/np.max(degreeC)))
    else:
        labels_node.append('')
        colors_node.append('#FFFFFF')



arrayC=C[np.argwhere(C>percentil90)[:,0],np.argwhere(C>percentil90)[:,1]]

fig, axcircle = plt.subplots(figsize=(6, 6), facecolor='white',
                        subplot_kw=dict(polar=True))
plot_connectivity_circle(arrayC,labels_node,indices=(indexes_i,indexes_j),
                          node_angles=nodes_order,
                          facecolor='white', textcolor='black',linewidth=1,
                          node_colors=colors_node, colormap='turbo', node_linewidth=0.5,
                          vmin=7,vmax=60, colorbar=False,
                          ax=axcircle,fontsize_names=8,fontsize_colorbar=8,padding=-0.2)
norm=plt.Normalize(vmin=np.min(colors_node)*np.max(degreeC)//5*5,vmax=(np.max(colors_node)*np.max(degreeC)//5+1)*5)
normW=plt.Normalize(vmin=7,vmax=60)
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greys),ax=axcircle,shrink=0.2,pad=0.01,anchor=(-0.1,0.5))
cbW=fig.colorbar(plt.cm.ScalarMappable(norm=normW, cmap=plt.cm.turbo),ax=axcircle,shrink=0.2,pad=0.08,anchor=(0.2,0.5))
cb.ax.tick_params(labelsize=8)
cb.set_label(r'degree $k_n$',fontsize=8)
cbW.ax.tick_params(labelsize=8)
cbW.set_label(r'weight $w_{nm}$',fontsize=8)
fig.tight_layout()
fig.savefig('SC_matrix_circular.png',bbox_inches='tight',dpi=300)
fig.savefig('FigS5.pdf',bbox_inches='tight',dpi=300)
fig.savefig('FigS5.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
plt.close(fig)

#%%
DG=np.copy(D[hemisphere_sorted,:][:,hemisphere_sorted])
DG[CG==0]=0
indexesD_i=np.argwhere(CG>1)[:,0]
indexesD_j=np.argwhere(CG>1)[:,1]
nonZeroDG=DG[DG>0]

colorsD_node=[]
labelsD_node=[]
data90D=np.ones((N,))
for node in range(N):
    if data90[hemisphere_sorted[node]]!=0:
        labelsD_node.append(node_labels[node])
        colorsD_node.append(plt.cm.Greys(intensities[hemisphere_sorted[node]]/np.max(intensities)))
    else:
        labelsD_node.append('')
        colorsD_node.append('#FFFFFF')

arrayD=DG[np.argwhere(CG>1)[:,0],np.argwhere(CG>1)[:,1]]

fig, axcircle = plt.subplots(figsize=(6, 6), facecolor='white',
                        subplot_kw=dict(polar=True))
plot_connectivity_circle(arrayD,labelsD_node,indices=(indexesD_i,indexesD_j),
                          node_angles=nodes_order,
                          facecolor='white', textcolor='black',linewidth=1,
                          node_colors=colorsD_node, colormap='turbo', node_linewidth=0.5,
                          vmin=0,vmax=2.5, colorbar_pos=(-0.8,0.5),
                          ax=axcircle,fontsize_names=6,fontsize_colorbar=6,padding=-0.2)
norm=plt.Normalize(vmin=np.min(colors_node)*np.max(intensities),vmax=np.max(colors_node)*np.max(intensities))
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greys),ax=axcircle,shrink=0.2,pad=0.1,anchor=(0,0.5))
cb.ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig('D_matrix_circular.png',bbox_inches='tight',dpi=600)
plt.close(fig)

plt.figure()
plt.plot(np.arange(90),intensities[participation],'o',label='strength')
plt.plot(np.arange(90),degreeC[participation],'o',label='degree')
plt.xticks(np.arange(90),node_labels)
plt.xlabel('Region')
plt.ylabel('Strength/Degree')
