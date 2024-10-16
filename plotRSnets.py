#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 03:41:42 2024

@author: felipe
"""
import sys
import os
sys.path.append(os.path.abspath('../../'))
import plot.networks as pltn 
import analysis.connectivityMatrices as connectivityMatrices
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%%
colors_rsn=[0.16,0.22,0.89,0.4,0.52,0.78,0.06,0.25,0.7,0.85] 
N=90
file1=np.load('./AAL90_resting_state_nets.npz')
RSnets_labels=file1['labels']
RSnets=file1['nets']
RSjoint=np.zeros((N,3))

RSjoint[(RSnets[:,0]+RSnets[:,6])>0,0]=1
RSjoint[(RSnets[:,1]+RSnets[:,3]+RSnets[:,4])>0,1]=1
RSjoint[(RSnets[:,2]+RSnets[:,5]+RSnets[:,4])>0,2]=1

RSnets=np.hstack((RSnets,RSjoint))
for rsn in range(10):
    fig=plt.figure(figsize=(1.2,1.2))
    gss=gridspec.GridSpec(1,1,wspace=0.01,left=0.01,right=0.99,top=0.99,bottom=0.15)
    gss.tight_layout(figure=fig,pad=0.02)
    axnet=fig.add_subplot(gss[0],projection='3d',frame_on=False,position=[0,0,1,1])
    pltn.plotAAL90Brain(RSnets[:,rsn]*colors_rsn[rsn],orientation=[90,270],interpolation='none',alpha=0.4,ax=axnet,cmap_name='nipy_spectral')
    axnet.set_axis_off()
    fig.savefig('RSnet%d.png'%(rsn),dpi=600)
    plt.close(fig)