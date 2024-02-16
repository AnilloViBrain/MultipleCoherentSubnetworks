#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:28:50 2023

@author: felipe
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../'))
import analysis.frequency as frequency
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio

K1_values=np.arange(0,11,1)
MD1_values=np.arange(0,41,1)

K2_values=np.arange(0.5,10,1)
MD2_values=np.arange(0,41,1)

K_all_values=np.arange(0,10.5,0.5)
MD_all_values=np.arange(0,41,1)

H1=np.zeros((len(K1_values),len(MD1_values)))
H2=np.zeros((len(K2_values),len(MD2_values)))
H_all=np.zeros((len(K_all_values),len(MD_all_values)))
# kop1=np.zeros((len(K1_values),len(MD1_values)))
kop2=np.zeros((len(K2_values),len(MD2_values)))
kop_all=np.zeros((len(K_all_values),len(MD_all_values)))
# sd_kop1=np.zeros((len(K1_values),len(MD1_values)))
sd_kop2=np.zeros((len(K2_values),len(MD2_values)))
sd_kop_all=np.zeros((len(K_all_values),len(MD_all_values)))
# pf1=np.zeros((len(K1_values),len(MD1_values)))
pf2=np.zeros((len(K2_values),len(MD2_values)))
mean_pf_all=np.zeros((len(K_all_values),len(MD_all_values)))

Pxx_all=np.zeros((len(K_all_values),len(MD_all_values),401))
Pxx_90_all=np.zeros((len(K_all_values),len(MD_all_values),90,401))

#file=np.load('KOPandFreqHeatmaps.npz')
file1=sio.loadmat('NodesSpectrums_largerMD')
Pxx1=file1['Pxx_nodes']
freqs=file1['freqs'][0]
kop1=file1['mean_kop']
sd_kop1=file1['std_kop']
pf_mean1=np.mean(file1['pf_nodes'],axis=2)

file2=sio.loadmat('NodesSpectrums_Kdotfive')
Hxx2=file2['Hxx_nodes']
Pxx2=file2['Pxx_nodes']
kop2=file2['mean_kop']
sd_kop2=file2['std_kop']
pf_mean2=np.mean(file2['pf_nodes'],axis=2)


for k,K in enumerate(K1_values):
    for md, MD in enumerate(MD1_values):
        Hxx=frequency.spectralEntropy(Pxx1[k,md,:,:])
        H1[k,md]=np.sum(Hxx)
        
for k,K in enumerate(K2_values):
    for md, MD in enumerate(MD2_values):
        H2[k,md]=np.sum(Hxx2[k,md,:])
        

fig1=plt.figure(figsize=(8,4))
gs=gridspec.GridSpec(len(K_all_values), len(MD_all_values))
for k,K in enumerate(K_all_values):
    for md, MD in enumerate(MD_all_values):
        ax=fig1.add_subplot(gs[len(K_all_values)-k-1,md])
        if K%1==0:
            print('1',K, MD)
            k_now=np.argwhere(K1_values==K)[0][0]
            MD_now=np.argwhere(MD1_values>=MD)[0][0]
            H_all[k,md]=H1[k_now,MD_now]
            kop_all[k,md]=kop1[k_now,MD_now]
            sd_kop_all[k,md]=sd_kop1[k_now,MD_now]
            mean_pf_all[k,md]=pf_mean1[k_now,MD_now]
            Pxx_all[k,md]=np.mean(Pxx1[k_now,MD_now,:,0:401],axis=0)
            Pxx_90_all[k,md,:,:]=Pxx1[k_now,MD_now,:,0:401]
            #print(np.shape(Pxx_all[k,md]))
            #print(np.shape(freqs[0:401]))
            ax.plot(freqs[0:401],Pxx_all[k,md],'k',linewidth=0.5)	
        elif K%0.5==0 and K%1!=0:
            print('2',K, MD)
            k_now=np.argwhere(K2_values==K)[0][0]
            MD_now=np.argwhere(MD2_values>=MD)[0][0]
            kop_all[k,md]=kop2[k_now,MD_now]
            sd_kop_all[k,md]=sd_kop2[k_now,MD_now]
            mean_pf_all[k,md]=pf_mean2[k_now,MD_now]
            H_all[k,md]=H2[k_now,MD_now]
            Pxx_all[k,md]=np.mean(Pxx2[k_now,MD_now,:,0:401],axis=0)
            Pxx_90_all[k,md,:,:]=Pxx2[k_now,MD_now,:,0:401]
            ax.plot(freqs[0:401],Pxx_all[k,md],'k',linewidth=0.5)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
fig2=plt.figure(figsize=(8,6))
im=plt.imshow(np.flipud(H_all),aspect='equal',cmap=plt.cm.turbo,vmin=np.min(H_all)//5*5,vmax=(np.max(H_all)//5+1)*5,interpolation='None')
plt.xticks(np.arange(0,len(MD_all_values),5),np.arange(0,len(MD_all_values),5),fontsize=8)
plt.yticks(np.arange(0,len(K_all_values),1),np.flip(np.arange(0,len(K_all_values),1))*0.5,fontsize=8)
plt.xlabel('mean delay (ms)',fontsize=8)
plt.ylabel('global coupling (K)',fontsize=8)
plt.colorbar(im,shrink=0.5,label='spectral entropy (nits)')
###
fig2.savefig('spectralEntropy40ms.pdf',dpi=300)
fig1.savefig('spectros40ms.png',dpi=300)

np.savez('heatmaps_spectrums.npz',H=H_all,KOP=kop_all,SD_KOP=sd_kop_all,Pxx=Pxx_all,Pxx90=Pxx_90_all,mean_pf=mean_pf_all)

