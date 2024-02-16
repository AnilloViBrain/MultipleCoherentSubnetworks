#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:28:55 2024

@author: felipe
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio

import sys
import os
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../'))

import plot.networks as pltn 
import analysis.connectivityMatrices as connectivityMatrices
import analysis.frequency as frequency

N=90
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/AAL90Heatmaps/'
K=360
MD=0.021
fs=1000
nperseg=5000
noverlap=2500

fileC=sio.loadmat('../../input_data/AAL_matrices.mat')
struct_connectivity=fileC['C']
C=struct_connectivity
C[np.diag(np.ones((N,)))==0] /= C[np.diag(np.ones((N,)))==0].mean()
idx,idy=np.nonzero(C)
values=np.ravel(C[idx,idy])
C_random=np.zeros((N,N))
C_plain=np.ones((N,N))
C_plain[C==0]=0
np.random.seed(2)
np.random.shuffle(values)
C_random[idx,idy]=values

#Test the degree/strength distribution
kC=np.sum(C,axis=1)
kCr=np.sum(C_random,axis=1)
kCp=np.sum(C_plain,axis=1)

fileUniformC=sio.loadmat(directory+'AAL90UniformC_300seg_freqMean40HZ_freqSD0HZ_K%.3F_MD%.3f.mat'%(K,MD))
theta_UniformC=fileUniformC['theta'][20000::,:]

fileUniformD=sio.loadmat(directory+'AAL90UniformD_300seg_freqMean40HZ_freqSD0HZ_K%.3F_MD%.3f.mat'%(K,MD))
theta_UniformD=fileUniformD['theta'][20000::,:]


fileRandom=sio.loadmat(directory+'AAL90RandomC_300seg_freqMean40HZ_freqSD0HZ_K%.3F_MD%.3f.mat'%(K,MD))
theta_Random=fileRandom['theta'][20000::,:]

fileRandom2=sio.loadmat(directory+'AAL90RandomC2_300seg_freqMean40HZ_freqSD0HZ_K%.3F_MD%.3f.mat'%(K,MD))
theta_Random2=fileRandom2['theta'][20000::,:]

file=sio.loadmat(directory+'AAL90_PYTHON_freqMean40HZ_freqSD0HZ_K%.3F_MD%.3f.mat'%(K,MD))
theta=file['theta'][20000::,:]

f,Pxx_uniformC,fpeaks=frequency.peak_freqs(theta_UniformC.T,fs=fs,nperseg=nperseg,noverlap=noverlap,applySin=True)
f,Pxx_uniformD,fpeaks=frequency.peak_freqs(theta_UniformD.T,fs=fs,nperseg=nperseg,noverlap=noverlap,applySin=True)
f,Pxx_Random,fpeaks=frequency.peak_freqs(theta_Random.T,fs=fs,nperseg=nperseg,noverlap=noverlap,applySin=True)
f,Pxx_Random2,fpeaks=frequency.peak_freqs(theta_Random2.T,fs=fs,nperseg=nperseg,noverlap=noverlap,applySin=True)
f,Pxx,fpeaks=frequency.peak_freqs(theta.T,fs=fs,nperseg=nperseg,noverlap=noverlap,applySin=True)


#%%
fig=plt.figure(figsize=(5,7.5))
gs=gridspec.GridSpec(4,2,hspace=0.35,wspace=0.35)
axA=fig.add_subplot(gs[0,0])
axB=fig.add_subplot(gs[0,1])
axC=fig.add_subplot(gs[1,0])
axD=fig.add_subplot(gs[1,1])
axE=fig.add_subplot(gs[2,0])
axF=fig.add_subplot(gs[2,1])
axG=fig.add_subplot(gs[3,0])
axH=fig.add_subplot(gs[3,1])



axA.imshow(np.ma.masked_equal(np.log10(C),0),cmap='turbo',aspect='equal',interpolation='none')

axC.imshow(np.ma.masked_equal(C_plain,0),cmap='turbo',aspect='equal',interpolation='none')
axE.imshow(np.ma.masked_equal(np.log10(C_random),0),cmap='turbo',aspect='equal',interpolation='none')
axG.imshow(np.ma.masked_equal(np.log10(C),0),cmap='turbo',aspect='equal',interpolation='none')

axA.set_xticks([])
axA.set_xticklabels([])
axA.set_yticks([])
axA.set_yticklabels([])

axC.set_xticks([])
axC.set_xticklabels([])
axC.set_yticks([])
axC.set_yticklabels([])

axE.set_xticks([])
axE.set_xticklabels([])
axE.set_yticks([])
axE.set_yticklabels([])

axG.set_xticks([])
axG.set_xticklabels([])
axG.set_yticks([])
axG.set_yticklabels([])

axA.text(-0.1,1,'A',transform=axA.transAxes)
axC.text(-0.1,1,'B',transform=axC.transAxes)
axE.text(-0.1,1,'C',transform=axE.transAxes)
axG.text(-0.1,1,'D',transform=axG.transAxes)
axB.plot(f,Pxx.T,':',linewidth=0.5,color=plt.cm.tab20(11))
axD.plot(f,Pxx_uniformC.T,':',linewidth=0.5,color=plt.cm.tab20(19))
axH.plot(f,Pxx_uniformD.T,':',linewidth=0.5,color=plt.cm.tab20(15))
axF.plot(f,Pxx_Random.T,':',linewidth=0.5,color=plt.cm.tab20(17))
# axF.plot(f,Pxx_Random2.T,':',linewidth=0.5,color=plt.cm.tab20(9))

axB.plot(f,np.mean(Pxx,axis=0),linewidth=2,color=plt.cm.tab20(10),label='AAL90')
axD.plot(f,np.mean(Pxx_uniformC,axis=0),linewidth=2,color=plt.cm.tab20(18),label='Homogeneous C')
axH.plot(f,np.mean(Pxx_uniformD,axis=0),linewidth=2,color=plt.cm.tab20(14),label='Homogeneous D')
axF.plot(f,np.mean(Pxx_Random,axis=0),linewidth=2,color=plt.cm.tab20(16),label='Shuffled C')
# axF.plot(f,np.mean(Pxx_Random2,axis=0),linewidth=2,color='C4',label='shuffled C trial 2')
axB.legend(loc='upper left',fontsize=7,handlelength=1.0,handletextpad=0.5)
axD.legend(loc='upper left',fontsize=7,handlelength=1.0,handletextpad=0.5)
axF.legend(loc='upper left',fontsize=7,handlelength=1.0,handletextpad=0.5)
axH.legend(loc='upper left',fontsize=7,handlelength=1.0,handletextpad=0.5)
axB.set_xlim([0,60])
axB.set_xlabel('frequency (Hz)',fontsize=8)
axB.set_ylabel(r'PSD ($u^2$/Hz)',fontsize=8)
axB.tick_params('both',labelsize=8)
axD.set_xlim([0,60])
axD.set_xlabel('frequency (Hz)',fontsize=8)
axD.set_ylabel(r'PSD ($u^2$/Hz)',fontsize=8)
axD.tick_params('both',labelsize=8)
axD.set_yticks([0,0.2,0.4,0.6,0.8])
axF.set_xlim([0,60])
axF.set_xlabel('frequency (Hz)',fontsize=8)
axF.set_ylabel(r'PSD ($u^2$/Hz)',fontsize=8)
axF.tick_params('both',labelsize=8)
axH.set_xlim([0,60])
axH.set_xlabel('frequency (Hz)',fontsize=8)
axH.set_ylabel(r'PSD ($u^2$/Hz)',fontsize=8)
axH.tick_params('both',labelsize=8)

fig.savefig('FigS11.pdf',dpi=700,bbox_inches='tight')
fig.savefig('FigS11.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
