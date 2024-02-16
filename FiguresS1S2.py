#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:43:56 2023

@author: felipe
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

file=np.load('heatmaps_spectrums.npz')
H=file['H']
KOP=file['KOP']
SD_KOP=file['SD_KOP']
Pxx=file['Pxx']
normPxx=Pxx/np.max(Pxx,axis=2,keepdims=True)
K_all_values=np.arange(0,10.5,0.5)
MD_all_values=np.arange(0,41,1)
f=np.arange(0,80.2,0.2)
f_peak=f[np.argmax(Pxx,axis=2)]
max_MD=21
max_K=4*2-1

normKOP=plt.Normalize(vmin=0,vmax=1)
normKOPSD=plt.Normalize(vmin=0,vmax=0.25)

figS1=plt.figure(figsize=(4.5,6))
g1s=gridspec.GridSpec(3, 1)
axA=figS1.add_subplot(g1s[0])
imA=axA.imshow(np.flipud(KOP),cmap=plt.cm.jet,aspect='equal',interpolation='none',norm=normKOP)
cbA=figS1.colorbar(imA,ax=axA,shrink=0.9)
axA.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
axA.set_ylabel('global coupling [K]',fontsize=8)
axA.set_yticks(np.arange(0,len(K_all_values),2))
axA.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
axA.set_xticks(np.arange(0,len(MD_all_values),5))
axA.set_xticklabels(np.arange(0,len(MD_all_values),5))
axA.tick_params('both',labelsize=8)
axA.plot(max_MD,19-max_K,'s',color='gray',mfc='none')
axA.text(-0.15,1,'A',transform=axA.transAxes)
cbA.set_label(r'$\langle$ KOP $\rangle$',fontsize=8)


axB=figS1.add_subplot(g1s[1])
imB=axB.imshow(np.flipud(SD_KOP),cmap=plt.cm.jet,aspect='equal',interpolation='none',norm=normKOPSD)
cbB=figS1.colorbar(imB,ax=axB,shrink=0.9)
axB.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
axB.set_ylabel('global coupling [K]',fontsize=8)
axB.set_yticks(np.arange(0,len(K_all_values),2))
axB.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
axB.set_xticks(np.arange(0,len(MD_all_values),5))
axB.set_xticklabels(np.arange(0,len(MD_all_values),5))
axB.tick_params('both',labelsize=8)
axB.plot(max_MD,19-max_K,'s',color='gray',mfc='none')
axB.text(-0.15,1,'B',transform=axB.transAxes)
cbB.set_label('sd(KOP)',fontsize=8)
cbB.ax.set_yticks(np.arange(0,0.26,0.05))


axC=figS1.add_subplot(g1s[2])
#Average of the peaks of each node
imC=axC.imshow(np.flipud(f_peak),cmap=plt.cm.jet,aspect='equal',interpolation='none',vmin=0,vmax=50)
#Peak of the average spectrum
# imC=axC.imshow(np.flipud(peak_Pxx),cmap=plt.cm.jet,aspect='equal',interpolation='None',vmin=0,vmax=40)
cbC=figS1.colorbar(imC,ax=axC,shrink=0.9)
axC.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
axC.set_ylabel('global coupling [K]',fontsize=8)
axC.set_yticks(np.arange(0,len(K_all_values),2))
axC.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
axC.set_xticks(np.arange(0,len(MD_all_values),5))
axC.set_xticklabels(np.arange(0,len(MD_all_values),5))
axC.tick_params('both',labelsize=8)
axC.plot(max_MD,19-max_K,'s',color='gray',mfc='none')
axC.text(-0.15,1,'C',transform=axC.transAxes)
cbC.set_label('Hz',fontsize=8)
cbC.ax.set_yticks(np.arange(0,51,10))
figS1.tight_layout()
figS1.savefig('FigS2.pdf',dpi=300,bbox_inches='tight')
figS1.savefig('FigS2.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')


#%%
figS2=plt.figure(figsize=(6,2.2))
g2s=gridspec.GridSpec(1, 3,wspace=0.5)
axA=figS2.add_subplot(g2s[0])
im=axA.imshow(normPxx[4,:,0:301].T,aspect='auto',interpolation='none',cmap='magma_r')
axA.set_yticks(np.arange(0,301,25))
axA.set_yticklabels(f[0:301:25])
axA.set_ylabel('frequency (Hz)',fontsize=8)
axA.set_xlabel('mean delay [MD] (ms)',fontsize=8)
axA.set_title('K=2',fontsize=8)
axA.set_xticks(np.arange(0,41,10))
axA.tick_params('both',labelsize=8)

axB=figS2.add_subplot(g2s[1])
axB.imshow(normPxx[8,:,0:301].T,aspect='auto',interpolation='none',cmap='magma_r')
axB.set_yticks(np.arange(0,301,25))
axB.set_yticklabels(f[0:301:25])
axB.set_xticks(np.arange(0,41,10))
# axB.set_ylabel('frequency (Hz)')
axB.set_xlabel('mean delay [MD] (ms)',fontsize=8)
axB.set_title('K=4',fontsize=8)
axB.tick_params('both',labelsize=8)


axC=figS2.add_subplot(g2s[2])
axC.imshow(normPxx[12,:,0:301].T,aspect='auto',interpolation='none',cmap='magma_r')
axC.set_yticks(np.arange(0,301,25))
axC.set_yticklabels(f[0:301:25])
axC.set_xticks(np.arange(0,41,10))
# axC.set_ylabel('frequency (Hz)')
axC.set_xlabel('mean delay [MD] (ms)',fontsize=8)
axC.set_title('K=6',fontsize=8)
axC.tick_params('both',labelsize=8)

# axD=figS2.add_subplot(g2s[3])
# axD.imshow(normPxx[16,:,0:301].T,aspect='auto',interpolation='none',cmap='magma_r')
# axD.set_yticks(np.arange(0,300,25))
# axD.set_yticklabels(f[0:300:25])
# axD.set_ylabel('frequency (Hz)')
# axD.set_xlabel('mean delay (ms)')
# axD.set_title('K=8')
# axD.tick_params('both',labelsize=8)

cb2=figS2.colorbar(im,ax=[axA,axB,axC],pad=0.01)
cbC.set_label(r'$u^2/Hz$',fontsize=8)

figS2.savefig('FigS1.pdf',dpi=300,bbox_inches='tight')
figS2.savefig('FigS1.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
