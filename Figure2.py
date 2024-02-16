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
#Spectrogram resolution #Test with a high resolution 1/20 = 0.05 Hz.
nperseg=5000 
noverlap=2500
th=0.232

init_time=30000
end_time=35000
delay_envelope=4000

file_labels=sio.loadmat('../../input_data/AAL_labels')
labels=file_labels['label90']
file_short_labels=sio.loadmat('../../input_data/AAL_short_labels')
short_labels=file_short_labels['label90']
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/AAL90HalfHours/'
filename=directory+'AAL90_Long_freqMean40HZ_freqSD0HZ_K%.3f_MD%.3f_seed%d'%(K*90,MD,seed)
file_dict=sio.loadmat(filename)
theta=file_dict['theta']
theta=theta[20*fs:20*fs+end_time+1200*fs,:]
f,t,Sxx_nodes=frequency.spectrogram(np.sin(theta.T),nperseg=nperseg,noverlap=noverlap)
Pxx_mean=np.mean(np.mean(Sxx_nodes,axis=-1),axis=0)
#Reduce the time length
theta=theta[0:end_time+8*fs,:]

indexes_peaks=[65,75,147,206,215]
freq_peaks=f[indexes_peaks] #13,15,29.5,43
fbands=[]
f_lows=np.zeros_like(freq_peaks)
f_highs=np.zeros_like(freq_peaks)
for fp,freq in enumerate(freq_peaks):
    f_lows[fp]=freq-0.5
    f_highs[fp]=freq+0.5
    fbands.append('%.2f-%.2f Hz'%(f_lows[fp],f_highs[fp]))

b30,a30=signal.butter(6,2*40/(fs),btype='lowpass')
#b30,a30=signal.butter(2,[2*12.5/fs,2*13.5/(fs)],btype='bandpass')
filtered30=signal.filtfilt(b30,a30,np.sin(theta.T))
filtered30=filtered30[:,4*fs:-4*fs]
tt=np.arange(0,53,dt)

#%%

selected_nodes=[0,1,4,5,8,9,18,19,32,33,34,35,36,37,40,41,42,43,48,49,52,53,66,67,76,77,78,79,80,81]
fig2=plt.figure(figsize=(6.2,8))
gs=gridspec.GridSpec(2,1,height_ratios=[0.14,1],hspace=0.13)


axA=fig2.add_subplot(gs[0])
axA.plot(f[0:300],Pxx_mean[0:300],'k',label='Nodes activity')
for m,fpeak in enumerate(indexes_peaks):
    axA.plot(freq_peaks[m],Pxx_mean[fpeak],'o',color=colors[m],alpha=0.8,label='peak = %.2f filter : %s'%(freq_peaks[m],fbands[m]))
    axA.fill_betweenx(y=np.linspace(0,np.max(Pxx_mean),10),x1=np.ones((10,))*f_lows[m],x2=np.ones((10,))*f_highs[m],color=colors[m],alpha=0.2)
axA.set_xlabel('frequency (Hz)',fontsize=8,labelpad=0)
axA.set_ylabel('Spectrum \n (units^2/Hz)',fontsize=8)
axA.tick_params('both',labelsize=8)
axA.legend(ncol=2,fontsize=8, columnspacing=0.5, handletextpad=0.3,
           loc='upper left', bbox_to_anchor=(0.0,0.85,1,1))

axA.text(-0.4,1,'A',transform=axA.transAxes)
axB=fig2.add_subplot(gs[1])
ytick_pos=[]
ytick_labels=[]

for n,node_n in enumerate(selected_nodes):
    axB.plot(tt[init_time:end_time]-tt[init_time],filtered30[node_n,init_time:end_time]+2.25*n,'k',linewidth=0.15)
    ytick_pos.append(2.25*n)
    ytick_labels.append(labels[node_n].strip()+' ('+short_labels[node_n].strip()+')')

prev_envelopes=0#dummy line, just to avoid IDE warnings
#For each frequency, caculate the envelopes
binary_envelopes=np.zeros((5,90,end_time))
for fb,fband in enumerate(fbands):
    print(fband)
    envelopes_low=synchronization.envelopesFrequencyBand(theta.T,f_low=f_lows[fb],f_high=f_highs[fb],fs=1000,applyLow=False)
    #envelopes_mask=np.ma.masked_where(envelopes_low<=th, envelopes_low)
    
    binary_envelopes[fb,envelopes_low>th]=1#envelopes_low[envelopes_low>th]

#Show only one envelope, but the node could be overlapping the threshold for several frequencies.
plot_binary_envelopes=np.argmax(binary_envelopes,axis=0)+1
#Remove the zeros
plot_binary_envelopes[np.sum(binary_envelopes,axis=0)==0]=0


for fb,fband in enumerate(fbands):
    for n,node_n in enumerate(selected_nodes):
        #Shadows
        axB.fill_between(tt[init_time:end_time]-tt[init_time],y1=2.25*n-1,y2=2.25*n+1,where=binary_envelopes[fb,node_n,init_time:end_time]==1,facecolor=plt.cm.tab10(fb/10),alpha=0.35)
axB.set_ylabel(r'$x_n(t)$ (low-pass filtered 40 Hz)',fontsize=8)
axB.set_xlabel('time (s)',fontsize=8,labelpad=0)
axB.tick_params('both',labelsize=8)
axB.set_yticks(ytick_pos)
axB.set_yticklabels(ytick_labels,fontsize=8)
axB.set_ylim([-2.25,len(selected_nodes)*2.26])
# axB.set_xlim([0,(end_time-init_time)/fs])
axB.text(-0.4,1,'B',transform=axB.transAxes)
fig2.savefig('Fig2.pdf',dpi=600,bbox_inches='tight')
fig2.savefig('Fig2.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
