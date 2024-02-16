#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:18:59 2023

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
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc
import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
import statsmodels.stats.multicomp as MultiComparison

import pandas as pd 
def relabel(labels,new_tags):
    N=np.max(labels)
    new_labels=np.zeros_like(labels)
    if N==np.max(new_tags):
        old_labels=np.copy(labels)
        for n in range(N+1):
            new_labels[old_labels==new_tags[n]]=n
    return new_labels

Nfb=5
N=90
dt=1e-3
dW=59e-3
time_windows=[231*2,200*2,102*2,73*2,70*2]
max_duration=[1.2*2,1.0*2,0.5*2,0.26*2,0.25*2]
seeds=[3,5,8,13,21,34,55,89,144,145,146,147,148,149,150,151,152,153,154,233]
nsamples=[5000,5000,18000,18000,18000]

#freq_labels=['12.5 - 13.5 Hz','14.5 - 15.5 Hz', '28.9 - 29.9 Hz', '40.7 - 41.7 Hz', '42.5 - 43.5 Hz']
freq_labels=['13.0 Hz','15.0 Hz','29.4 Hz', '41.2 Hz', '43.0 Hz']
nnets_fb=[4,5,4,7,7]
alpha=0.01/np.array(nnets_fb)
sorted_indexes=np.load('index_sortFCs_Coherence2_corrected.npz',allow_pickle=True)['sorted_index'][()]
counts=np.load('count_Coherence2_corrected_networks_scaled.npz',allow_pickle=True)['counts'][()]
occupancy={}
colors=[plt.cm.tab10(0.0),plt.cm.tab10(0.1),plt.cm.tab10(0.2),plt.cm.tab10(0.3),plt.cm.tab10(0.4),plt.cm.tab10(0.5)]
Nseeds=len(seeds)
for fb in range(Nfb):
    occupancy[fb]=np.zeros((Nseeds,nnets_fb[fb]))
    
for nseed in range(Nseeds):
    for fb in range(Nfb):
        if nseed==0:
            occupancy[fb][nseed,:]=counts[nseed][fb]
        else:
            index=sorted_indexes[fb][nnets_fb[fb]][nseed]
            occupancy[fb][nseed,:]=counts[nseed][fb][index]
fig=plt.figure(figsize=(6.5,2.3))
gs=gridspec.GridSpec(2,Nfb,hspace=0.7,wspace=0.75)
axes=[]
max_occupancy=np.zeros((Nfb,))
for fb in range(Nfb):
    ax=fig.add_subplot(gs[0,fb])
    bp=ax.violinplot(occupancy[fb]/np.sum(occupancy[fb],axis=1)[0]*100,showmedians=True)
    print(np.median(occupancy[fb]/np.sum(occupancy[fb],axis=1)[0]*100,axis=0))
    max_occupancy[fb]=np.max(occupancy[fb]/np.sum(occupancy[fb],axis=1)[0])*100
    patch=bp['cmedians']
    patch.set(color=colors[fb],linewidth=0.5)
    patch=bp['cmaxes']
    patch.set(color=colors[fb],linewidth=0.5)
    patch=bp['cmins']
    patch.set(color=colors[fb],linewidth=0.5)
    patch=bp['cbars']
    patch.set(color=colors[fb],linewidth=0.5)         
    for patch in bp['bodies']:
        patch.set_color(colors[fb])
    ax.set_xlabel('subnetwork',fontsize=8)
    ax.set_ylabel('FO (%)',fontsize=8)
    ax.set_title(freq_labels[fb],fontsize=8)
    ax.tick_params('both',labelsize=8)
    axes.append(ax)


p={}
pw={}
for fb in range(Nfb):
    p[fb]={}
    pw[fb]=np.zeros((nnets_fb[fb],nnets_fb[fb]))
    s,p[fb]=stats.normaltest(occupancy[fb],axis=0)
    height=2
    for j in range(nnets_fb[fb]):
        for k in range(j+1,nnets_fb[fb]):
            #w,pw[fb][j,k]=stats.mannwhitneyu(occupancy[fb][:,j],occupancy[fb][:,k],use_continuity=False)
            wilc,pw[fb][j,k]=stats.wilcoxon(occupancy[fb][:,j],occupancy[fb][:,k])
            if fb<3:
                if pw[fb][j,k]<alpha[fb]:
                    axes[fb].hlines([max_occupancy[fb]+height],xmin=j+1,xmax=k+1,color='k',linewidth=0.5)
                    axes[fb].vlines([j+1,k+1],ymin=max_occupancy[fb]+height-1,ymax=max_occupancy[fb]+height,color='k',linewidth=0.5)
                    height=height+2.5
                axes[fb].plot(nnets_fb[fb]/2+0.5,5,'dk',markersize=2.5)
axes[0].set_ylim([-0.5,60.5])
axes[1].set_ylim([-0.5,60.5])
axes[2].set_ylim([-0.5,60.5])
axes[3].set_ylim([-0.5,30.5])
axes[4].set_ylim([-0.5,30.5])
axes[0].set_yticks(np.arange(0,61,20))
axes[1].set_yticks(np.arange(0,61,20))
axes[2].set_yticks(np.arange(0,61,20))
axes[3].set_yticks(np.arange(0,31,10))
axes[4].set_yticks(np.arange(0,31,10))
axes[0].text(-0.7,1.1,'A',transform=axes[0].transAxes)
axes[0].set_xticks(np.arange(1,nnets_fb[0]+1))
axes[0].set_xticklabels(np.arange(1,nnets_fb[0]+1))
# axes[1].text(-0.37,1,'C',transform=axes[1].transAxes)
axes[1].set_xticks(np.arange(1,nnets_fb[1]+1))
axes[1].set_xticklabels(np.arange(1,nnets_fb[1]+1))
axes[2].set_xticks(np.arange(1,nnets_fb[2]+1))
axes[2].set_xticklabels(np.arange(1,nnets_fb[2]+1))
axes[3].set_xticks(np.arange(1,nnets_fb[3]+1))
axes[3].set_xticklabels(np.arange(1,nnets_fb[3]+1))
axes[4].set_xticks(np.arange(1,nnets_fb[4]+1))
axes[4].set_xticklabels(np.arange(1,nnets_fb[4]+1))



# axes[2].text(-0.37,1,'E',transform=axes[2].transAxes)
# axes[3].text(-0.37,1,'G',transform=axes[3].transAxes)
# axes[4].text(-0.37,1,'I',transform=axes[4].transAxes) 

##Durations
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/FC_coherence_clusters/'
# n_events=np.zeros((10,5,7))
# mean_duration=np.zeros((10,5,7))
transitions={}
all_transitions={}
durations={}
for j in range(len(seeds)):
    transitions[j]={}
    durations[j]={}
for fb in range(Nfb):
    all_transitions[fb]=np.zeros((nnets_fb[fb],nnets_fb[fb]))
for nseed,seed in enumerate(seeds):
    for fb in range(Nfb):
        file=np.load(directory+'FC_clusters_Coherence2_Amplitude_fb%d_seed%d.npz'%(fb,seed),allow_pickle=True)  
        labels=file['labels'][()]
        nnet=nnets_fb[fb]
        if nseed>0:
            labels_case=relabel(labels[nnet-2],sorted_indexes[fb][nnet][nseed])
        else:
            labels_case=labels[nnet-2]
        durations[nseed][fb]=synchronization.durationfromLabels(labels_case,time_window=time_windows[fb],overlap=0.5) 
        transitions[nseed][fb]=synchronization.transitionsfromLabels(labels_case)
        all_transitions[fb]+=transitions[nseed][fb]
            # for nnnet in range(nnet):
            #     n_events[j,fb,nnnet]=len(durations[j][fb][nnnet])
            #     mean_duration[j,fb,nnnet]=np.mean(durations[j][fb][nnnet])*dt
            
axesDuration=[]
alpha=0.01/(np.array(nnets_fb))
for fb in range(Nfb):
    if fb>2:
    	height_add=0.03
    	height=0.03
    elif fb>1:
        height_add=0.03
        height=0.03
    else:
        height_add=0.06
        height=0.06
    durations_fb=[]
    nnet=nnets_fb[fb]
    
    pwilc=np.zeros((nnet,nnet))
    ax=fig.add_subplot(gs[1,fb])
    boxprops=dict(linewidth=0.5)
    for nnnet in range(nnet):
        durations_fb_nnet=[]
        durations_seeds_fb=np.zeros((len(seeds),500))
        for j in range(len(seeds)):
            for dur in durations[j][fb][nnnet]:
                durations_fb_nnet.append(dur/1000)
        # plt.figure()
        # plt.boxplot(durations_seeds_fb.T)
        bp=ax.boxplot(durations_fb_nnet,positions=[nnnet+1],sym='', boxprops=boxprops,whis=(0,99))
        durations_fb.append(durations_fb_nnet)
        print(fb, nnnet,np.min(durations_fb_nnet),np.max(durations_fb_nnet),np.median(durations_fb_nnet),np.mean(durations_fb_nnet))
        for patch in bp['boxes']:
            patch.set_color(colors[fb])
        for patch in bp['medians']:
            patch.set_color(colors[fb])
        for patch in bp['whiskers']:
            patch.set_color(colors[fb])
            patch.set_linewidth(0.5)
        for patch in bp['fliers']:
            patch.set_color(colors[fb])
            patch.set_markersize(0.1)
    ax.set_ylabel(r'duration (s)',fontsize=8)
    ax.set_xlabel('subnetwork',fontsize=8)
    
    ax.tick_params('both',labelsize=8)
    #Statistical values
    for i_nnnet in range(nnet):
        for j_nnnet in range(i_nnnet+1,nnet):
            #t,p[i_nnnet,j_nnnet]=stats.mannwhitneyu(np.array(durations_fb[i_nnnet])[0:nsamples[fb]],np.array(durations_fb[j_nnnet])[0:nsamples[fb]],use_continuity=False)
            wilc,pwilc[i_nnnet,j_nnnet]=stats.wilcoxon(durations_fb[i_nnnet][0:nsamples[fb]],durations_fb[j_nnnet][0:nsamples[fb]])
            if fb!=3: #no multi-group difference
                if pwilc[i_nnnet,j_nnnet]<alpha[fb]:
                    ax.hlines([max_duration[fb]+height],xmin=i_nnnet+1,xmax=j_nnnet+1,color='k',linewidth=0.5)
                    ax.vlines([i_nnnet+1,j_nnnet+1],ymin=max_duration[fb]+height-0.005,ymax=max_duration[fb]+height,color='k',linewidth=0.5)
                    height=height+height_add
    ax.set_ylim([0.05,max_duration[fb]+height])
    axesDuration.append(ax)
    if fb==2:
        ax.set_yticks([0,0.4,0.8,1.2])
    elif fb>2:
        ax.set_yticks([0,0.4,0.8])
    else:
        ax.set_yticks([0,1.0,2.0])
    if fb!=3:
        ax.plot(nnets_fb[fb]/2+0.5,0.02+(5-fb)*0.029,'dk',markersize=2.2)
axesDuration[0].text(-0.7,1.1,'B',transform=axesDuration[0].transAxes)
# axesDuration[1].text(-0.37,1,'D',transform=axesDuration[1].transAxes)
# axesDuration[2].text(-0.37,1,'F',transform=axesDuration[2].transAxes)
# axesDuration[3].text(-0.37,1,'H',transform=axesDuration[3].transAxes)
# axesDuration[4].text(-0.37,1,'J',transform=axesDuration[4].transAxes) 

           
fig.savefig('Fig4.pdf',dpi=300,bbox_inches='tight')   
fig.savefig('Fig4.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')                


#%%
## Re-test statistics

for fb in range(Nfb):
    occ=np.ravel(occupancy[fb])
    nets=np.zeros_like(occ)
    tseeds=np.zeros_like(occ)
    nets=np.tile(np.arange(nnets_fb[fb]),len(seeds))
    n=0
    for m in range(len(seeds)):
        tseeds[m]=n
        if (m+1)%nnets_fb[fb]==0:
            n+=1
    dfOccupancy=pd.DataFrame({'Occupancy':occ,'net':nets,'seed':tseeds})
    comp=MultiComparison.MultiComparison(dfOccupancy['Occupancy'],dfOccupancy['net'])
    tbl, a1, a2 = comp.allpairtest(stats.wilcoxon, alpha=0.01, method= "bonf")
    print(tbl)
    pk=comp.kruskal()
    print('kruskal:',pk)
    # comp=MultiComparison.MultiComparison(dfOccupancy['Occupancy'],dfOccupancy['seed'])
    # tbl, a1, a2 = comp.allpairtest(stats.wilcoxon, method= "bonf")
    # print(tbl)

#%%
###
print('====================')
print('Duration')
print('====================')
nsamples=[400,400,1000,1470,1600]
for fb in range(Nfb):
    nnet=nnets_fb[fb]
    dd=np.zeros((nnet,len(seeds),nsamples[fb]))
    nets=np.zeros((nnet,len(seeds),nsamples[fb]))
    s=np.zeros((nnet,len(seeds),nsamples[fb]))
    for nnnet in range(nnet):
        for j in range(len(seeds)):
            dd[nnnet,j,:]=durations[j][fb][nnnet][0:nsamples[fb]]
            nets[nnnet,j,:]=nnnet
            s[nnnet,j,:]=j
    dfDuration=pd.DataFrame({'Duration':np.ravel(dd),'net':np.ravel(nets),'seed':np.ravel(s)})
    comp=MultiComparison.MultiComparison(dfDuration['Duration'],dfDuration['net'])
    pk=comp.kruskal()
    print('kruskal:',pk)
    tbl, a1, a2 = comp.allpairtest(stats.wilcoxon, alpha=0.01, method= "bonf")
    print(tbl)
    
