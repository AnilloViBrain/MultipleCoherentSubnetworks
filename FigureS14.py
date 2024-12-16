import numpy as np
import matplotlib.pyplot as plt

file=np.load('heatmap_SEModularity.npz')
SE=file['SE']
modularity=file['modularity']

file1=np.load('heatmap_SEModularity_0.9.npz')
modularity1=file1['modularity']

file2=np.load('heatmaps_spectrums.npz')
SD_KOP=file2['SD_KOP']
KOP=file2['KOP']


K_all_values=np.arange(0,10.1,0.5)
MD_all_values=np.arange(0,41,1)

fig,ax=plt.subplots(1,2,figsize=(5,2.5))

axA=ax[0]
im=axA.imshow(np.flipud(modularity1),aspect='auto',cmap='turbo',interpolation='none',vmin=0,vmax=0.2)
axA.set_xlabel('mean dealy [MD] (ms)',fontsize=8)
axA.set_ylabel('global coupling [K]',fontsize=8)
axA.tick_params('both',labelsize=8)
axA.set_xticks(np.arange(0,len(MD_all_values),10))
axA.set_xticklabels(np.arange(0,len(MD_all_values),10),fontsize=8)
axA.set_yticks(np.arange(0,len(K_all_values),5))
axA.set_title('FCD modularity')
ylabels=[]
for yl in np.flip(K_all_values[::5]):
          ylabels.append('%.1f'%yl)
axA.set_yticklabels(ylabels,fontsize=8)
axA.text(-0.35,1,'a',transform=axA.transAxes)
clb=plt.colorbar(im,ax=axA)
clb.ax.tick_params('both',labelsize=8)

axB=ax[1]
plt.plot(np.ravel(SE),np.ravel(modularity1),'.k',markersize=1,label='other K, MD')
axB.plot(SE[8,:],modularity1[8,:],'.c',markersize=3,label='K=4')
axB.plot(SE[:,21],modularity1[:,21],'.m',markersize=3,label='MD=21 ms')
axB.set_xlabel('Spectral entropy (nits)',fontsize=8)
axB.set_ylabel('FCD modularity',fontsize=8)
axB.set_ylim([0,0.2])
axB.legend(fontsize=6)
axB.tick_params('both',labelsize=8)
axB.text(-0.3,1,'b',transform=axB.transAxes)
fig.tight_layout()
fig.savefig('SE_vs_modularity.png',dpi=600,bbox_inches='tight')
fig.savefig('FigS14.pdf',dpi=600,bbox_inches='tight')
fig.savefig('FigS14.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
