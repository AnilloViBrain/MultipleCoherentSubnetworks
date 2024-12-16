#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:10:27 2023

@author: felipe
"""


import numpy as np
import matplotlib.pyplot as plt

#Load pre-calculated metrics
file=np.load('sd_T_matrix.npz')
var_tensor=file['var_tensor']
var_vector=file['var_vector']
sq_var_tensor=file['sq_var_tensor']
sq_var_vector=file['sq_var_vector']
count_tensor=file['count_tensor']


file2=np.load('correlation_matrix.npz')
inter_seed_score=file2['inter_seed_score']

#Empty metric matrices
var_sum_matrix=np.zeros((20,5,10))
var_sum1_matrix=np.zeros((20,5,10))
internal_var_sum_matrix=np.zeros((20,5,10))
r_matrix=np.zeros((20,5,10))
rone_matrix=np.zeros((20,5,10))
r_all=(var_vector**2)/sq_var_vector

#Fill the metrics
for nseed in range(20):
    for fb in range(5):
        for nn in range(10):
            var_sum_matrix[nseed,fb,nn]=np.sum(var_tensor[nseed,fb,nn,1:nn+2],axis=-1)
            var_sum1_matrix[nseed,fb,nn]=np.sum(var_tensor[nseed,fb,nn,0:nn+1],axis=-1)
            r_matrix[nseed,fb,nn]=np.sum(var_tensor[nseed,fb,nn,0:nn+2]**2/sq_var_tensor[nseed,fb,nn,0:nn+2],axis=-1)
            rone_matrix[nseed,fb,nn]=np.mean(var_tensor[nseed,fb,nn,0:nn+2]**2/sq_var_vector[nseed,fb],axis=-1)

#Compute the scores
corr_scores=np.zeros((5,10))
score_var=np.zeros((5,10))
score_internal_var=np.zeros((5,10))
score=np.zeros((5,10))
score_r=np.zeros((5,10))
count_corr=np.zeros((5,10))
for fb in range(5):
    for n_cluster in range(10):
        score_var[fb,n_cluster]=np.mean((var_sum_matrix[:,fb,n_cluster]+var_sum1_matrix[:,fb,n_cluster])/(2*var_vector[:,fb]),axis=0)
        corr_scores[fb,n_cluster]=np.mean(inter_seed_score[fb,n_cluster,0:n_cluster+2],axis=-1)
        corr_count_fb=np.corrcoef(count_tensor[:,fb,n_cluster,:])
        count_corr[fb,n_cluster]=np.mean(corr_count_fb[np.tril_indices(10,k=-1)])


#Plot
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.plot(np.arange(3,12),corr_scores[:,1::].T, ':P')
plt.xlabel('M',fontsize=8)
plt.ylabel('Inter-seed functional networks \n correlation of cluster centroids',fontsize=8)
plt.text(-0.1,0.99,'a')
plt.xticks(np.arange(3,12))
plt.subplot(1,2,2)
plt.plot(np.arange(3,12),count_corr[:,1::].T, ':P')
plt.xlabel('M',fontsize=8)
plt.xticks(np.arange(3,12))
plt.ylabel('Inter-seed functional networks \n correlation of occurrences',fontsize=8)
plt.text(-0.1,1,'b')
plt.tight_layout()
plt.savefig('FigS12.pdf',dpi=300,bbox_inches='tight')
plt.savefig('FigS12.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')
