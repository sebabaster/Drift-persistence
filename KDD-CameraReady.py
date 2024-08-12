# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:39:47 2023
@author: sebbas
"""

# Data tools
import numpy as np
np.float = float  # small correction to make skmultiflow work (problems of versions)
from scipy.io import arff
from scipy.stats import norm, wasserstein_distance
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

# Plot tools
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

# TDA tools
import ripser
import random
from persim.persistent_entropy import *
from ripser import ripser
from persim import plot_diagrams
from scipy import stats

#
# Read the data
data_path = 'C:/Users/sebbas/PycharmProjects/pythonProject/DELTA-Workshop2024/Projections/'
caseStudy="A"
file_X="data_Mnist_"+caseStudy+"_X.npy"
file_y="data_Mnist_"+caseStudy+"_y.npy"
dist_matrix_SOM = np.load(data_path + "SOM_Case_"+caseStudy+"_dist_data_Mnist_X.npy")
dist_matrix_PCA = np.load(data_path + "PCA_Case_"+caseStudy+"_dist_data_Mnist_X.npy")
dist_matrix_KPCA = np.load(data_path + "KernelPCA_Case_"+caseStudy+"_dist_data_Mnist_X.npy")
#
MSOM=np.reshape(dist_matrix_SOM, (dist_matrix_SOM.shape[0],10,10))
MPCA=np.reshape(dist_matrix_PCA, (dist_matrix_PCA.shape[0],10,10))
MKPCA=np.reshape(dist_matrix_KPCA, (dist_matrix_KPCA.shape[0],10,10))
#
# To work with list for using persim package.
listSOM=[]
listPCA=[]
listKPCA=[]
for i in range(MSOM.shape[0]):
	listSOM.append(MSOM[i,:,:].reshape(50,2))
	listPCA.append(MPCA[i,:,:].reshape(50,2))
	listKPCA.append(MKPCA[i,:,:].reshape(50,2))

print("SOM shape", MSOM.shape)
print("PCA shape", MPCA.shape)
#######
chunks=[50,100,250]
for chunkSize in chunks:
    p = 0
    top=int(np.trunc(MSOM.shape[0]/chunkSize))-1
    #
    pValueSOM=np.zeros(shape=(1,top))
    pValuePCA=np.zeros(shape=(1,top))
    pValueKPCA=np.zeros(shape=(1,top))
    # Per two consecutives chunks
    for k in range(top):
        # Per sample in the chunks
        dgm_SOM1 = []
        dgm_SOM2 = []
        dgm_PCA1 = []
        dgm_PCA2 = []
        dgm_KPCA1 = []
        dgm_KPCA2 = []
        for i in range(chunkSize):
            dgm_SOM1.append(ripser(listSOM[k*chunkSize+i])['dgms'][p])
            dgm_SOM2.append(ripser(listSOM[(k+1)*chunkSize+i])['dgms'][p])
            #
            dgm_PCA1.append(ripser(listPCA[k * chunkSize + i])['dgms'][p])
            dgm_PCA2.append(ripser(listPCA[(k + 1) * chunkSize + i])['dgms'][p])
            #
            dgm_KPCA1.append(ripser(listKPCA[k * chunkSize + i])['dgms'][p])
            dgm_KPCA2.append(ripser(listKPCA[(k + 1) * chunkSize + i])['dgms'][p])
            # Calculate their persistent entropy.

        pValueSOM[0,k]=stats.mannwhitneyu(persistent_entropy(dgm_SOM1), persistent_entropy(dgm_SOM2))[1]
        pValuePCA[0,k]=stats.mannwhitneyu(persistent_entropy(dgm_PCA1), persistent_entropy(dgm_PCA2))[1]
        pValueKPCA[0,k]=stats.mannwhitneyu(persistent_entropy(dgm_KPCA1), persistent_entropy(dgm_KPCA2))[1]

    np.save(data_path+"pValueSOM_"+str(chunkSize)+"_Case_"+caseStudy+".npy",pValueSOM[0,:])
    np.save(data_path+"pValuePCA_"+str(chunkSize)+"_Case_"+caseStudy+".npy",pValuePCA[0,:])
    np.save(data_path+"pValueKernelPCA_"+str(chunkSize)+"_Case_"+caseStudy+".npy",pValueKPCA[0,:])

