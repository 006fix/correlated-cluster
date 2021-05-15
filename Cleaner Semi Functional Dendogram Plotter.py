# -*- coding: utf-8 -*-
"""
Created on Sat May 15 17:25:29 2021

@author: pyeac
"""

#the following is intended to be a clean easy to use version of the K-means
#variance detection tool.

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


Source_Set = pd.read_csv("C:/Users/pyeac/Downloads/Bike-Sharing-Dataset/day.csv")
Source_Set_Sub = Source_Set[['temp', 'cnt', 'weathersit', 'mnth', 'hum', 'windspeed', 'cnt']]


def normalise_data(dataset):
    scaler=StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data
    
def PCA_data(normalised_dataset):
    #has option to alter the PCA of components via modification here
    reduced_data = PCA(n_components = 2).fit_transform(normalised_dataset)
    return reduced_data

def Dendrogram(reduced_data):
    plt.figure(figsize=(10,7))
    plt.title('Trial Dendogram')
    dend = shc.dendrogram(shc.linkage(Source_Set_Sub, method='ward'))
    
    
#i'll need tto add flexibility here for N_clusters 
def Base_Cluster_Gen(pca_data):
    cluster = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean', linkage='ward')
    cluster.fit_predict(pca_data)
    return cluster

def Plot_Clusters(pca_data, clusters):
    plt.figure(figsize = (10,8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters.labels_, cmap = 'rainbow')
    
def Hierarchical_Clusterer(Source_Set_Sub):
    x1 = normalise_data(Source_Set_Sub)
    x2 = PCA_data(x1)
    x3 = Dendrogram(x2)
    print('x3 done')
    x4 = Base_Cluster_Gen(x2)
    print('x4 done')
    x5 = Plot_Clusters(x2, x4)
    print('x5 done')
    

Hierarchical_Clusterer(Source_Set_Sub)