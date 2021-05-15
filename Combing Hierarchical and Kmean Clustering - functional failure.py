# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:44:02 2021

@author: pyeac
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

Source_Set = pd.read_csv("C:/Users/pyeac/Downloads/Bike-Sharing-Dataset/day.csv")
Source_Set_Sub = Source_Set[['temp', 'cnt', 'weathersit', 'mnth', 'hum', 'windspeed', 'cnt']]

kmeans_kwargs = {
    "init": "random",
    "n_init": 3,
    }

def normalise_data(dataset):
    scaler=StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data
    
def PCA_data(normalised_dataset):
    #has option to alter the PCA of components via modification here
    reduced_data = PCA(n_components = 2).fit_transform(normalised_dataset)
    return reduced_data

def Initial_plot(PCA_dataset):
    plt.scatter(PCA_dataset[:,0], PCA_dataset[:,1])
    plt.show()

def Plot_Elbow_Graph(Elbow_Data):
    plt.plot(range(1,11), Elbow_Data)
    plt.xticks(range(1,11))
    plt.show()

def Plot_Sil_Graph(Sil_Score):
    plt.plot(range(2,11), Sil_Score)
    plt.xticks(range(2,11))
    plt.show()
    
def K_mean_prestages(PCA_dataset):
    
    Elbow_Data = [] #holder for elbow graph data
    
    for k in range(1,11):
        kmeantrial = KMeans(n_clusters = k, **kmeans_kwargs)
        kmeantrial.fit(PCA_dataset)
        Elbow_Data.append(kmeantrial.inertia_)
    
    Plot_Elbow_Graph(Elbow_Data)
    
    Sil_Score = [] #holder for sillhouete score
    
    for k in range(2,11):
        kmeantrial = KMeans(n_clusters = k, **kmeans_kwargs)
        kmeantrial.fit(PCA_dataset)
        score = silhouette_score(PCA_dataset, kmeantrial.labels_)
        Sil_Score.append(score)
    
    Plot_Sil_Graph(Sil_Score)
    
#below follows the number of clusters and number of iterations to be used
Clusters = 3
Iterations_Used = 10

def K_Mean_Iterations(PCA_dataset):
    
    Output_Values = []
    
    for k in range(1, 11):
        kmean_loop = KMeans(n_clusters = 3, **kmeans_kwargs)
        kmean_loop.fit(PCA_dataset)
        Output_Values.append(kmean_loop.labels_)
        
    PCA_dataset_c1 = pd.DataFrame(data=PCA_dataset)
    
    PCA_dataset_c1['rows'] = PCA_dataset_c1.reset_index().index
    
    Output_Values_df = pd.DataFrame(data=Output_Values)

    Output_Values_df_2 = Output_Values_df.transpose()
    
    Output_Values_df_2['rows'] = Output_Values_df_2.reset_index().index
    
    Joined_Data = PCA_dataset_c1.merge(Output_Values_df_2, on='rows')
    
    New_Joined_Data_Labs = Joined_Data.drop(['rows'], axis=1)
    
    return New_Joined_Data_Labs

def Generate_Iteration_Strings(Base_Iteration_Dataset):
    
    Base_Iteration_Dataset['Check_Var'] = Base_Iteration_Dataset.iloc[:,2].astype(str) + Base_Iteration_Dataset.iloc[:,3].astype(str) + Base_Iteration_Dataset.iloc[:,4].astype(str) + Base_Iteration_Dataset.iloc[:,5].astype(str) + Base_Iteration_Dataset.iloc[:,6].astype(str) + Base_Iteration_Dataset.iloc[:,7].astype(str) + Base_Iteration_Dataset.iloc[:,8].astype(str) + Base_Iteration_Dataset.iloc[:,9].astype(str) + Base_Iteration_Dataset.iloc[:,10].astype(str) + Base_Iteration_Dataset.iloc[:,11].astype(str)
    
    return Base_Iteration_Dataset
    
def Generate_Count_by_Cluster(String_Coded_Dataset):
    
    Total_Counts = String_Coded_Dataset.groupby(['Check_Var']).count()
    
    return Total_Counts

def Name_Clusters(Counted_Clusters_Dataset):
    

    Total_Counts2 = Counted_Clusters_Dataset.sort_values(by=['0_y'], ascending=False)
    
    Total_Counts2['clus_num'] = np.arange(len(Total_Counts2))
    Total_Counts2.reset_index(level=0, inplace=True)
    return Total_Counts2

def Core_Clusters(Named_Clusters):
    
    Total_Counts_Cluster_Sets = Named_Clusters.head(Clusters)
    
    return Total_Counts_Cluster_Sets

def Ungrouped_Clusters(Named_Clusters, Core_Clusters):
    
    Ungrouped_Clusters = Named_Clusters[~Named_Clusters.isin(Core_Clusters)].dropna()
    
    return Ungrouped_Clusters

def Gen_Plotting_Dataset(Kmeans_Iterations, Name_Clusters):
    
    Points_With_Labels = Kmeans_Iterations[['0_x', '1_x', 'Check_Var']].merge(Name_Clusters[['Check_Var', 'clus_num']], on='Check_Var')
    
    return Points_With_Labels

def Print_Final_Plot(Gen_Plotting_Dataset):
    
    colour_dict = {0:'b', 1:'g', 2:'r', 3:'c',4:'m', 5:'y', 6:'bisque', 7:'lime', 8:'lavender'}
    
    f = plt.figure(figsize=(10,10))
    plt.subplot(111)
    plt.scatter(Gen_Plotting_Dataset['0_x'], Gen_Plotting_Dataset['1_x'],c=Gen_Plotting_Dataset['clus_num'])
    for i in Gen_Plotting_Dataset['clus_num'].unique():
        if len(Gen_Plotting_Dataset[Gen_Plotting_Dataset['clus_num'] == i]) > 2:
            points = Gen_Plotting_Dataset[Gen_Plotting_Dataset['clus_num'] == i][['0_x', '1_x']].values
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices,0],
                               points[hull.vertices,0][0])
            y_hull = np.append(points[hull.vertices,1],
                               points[hull.vertices,1][0])
            plt.fill(x_hull, y_hull, alpha=0.3, c=colour_dict[i])
    return f    
            
def Full_Kmeans_Repeated_Sampling_Function(Source_Set_Sub):
    x1 = normalise_data(Source_Set_Sub)
    x2 = PCA_data(x1)
    x3 = K_Mean_Iterations(x2)
    x4 = Generate_Iteration_Strings(x3)
    x5 = Generate_Count_by_Cluster(x4)
    x6 = Name_Clusters(x5)
    x7 = Core_Clusters(x6)
    x8 = Ungrouped_Clusters(x6, x7)
    x9 = Gen_Plotting_Dataset(x3, x6)
    x10 = Print_Final_Plot(x9)
    
    
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
    

#Hierarchical_Clusterer(Source_Set_Sub)
    
#Full_Kmeans_Repeated_Sampling_Function(Source_Set_Sub)

def Print_Final_Plot_v22(Gen_Plotting_Dataset, clusters):
    
    colour_dict = {0:'b', 1:'g', 2:'r', 3:'c',4:'m', 5:'y', 6:'bisque', 7:'lime', 8:'lavender'}
    
    f = plt.figure(figsize=(10,10))
    plt.subplot(111)
    plt.scatter(Gen_Plotting_Dataset['0_x'], Gen_Plotting_Dataset['1_x'],c=clusters.labels_, cmap = 'rainbow')
    for i in Gen_Plotting_Dataset['clus_num'].unique():
        if len(Gen_Plotting_Dataset[Gen_Plotting_Dataset['clus_num'] == i]) > 2:
            points = Gen_Plotting_Dataset[Gen_Plotting_Dataset['clus_num'] == i][['0_x', '1_x']].values
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices,0],
                               points[hull.vertices,0][0])
            y_hull = np.append(points[hull.vertices,1],
                               points[hull.vertices,1][0])
            plt.fill(x_hull, y_hull, alpha=0.3, c=colour_dict[i])
    return f

def Kmeans_Hierarchy_Comparison (Source_Set_Sub):
    x1 = normalise_data(Source_Set_Sub)
    x2 = PCA_data(x1)
    x3 = K_Mean_Iterations(x2)
    x3_2 = Dendrogram(x2)
    x4 = Generate_Iteration_Strings(x3)
    x4_2 = Base_Cluster_Gen(x2)
    x5 = Generate_Count_by_Cluster(x4)
    x5_2 = Plot_Clusters(x2, x4_2)
    x6 = Name_Clusters(x5)
    x7 = Core_Clusters(x6)
    x8 = Ungrouped_Clusters(x6, x7)
    x9 = Gen_Plotting_Dataset(x3, x6)
    x10 = Print_Final_Plot_v22(x9, x4_2)
    
Kmeans_Hierarchy_Comparison(Source_Set_Sub)

#initial attempt 1 - absolute and total failure- why?????