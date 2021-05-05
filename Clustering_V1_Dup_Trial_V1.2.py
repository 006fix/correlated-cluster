# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:47:21 2021

@author: pyeac
"""

#step 1 - import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#step 2 - import dataset

Bikes_Daily = pd.read_csv("C:/Users/pyeac/Downloads/Bike-Sharing-Dataset/day.csv")

#step 3 - subset the dataset

Bikes_Daily_Sub = Bikes_Daily[['temp', 'cnt', 'weathersit', 'mnth', 'hum', 'windspeed', 'cnt']]

#step 4 - scale data

scaler=StandardScaler()
scaled_data = scaler.fit_transform(Bikes_Daily_Sub)

#step 5 - PCA the data
#at present, we will do this down to 2 axes, for ease of visualisation

#for future steps, I should maybe consider what the most appropriate reduction should be

reduced_data = PCA(n_components = 2).fit_transform(scaled_data)

#step 6 - plot the reduced data, just to make sure it looks ok

plt.scatter(reduced_data[:,0], reduced_data[:,1])
plt.show()

#step 6 - carry out analyses of optimal number of clusters to utilise

#step 6.0.1 - generate input variables to feed in for function arguments

kmeans_kwargs = {
    "init": "random",
    "n_init": 20,
    }

#step 6.1.1 - generate elbow graph

SSE = []
#holder variable

for k in range(1,11):
    kmeantrial = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeantrial.fit(reduced_data)
    SSE.append(kmeantrial.inertia_)


#step 6.1.2 - plots the eblow graph
plt.plot(range(1,11), SSE)
plt.xticks(range(1,11))
plt.show()

#step 6.2.1 - generate sillhouete score

Sil_Col = []
#holder var

for k in range(2,11):
    kmeantrial = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeantrial.fit(reduced_data)
    score = silhouette_score(reduced_data, kmeantrial.labels_)
    Sil_Col.append(score)
    
#similar to above but runs from N=2 since it requires 2+ clusters

# step 6.2.2 - plot sillhouete score
plt.plot(range(2,11), Sil_Col)
plt.xticks(range(2,11))
plt.show()

#step 6.3.1 - analyse results

#based on the above, it apears that 3 clusters is optimal for this dataset.

#proceeding forward, we shall use N = 3.

#step 6.3.2 - create variable to hold number of clusters

Clusters = 3

#step 6.3.3 - create variable to hold number of iterations conducted

Iterations_used = 10

#step 6.4 - generate K-means clustering, run it Z times, where Z = number of runs, and for each run, append the labels to a new dataset
#for this trial, we'll do 10 runs.


#holder var for our numerous outputs to be apended to
Output_Values = []

for k in range(1, 11):
    kmean_loop = KMeans(n_clusters = 3, **kmeans_kwargs)
    kmean_loop.fit(reduced_data)
    Output_Values.append(kmean_loop.labels_)


#the below simply serves to check that the size of the label sets and the original lists are the same
print(len(reduced_data))
print(len(Output_Values[0]))

#step 6.5.1 - create a new subset list to append onto, writing to pandas dataframe
#also add in a rownum to join on

reduced_data_copy1 = pd.DataFrame(data=reduced_data)
reduced_data_copy1['rows'] = reduced_data_copy1.reset_index().index

#step 6.5.2 - write the numpy array to a pandas dataframe
#also add in a rownum to join on
#also transpose it since otherwise they're the wrong way round

Output_Values_df = pd.DataFrame(data=Output_Values)

Output_Values_df_2 = Output_Values_df.transpose()
Output_Values_df_2['rows'] = Output_Values_df_2.reset_index().index

#step 6.5.3 - check to make sure the dataframes are the same size

print('DETAILS FOLLOW')
print(reduced_data_copy1.shape)
print(Output_Values_df_2.shape)

#step 6.5.4 - merge the two datasets together

Joined_Data = reduced_data_copy1.merge(Output_Values_df_2, on='rows')

#step 6.5.5 - check to ensure accurate merge

print(Joined_Data.head(15))

#step 6.6.0 - identify degree of concordance
#create 3 columns, 1 for each cluster, identify % of members present
#I think I can do this with Bool masks, then summing along axis=1.
#I really ought to map this within a function, that iterates the same thing across all N clusters
 
#step 6.6.1 - create datasubset containing just the labels
#the below seems to work fine, however 3: needs to become variable
#MAKE IT VARIABLE
#ISN'T IT ALREADY VARIABLE? 3RD COL ONWARDS

Joined_Data_Labs = Joined_Data.iloc[:, 3:]

print(Joined_Data_Labs)

#step 6.6.1 - generate new dataframe to contain copy of joined data to add results back to

Joined_Data_v2 = Joined_Data

#step 6.7.0 - generate meta function to iterate over n of clusters
#step 6.7.1 - dynamically generate new variables to represent % of each row that equals that cluster number
#step 6.7.2 - dynamically generate new variables to represent each of the above as a pandas dataframe
#step 6.7.3 - modify variables from 6.7.2 to include a row number to allow joining onto base dataset
#step 6.7.4 - modify the 0th indexed 
#step 6.7.5 - join in these new variables back into the base dataset
    
# for i in range(0, Clusters):
#     globals()['Cluster_{}'.format(i)] = 100*(Joined_Data_Labs[Joined_Data_Labs == i].count(axis = 1))/10
#     #creates new variable, as array of %'s of count of clusters
#     #change the /10 to/ N of loops
#     #MAKE IT VARIABLE
#     globals()['Cluster_df_{}'.format(i)] = pd.DataFrame(data=(globals()['Cluster_{}'.format(i)]))
#     #creates additional new variable as pd dataframe of above arrays
#     (globals()['Cluster_df_{}'.format(i)])['rows'] = (globals()['Cluster_df_{}'.format(i)]).reset_index().index
#     #generates a row number for each variable
#     (globals()['Cluster_df_{}'.format(i)]).rename(columns={0:('Cluster ' + str(i) + ' percentage')}, inplace=True)
#     Joined_Data_v2 = Joined_Data_v2.merge((globals()['Cluster_df_{}'.format(i)]), on='rows')

# print('Represents the output dataset')           
# print(Joined_Data_v2.iloc[:,6:])

#pausing for now: for next time
#the above was fascinating but ultimately useless. i'm not even really sure it needs to exist
#what I really need to do is generate string concats for each of the rows
#this will need to be done such that if n < 10 (true num format), it gets a 0 appended, to allow for n clusters = > 10
#once that's done, I need to count distinct string_concats, then ideally find a way to measure difference

#step 6.7.0.1 - generate new column as string concat of prev rows
#step 6.7.0.2 - this ideallly needs to be flexible to the number of iterations conducted
#step 6.7.0.3 - this also needs to modify strings such that if N < 10, string = 0&val. 
#but lets leave step 6.7.0.2 and 6.7.03 aside for now
#seriously though, with the solution below how on EARTH will i handle 6.7.0.2????

Joined_Data_Labs['Check_Var'] = Joined_Data_Labs.iloc[:,0].astype(str) + Joined_Data_Labs.iloc[:,1].astype(str) + Joined_Data_Labs.iloc[:,2].astype(str) + Joined_Data_Labs.iloc[:,3].astype(str) + Joined_Data_Labs.iloc[:,4].astype(str) + Joined_Data_Labs.iloc[:,5].astype(str) + Joined_Data_Labs.iloc[:,6].astype(str) + Joined_Data_Labs.iloc[:,7].astype(str) + Joined_Data_Labs.iloc[:,8].astype(str) + Joined_Data_Labs.iloc[:,9].astype(str)
        
print(Joined_Data_Labs)

Total_Counts = Joined_Data_Labs.groupby(['Check_Var']).count()

print(Total_Counts)

#the below will need to be changed to be a variable, or at least a
#consistent first variable name

#step 6.7.1 - sort by highest to lowest
#aim is that the most common clusters should all have the highest value counts
#honestly I feel like this is trash, and should maybe be replaced by something
#utilising the average size of each dataset in each run?
#might be interesting to code
Total_Counts2 = Total_Counts.sort_values(by=['0_y'], ascending=False)

#this appears to provide a good view of the unique outputs and their frequency
#from our example we can see that only one row varies, with the final cluster alternating between 0 and 2
#step 6.7.0.3.1 - could probably tidy up the graph, it only really needs one column which could easily be renamed.

#the trouble is, my results appear to suggest the data I've got at present is really, REALLY stable, with only the smallest of variations

#step 6.7.2 - select the first Nclusters rows, based on the above
#but as mentioned I bet there's a better way
Total_Counts_Cluster_Sets = Total_Counts2.head(Clusters)

#step 6.7.3.1 - selects the rows which do not exist in the set of main clusters
#this allows us to identify which of the main clusters they are closest to
Ungrouped_Clusters = Total_Counts2[~Total_Counts2.isin(Total_Counts_Cluster_Sets)].dropna()
print(Ungrouped_Clusters)

#step 6.7.3.2 - turn the index 'Check_Var' of my two datasets into an actual column
#FOR LATER - MAYBE THIS NEEDS TO BE EARLIER???
Total_Counts_Cluster_Sets.reset_index(level=0, inplace=True)
Ungrouped_Clusters.reset_index(level=0, inplace=True)

#step 6.8.0 - identify closest clusters
#this can be rephrased as "sort by min->max number of changes required to match each of the top clusters

#step 6.8.1 - the following function will calculate hamming distance
#minimum hamming distance = closest match
def hamming_calc(s1,s2):
    result=0
    if(len(s1) != len(s2)):
        print('SOMETHINGS GONE WRONG')
    else:
        for x,(i,j) in enumerate(zip(s1,s2)):
            if i!=j:
                result+=1
    return result

#step 6.8.2 - now I need a function that checks the various possible values and shows match %'s

#even if this works I should probably just make it so i can rip all the lists out of the functions
#because.
#somethings going wrong with the list of list generation.
def hamming_comparator(cluster_variables, unmapped_variables):
    Unmapped_Checks = []
    Cluster_Checks = []
    Distances = []
    for i in range(len(unmapped_variables.iloc[:,0])):
                 for j in range(len(cluster_variables.iloc[:,0])):
                     temp = hamming_calc(unmapped_variables.iloc[i,0],cluster_variables.iloc[j,0])
                     Unmapped_Checks.append(unmapped_variables.iat[i,0])
                     Cluster_Checks.append(cluster_variables.iat[j,0])
                     Distances.append(temp)
    Final_LOL = [Unmapped_Checks, Cluster_Checks, Distances]
    return Final_LOL

Comparison_Dataset = pd.DataFrame(data=(hamming_comparator(Total_Counts_Cluster_Sets, Ungrouped_Clusters)))
Comparison_Dataset = Comparison_Dataset.transpose()
Comparison_Dataset.columns = ['Unassigned_Value', 'Cluster_Values', 'Distance']
print(Comparison_Dataset)


#IT WORKS!!!!!!!!!