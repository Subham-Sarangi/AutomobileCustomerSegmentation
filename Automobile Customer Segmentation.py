#!/usr/bin/env python
# coding: utf-8

# In[86]:


import os
import warnings
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import pickle
import plotly.graph_objects as go
import plotly.express as px 
from sklearn.svm import SVC
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')


# ## <font color='Green'>1.0 Loading Data</font>

# In[87]:


df =pd.read_csv('automobile.csv')
df.head(5)


# ## <font color='Green'>1.1 Understanding the Dataset</font>

# In[88]:


df.describe()


# In[89]:


df.info()


# In[ ]:





# In[90]:


#Dropping ID which is not required for the analysis
df.drop('ID', axis=1, inplace=True)


# In[31]:


cont_features = []
cat_features = []

for c in df.columns:
    if df[c].dtype == 'int64':
        print(df[c].dtype)
        cont_features += [c]
    else:
        print(df[c].dtype)
        cat_features += [c]
# df[cat_features].nunique()
df[cont_features].nunique()


# ## Checking for Missing Values

# In[91]:


df.isna().sum()


# ### Missing values: 
# normalized-losses: 41, num-of-doors: 2, bore: 4, stroke: 4, horsepower: 2, peak-rpm: 2, price: 4

# ## Finding proportion of missing values

# In[93]:


# Finding proportion of missing values in entire data

# Size and shape of the dataframe
print("Size of the dataframe:", df.size)
print("Shape of the dataframe:", df.shape)

# Overall dataframe
print("Count of all missing values in dataframe: ", df.isnull().sum().sum())

# Overall % of missing values in the dataframe
print("% of missing values in dataframe: ", round((df.isnull().sum().sum()/df.size)*100,2),"%")


# Overall missing values is < 10%

# ## <font color='Green'>1.2 Missing Value Analysis</font>

# ### <font color='Blue'>Complete Cases Approach</font>
# We ignore the cases with missing values and use the rest

# In[94]:


df_cc = df.dropna()
print("Original data:",df.shape)
print("After removing cases with missing values:",df_cc.shape)


# In[95]:


df_cc.isna().sum()


# ## <font color='Green'>1.3 Data Visualisation</font>

# In[96]:


import seaborn as sn

correlations=df_cc.corr()
f,ax=plt.subplots(figsize=(20,10))

sn.heatmap(correlations,annot=True)


# ## <font color='Green'>1.4 Detecting and Removing Outliers</font>

# In[97]:


from scipy.stats import zscore
import matplotlib.pyplot as plt
columns = ['wheel-base','length','width','height','bore','stroke','compression-ratio','horsepower','peak-rpm',
           'city-mpg','highway-mpg','price']
df_cc1 = df_cc[columns]
df_cc1 = df_cc1.apply(zscore)
df_cc1.head()


# In[98]:


fig = plt.figure(figsize = (15,15))
box = plt.boxplot(df_cc1, vert=True, patch_artist=True, labels=df_cc1.columns);
fig = plt.ylabel('Z_Score', fontsize=10);
fig = plt.grid()


# <b> Inference </b>
# 
# From the box plot, we can notice that fewer variables 'wheel-base','length','width','stroke','compression-ratio' have several outliers. The outliers have to removed in the data processing step.

# <b> Cut-offs for Outliers </b>
# 

# In[99]:


df_cc['z_score_tc'] = zscore( df_cc['price'] )


# In[100]:


df_cc[ ( df_cc.z_score_tc > 3.0) | ( df_cc.z_score_tc < -3.0) ]


# In[101]:


df_cc = df_cc.drop([46,69,71])
df_cc = df_cc.drop(['z_score_tc'],axis=1)
df_cc.shape 


# ## <font color='Green'>1.5 Scaling the Data</font>
# 

# In[103]:


scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cc1),columns = df_cc1.columns)
df_scaled.describe()


# ## <font color='Green'>1.6 Dimensionality Reduction</font>

# #### K-means Clustering, Hierarchial clustering use the Euclidean distance, which gets affected as the number of dimensions increase. So, before using these methods, we have to reduce the number of dimensions. Hence, we are using PCA which is by far the most popular dimensionality reduction algorithm.

# In[104]:


from sklearn.decomposition import PCA
pca= PCA()
pca.fit(df_scaled)


# In[105]:


pca.components_


# In[106]:


pca.explained_variance_


# ### Analyzing Results - Explained Variance

# In[107]:


explained_variance = pca.explained_variance_ratio_
explained_variance


# ### Analyzing Results - Scree Plot

# In[108]:


plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# <b> We can see that at n_components = 7, we can capture 95% of variance in the data. </b>

# ### Executing PCA with 7 components

# In[109]:


pca = PCA(n_components=7)
labels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7']
df_scaled_PCA = pd.DataFrame(pca.fit_transform(df_scaled),columns = labels)


# ### Understanding the PCA data

# In[110]:


df_scaled_PCA.describe()


# ### Finding which features contribute to the PCA component

# In[128]:


fig, ax = plt.subplots(figsize=(24, 16))
plt.imshow(pca.components_.T,
           cmap="Spectral",
           vmin=-1,
           vmax=1,
          )
plt.yticks(range(len(df_scaled.columns)), df_scaled.columns)
plt.xticks(range(len(df_scaled_PCA.columns)), df_scaled_PCA.columns)
plt.xlabel("Principal Component")
plt.ylabel("Contribution")
plt.title("Contribution of Features to Components")
plt.colorbar()


# ## <font color='Green'>Model Building</font>

# ## <font color='Green'>2.1 Building K-Means Cluster Model</font>

# In[118]:


import collections 
from sklearn.cluster import KMeans
def CountFrequency(arr): 
    return collections.Counter(arr)


# In[119]:


cluster_range = range( 1, 11 )
cluster_errors = []  # Captures WSS

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( df_scaled_PCA )
    cluster_errors.append( clusters.inertia_ )
    print(CountFrequency(clusters.labels_))
   
print("cluster_errors:", cluster_errors)
plt.figure(figsize=(6,4))
plt.plot( cluster_range, cluster_errors, marker = "o" );
plt.xlabel('Number of clusters');
plt.ylabel('WCSS');
plt.title( "Scree Plot");


# Size wise = cluster solutions containing 4 and 5 clusters look good 

# Let's see the performance Measures and select the best cluster solution

# In[129]:


model_3clusters = KMeans(n_clusters=3).fit(df_scaled_PCA).labels_
model_4clusters = KMeans(n_clusters=4).fit(df_scaled_PCA).labels_


# ###  <font color='Green'>2.2 Measuring the Performance</font>

# ###  Performance Measure: Silhouette Score

# In[132]:


from sklearn import metrics
print("Silhouette Coefficient for 3 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, model_3clusters))
print("Silhouette Coefficient for 4 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, model_4clusters))

## Silhouette score between -1 and 1


# ###  Performance Measure: Calinski-Harabasz index

# In[133]:


print("Calinski-Harabasz index of 3 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, model_3clusters))
print("Calinski-Harabasz index of 4 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, model_4clusters))


#     From the performance metrics, we can notice that 3 cluster solution is performing better and is also the elbow point

# ###  <font color='Green'>2.3 Examining Chararcteristics with 3 cluster solution</font>

# In[135]:


df_scaled["model_3clusters"] = model_3clusters
cluster_size3 = df_scaled.groupby(['model_3clusters']).size() 
print(cluster_size3)
print("")


# In[136]:


values=['wheel-base','length','width']
index =['model_3clusters']
aggfunc={'wheel-base': np.mean,
         'length': np.mean,
         'width': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size3
result = result.round(2)
result


# <b> Insights: </b>
# 
# Classification on the basis of cluster size, cluster 0 is the larger cluster containing 63 data points followed by cluster 2 with 52 data points and the last is cluster 1 with 44 data points.
# 
# We can notice that cluster 0 & 2 has lowest length, width & wheel base of the automobile model whereas cluster 1 has all the three features higher than other two.

# In[137]:


values=['height','bore','stroke']
index =['model_3clusters']
aggfunc={'height': np.mean,
         'bore': np.mean,
         'stroke': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size3
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 1 has models with highest diameter of each wheel in the automobile (bore), with highest height and highest number of phases in engine's cycle(stroke).
# 
# Cluster 2 has models with lowest diameter of each wheel(bore), lowest height of the model and low number of phases in engine's cycle(stroke).
# 
# Cluster 0 has models with less height but medium diameter of each wheel and number of phases in engine's cycle of the model.
# 

# In[138]:


values=['compression-ratio','horsepower','peak-rpm']
index =['model_3clusters']
aggfunc={'compression-ratio': np.mean,
         'horsepower': np.mean,
         'peak-rpm': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size3
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 1 has models with highest volume of cylinder and chamber (compression-ratio) in the engine with highest power automobile (horsepower) but low revolutions per minute (peak-rpm)
# 
# Cluster 0 has models with lowest volume of cylinder and chamber in the engine (compression-ratio), low power automobile (horsepower) and low revolutions per minute ((peak-rpm)
# 
# Cluster 2 has models with lowest power automobile (horsepower) but medium volume of cylinder and chamber of the engine (compression-ratio) with medium revolutions per minute (peak-rpm)

# In[189]:


values=['city-mpg','highway-mpg','price']
index =['model_3clusters']
aggfunc={'city-mpg': np.mean,
         'highway-mpg': np.mean,
         'price': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size3
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 2 has cheapest automobile models (price) with highest scoring car in an average city (city-mpg) & highway (highway-mpg)
# 
# Cluster 1 are the most expensive automobile models with lowest scoring car in an average city or highway as it is pretty obviuos that in an average city or highway the number of expensive cars would be less due to affordability of the citizens and road infrastructure
# 
# Cluster 0 has average price automobile model with medium score in city & highway and highest number of data points

# ### CONCLUSION

# <ul>
#     <li>Cluster 0: It has the highest number of data points </li>
#     <li>Cluster 1: </li>
#         <li>Cluster 2: </li>  
# </ul>    

# ## <font color='Green'>3.1 Building Hierarchial Clustering Model</font>

# In[177]:


from sklearn.cluster import AgglomerativeClustering
clusterid3 = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid4 = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid5 = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid6 = AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid7 = AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid8 = AgglomerativeClustering(n_clusters=8,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid9 = AgglomerativeClustering(n_clusters=9,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_
clusterid10 = AgglomerativeClustering(n_clusters=10,affinity='euclidean',linkage='ward').fit(df_scaled_PCA).labels_


# # Assign Cluster Labels

# In[178]:


df_scaled_PCA["clusterid3"] = clusterid3
df_scaled_PCA["clusterid4"] = clusterid4
df_scaled_PCA["clusterid5"] = clusterid5
df_scaled_PCA["clusterid6"] = clusterid6
df_scaled_PCA["clusterid7"] = clusterid7
df_scaled_PCA["clusterid8"] = clusterid8
df_scaled_PCA["clusterid9"] = clusterid9
df_scaled_PCA["clusterid10"] = clusterid10

cluster_size3 = df_scaled_PCA.groupby(['clusterid3']).size() 
cluster_size4 = df_scaled_PCA.groupby(['clusterid4']).size() 
cluster_size5 = df_scaled_PCA.groupby(['clusterid5']).size() 
cluster_size6 = df_scaled_PCA.groupby(['clusterid6']).size()
cluster_size7 = df_scaled_PCA.groupby(['clusterid7']).size()
cluster_size8 = df_scaled_PCA.groupby(['clusterid8']).size()
cluster_size9 = df_scaled_PCA.groupby(['clusterid9']).size()
cluster_size10 = df_scaled_PCA.groupby(['clusterid10']).size()

print(cluster_size3)
print("")
print(cluster_size4)
print("")
print(cluster_size5)
print("")
print(cluster_size6)
print("")
print(cluster_size7)
print(cluster_size8)


# ## <font color='Green'>3.2 Measuring the Performance</font>

# ###  Performance Measure: Silhouette Score

# In[179]:


print("Silhouette Coefficient of 3 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid3))
print("Silhouette Coefficient of 4 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid4))
print("Silhouette Coefficient of 5 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid5))
print("Silhouette Coefficient of 6 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid6))
print("Silhouette Coefficient of 7 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid7))
print("Silhouette Coefficient of 8 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid8))
print("Silhouette Coefficient of 9 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid9))
print("Silhouette Coefficient of 10 clusters: %0.3f"% metrics.silhouette_score(df_scaled_PCA, clusterid10))


# ###  Performance Measure: Calinski-Harabasz Index

# In[180]:


print("Calinski-Harabasz index of 3 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid3))
print("Calinski-Harabasz index of 4 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid4))
print("Calinski-Harabasz index of 5 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid5))
print("Calinski-Harabasz index of 6 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid6))
print("Calinski-Harabasz index of 7 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid7))
print("Calinski-Harabasz index of 8 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid8))
print("Calinski-Harabasz index of 9 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid9))
print("Calinski-Harabasz index of 10 clusters: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, clusterid10))


# <b> From the above metrics, we can observe that 9 cluster solution is good.</b>

# In[181]:


df_scaled["clusterid9"] = clusterid9
cluster_size9 = df_scaled.groupby(['clusterid9']).size() 
print(cluster_size9)
print("")


# ###  <font color='Green'>3.3 Examining Chararcteristics</font>

# In[184]:


values=['wheel-base','length','width']
index =['clusterid9']
aggfunc={'wheel-base': np.mean,
         'length': np.mean,
         'width': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size9
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 2 & 3 have automobile models of highest length while Cluster 1 & 5 have medium length models whereas cluster 0,4,6,7,8 have lowest length automobile models
# 
# It can be observed that clusters with highest length have highest type of wheel base as well (cluster 2 & 3), cluster 1 & 5 have medium wheel base model whereas cluster 0,4,6,7,8 have low type of wheel base
# 
# Cluster 1 & 2 have highest width of the automobile model while cluster 3 & 5 have medium width models wheras cluster 0,4,6,7,8 have lowest width automobile models

# In[185]:


values=['height','bore','stroke']
index =['clusterid9']
aggfunc={'height': np.mean,
         'bore': np.mean,
         'stroke': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size9
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 6 & 3 have the models with highest diameter of each wheel, whereas cluster 8 and 7 have models with lowest diameter of each wheel
# 
# Cluster 2 & 3 have models with highest height, whereas cluster 8 & 1 have lowest height models
# 
# Cluster 2 & 5 have models with more number of phases in engine's cycle than cluster 6 & 3 have models with least number of phases in engine's cycle

# In[187]:


values=['compression-ratio','horsepower','peak-rpm']
index =['clusterid9']
aggfunc={'compression-ratio': np.mean,
         'horsepower': np.mean,
         'peak-rpm': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size9
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 3 & 2 have models with highest volume of cylinder and chamber in the engine whereas cluster 1 & 4 have models with lowest volume of cylinder and chamber in the engine
# 
# It can be observed that Cluster 1 & 3 have models with highest power of engine despite having low volume of chamber of the engine, also cluster 7 having haighest volume of cylinder and chamber in the engine have lowest power engine
# 
# Also Cluster 4 have lowest volume of chamber in the engine but have highest revolutiuons per minute

# In[188]:


values=['city-mpg','highway-mpg','price']
index =['clusterid9']
aggfunc={'city-mpg': np.mean,
         'highway-mpg': np.mean,
         'price': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size9
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 7 & 8 have lowest price and highest score in number of car in an average city and on a highway whereas cluster 1 & 2 have the models with the most expensive cars and the lowest in number in an average city and on a highway
# 
# Inference : The more expensive car is the least is the number of car in an average city or on a highway
# 
# 
# 

# ### CONCLUSION

# <ul>
#     <li>Cluster 0: </li>
#     <li>Cluster 1: </li>
#     <li>Cluster 2: </li>
#     <li>Cluster 3: </li>
#     <li>Cluster 4: </li>
#     <li>Cluster 5: </li>
#     <li>Cluster 6: </li>
#     <li>Cluster 7: </li>
#     <li>Cluster 8: </li>
# 
#     
# </ul>    

# In[191]:


from pyclustering.cluster.kmeans import kmeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pyclustering.utils.metric import distance_metric
from pyclustering.utils.metric import type_metric
from pyclustering.utils import calculate_distance_matrix

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample


# ## <font color='Green'>4.1 Executing Clustering</font>

# ## <font color='Green'>Prepare initial centers using K-Means++ method</font>

# In[192]:


initial_centers = kmeans_plusplus_initializer(df_scaled_PCA, 4).initialize()


# ## <font color='Green'>4.2 create metric that will be used for clustering</font>

# In[194]:


gower_metric = distance_metric(type_metric.GOWER,data=df_scaled_PCA)


# ## <font color='Green'>4.3 Create instance of K-Means using specific distance metric</font>

# In[195]:


kmeans_instance = kmeans(df_scaled_PCA, initial_centers, metric=gower_metric)


# ## <font color='Green'>4.4 Run cluster analysis and obtain results</font>

# In[196]:


kmeans_instance.process()
clusters = kmeans_instance.get_clusters()


# ## <font color='Green'>4.5 Show Allocated Clusters</font>

# In[197]:


print(clusters)


# ## <font color='Green'>4.6 Adding the cluster lables to dataframe df for analysis</font>

# In[198]:


df=df_scaled
df['clusterid'] = ''
for x in df.index.values:
    if x in clusters[0]:
       df['clusterid'][x] = 0
    elif x in clusters[1]:
       df['clusterid'][x] = 1
    elif x in clusters[2]:
       df['clusterid'][x] = 2
    else: 
       df['clusterid'][x] = 3

cluster_size = df.groupby(['clusterid']).size() 
print(cluster_size)


# ## <font color='Green'>4.7 Performance Measures: Silhouette Score</font>

# ### Performance Measures: Silhouette Score

# In[200]:


print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(df_scaled_PCA, df['clusterid']))
# Silhouette score between -1 and 1


# ### Performance Measure: Calinski-Harabasz

# In[201]:


print("Calinski-Harabasz index: %0.3f"% metrics.calinski_harabasz_score(df_scaled_PCA, df['clusterid']))


# ## <font color='Green'>4.8 Examining Chararcteristics</font>

# In[202]:


values=['wheel-base','length','width']
index =['clusterid']
aggfunc={'wheel-base': np.mean,
         'length': np.mean,
         'width': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size
result = result.round(2)
result


# <b> Insights: </b>
# 
# Classification on the basis of cluster size, cluster 0 is the larger cluster containing 60 data points followed by cluster 1 with 47 data points then cluster 3 with 29 data points and the last is cluster 2 with 23 data points
# 
# We can notice that cluster 2 & 3 has lowest length, width & wheel base of the automobile model whereas cluster 1 has all the three features higher than other two.

# In[203]:


values=['height','bore','stroke']
index =['clusterid']
aggfunc={'height': np.mean,
         'bore': np.mean,
         'stroke': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 1 has models with highest diameter of each wheel in the automobile (bore), with highest height but low number of phases in engine's cycle(stroke).
# 
# Cluster 2 has models with lowest diameter of each wheel(bore), lowest height of the model but moderate number of phases in engine's cycle(stroke).
# 
# Cluster 0 has models with less height but moderate diameter of each wheel and highest number of phases in engine's cycle of the model.
# 
# Cluster 3 has models with moderate height but lowest diameter of each wheel in the automobile

# In[204]:


values=['compression-ratio','horsepower','peak-rpm']
index =['clusterid']
aggfunc={'compression-ratio': np.mean,
         'horsepower': np.mean,
         'peak-rpm': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 2 has models with highest volume of cylinder and chamber (compression-ratio) in the engine and low revolutions per minute (peak-rpm) but lowest power automobile (horsepower)  
# 
# Cluster 1 has models with moderate volume of cylinder and chamber (compression-ratio) in the engine and low revolutions per minute (peak-rpm) but have highest power automobile (horsepower)  
# 
# Cluster 0 has models with lowest volume of cylinder and chamber in the engine (compression-ratio), low power automobile (horsepower) but moderate revolutions per minute ((peak-rpm)
# 
# Cluster 3 has models with low power automobile (horsepower),low volume of cylinder and chamber of the engine (compression-ratio) with lowest revolutions per minute (peak-rpm)

# In[205]:


values=['city-mpg','highway-mpg','price']
index =['clusterid']
aggfunc={'city-mpg': np.mean,
         'highway-mpg': np.mean,
         'price': np.mean}
result = pd.pivot_table(df_scaled,values=values,
                             index =index,
                             aggfunc=aggfunc,
                             fill_value=0)
result['cluster_size'] = cluster_size
result = result.round(2)
result


# <b> Insights: </b>
# 
# Cluster 2 has cheapest automobile models (price) with highest scoring car in an average city (city-mpg) & highway (highway-mpg)
# 
# Cluster 1 are the most expensive automobile models with lowest scoring car in an average city or highway as it is pretty obviuos that in an average city or highway the number of expensive cars would be less due to affordability of the citizens and road infrastructure
# 
# Cluster 3 has moderate number of cars in average city and highway with moderate pricing
# 
# Cluster 0 has low price automobile model with low score in city & highway and highest number of data points
# 

# ### CONCLUSION

# <ul>
#     <li>Cluster 0: </li>
#     <li>Cluster 1: </li>
#     <li>Cluster 2: </li>
#     <li>Cluster 3: </li>
# 
# </ul>    

# In[ ]:




