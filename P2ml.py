#!/usr/bin/env python
# coding: utf-8

# In[81]:


#importing prerequisite modules

import pandas as pd
import numpy as np

final = pd.read_csv('datasets/final/final.csv')
metadata = pd.read_csv('datasets/final/metadata.csv')
#We will be using K Means Algorithm fro model selection

from sklearn.cluster import KMeans
from sklearn.utils import shuffle

final = shuffle(final)

X = final.loc[[i for i in range(0, 6000)]]
Y = final.loc[[i for i in range(6000, final.shape[0])]]
X = shuffle(X)
Y = shuffle(Y)

metadata.head()
metadata = metadata.set_index('track_id')

X.drop(['label'], axis= 1, inplace= True)
kmeans = KMeans(n_clusters=6)
Y.head()

def func1(df, algo, flag=0): #fit function
    if flag:
        algo.func1(df)
    else:
         algo.partial_func1(df)          
    df['label'] = algo.labels_
    return (df, algo)

def func2(t, Y): #prediction fn
    y_pred = t[1].func2(Y)
    mode = pd.Series(y_pred).mode()
    return t[0][t[0]['label'] == mode.loc[0]]

def func3(recommendations, meta, Y): #fn for recommendations
    dat = []
    for i in Y['track_id']:
        dat.append(i)
    genre_mode = meta.loc[dat]['genre'].mode()
    artist_mode = meta.loc[dat]['artist_name'].mode()
    return meta[meta['genre'] == genre_mode.iloc[0]], meta[meta['artist_name'] == artist_mode.iloc[0]], meta.loc[recommendations['track_id']]

t = func1(X, kmeans, 1)
recommendations = func2(t, Y)
output = func3(recommendations, metadata, Y)
genre_recommend, artist_name_recommend, mixed_recommend = output[0], output[1], output[2]
genre_recommend.shape
artist_name_recommend.shape
mixed_recommend.shape


genre_recommend.head() # for genre wise recommendations
artist_name_recommend.head() # for artist wise recommendations
mixed_recommend.head() # for mixed recommendations

recommendations.head()
artist_name_recommend['artist_name'].value_counts()
genre_recommend['genre'].value_counts()
genre_recommend['artist_name'].value_counts()

#testing underway
testing = Y.iloc[6:12]['track_id']
testing
ids = testing.loc[testing.index]
songs = metadata.loc[testing.loc[list(testing.index)]]
songs
re = func2(t, Y.iloc[6:12])
output = func3(re, metadata, Y.iloc[6:12])
ge_re, ge_ar, ge_mix = output[0], output[1], output[2]
ge_re.head()
ge_ar.head(10)
ge_mix.head(10)
ge_re.shape
ge_ar.shape
ge_mix.shape
#Model Selection => MiniBatchKMeans, importing it

from sklearn.cluster import MiniBatchKMeans
mini = MiniBatchKMeans(n_clusters = 6)
X.drop('label', axis=1, inplace=True)

# now dividing intital dataset into pieces to demonstrate online learning
part_1, part_2, part_3 = X.iloc[0: 2000], X.iloc[2000:4000], X.iloc[4000:6000]

for i in [part_1, part_2, part_3]:
    t = func1(i, mini)
    mini = t[1]
    i = t[0]

X = pd.concat([part_1, part_2, part_3])
X.columns
X.head(3)
X['label'].value_counts()
recommendations = func2((X, mini), Y)
output = func3(recommendations, metadata, Y)
genre_recommend_mini, artist_name_recommend_mini, mixed_mini = output[0], output[1], output[2]
genre_recommend_mini.shape
artist_name_recommend_mini.shape

genre_recommend_mini.head()# for genre wise recommendations
artist_name_recommend_mini.head() # for artist wise recommendations
mixed_mini.head() #for mixed recommendations
#Model Selection => Birch

from sklearn.cluster import Birch
birch = Birch(n_clusters = 6)
X.drop('label', axis=1, inplace=True)

# will divide the intital dataset into pieces to demonstrate online learning
part_1, part_2, part_3 = X.iloc[0: 2000], X.iloc[2000:4000], X.iloc[4000:6000]

for i in [part_1, part_2, part_3]:
    t = func1(i, birch)
    mini = t[1]
    i = t[0]

X = pd.concat([part_1, part_2, part_3])
X.columns
X.head(3)
X['label'].value_counts()
recommendations = func2((X, birch), Y)
output = func3(recommendations, metadata, Y)
genre_recommend_birch, artist_name_recommend_birch, mixed_birch = output[0], output[1], output[2]
genre_recommend_birch.shape
artist_name_recommend_birch.shape

genre_recommend_birch.head() # for genre wise recommendations
artist_name_recommend_birch.head() # for artist wise recommendations
mixed_birch.head() # for mixed recommendations


# In[ ]:





# In[ ]:





# In[ ]:




