#!/usr/bin/env python
# coding: utf-8

# In[11]:


#obtaining data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import missingno as ms #you may need to explicitly install this
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 2000)

echonest = pd.read_csv('datasets/raw/echonest.csv')
features = pd.read_csv('datasets/raw/features.csv')
genres = pd.read_csv('datasets/raw/genres.csv')
tracks = pd.read_csv('datasets/raw/tracks.csv')
#working with echonest dataset
#analyzing data
echonest.info()
features.info()
genres.info()
tracks.info()
echonest.head(10)
#feature engg
ms.matrix(echonest)
echonest.drop(['echonest.8', 'echonest.9', 'echonest.15', 'echonest.16', 'echonest.17', 'echonest.18', 'echonest.19'], axis=1, inplace=True)
echonest.tail(15)
ms.matrix(echonest.iloc[:, 0:15])
echonest.drop(['echonest.10', 'echonest.11', 'echonest.12'], axis=1, inplace=True)
ms.matrix(echonest)
echonest.head(10)
echonest.drop(0, axis=0, inplace=True)
echonest.iloc[0, 0]
echonest.iloc[1, 0]
echonest.iloc[0, 0] = echonest.iloc[1, 0]
echonest.head()
echonest.drop(2, axis=0, inplace=True)
echonest.columns = echonest.iloc[0]

echonest.head()
echonest.drop(1, axis=0, inplace=True)
echonest.head()
echonest.reset_index(inplace=True)
echonest.drop('index', inplace=True, axis=1)
echonest.head()
type(echonest['acousticness'][0])
def convert_to_float(df, columns):
    for i in columns:
        df[i] = df[i].astype('float')
    return df
echonest = convert_to_float(echonest, set(echonest.columns) - set(['track_id', 'artist_name', 'release']))
echonest.head()
echonest.info()
#Working with 'Features' dataset
#Analysing Data
features.info()
features.head(10)
ms.matrix(features.iloc[:, 21:40])
#Feature Engineering
features.iloc[0,0] = features.iloc[2, 0]
features.head(3)
features.drop(2, inplace=True)
len(features.columns)
len(features.iloc[0])
def combine_two_rows(df):
    columns = list(df.columns)
    for i in range(0, 519):
        columns[i] = columns[i] + " " + df.iloc[0, i]
    return columns

features.columns = combine_two_rows(features)
features.drop([0, 1], inplace=True)
features.reset_index(inplace=True)
features.drop('index', axis=1, inplace=True)
features.head()
features = features.astype(dtype='float')
features['feature track_id'] = features['feature track_id'].astype('int')
ms.matrix(features)

features.head(3)
#Working with 'Tracks' dataset
#Analysing Data
tracks.info()
tracks.head()
tracks.iloc[0,0] = tracks.iloc[1, 0]
tracks.drop(1, axis=0, inplace=True)
tracks.head()
#Feature Engineering
len(tracks.columns)
def combine_one_row(df):
    columns = list(df.columns)
    for i in range(0, 53):
        if i == 0:
            columns[i] = df.iloc[0, i]
        else:
            columns[i] = columns[i] + " " + df.iloc[0, i]
    return columns

tracks.columns = combine_one_row(tracks)
tracks.drop(0, inplace=True)
tracks.reset_index(inplace=True)
tracks.drop(['index'], axis=1, inplace=True)
ms.matrix(tracks.iloc[0: 10])
tracks.head()
tracks['track.7 genre_top'].value_counts()
track_title = pd.DataFrame(tracks['track.19 title'])
track_title['track_id'] = tracks['track_id']
track_title.head()
track_title.tail()
track_title.shape
tracks.drop(['album comments','album.1 date_created', 
             'album.2 date_released', 'album.11 tracks', 
             'album.9 tags', 'album.8 producer', 'album.3 engineer', 'album.6 information',
             'artist active_year_begin', 'artist.1 active_year_end', 'artist.2 associated_labels',
             'artist.3 bio','artist.4 comments','artist.5 date_created', 'artist.7 id',
             'artist.8 latitude','artist.9 location','artist.10 longitude', 'artist.11 members',
             'artist.13 related_projects', 'artist.14 tags','artist.15 website','artist.16 wikipedia_page',
             'set.1 subset', 'track.1 comments', 'track.2 composer', 'track.3 date_created', 'track.4 date_recorded',
             'track.10 information', 'track.13 license', 'track.15 lyricist', 'track.17 publisher', 'track.18 tags',
             'track.19 title'], axis=1, inplace=True)
tracks.info()
ms.matrix(tracks)
tracks['album.12 type'].value_counts()
tracks['album.10 title'].value_counts()
tracks['album.10 title'].fillna(method='ffill', inplace=True)
tracks.drop(['track.12 language_code', 'album.12 type'], axis=1, inplace=True)
tracks.drop('track.9 genres_all', axis=1, inplace=True)
ms.matrix(tracks)
tracks['track.8 genres'].unique()
genres.info()
type(tracks['track.7 genre_top'].iloc[27])
def getList(cd):
    return cd[1:-1].split(',')
for i in range(0, 106574):
    if type(tracks['track.7 genre_top'][i]) == float:
        genre_list = getList(str(tracks['track.8 genres'][i]))
        count = len(genre_list)
        title = ""
        for j in range(0, count):
            title = title + str(genres['title'][j]) + str('|')
        tracks['track.7 genre_top'][i] = title
#Working with 'Genre' dataset
#Analysing Data
genres.info()
ms.matrix(genres)
genres.head()
#Feature Engineering
#Nothing to engineer!

#Combining all datasets into a single entity
#Analysing Data
echonest.info()
tracks.info()
tracks.head()
echonest.head()
genres.info()
features.info()
#Feature Engineering
features.columns = ['track_id'] + list(features.columns[1:])
features.head()
type(echonest['track_id'].iloc[0])
echonest['track_id'] = echonest['track_id'].astype('int')
tracks['track_id'] = tracks['track_id'].astype('int')
features.sort_values(by='track_id', inplace=True)
tracks.sort_values(by='track_id', inplace=True)
echonest.sort_values(by='track_id', inplace=True)
features.head()
tracks.head()
count = 0
for i in range(0, 106574):
    if features['track_id'][i] == tracks['track_id'][i]:
        count += 1
    else:
        print(features['track_id'][i], tracks['track_id'][i])

final = pd.concat([features, tracks.drop('track_id', axis=1)], axis=1)
final.shape
final.head()
echonest.tail(3)
echonest.drop(['artist_name', 'release'], axis=1, inplace=True)
tracks.tail(3)
features.head(1)
final = echonest.merge(final, on='track_id')
final.shape
ms.matrix(final)
#Analysing Data
final.head()
final.shape
final.info()
final.drop('track.8 genres', axis=1, inplace=True)
final.shape
final.head()
final['track.7 genre_top'].value_counts()
#Feature Engineering
def format_strings(x):
    if '-' in x:
        return ''.join(x.split('-'))
    if x.find('/'):
        return '|'.join(x.split('/'))
    return x

def modifyString(serie, val):
    for i in range(0, val):
        if serie[i] == 'Old-Time / Historic':
            serie[i] = 'OldTime|Historic'
    return serie

final['track.7 genre_top'] = modifyString(final['track.7 genre_top'], 13129)
final['track.7 genre_top'] = final['track.7 genre_top'].apply(format_strings)
final['track.7 genre_top'].value_counts()
final.head()
metadata = pd.DataFrame()
metadata['track_id'] = final['track_id']
metadata.shape
track_title.shape
track_title = track_title.set_index('track_id')
track_title.head()
track_title.index = [int(i) for i in track_title.index]
track_title.head()
metadata.head()
metadata['album_title'] = final['album.10 title']
metadata['artist_name'] = final['artist.12 name']
metadata['genre'] = final['track.7 genre_top']
metadata = metadata.set_index('track_id')
metadata.tail()
metadata.head()
metadata['track_title'] = track_title.loc[metadata.index]['track.19 title']
metadata.tail()
metadata.head()
len(metadata[metadata['genre'].isnull()])
final.drop('album.10 title', axis=1, inplace=True)
final.head()
final.info()
final.drop('artist.12 name', axis=1, inplace=True)
final.info()
final.head()
k = final # Restore point # Removed Label Encoding
final.head()
final.drop('set split', axis=1, inplace=True)
final.info()
final.info()
genres['title'].count()
genre_dummy = pd.DataFrame(data= np.zeros((13129, 163)), columns= list(genres['title'].unique()))
genre_dummy.head()
genre_list = pd.Series(data= genre_dummy.columns)
genre_list = modifyString(genre_list, 163)
genre_list = genre_list.apply(format_strings)
genre_dummy.columns= genre_list
# columns converted successfully
genre_list = list(genre_list)
final
for i in range(0, 13129):
    if '|' in final['track.7 genre_top'][i]:
        divided_list = str(final['track.7 genre_top'][i]).split('|')
        count = len(divided_list)
        for j in range(0, count):
            if divided_list[j] in genre_list:
                location = genre_list.index(divided_list[j])
                genre_dummy.iloc[i, location] = 1
    else:
        location = genre_list.index(final['track.7 genre_top'][i])
        genre_dummy.iloc[i, location] = 1

genre_list.index(final['track.7 genre_top'][0])
final.drop(['track.7 genre_top'], axis= 1, inplace= True)
final = pd.concat([final, genre_dummy], axis= 1)
final.head()
#Writing final data to .csv files
import os
if not os.path.isdir(os.path.join('datasets','final')):
    os.makedirs(os.path.join('datasets','final'))
    
metadata.to_csv('datasets/final/metadata.csv')
final.to_csv('datasets/final/final.csv')











