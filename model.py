#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.feature_extraction.text import TfidfVectorizer


new_movies = pd.read_csv('new_movies.csv')

#Feature extraction
tfidf=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2))
feature_matrix=tfidf.fit_transform(new_movies.clean.values.astype('U'))

#compute using the linear kernel,different metrics can also be used
linear_kernel=pairwise_kernels(feature_matrix,feature_matrix,metric='linear')

#Recommendation
def recommendation(movie,sim_matrix=linear_kernel):
    movie_indces=pd.Series(new_movies.index,index=new_movies.original_title).drop_duplicates() 
    movie_index=movie_indces[movie]
    sim_scores=list(enumerate(sim_matrix[movie_index]))
    sim_movies=sorted(sim_scores,key=lambda x:x[1],reverse=True)[1:11]
    index_movies=[i[0] for i in sim_movies]
    return new_movies['original_title'].iloc[index_movies]
