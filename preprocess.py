from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer  # root form of words
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import numpy as np


credit=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')

#merge credits and movies
credit=credit.rename(columns={'movie_id':'id'})
movies=movies.merge(credit,on='id')
movies=movies.drop(columns=['title_x','title_y','production_companies','spoken_languages','production_countries','homepage','tagline'])
movies.head()

from ast import literal_eval
list_eval=['cast','crew','keywords','genres']
for i in list_eval:
    movies[i]=movies[i].apply(literal_eval)
movies.head()

new_movies=movies.drop(columns=['budget', 'genres', 'keywords','runtime', 'status', 'vote_count', 'cast', 'crew','revenue','popularity'])


#Preprocessing
sno=SnowballStemmer('english')
def get_director(crew):
    for i in crew:
        if i['job']=='Director':
            return i['name']
        None

def get_composer(crew):
    for i in crew:
        if i['job']=='Original Music Composer':
            return i['name']
         np.nan

def genres(genre):
    if isinstance(genre,list):
        genres=' '.join([j['name'] for j in genre])
        return genres
    return ' '
def keywords(keywords):
    if isinstance(keywords,list): 
        keyword=[j['name'] for j in keywords]
        return ' '.join(keyword)
    return ' '
def merge(new_movies):
    return (new_movies['keywords']+' '+new_movies['genres']+' '+new_movies['director'])
def preprocessing(final_array):  
    for sentence in final_array:
        clean_sent=' '
        for word in sentence.split():
            if word.isalpha():
                clean_sent=clean_sent+ ' '+(sno.stem(word))
            continue
        clean_sentence.append(clean_sent)
    return clean_sentence

clean_sentence=[]
new_movies['director']=movies['crew'].apply(get_director)
#new_movies['composer']=movies['crew'].apply(get_composer)
new_movies['genres']=movies['genres'].apply(genres)
new_movies['keywords']=movies['keywords'].apply(keywords)

new_movies['final']=merge(new_movies)
new_movies['final']=new_movies.final.fillna("")
new_movies['clean']=preprocessing(new_movies.final.values)


new_movies.to_csv('new_movies.csv',index=False)




