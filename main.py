import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

data1 = pd.read_csv('ratings.csv')
data2 = pd.read_csv('movies.csv')
data1_set= data1[['userId', 'movieId', 'rating']]
data2_set = data2[['movieId', 'title', 'genres']]
result = pd.merge(data1, data2, on='movieId')
pivot_table = result.pivot_table(index='userId', columns='movieId', values='rating')

user_similarity = cosine_similarity(pivot_table.fillna(0))
train_data, test_data = train_test_split(result, test_size=0.2)
item_similarity = cosine_similarity(train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0))
item_similarity_dict = {}
for i in range(len(item_similarity)):
    item_similarity_dict[i] = item_similarity[i]

def get_recommendations(movieId, item_similarity_dict, num_recommendations=10):
    similarity_hasil = item_similarity_dict[movieId]
    similarity_hasil_sort = sorted(list(enumerate(similarity_hasil)), key=lambda x: x[1], reverse=True)
    similarity_movie_ids = [i[0] for i in similarity_hasil_sort]
    top_n_similar_movie_ids = similarity_movie_ids[1:num_recommendations+1]
    top_n_similar_movie_titles = list(data2_set.loc[data2_set['movieId'].isin(top_n_similar_movie_ids), 'title'])
    top_n_similar_movie_genres = list(data2_set.loc[data2_set['movieId'].isin(top_n_similar_movie_ids), 'genres'])
    top_n_similar_movies = pd.DataFrame({'Title': top_n_similar_movie_titles, 'Genre': top_n_similar_movie_genres})
    return top_n_similar_movies

recommended_movies = get_recommendations(1, item_similarity_dict, 10)
print(recommended_movies)
