import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['id', 'original_title', 'overview', 'genres']]

# Combine 'genres' and 'overview' into 'tags'
movies['genres'] = movies['genres'].fillna('').astype(str)
movies['overview'] = movies['overview'].fillna('').astype(str)
movies['tags'] = movies['genres'] + " " + movies['overview']

# Drop unused columns
new_data = movies.drop(columns=['genres', 'overview'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(new_data['tags'].values.astype('U'))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Instead of saving full cosine_sim, just save top-N similar indices
top_n = 20
top_sim_indices = []

for row in cosine_sim:
    top_indices = np.argsort(row)[::-1][1:top_n+1]
    top_sim_indices.append(top_indices)

# Save to lightweight .pkl files (under 25MB)
pickle.dump(new_data, open("movieslist.pkl", "wb"))
pickle.dump(top_sim_indices, open("top_sim_indices.pkl", "wb"))
