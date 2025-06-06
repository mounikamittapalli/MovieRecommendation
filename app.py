

import streamlit as st
import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Streamlit UI
movies=pickle.load(open("movies_list.pkl",'rb'))
similarity=pickle.load(open("similarity.pkl",'rb'))
movie_list = movies['original_title'].values


def recommand(movie):
    index=movies[movies['original_title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), key=lambda vector: vector[1], reverse=True)
    recommend_movie=[]
# Print top 5 results (after the first one which is usually the same item)
    for i in distance[0:5]:
        recommend_movie.append(movies.iloc[i[0]].original_title)
    return recommend_movie

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Show Recommendations"):
    recommendations = recommand(movie_list)
    st.subheader("Top 5 similar movies:")
    for movie in recommendations:
        st.write(f"🎬 {movie}")





