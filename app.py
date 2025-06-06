import streamlit as st
import pickle
import pandas as pd

# Load pickled data
movies = pickle.load(open("movieslist.pkl", "rb"))
top_sim_indices = pickle.load(open("top_sim_indices.pkl", "rb"))

# List of movie titles for dropdown
movie_list = movies['original_title'].values

def recommend(movie_title):
    index = movies[movies['original_title'] == movie_title].index[0]
    recommended = [movies.iloc[i].original_title for i in top_sim_indices[index][:5]]
    return recommended

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 similar movies:")
    for movie in recommendations:
        st.write(f"🎬 {movie}")
