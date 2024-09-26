import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def recommend_movies_by_title(movie_title, df):
            if movie_title not in df['Title'].values:
                return "Başlık bulunamadı."
            
            idx = df[df['Title'] == movie_title].index[0]

            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df['Genres'].astype(str))
            similarity_matrix = cosine_similarity(tfidf_matrix)

            similarity_scores = list(enumerate(similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            filtered_scores = [score for score in similarity_scores if df.iloc[score[0]]['Title'] != movie_title]

            top_movies = filtered_scores[:5]

            movie_indices = [i[0] for i in top_movies]
            recommended_movies = df[['Title']].iloc[movie_indices]
        
            return recommended_movies


df = pd.read_csv('data/netflix4app.csv')

st.title("Netflix Recommendation System")

movie_title = st.selectbox("Select Movie Title", df['Title'].values)

if not df.empty:
    if movie_title:
        recommended_movies = recommend_movies_by_title(movie_title, df)

        st.subheader(f"Selected Movie: {movie_title}")
        st.write("Description:", df[df['Title'] == movie_title]['Description'].values[0])

        if isinstance(recommended_movies, pd.DataFrame):
            st.write("Recommmended Movies:")
            st.dataframe(recommended_movies)
        else:
            st.write(recommended_movies) 
