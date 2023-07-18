import pickle#  note: to run write stramlit run slite.p
import streamlit as st
import requests
import pandas as pd


def recommend(movie):               #Function to recommend movies
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index] 
    mlist=sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:9]

    top_fivemovies_bucket=[]
    for i in mlist:
        top_fivemovies_bucket.append(movies.iloc[i[0]].title)

    return top_fivemovies_bucket    





movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies=pd.DataFrame(movies_dict)
similarity=pickle.load(open('similarity.pkl','rb'))
movies_list=movies['title'].values  

st.title("Movie Recommender system") 
choosen_movie_name=st.selectbox(
 'Select Movies',
 movies_list   
)


if st.button('Recommand'):
    recommendations=recommend(choosen_movie_name)

    for i in recommendations:
        st.write(i)

