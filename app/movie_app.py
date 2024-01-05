import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scrape_ratings
from io import StringIO
import predictor

# Initialize the MovieAnalysis predictor
movie_predictor = predictor.MovieAnalysis()

if 'trained' not in st.session_state:
    st.session_state.trained = False


def load_data():
    data = pd.read_csv(uploaded)
    if 'Unnamed: 0' in data.columns:
        data=data.drop(columns=['Unnamed: 0'], index=1)
    return data


@st.cache_data(show_spinner=False)
def scrape_data(user_name):
    with st.spinner('Scraping your Letterboxd profile...'):
        data = scrape_ratings.scrape(user_name)
        data = data.to_csv(index=False)
        st.sidebar.write('**Scraping Completed!**')
        data = pd.read_csv(StringIO(data))
    st.success('Done!')
    return data


def show_data(data):
    if type(data) == pd.DataFrame:
        if  all(col in ['Title', 'Rating', 'URL', 'Cast', 'Crew', 'Studios', 'Genres'] for col in data.columns if col != 'Duration'):
            st.write('Data is in valid format for training.')
            st.dataframe(data.head())
            return True
        else:
            st.write("Your data is in incorrect format. Make sure you have **'Title', 'Rating', 'URL', 'Cast', 'Crew', 'Studios', 'Genres'** columns.")
            return False


def train(data):
    with st.spinner('Training the model...'):
        model = movie_predictor.train(data)
    return model
    

st.write('''
# Movie Rating Prediction App
This app predicts **your movie ratings** by using your **Letterboxd** profile.

1. **Upload** your pre-scraped dataset, or **enter your Letterboxd username** to scrape the data. Once the scraping process is completed, you can **download the file** for future use.

2. **Train the predictor model** with the dataset.

3. **Download the predictor model** if desired.
---
''')

    
st.sidebar.header('''User Input Features''')
uploaded = st.sidebar.file_uploader('Upload your scraped Letterboxd csv file')

if uploaded:
    data = load_data()
    st.sidebar.text('To scrape new data, \nclear the file selection.')  
else:
    user_name = st.sidebar.text_input("or \n\n Enter **Letterboxd username 👇**")
    if user_name:
        data = scrape_data(user_name)
        csv_data = data.to_csv(index=False, encoding='utf-8')
        download_button = st.sidebar.download_button('Download your scraped Letterboxd file', csv_data, f'{user_name}.csv')

if 'data' not in vars():
    st.session_state.trained = False
else:   
    proceed = show_data(data)
    if proceed:
        col1, col2 = st.columns(2)
        with col1:
            train_button = st.button('Train Predictor Model')
            st.session_state.trained = train_button
        if train_button:
            st.session_state.model = train(data)
            st.session_state.model_filename = f'{st.session_state.model}.pkl'
            

        if st.session_state.trained:
            with col2:
                with open(st.session_state.model_filename, 'wb') as model_file:
                    pickle.dump(st.session_state.model, model_file)
                download_button = st.download_button('Download Model', st.session_state.model_filename, f'{st.session_state.model_filename}')