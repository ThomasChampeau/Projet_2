import streamlit as st
import pandas as pd
import numpy as np
import re
import ast
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zipfile import ZipFile
from deep_translator import GoogleTranslator

def main() :

    @st.cache_resource
    def load_data():
    
        sample = pd.read_csv('BDD_movies_clean.csv', encoding ='utf-8', na_values=["\\N"])
        sample['runtime'] = pd.to_numeric(sample['runtime'], errors="coerce").fillna(0).astype(int)
        sample['vote_average'] = pd.to_numeric(sample['vote_average'], errors="coerce").fillna(0.0)
        return sample


    
    # Function to load custom CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    
    sample = load_data()
    #sidebar.header va permettre de mettre une bordure sur le coté
    chk_id = st.selectbox("Selectionnez un film que vous aimez", options= sample["original_title"], index= 120, placeholder="Sectionnez un film", key="selection")
    
    #On creée la fonction de traduction pour les résumés des films
    def trad(description):
        txt = GoogleTranslator(source="auto", target="fr").translate(description) 
        st.write(txt)
    
       
    #On va affiche les infos du film selectionné
    def infos_film(title):
        st.header(title)
        st.image(sample[sample["original_title"] == title]["poster_ok"].iloc[0], width=200)
        st.write("Date de sortie :", sample[sample["original_title"] == title]["release_date"].iloc[0])
        st.write(GoogleTranslator(source="auto", target="fr").translate(sample[sample["original_title"] == title]["overview"].iloc[0]))
        st.write("Avec :", sample[sample["original_title"] == title]["primaryName"].iloc[0])       
        st.write(sample[sample["original_title"] == title]["runtime"].iloc[0], "minutes")
    
    st.markdown(f"### {chk_id}")
    st.image(sample[sample["original_title"] == chk_id]["poster_ok"].iloc[0], width=200)
    st.write("Date de sortie :", sample[sample["original_title"] == chk_id]["release_date"].iloc[0])
    st.write(GoogleTranslator(source="auto", target="fr").translate(sample[sample["original_title"] == chk_id]["overview"].iloc[0]))
    actors = sample[sample['original_title'] ==chk_id]["actors"].iloc[0]
    actors = ast.literal_eval(actors)
    st.write("Avec :", ", ".join(actors[:6]))       
    st.write(str(sample[sample["original_title"] == chk_id]["runtime"].iloc[0]), "minutes")
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text
     
    #on va créer une checkbox
    chk_voisins2 = st.checkbox("Afficher les films similaires (cocher la case)")   
    
    #le if sert a mettre la condition "si la checkbox est selectionner"
    if chk_voisins2:
        top = 15
        vectorizer = TfidfVectorizer(stop_words= "english")
        tfidf_matrix = vectorizer.fit_transform(sample['NLP'])
        if chk_id not in sample['original_title'].values:
            st.write("Film non trouvé !")
        idx_film = sample[sample['original_title'] == chk_id].index[0]
        titre = clean_text(chk_id)
        titre_vector = vectorizer.transform([chk_id])
        similarities = cosine_similarity(titre_vector, tfidf_matrix)
        similarity_scores = similarities.flatten()
        idxs_sorted = similarity_scores.argsort()[::-1][:top]
        idxs_excluded =  idxs_sorted[idxs_sorted != idx_film]
        
        cols = st.columns(5)
        
        movies_to_show = idxs_excluded[:11]

        for row in range(2):
            cols = st.columns(5)

            for col in range(5):
                idx = row * 5 + col

                if idx < len(movies_to_show):
                    movie_idx = movies_to_show[idx]

                    with cols[col]:

                        poster = sample.iloc[movie_idx]["poster_ok"]

                        if poster == "https://media.themoviedb.org/t/p/w220_and_h330_facehttps://i.postimg.cc/W1wtX9W5/Affiche-non-dispo-(2).jpg":
                            poster = "https://i.postimg.cc/W1wtX9W5/Affiche-non-dispo-(3).jpg"

                        st.markdown(f"""
                        <div style="width:150px; height:225px; display:flex; align-items:center; justify-content:center;">
                            <img src="{poster}" style="max-width:140px; max-height:225px; object-fit:contain;">
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write(sample.iloc[movie_idx]["original_title"])
                        st.write("⭐", sample.iloc[movie_idx]["vote_average"])
                        

                       
                            #st.write("Date de sortie :", sample.iloc[movie_idx]["release_date"])
                            #st.write(GoogleTranslator(source="auto", target="fr").translate(sample.iloc[movie_idx]["overview"]))
                            #actors = sample.iloc[movie_idx]["actors"]
                            #actors = ast.literal_eval(actors)
                            #st.write("Avec :", ", ".join(actors[:6]))
                            #st.write("Avec :", sample.iloc[movie_idx]["actors"])
                            #st.write(str(sample.iloc[movie_idx]["runtime"]), "minutes")
                            #st.write("Plus d'infos sur [IMDb.com](https://www.imdb.com/fr/) ou [TMDB.com](https://www.themoviedb.org)")      


if __name__ == '__main__':
    main()
    