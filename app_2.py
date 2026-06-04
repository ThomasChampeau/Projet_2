import streamlit as st
import pandas as pd
import numpy as np
import re
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
    
        sample = pd.read_csv('BDD_movies_clean.zip', compression="zip", encoding ='utf-8', na_values=["\\N"])
        sample['runtime'] = pd.to_numeric(sample['runtime'], errors="coerce").fillna(0).astype(int)
        sample['vote_average'] = pd.to_numeric(sample['vote_average'], errors="coerce").fillna(0.0)
        return sample


    #on va intégrer du code html dans streamlit pour avoir un meilleur visuel
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">RECOMMANDATIONS DE FILMS</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Faites votre choix !</p>
    """
    #Pour que le html soit integré on va le mettre dans un markdown qui sert a mettre en forme
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Function to load custom CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    local_css("design.css")
    
    with st.sidebar:
        st.button("Click Me", key="green_button")

    
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
    st.write("Avec :", sample[sample["original_title"] == chk_id]["actors"].iloc[0])       
    st.write(sample[sample["original_title"] == chk_id]["runtime"].iloc[0], "minutes")
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text
     
    #on va créer une checkbox
    chk_voisins2 = st.checkbox("Afficher les films similaires (cocher la case)")   
    
    #le if sert a mettre la condition "si la checkbox est selectionner"
    if chk_voisins2:
        top = 10
        vectorizer = TfidfVectorizer(stop_words= "english")
        tfidf_matrix = vectorizer.fit_transform(sample['clean_overview'])
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
                            st.markdown(sample.iloc[movie_idx]["original_title"])
                            st.image(sample.iloc[movie_idx]["poster_ok"], width=200)
                            st.write('⭐', str(sample.iloc[movie_idx]['vote_average']))
                            st.write("Date de sortie :", sample.iloc[movie_idx]["release_date"])
                            #st.write(GoogleTranslator(source="auto", target="fr").translate(sample.iloc[movie_idx]["overview"]))
                            #st.write("Avec :", sample.iloc[movie_idx]["actors"])
                            st.write(sample.iloc[movie_idx]["runtime"], "minutes")
                            st.write("Plus d'infos sur [IMDb.com](https://www.imdb.com/fr/) ou [TMDB.com](https://www.themoviedb.org)")
                            st.write("---")        


if __name__ == '__main__':
    main()
    