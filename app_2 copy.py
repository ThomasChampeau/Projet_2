import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from zipfile import ZipFile
from deep_translator import GoogleTranslator

def main() :

    @st.cache_data
    def load_data():
    
        sample = pd.read_csv('merge_V6.csv', encoding ='utf-8')
       
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

    
    sample = load_data()
    title = sample.primaryTitle
    #sidebar.header va permettre de mettre une bordure sur le coté
    chk_id = st.selectbox("Selectionnez un film que vous aimez", options= sample["primaryTitle"])
    
    #On creée la fonction de traduction pour les résumés des films
    def trad(description):
        txt = GoogleTranslator(source="auto", target="fr").translate(description) 
        st.write(txt)
    
       
    #On va affiche les infos du film selectionné
    def infos_film(title):
        st.header(title)
        st.image(sample[sample["primaryTitle"] == title]["poster_ok"].iloc[0], width=200)
        st.write("Date de sortie :", sample[sample["primaryTitle"] == title]["release_date"].iloc[0])
        st.write(GoogleTranslator(source="auto", target="fr").translate(sample[sample["primaryTitle"] == title]["overview"].iloc[0]))
        st.write("Avec :", sample[sample["primaryTitle"] == title]["primaryName"].iloc[0])       
        st.write(sample[sample["primaryTitle"] == title]["runtimeMinutes"].iloc[0], "minutes")
    
    st.header(chk_id)
    st.image(sample[sample["primaryTitle"] == chk_id]["poster_ok"].iloc[0], width=200)
    st.write("Date de sortie :", sample[sample["primaryTitle"] == chk_id]["release_date"].iloc[0])
    st.write(GoogleTranslator(source="auto", target="fr").translate(sample[sample["primaryTitle"] == title]["overview"].iloc[0]))
    st.write("Avec :", sample[sample["primaryTitle"] == chk_id]["primaryName"].iloc[0])       
    st.write(sample[sample["primaryTitle"] == chk_id]["runtimeMinutes"].iloc[0], "minutes")
     
    #on va créer une checkbox
    chk_voisins2 = st.checkbox("Afficher les films similaires (cocher la case)")   
    
    #le if sert a mettre la condition "si la checkbox est selectionner"
    if chk_voisins2:

        # Transformation des genres en vecteurs (one-hot encoding)
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(sample["genres_x"])

        # Modèle KNN
        knn = NearestNeighbors(n_neighbors=10, metric='cosine')
        knn.fit(genre_matrix)

        # Index du film
        idx = sample[sample["primaryTitle"] == chk_id].index[0]

        # Trouver les voisins
        distances, indices = knn.kneighbors([genre_matrix[idx]], n_neighbors=10)

        # Exclure le film lui-même
        for i in indices[0][1:]:
            st.header(sample.iloc[i]["primaryTitle"])
            st.image(sample.iloc[i]["poster_ok"], width=200)
            st.write("Date de sortie :", sample.iloc[i]["release_date"])
            st.write(GoogleTranslator(source="auto", target="fr").translate(sample.iloc[i]["overview"]))
            st.write("Avec :", sample.iloc[i]["primaryName"])
            st.write(sample.iloc[i]["runtimeMinutes"], "minutes")
            st.write("Plus d'infos sur [IMDb.com](https://www.imdb.com/fr/) ou [TMDB.com](https://www.themoviedb.org)")
            st.write("---")

        


if __name__ == '__main__':
    main()