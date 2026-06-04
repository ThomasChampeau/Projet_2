import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from zipfile import ZipFile


def main() :

    @st.cache
    def load_data():
    
        z = ZipFile("df_affichage.zip")
        sample = pd.read_csv(z.open('df_affichage.csv'), encoding ='utf-8')
       
        return sample


    #on va intégrer du code html dans streamlit pour avoir un meilleur visuel
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Trop bien ça marche !</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">c'est super bien !</p>
    """
    #Pour que le html soit integré on va le mettre dans un markdown qui sert a mettre en forme
    st.markdown(html_temp, unsafe_allow_html=True)

    sample = load_data()
    title = sample.title
    #sidebar.header va permettre de mettre une bordure sur le coté
    st.sidebar.header("**General Information**")

    #Ici on va créer une liste pour séléctionné notre film
    chk_id = st.sidebar.selectbox("Selectionnez un Film", title)
 
    
       
    #On va affiche le film qui a été selectionné
    st.write("Selection du film :", chk_id)           
            
    
     #on va créer une checkbox
    chk_voisins2 = st.checkbox("Voir les films similaires?")   
    
    #le if sert a mettre la condition "si la checkbox est selectionner"
    if chk_voisins2:

        # Transformation des genres en vecteurs (one-hot encoding)
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(sample["genres"])

        # Modèle KNN
        knn = NearestNeighbors(n_neighbors=10, metric='cosine')
        knn.fit(genre_matrix)

        # Index du film
        idx = sample[sample["title"] == chk_id].index[0]

        # Trouver les voisins
        distances, indices = knn.kneighbors([genre_matrix[idx]], n_neighbors=10)

        # Exclure le film lui-même
        recommendations = []
        for i in indices[0][1:]:
            recommendations.append(sample.iloc[i]["title"])

        st.markdown(recommendations)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
    st.markdown('***')


if __name__ == '__main__':
    main()