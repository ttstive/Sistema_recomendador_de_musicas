import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
from io import BytesIO
import requests
import plotly.express as px
import plotly.graph_objects as go
# Função para carregar dados com cache
@st.cache_data
def load_data():
    dados_gerais = pd.read_csv('https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/Dados_totais.csv')
    dados_generos = pd.read_csv('https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/data_by_genres.csv')
    dados_anos = pd.read_csv('https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/data_by_year.csv')
    
    dados_gerais = dados_gerais.drop(["explicit", "key", "mode"], axis=1)
    dados_anos.reset_index()
    dados_generos1 = dados_generos.drop('genres', axis=1)
    
    return dados_gerais, dados_generos, dados_anos, dados_generos1

# Função para preparar os dados e executar PCA e KMeans
@st.cache_data
def preprocess_data(dados_gerais, dados_generos1):
    SEED = 1224
    np.random.seed(SEED)
    
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])
    genre_training_pca = pca_pipeline.fit_transform(dados_generos1)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_training_pca)
    
    kmeans_pca = KMeans(n_clusters=5, verbose=True, random_state=SEED)
    kmeans_pca.fit(projection)
    dados_generos['cluster_pca'] = kmeans_pca.predict(projection)
    projection['cluster_pca'] = kmeans_pca.predict(projection)
    
    ohe = OneHotEncoder(dtype=int)
    colunas_ohe = ohe.fit_transform(dados_gerais[['artists']]).toarray()
    dados2 = dados_gerais.drop('artists', axis=1)
    dados_musicas_dummies = pd.concat([dados2, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['artists']))], axis=1)
    
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])
    music_embedding_pca = pca_pipeline.fit_transform(dados_musicas_dummies.drop(['id', 'artists_song', 'name'], axis=1))
    projection_m = pd.DataFrame(data=music_embedding_pca)
    
    kmeans_pca_pipeline = KMeans(n_clusters=50, verbose=False, random_state=SEED)
    kmeans_pca_pipeline.fit(projection_m)
    dados_gerais['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
    projection_m['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
    
    projection_m['artists'] = dados_gerais['artists']
    projection_m['song'] = dados_gerais['artists_song']
    
    return dados_gerais, dados_generos, projection, projection_m

# Carregar dados
dados_gerais, dados_generos, dados_anos, dados_generos1 = load_data()
dados_gerais, dados_generos, projection, projection_m = preprocess_data(dados_gerais, dados_generos1)

# Configurar Spotify API
scope = "user-library-read playlist-modify-private"
OAuth = SpotifyOAuth(scope=scope,
                     client_id='195ecc33d37b409cacdb0db8da73d596',
                     client_secret='177ed513265949b6af33b5299a3f529e',
                     redirect_uri='http://localhost:8888/callback')

client_credentials_manager = SpotifyClientCredentials(client_id='195ecc33d37b409cacdb0db8da73d596',
                                                      client_secret='177ed513265949b6af33b5299a3f529e')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Função para recomendar músicas
@st.cache_data
def recommend_music(nome_musica):
    # Filtra as músicas pelo nome
    filtered = projection_m[projection_m['song'] == nome_musica]
    
    if not filtered.empty:
        # Obtém o cluster da primeira música encontrada
        cluster = filtered.iloc[0]['cluster_pca']
        
        # Filtra as músicas pelo cluster encontrado
        musicas_recomendadas = projection_m[projection_m['cluster_pca'] == cluster][[0, 1, 'song']]
        
        # Calcula as distâncias euclidianas
        x_musica = filtered.iloc[0][0]
        y_musica = filtered.iloc[0][1]
        distancias = euclidean_distances(musicas_recomendadas[[0, 1]], [[x_musica, y_musica]])
        
        # Adiciona as colunas de id e distâncias
        musicas_recomendadas['id'] = dados_gerais['id']
        musicas_recomendadas['distancias'] = distancias
        
        # Ordena as músicas recomendadas por distância
        recomendada = musicas_recomendadas.sort_values('distancias').head(10)
        
        # Obtém as informações das músicas recomendadas
        playlist_id = recomendada['id']
        url = []
        name = []
        for i in playlist_id:
            track = sp.track(i)
            url.append(track["album"]["images"][1]["url"])
            name.append(track["name"])
        
        return name, url
    else:
        return [], []

# Função para exibir imagens das músicas recomendadas
def display_images(name, url):
      if not name or not url:
        st.write("Nenhuma recomendação encontrada.")
      else:

        # Cria um seletor para navegar entre as imagens
        selected_index = st.slider("Escolha uma música para ver a imagem:", 0, len(name) - 1, 0)

        # Exibe a imagem selecionada
        st.write(name[selected_index])
        response = requests.get(url[selected_index])
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=name[selected_index], use_column_width=True)
    

# Aplicação Streamlit
# Caminho para a imagem sem fundo
logo = "https://styles.redditmedia.com/t5_2h522r/styles/communityIcon_nnizf8q0jbl41.png"

# Título da aplicação
st.title('Recomendador de Músicas')

# HTML e CSS para centralizar a imagem na sidebar
st.sidebar.markdown(
    f"""
    <style>
    .sidebar .left {{
        display: flex;
        justify-content: flex-start;
        align-items: center;
    }}
    .sidebar .left img {{
        width: 50px;  /* Ajuste o tamanho conforme necessário */
    }}
    </style>
    <div class="left">
        <img src="{logo}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)
nome_musica = st.text_input('Digite o nome de uma música:')

if nome_musica:
    st.write(f'Recomendações para: {nome_musica}')
    name, url = recommend_music(nome_musica)
    display_images(name, url)



##mostrar dois graficos que simbolizem a distancia euclidiana entre as musicas sugeridas
    

def mostrar_graficos():
    st.write("Gráfico de dispersão dos gêneros:")
    st.write("Cada cor representa um cluster")
    
    # Cria o seletor de tipo de gráfico
    tipo_grafico = st.selectbox("Selecione o tipo de gráfico", ['Gráfico de dispersão', 'Gráfico de Correlação'])
    
    if tipo_grafico == 'Gráfico de dispersão':
        # Gráfico de dispersão
        fig1 = px.scatter(projection, x='x', y='y', color_discrete_sequence=['Green'])
        st.plotly_chart(fig1)
    elif tipo_grafico == 'Gráfico de Correlação':
        # Gráfico de correlação
        fig2 = px.imshow(projection.corr())  # Assumindo que projection é um DataFrame com dados numéricos
        st.plotly_chart(fig2)
                         




 
## criar função para listar musicas na lateral da tela e algumas opções de filtros


def mostrar_musicas():
    st.filtrar_musicas = st.sidebar.selectbox('Buscar inspirações por:', ['Artista', 'Nome da música'])

    if st.filtrar_musicas == 'Nome da música':
        # Filtragem por nome da música
        musica = st.sidebar.text_input('Digite o nome da música:')
        df = dados_gerais[dados_gerais['artists_song'].str.contains(musica, case=False)][['artists', 'artists_song']]

        st.subheader('Músicas encontradas, selecione alguma e cole no seletor de cima, para ver recomendações:')

    else:
        # Filtragem por nome do artista
        artista = st.sidebar.text_input('Digite o nome do artista:')
        df = dados_gerais[dados_gerais['artists'].str.contains(artista, case=False)][['artists_song', 'artists']]

        st.subheader('Músicas encontradas, selecione alguma e cole no seletor de cima, para ver recomendações:')

    # Cria a tabela com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='Black',
                    font=dict(color='white'),
                    font_size=20,
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='Black',
                   font=dict(color='white'),
                   font_size=12,
                   align='left'))
    ])
    # Exibe a tabela com Streamlit
    st.write(fig)

mostrar_graficos()
mostrar_musicas()

st.write("\n")
st.write("\n")
st.write("\n")

st.subheader("Desenvolvido por: Estevão Lins Maia")
