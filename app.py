import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load CSV files
rating_df = pd.read_csv("RatingPengunjung.csv")
mapping_wisata_df = pd.read_csv("dataset_tempat_wisata3.csv")
mapping_user_df = pd.read_csv("MappingUser.csv")

# Sidebar for filters
st.sidebar.header("Filter Rekomendasi")

# Dropdown for jenis wisata
jenis_wisata_options = mapping_wisata_df['jenis_wisata'].unique()
selected_jenis_wisata = st.sidebar.selectbox("Pilih Jenis Wisata:", jenis_wisata_options)

# Dropdown for asal daerah
asal_options = mapping_user_df['Asal'].unique().tolist()
asal_options.append("Asal selain daerah tertera")
selected_asal = st.sidebar.selectbox("Pilih Asal Daerah:", asal_options)

# Filter MappingWisata by selected jenis wisata
filtered_wisata = mapping_wisata_df[mapping_wisata_df['jenis_wisata'] == selected_jenis_wisata]

# Function to calculate average rating for each wisata
def calculate_average_rating():
    rating_df['Rating'] = pd.to_numeric(rating_df['Rating'], errors='coerce')
    valid_ratings = rating_df.dropna(subset=['Rating'])
    avg_rating = valid_ratings.groupby('ItemID')['Rating'].mean().reset_index()
    avg_rating.columns = ['ItemID', 'avg_rating']
    return avg_rating

avg_rating_df = calculate_average_rating()

# Merge filtered wisata with average ratings
filtered_wisata = pd.merge(filtered_wisata, avg_rating_df, on='ItemID', how='left')

# Handling Asal Daerah filter
if selected_asal != "Asal selain daerah tertera":
    user_ids = mapping_user_df[mapping_user_df['Asal'] == selected_asal]['UserID']
    filtered_ratings = rating_df[rating_df['UserID'].isin(user_ids)]
    avg_rating_by_asal = filtered_ratings.groupby('ItemID')['Rating'].mean().reset_index()
    avg_rating_by_asal.columns = ['ItemID', 'avg_rating']
    filtered_wisata = pd.merge(filtered_wisata, avg_rating_by_asal, on='ItemID', how='left', suffixes=('', '_by_asal'))
    filtered_wisata['avg_rating'] = filtered_wisata['avg_rating_by_asal'].fillna(filtered_wisata['avg_rating'])
    filtered_wisata.drop(columns=['avg_rating_by_asal'], inplace=True)

# Sort by average rating
filtered_wisata = filtered_wisata.sort_values(by='avg_rating', ascending=False)

# Display recommendations
st.title("Sistem Rekomendasi Tempat Wisata Di Kab. Mojokerto")

st.header(f"Rekomendasi untuk Jenis Wisata '{selected_jenis_wisata}'")
if selected_asal == "Asal selain daerah tertera":
    st.subheader("(Rekomendasi Umum)")
else:
    st.subheader(f"(Rekomendasi untuk Asal Daerah '{selected_asal}')")

if not filtered_wisata.empty:
    for _, row in filtered_wisata.iterrows():
        st.write(f"### {row['Nama_Tempat_Wisata']}")
        st.write(f"Deskripsi: {row['Deskripsi']}")
        st.write(f"Jenis Wisata: {row['jenis_wisata']}")
        st.write(f"Rating: {row['avg_rating']:.2f}" if not pd.isna(row['avg_rating']) else "Rating: Belum Ada")
        st.image(row['url_gambar'], use_container_width=True)
        st.write("---")
else:
    st.write("Tidak ada rekomendasi yang tersedia.")

# Content-Based Filtering
st.header("Rekomendasi Tambahan dengan Content-Based Filtering")

# Combine features into a single text
mapping_wisata_df['combined_features'] = mapping_wisata_df['Deskripsi'] + " " + mapping_wisata_df['jenis_wisata']

# Ambil stop words untuk bahasa Indonesia dari NLTK
indonesian_stopwords = stopwords.words('indonesian')

# TF-IDF Vectorizer dengan stop words custom
tfidf = TfidfVectorizer(stop_words=indonesian_stopwords)
tfidf_matrix = tfidf.fit_transform(mapping_wisata_df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get indices for mapping
indices = pd.Series(mapping_wisata_df.index, index=mapping_wisata_df['Nama_Tempat_Wisata']).drop_duplicates()

# Function to get similar places and similarity scores
def get_similar_places(name, cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar places
    similar_indices = [i[0] for i in sim_scores]
    similar_scores = [i[1] for i in sim_scores]
    return mapping_wisata_df.iloc[similar_indices], similar_scores

# Example: Content-Based Filtering for a place
if not filtered_wisata.empty:
    first_place = filtered_wisata.iloc[0]['Nama_Tempat_Wisata']
    st.subheader(f"Tempat Wisata Serupa dengan '{first_place}'")
    
    # Get similar places and scores
    similar_places, scores = get_similar_places(first_place)

    for i, (_, row) in enumerate(similar_places.iterrows()):
        st.write(f"### {row['Nama_Tempat_Wisata']}")
        st.write(f"Deskripsi: {row['Deskripsi']}")
        st.write(f"Jenis Wisata: {row['jenis_wisata']}")
        st.write(f"**Skor Similarity:** {scores[i]:.4f}")
        st.image(row['url_gambar'], use_container_width=True)
        st.write("---")

