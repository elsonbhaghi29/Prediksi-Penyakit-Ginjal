import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# Judul aplikasi
st.title("Prediksi Penyakit Ginjal")

# Memuat model Decision Tree
model = joblib.load('decision_tree_model.pkl')

# Memuat dataset
file_path = 'kidney_disease.csv'
data = pd.read_csv(file_path)

# Menghapus duplikat dan mengisi missing values seperti yang Anda lakukan sebelumnya
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
