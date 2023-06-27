import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Judul web
st.title('Home Checking Your Heart')

file = st.file_uploader("Database Penyakit Jantung", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.write(df)  # Menampilkan dataframe di aplikasi Streamlit

# Load dataset
heart_data = pd.read_csv("heart_cleveland_upload.csv")

# Preprocessing
X = heart_data.drop(columns='condition', axis=1)
Y = heart_data['condition']

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.14, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Perform predictions on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Perform predictions on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Streamlit app
def main():
    # Display training data
    st.subheader("Data Training")
    training_data = pd.concat([X_train, Y_train], axis=1)  # Menggabungkan fitur X_train dan target Y_train
    st.write(training_data)

    # Display testing data
    st.subheader("Data Testing")
    testing_data = pd.concat([X_test, Y_test], axis=1)  # Menggabungkan fitur X_test dan target Y_test
    st.write(testing_data)

    # Display evaluation results
    st.subheader("Evaluasi Model")
    st.write("Akurasi Data Testing:", test_data_accuracy)

if __name__ == "__main__":
    main()



col1, col2, col3 = st.columns(3)

with col1 :
    age = st.number_input('Usia')
with col2 : 
    sex = st.number_input('Jenis Kelamin (1: Pria ; 0: Wanita)')
with col3 :
    cp = st.number_input ('Jenis Nyeri Dada')
with col1 :
    trestbps = st.number_input ('Tekanan Darah')
with col2 :
    chol = st.number_input ('Nilai Kolestrol')
with col3 :
    fbs = st.number_input ('Gula Darah')
with col1 :
    restecg = st.number_input ('Hasil Elektrokadiografi')
with col2 :
    thalach = st.number_input ('Detak Jantung Max.')
with col3 :
    exang = st.number_input ('Induksi Angina')
with col1 :
    oldpeak = st.number_input ('ST Depression')
with col2 :
    slope =  st.number_input ('Slope')
with col3 :
    ca = st.number_input ('Nilai CA')
with col1 :
    thal = st.number_input ('Nilai Thal')

# Kode Prediksi
heart_diagnosis = ''

# Membuat tombol prediksi
if st.button ('Prediksi Penyakit Jantung'):
    heart_prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if(heart_prediction[0] == 1):
        heart_diagnosis = 'Pasien Terindikasi Penyakit Jantung'
    else :
        heart_diagnosis = 'Pasien Tidak Terindikasi Penyakit Jantung'
st.success(heart_diagnosis)

# Menampilkan data prediksi
st.subheader("Hasil Prediksi")
st.write("Data Pasien:")
st.write("Usia:", age)
st.write("Jenis Kelamin (1: Pria ; 0: Wanita):", sex)
st.write("Jenis Nyeri Dada:", cp)
st.write("Tekanan Darah:", trestbps)
st.write("Nilai Kolesterol:", chol)
st.write("Gula Darah:", fbs)
st.write("Hasil Elektrokadiografi:", restecg)
st.write("Detak Jantung Max.:", thalach)
st.write("Induksi Angina:", exang)
st.write("ST Depression:", oldpeak)
st.write("Slope:", slope)
st.write("Nilai CA:", ca)
st.write("Nilai Thal:", thal)
st.write("Diagnosis Penyakit Jantung:", heart_diagnosis)