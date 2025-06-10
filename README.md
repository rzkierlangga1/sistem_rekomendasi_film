# Implementasi Algoritma Machine Learning pada Sistem Rekomendasi Film Berbasis Website Interaktif - Moodflix 

    Overview

## Tools

    1. Google Colab
    2. 
    3.

## Dataset

    Dataset yang digunakan dalam proyek ini diperoleh dari TMDB (The Movie Database) melalui proses web scraping. Dataset terdiri dari 15.000 baris dan 6 kolom, dan dapat diakses melalui tautan GitHub publik berikut:

**https://raw.githubusercontent.com/ilyasbrhm/Dataset/refs/heads/main/tmdb_dataset.csv**

**Fitur Dataset:**

    1. title: Judul film
    2. genres: Daftar genre film
    3. rating: Rating (0–10)
    4. popularity: Skor popularitas dari TMDB
    5. release_year: Tahun rilis
    6. poster_url: Tautan gambar poster film

## Pipeline ML

### 1. Import Libraries

Pastikan semua library telah terinstal. Jalankan perintah di bawah ini untuk instalasi **(jika belum)**:

```bash
!pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
``` 

**Libraries yang digunakan:**

```bash
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import pickle
```

### 2. Exploratory Data Analysis (EDA)

Tahap ini bertujuan untuk memahami struktur, isi, dan kualitas data.

**a. Data Exploration:**

    - Menampilkan 5 baris pertama
    - Melihat jumlah baris dan kolom 
    - Memeriksa tipe data 
    - Menambahkan kolom id (jika belum ada)

**b. Data Cleaning:**

    - Memeriksa data yang hilang (missing values)
    - Menghapus entri dengan nilai kosong 
    - Menghapus data duplikat
    - Membersihkan dan memformat kolom genres yang awalnya berbentuk string yang menyerupai list dikonversi menjadi list Python asli.
    - Mengekstrak dan menghitung kemunculan setiap genre
    - Reset indeks dataframe hasil pembersihan

    Output dari tahap ini adalah DataFrame yang sudah dibersihkan (df_cleaned).

**c. Visualization (opsional)**

    - Mengevaluasi sebaran rating film
    - Melihat tren jumlah film berdasarkan tahun rilis
    - Mengidentifikasi genre paling umum
    - Meninjau hubungan antara rating dan popularitas

**Perlu dicatat bahwa beberapa langkah dalam tahap EDA dapat disesuaikan berdasarkan jenis dan struktur dataset yang digunakan.**

### 3. Data Preparation

Tahap ini bertujuan untuk menyiapkan data yang sudah dibersihkan sebelum digunakan dalam proses pelatihan model.

    - Mapping Mood Berdasarkan Genre: Memberikan label target yang lebih bermakna, dilakukan pemetaan genre film ke dalam kategori mood atau suasana film.
    - Multilabel Binarization (One-Hot Encoding Genre): Karena setiap film bisa memiliki lebih dari satu genre, digunakan MultiLabelBinarizer untuk mengubah daftar genre menjadi bentuk biner yang dapat digunakan sebagai fitur numerik.
    - Feature Engineering: Fitur akhir yang digunakan dalam model merupakan gabungan dari fitur numerik (release_year, popularity, rating) dan hasil one-hot encoding dari genre
    - Standardization: Fitur numerik dinormalisasi menggunakan StandardScaler agar berada pada skala yang seragam:
    - Label Encoding Mengonversi target mood yang masih berupa kategori (string) menjadi bentuk numerik.
    - One Hot Encoding untuk Target (mood): Mengubah angka hasil Label Encoding menjadi vektor biner.
    - Data Splitting: Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20.

**Beberapa proses pada tahap ini juga bisa bervariasi tergantung pada struktur dataset yang digunakan dan tujuan proyek masing-masing.**

### 4. Modelling

Model yang digunakan adalah Artificial Neural Network (ANN) menggunakan arsitektur Sequential dari TensorFlow/Keras.

**a. Struktur Model:**

![Screenshot 2025-06-10 125644](https://github.com/user-attachments/assets/61edcaf6-cfab-4bfe-9897-efd68a99bd67)

    - Dense(128, activation='relu') → Layer 1 (Hidden Layer 1)
    - Dropout(0.3) → Layer 2 (Dropout Layer)
    - Dense(64, activation='relu') → Layer 3 (Hidden Layer 2)
    - Dense(output, activation='softmax') → Layer 4 (Output Layer)

**b. Kompilasi Model:**

![Screenshot 2025-06-10 125801](https://github.com/user-attachments/assets/79edcbea-76a6-44fd-b728-13447ff62dca)

    - Optimizer: adam
    - Loss Function: categorical_crossentropy
    - Metrik Evaluasi: accuracy

**c. Proses Training Model:**   

![Screenshot 2025-06-10 125842](https://github.com/user-attachments/assets/78f44440-42b7-487f-a70f-fcc7df814652)

    - Epochs: 20
    - Batch Size: 32
    - Validation Split: 0.2

**d. Alasan memilih model ini:**
    
    - Fleksibel terhadap Berbagai Jenis Fitur: ANN mampu mengolah fitur numerik maupun kategori dan menemukan pola kompleks di antaranya, termasuk korelasi non-linear.
    - Mendukung Klasifikasi Multi-Kelas: Arsitektur softmax pada output layer menjadikan ANN sangat cocok untuk kasus klasifikasi dengan lebih dari dua kelas (seperti mood: emotional, cheerful, reflective, dll).
    - Menghasilkan Akurasi yang Tinggi: Dengan jumlah layer dan neuron yang memadai, ANN dapat mengungguli model klasik lainnya, terutama untuk dataset yang besar dan kompleks.
    - Daya Generalisasi Tinggi: Dengan teknik seperti dropout dan penggunaan validation set, model ANN lebih tahan terhadap overfitting dan mampu memberikan prediksi yang lebih akurat pada data baru yang belum pernah dilihat.

**Perlu diingat bahwa tahapan modelling dapat berbeda-beda tergantung preferensi, tujuan, dan pendekatan dari masing-masing proyek yang dikembangkan. Pemilihan model, struktur arsitektur, parameter training, hingga metrik evaluasi bisa disesuaikan dengan karakteristik data, kebutuhan akurasi, serta kompleksitas masalah yang dihadapi.**

### 5. Evaluation

Pada tahap ini, dilakukan evaluasi terhadap performa model yang telah dibuat. Beberapa metrik yang digunakan, yaitu:

    - Akurasi pada data uji (testing accuracy) untuk mengukur seberapa baik model memprediksi label pada data yang belum pernah dilihat sebelumnya.
    - Classification report yang mencakup metrik precision, recall, dan F1-score untuk masing-masing kelas.
    - Confusion matrix untuk memberikan gambaran visual mengenai jumlah prediksi yang benar dan salah dari setiap kelas.

## Output ML

Adapun output yang dihasilkan dan disimpan dari pipeline Machine Learning dalam proyek ini:
    
    - dataset_fix.csv = Dataset bersih (df_cleaned).
    - scaler.pkl = Menyimpan objek scaler untuk standarisasi fitur input.
    - label_encoder.pkl	= Encoder untuk mengubah label kategori (single label) jadi numerik.
    - mlb.pkl = Encoder multi-label untuk label dengan lebih dari satu kelas.
    - sistem_rekomendasi_model.h5	Model neural network hasil pelatihan, berisi arsitektur & bobot.

## Pipeline Backend
