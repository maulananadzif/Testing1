# Import library yang diperlukan
import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

# Membaca data dari file "hungarian.data"
with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

# Membuat generator menggunakan itertools.takewhile untuk mengambil bagian dari lines dengan panjang 76
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

# Memproses dan membersihkan data
df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)

# Mengganti nilai -9.0 dengan NaN
df.replace(-9.0, np.NaN, inplace=True)

# Memilih kolom yang relevan untuk analisis
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

# Membuat peta kolom untuk mengganti nama kolom
column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

# Mengganti nama kolom sesuai dengan peta
df_selected.rename(columns=column_mapping, inplace=True)

# Menghapus kolom yang tidak diperlukan
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

# Menghitung nilai rata-rata untuk beberapa kolom numerik
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

# Membulatkan nilai rata-rata
meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

# Mengisi nilai NaN dengan nilai rata-rata yang telah dihitung
fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

df_clean = df_selected.fillna(value=fill_values)
# Menghapus duplikat baris
df_clean.drop_duplicates(inplace=True)

# Membagi data menjadi fitur (X) dan target (y)
X = df_clean.drop("target", axis=1)
y = df_clean['target']
# Oversampling menggunakan SMOTE untuk menyeimbangkan distribusi kelas
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

model = pickle.load(open("model/xgb_oversample_model.pkl", 'rb')) # Memuat model XGBoost yang telah dilatih sebelumnya

y_pred = model.predict(X) # Memprediksi hasil

# Menghitung akurasi model
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

# Membuat DataFrame hasil akhir dengan fitur (X) dan target (y)
df_final = X
df_final['target'] = y

# ========================================================================================================================================================================================

# STREAMLIT: Konfigurasi halaman dan tampilan aplikasi web Streamlit

# Menetapkan judul dan ikon halaman
st.set_page_config(
  page_title = "Hungarian Heart Disease",
  page_icon = ":heart:"
)

# Menampilkan judul utama aplikasi
st.title("Hungarian Heart Disease")

# Menampilkan akurasi model sebagai informasi
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Do not copy outright_])")
st.write("")

# Membuat dua tab: Single-predict dan Multi-predict
tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

# Konfigurasi tab Single-predict
with tab1:
  st.sidebar.header("**User Input** Sidebar") # Menampilkan sidebar untuk input pengguna

  # Input umur (age)
  age = st.sidebar.number_input(label=":violet[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")

  # Input jenis kelamin (sex)
  sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  # Input jenis nyeri dada (chest pain type)
  cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  # Input tekanan darah istirahat (resting blood pressure)
  trestbps = st.sidebar.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")

  # Input serum kolesterol (serum cholestoral)
  chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")

  # Input kadar gula darah puasa (fasting blood sugar)
  fbs_sb = st.sidebar.selectbox(label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  # Input hasil elektrokardiografi istirahat (resting electrocardiographic results)
  restecg_sb = st.sidebar.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  # Input detak jantung maksimal (maximum heart rate achieved)
  thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")

  # Input angina yang dipicu oleh olahraga (exercise induced angina)
  exang_sb = st.sidebar.selectbox(label=":violet[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  # Input depresi ST yang diinduksi oleh olahraga (ST depression induced by exercise relative to rest)
  oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")

  # Menampilkan input pengguna sebagai DataFrame
  data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
  }

  # Menampilkan input pengguna sebagai DataFrame
  preview_df = pd.DataFrame(data, index=['input'])

  st.header("User Input as DataFrame")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  result = ":violet[-]"

  predict_btn = st.button("**Predict**", type="primary")

  st.write("")
  if predict_btn:
    #Melakukan prediksi menggunakan model
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    # Menampilkan status prediksi secara visual
    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    # Mengubah hasil prediksi menjadi label yang dapat dimengerti
    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]"

  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

# Konfigurasi tab Multi-predict
with tab2:
  st.header("Predict multiple data:")

  # Menghasilkan contoh data CSV untuk diunduh
  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  
  # Mengunggah file CSV dari pengguna
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    # Menampilkan status pengolahan data secara visual
    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    # Mengubah hasil prediksi untuk setiap baris data
    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    # Menampilkan status pengolahan data secara visual
    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()
        
    # Menampilkan hasil prediksi dan data yang diunggah
    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
