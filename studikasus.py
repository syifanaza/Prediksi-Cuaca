import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import joblib
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fungsi untuk membaca dataset (diperbarui dengan st.cache_data)
@st.cache_data
def load_data():
    df = pd.read_csv("weatherHistory.csv")
    return df

# Fungsi untuk menampilkan halaman dataset
def show_dataset(df):
    st.title("Dataset Cuaca")
    st.write("Data Cuaca")
    st.dataframe(df)

    st.write("Missing Data")
    st.write(df.isnull().sum())

    st.write("Statistik Deskriptif")
    st.write(df.describe())

    st.write("Tipe Data")
    st.write(df.dtypes)

# Fungsi untuk menampilkan visualisasi data
def show_visualizations(df):
    st.title("Visualisasi Data Cuaca")
    
    # Grafik distribusi suhu
    st.write("Grafik Distribusi Suhu")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Temperature (C)'], ax=ax, kde=True, color="skyblue")
    ax.set_title('Temperature Distribution Plot')
    st.pyplot(fig)

    # Grafik distribusi jenis cuaca
    st.write("Distribusi Jenis Cuaca")
    weather_counts = df['Summary'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    weather_counts.plot(kind="bar", ax=ax)
    ax.set_title("Weather Summary Distribution")
    ax.set_xlabel("Summary")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Word Cloud untuk jenis cuaca
    st.write("Word Cloud Jenis Cuaca")
    weather_summaries = ' '.join(df['Summary'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(weather_summaries)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Weather Summary Word Cloud')
    st.pyplot(fig)

    # Scatter plot antara suhu dan kelembapan
    st.write("Scatter Plot: Temperature vs Humidity")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['Temperature (C)'], df['Humidity'], color='orange')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Humidity')
    ax.set_title('Temperature vs Humidity')
    st.pyplot(fig)

# Fungsi untuk menampilkan prediksi suhu nyata
def show_predictions(df):
    st.title("Prediksi Suhu Nyata Berdasarkan Data Cuaca")

    # Split data menjadi training dan testing
    x = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
    y = df['Apparent Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Membangun model regresi linear
    model_regresi = LinearRegression()
    model_regresi.fit(X_train, y_train)

    # Simpan model
    joblib.dump(model_regresi, 'weather_model.sav')

    # Input nilai untuk prediksi
    # Mengatur input agar muncul vertikal (berurutan)
    temperature = st.slider('Masukkan Temperature (C)', min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
    humidity = st.slider('Masukkan Humidity', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    wind_speed = st.slider('Masukkan Wind Speed (km/h)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    if st.button('Prediksi'):
        # Prediksi suhu nyata
        prediction = model_regresi.predict([[temperature, humidity, wind_speed]])
        predicted_temperature = float(prediction[0])
        st.write(f'Suhu nyata yang diprediksi adalah: {predicted_temperature:.2f}°C')

    # Evaluasi model
    model_regresi_pred = model_regresi.predict(X_test)
    mae = mean_absolute_error(y_test, model_regresi_pred)
    mse = mean_squared_error(y_test, model_regresi_pred)
    rmse = np.sqrt(mse)

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Fungsi untuk menampilkan halaman tentang aplikasi
def show_about():
    st.title("Tentang Aplikasi")
    st.write("""  
    Aplikasi ini dirancang untuk memprediksi suhu nyata (Apparent Temperature) berdasarkan data cuaca yang diberikan.
    Anda dapat menjelajahi dataset, melihat visualisasi, dan melakukan prediksi melalui menu yang tersedia.
    """)
    st.write("Model yang digunakan: Linear Regression")

# Main Program
def main():
    st.sidebar.title("Navigasi")
    menu = st.sidebar.selectbox(
        "Pilih Menu",
        ["Dataset", "Visualisasi", "Prediksi", "Tentang Aplikasi"]
    )

    df = load_data()

    if menu == "Dataset":
        show_dataset(df)
    elif menu == "Visualisasi":
        show_visualizations(df)
    elif menu == "Prediksi":
        show_predictions(df)
    elif menu == "Tentang Aplikasi":
        show_about()

if __name__ == "__main__":
    main()
