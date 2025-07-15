# 🎵 Spotify Song Popularity Predictor

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-1DB954?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)
![Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)

A machine learning-powered web app that predicts the **popularity score** of Spotify songs based on genre, duration, and whether the song is explicit. This app uses **Linear, Ridge, and Lasso Regression** models and is built with **Streamlit** and **scikit-learn**.

---

## 🚀 Live Demo

👉 [Launch App](https://spotify-app-by-narasimhamanam.streamlit.app/)

---

## 🧠 Features

- 🎧 Predict individual song popularity based on:
  - Genre
  - Duration
  - Explicit content
- 📈 View model performance (R² Score & MSE)
- 🔁 Switch between Linear, Ridge, and Lasso Regression models
- 📊 Visualize popularity distribution and genre-based trends
- 📂 Upload CSV file for **batch predictions**
- 📥 Download predicted results as a CSV
- 💅 Stylish UI with custom CSS and Spotify-themed visuals

---

## 📁 Dataset

The app uses a dataset named `spotify.csv` with features like:
- `name`: Song title
- `genre`: Song genre
- `duration_ms`: Duration of the song in milliseconds
- `explicit`: Whether the song has explicit content
- `popularity`: Popularity score (target variable)

> ⚠️ Make sure the `spotify.csv` file is placed in the root directory before running the app.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/spotify-popularity-predictor.git
cd spotify-popularity-predictor

pip install -r requirements.txt

streamlit run app.py
