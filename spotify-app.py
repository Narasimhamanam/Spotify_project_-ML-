import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import io


st.set_page_config(page_title="Spotify Song Popularity Predictor", layout="wide")
# ------------------------------------
# 🎨 Inject Custom CSS
# ------------------------------------
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: white;
    }
    .main {
        background-color: #0f1117;
    }
    h1, h2, h3, h4 {
        color: #1DB954;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 16px;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        color: black;
        transform: scale(1.03);
    }
    .css-1aumxhk {
        padding: 2rem;
    }
    .css-1cpxqw2 {
        background-color: #191c24 !important;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    .stDownloadButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDownloadButton>button:hover {
        background-color: #1ed760;
        color: black;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-image:url("assets/Spotify-Logo.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ------------------------------------
# 🔹 Load Dataset
# ------------------------------------
df = pd.read_csv("spotify.csv")
df.dropna(inplace=True)

# Encode genre
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# Features and Target
X = df[['genre_encoded', 'duration_ms', 'explicit']]
y = df['popularity']

# Scale duration for better performance
scaler = StandardScaler()
X[['duration_ms']] = scaler.fit_transform(X[['duration_ms']])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# 🔹 Streamlit App Layout
# ------------------------------------
st.set_page_config(page_title="Spotify Song Popularity Predictor", layout="wide")
st.title("🎵 Spotify Song Popularity Predictor (ML Project)")




st.markdown("""
Use this app to predict how popular a song might be based on genre, duration, and whether it's explicit or not.
""")

# ------------------------------------
# 🔹 Model Selector
# ------------------------------------
model_option = st.selectbox("Choose Regression Model", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

if model_option == "Linear Regression":
    model = LinearRegression()
elif model_option == "Ridge Regression":
    model = Ridge()
else:
    model = Lasso()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.markdown(f"**Model Performance:**  \nR² Score: `{r2:.2f}` | MSE: `{mse:.2f}`")

# ------------------------------------
# 🔹 Predict Individual Song Popularity
# ------------------------------------
st.header("🎧 Predict Popularity for a Song")

col1, col2, col3 = st.columns(3)

with col1:
    selected_genre = st.selectbox("Genre", df['genre'].unique())
with col2:
    duration = st.slider("Duration (ms)", 30000, 500000, step=10000, value=210000)
with col3:
    explicit_flag = st.radio("Explicit?", ['No', 'Yes'])
explicit_binary = 1 if explicit_flag == 'Yes' else 0
genre_encoded = le.transform([selected_genre])[0]

# Scale duration for prediction
duration_scaled = scaler.transform([[duration]])[0][0]

if st.button("Predict Popularity"):
    input_data = np.array([[genre_encoded, duration_scaled, explicit_binary]])
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Popularity Score: {prediction:.2f}")

# ------------------------------------
# 📊 Popularity Distribution
# ------------------------------------
st.header("📊 Popularity Distribution")
st.bar_chart(df['popularity'].value_counts().sort_index())

# ------------------------------------
# 📈 Average Popularity by Genre
# ------------------------------------
st.header("🎼 Average Popularity by Genre")
avg_pop = df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
st.bar_chart(avg_pop)

# ------------------------------------
# 🔝 Top 5 Most Popular Songs
# ------------------------------------
st.header("🔥 Top 5 Most Popular Songs")
top_songs = df.sort_values(by='popularity', ascending=False).head(5)
st.table(top_songs[['name', 'genre', 'popularity']])

# ------------------------------------
# 📂 Upload CSV for Batch Predictions
# ------------------------------------
st.header("📂 Batch Predictions from Your CSV")

uploaded_file = st.file_uploader("Upload CSV with columns: name, genre, duration_ms, explicit", type="csv")

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    try:
        user_df['genre_encoded'] = le.transform(user_df['genre'])
        user_df['explicit'] = user_df['explicit'].apply(lambda x: 1 if str(x).lower() == 'yes' or str(x) == '1' else 0)
        user_df[['duration_ms']] = scaler.transform(user_df[['duration_ms']])
        preds = model.predict(user_df[['genre_encoded', 'duration_ms', 'explicit']])
        user_df['predicted_popularity'] = preds

        st.success("✅ Predictions done!")
        st.dataframe(user_df[['name', 'genre', 'predicted_popularity']])

        # Download button
        csv = user_df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "predicted_popularity.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# ------------------------------------
# 📂 Resources and ML concepts
# ------------------------------------
st.markdown("---")
st.subheader("📚 Resources & ML Concepts Used")

st.markdown("""
- 🤖 **Linear Regression**: [scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- 📉 **Ridge & Lasso Regression**: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- 📊 **Data Preprocessing**:
  - [Label Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
  - [Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- 📂 **Train-Test Split**: [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- 📈 **Metrics**:
  - [R² Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
  - [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- 🎨 **Streamlit App Framework**: [Streamlit Docs](https://docs.streamlit.io/)
- 📁 **Batch File Upload + Download**: [st.file_uploader](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader) & [st.download_button](https://docs.streamlit.io/library/api-reference/widgets/st.download_button)
""", unsafe_allow_html=True)



st.markdown(
    """
    <hr style='margin-top: 50px;'>
    <div style='text-align: center; color: #999; font-size: 14px;'>
        🔋 Powered by <strong>Narasimha Manam</strong> | Built with ❤️ using Streamlit & scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)


