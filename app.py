import streamlit as st
import joblib
import re
import nltk   

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 🔹 Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# 🔹 Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🔹 Emotion → Emoji mapping
emoji_map = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲"
}

# 🔹 Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# 🔹 Page Config
st.set_page_config(page_title="Emotion AI", page_icon="💬", layout="centered")

# 🔹 Custom Styling (🔥 UI BOOST)
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 50px !important;   /* 🔥 FORCE BIG SIZE */
    font-weight: 900 !important;
    margin-bottom: 10px;
}

/* colors */
.blue {
    color: #38bdf8 !important;
}

.highlight {
    color: #a78bfa !important;
}

/* subtitle */
.subtitle {
    text-align: center;
    font-size: 18px !important;
    color: #cbd5f5 !important;
}
</style>
""", unsafe_allow_html=True)

# 🔹 Header
st.markdown(
    '<p class="title">💬 <span class="blue">Emotion</span> <span class="highlight">Detection AI</span></p>',
    unsafe_allow_html=True
)

# 🔹 Subtitle
st.markdown(
    '<p class="subtitle">Understand emotions from text using Machine Learning</p>',
    unsafe_allow_html=True
)
# 🔹 Input Box
user_input = st.text_area("✍️ Enter your text below:", height=180)

# 🔹 Predict Button
if st.button("🚀 Predict Emotion"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        emotion = le.inverse_transform([pred])[0]

        emoji = emoji_map.get(emotion, "")

        # 🔥 Display Result
        st.markdown("---")
        st.success(f"🎯 **Predicted Emotion:** {emotion.upper()} {emoji}")
