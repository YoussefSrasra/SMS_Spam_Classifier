
import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="📩 SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below and see if it's spam or not.")

# Text input
user_input = st.text_area("Message:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform & predict
        vector = vectorizer.transform([user_input])
        pred = model.predict(vector)
        label = label_encoder.inverse_transform(pred)[0]

        if label == "spam":
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is **HAM** (not spam).")
