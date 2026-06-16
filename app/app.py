import streamlit as st
import joblib

# Load saved model
model = joblib.load("model/sentiment_model.pkl")

# Load saved vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Page title
st.title("🤖 AI Sentiment Analysis System")

st.write(
    "Analyze whether a review is Positive, Negative, or Neutral."
)

# User input
user_text = st.text_area(
    "Enter your review:"
)

# Button
if st.button("Analyze Sentiment"):

    if user_text.strip():

        user_vector = vectorizer.transform(
            [user_text]
        )

        prediction = model.predict(
            user_vector
        )[0]

        probabilities = model.predict_proba(
            user_vector
        )[0]

        confidence = max(probabilities) * 100

        # Emoji
        if prediction == "Positive":
            emoji = "😊"
        elif prediction == "Negative":
            emoji = "😞"
        else:
            emoji = "😐"

        st.subheader("Prediction")

        st.success(
            f"{prediction} {emoji}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.subheader(
            "Sentiment Probabilities"
        )

        for sentiment, probability in zip(
            model.classes_,
            probabilities
        ):
            st.write(
                f"**{sentiment}** : {probability*100:.2f}%"
            )