import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "dataset/twitter_training.csv",
    header=None,
    names=["ID", "Entity", "Sentiment", "Tweet"]
)

# =========================
# DATA CLEANING
# =========================

df = df.dropna(subset=["Tweet"])
df = df[df["Sentiment"] != "Irrelevant"]

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["Tweet"] = df["Tweet"].apply(clean_text)

# =========================
# FEATURES AND LABELS
# =========================

X = df["Tweet"]
y = df["Sentiment"]

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)

# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL TRAINING
# =========================

model = LogisticRegression(
    C=5,
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs"
)

model.fit(X_train, y_train)

# =========================
# MODEL EVALUATION
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print(" AI SENTIMENT ANALYSIS SYSTEM")
print("======================================")
print("Dataset Size :", len(df))
print("Model Accuracy :", round(accuracy * 100, 2), "%")
print("======================================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and Vectorizer saved successfully.\n")

# =========================
# USER PREDICTION LOOP
# =========================

while True:

    user_text = input(
        "Enter a review (or type 'exit' to quit): "
    )

    if user_text.lower() == "exit":
        print("\nThank you for using the system!")
        break

    cleaned_text = clean_text(user_text)

    user_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(user_vector)[0]

    probabilities = model.predict_proba(user_vector)[0]

    confidence = max(probabilities) * 100

    print("\n----------- RESULT -----------")
    print("Review      :", user_text)
    print("Sentiment   :", prediction)
    print("Confidence  :", round(confidence, 2), "%")

    print("\nSentiment Probabilities:")

    for sentiment, probability in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"{sentiment:<10}: {probability*100:.2f}%"
        )

    print("------------------------------\n")