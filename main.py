import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. LOAD DATA
DATA_PATH = "SMSSpamCollection"        
df = pd.read_csv(DATA_PATH,
                 sep="\t",
                 header=None,
                 names=["label", "message"])


# 2. CLEAN TEXT
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)                              # drop digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


df["message_clean"] = df["message"].apply(clean_text)


# 3. ENCODE LABELS  (spam = 1, ham = 0)
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])


# 4. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["message_clean"],
    df["label_encoded"],
    test_size=0.20,
    random_state=42
)


# 5. TEXT → TF-IDF  (fit on train, transform test)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


# 6. TRAIN MODEL
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# 7. EVALUATE
y_pred = model.predict(X_test_tfidf)

print("\n===== Evaluation on Hold-out Test Set =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification report\n", classification_report(y_test, y_pred))
print("Confusion matrix\n", confusion_matrix(y_test, y_pred))


# 8. SAVE ARTEFACTS
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\n✅ Model, vectorizer and encoder saved!")
