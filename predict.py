import sys
import joblib

# 1. LOAD ARTEFACTS
model          = joblib.load("spam_classifier_model.pkl")
tfidf          = joblib.load("tfidf_vectorizer.pkl")
label_encoder  = joblib.load("label_encoder.pkl")

# 2. GET MESSAGES TO CLASSIFY
if len(sys.argv) > 1:
    # messages passed as command-line arguments
    new_messages = sys.argv[1:]
else:
    # prompt user, keep asking until blank line
    print("Type/paste SMS messages to classify.  blank line = quit")
    new_messages = []
    while True:
        line = input("📩 > ").strip()
        if not line:
            break
        new_messages.append(line)

if not new_messages:
    print("No messages supplied – nothing to do.")
    sys.exit(0)

# 3. TRANSFORM & PREDICT
new_vectors = tfidf.transform(new_messages)
predictions = model.predict(new_vectors)
labels      = label_encoder.inverse_transform(predictions)

# 4. OUTPUT
for msg, lab in zip(new_messages, labels):
    print(f"\n📩  {msg}")
    print(f"   → 🚨 {lab.upper()}")
