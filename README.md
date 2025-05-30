#  SMS Spam Classifier

An AI-powered SMS classifier that detects spam messages using a Naive Bayes model and displays results via a Streamlit web interface.

---

##  Features

- Cleaned and processed SMS dataset (spam vs ham)
- TF-IDF vectorization
- Naive Bayes classification
- Evaluation with accuracy, precision, recall
- Live predictions with a **web interface**
- Modular structure: training, prediction, and interface separated

---

## 🖥 Demo

<img src="https://img.shields.io/badge/Streamlit-Running-green" />
> Run locally with `streamlit run app.py`

---

##  Dataset

From the [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---


## 🔧 Installation

```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-classifier.git
cd sms-spam-classifier
python -m venv venv
venv\Scripts\activate
pip install pandas scikit-learn joblib streamlit
streamlit run app.py
