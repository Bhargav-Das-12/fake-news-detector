# fake-news-detector
A machine learning-based Fake News Detector using TF-IDF and Logistic Regression. This project helps identify whether a news article is real or fake with high precision.
# 📰 Fake News Detector using Machine Learning

---

## 📌 Problem Statement

In today’s digital world, the rapid spread of fake news has become a major threat to public awareness, affecting opinions, emotions, and even social harmony. With thousands of articles being published online every day, it's practically impossible to manually verify each one. Hence, an automated and intelligent system for fake news detection is urgently needed.

---

## 💡 Proposed Solution

This project proposes a machine learning-based classifier that processes and analyzes the text of news articles, then labels them as *fake* or *real*. It leverages Natural Language Processing (NLP) and a logistic regression model trained on thousands of real-world news articles.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- TfidfVectorizer for text representation
- Matplotlib & Seaborn for visualization
- Jupyter Notebook / VS Code for development

---

## 🧠 Algorithm & Deployment

- **Vectorization**: Used `TfidfVectorizer` to convert raw text into numerical features.
- **Model**: Trained a `Logistic Regression` model to classify articles as fake or real.
- **Evaluation**: Evaluated model using accuracy, confusion matrix, precision & recall metrics.
- **Deployment**: (Optional) This project can be converted into a web app using Flask or Streamlit.
---
## How the Model Classifies News:
Preprocessing:
Converts news text to lowercase, removes punctuation, numbers, stopwords, and unwanted characters.

Vectorization:
Uses TF-IDF to turn cleaned text into numerical features highlighting important words.

Model Training:
Trains a Logistic Regression model to learn patterns from labeled real and fake news.

Prediction:
For new input, the same preprocessing and vectorization are applied, and the model predicts “Fake” or “Real”.

Evaluation:
Performance measured using Accuracy, Precision, Recall, F1-score, confusion matrix, and bar charts.

---

## 📊 Results

The model achieved strong performance on the validation set, with good accuracy, precision, and recall values. It was able to identify fake articles with high confidence.

Sample results:
- Confusion Matrix and Precision/Recall Visuals (see images in the repo)
- Accuracy: **~96%**

---

## 📌 Conclusion

This project demonstrates how machine learning can be effectively used to detect fake news. While not a replacement for human fact-checkers, such tools are extremely helpful in flagging potentially false content for review.

---

## 🔗 References

- [Dataset from Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)

---

📁 **Note**: Large CSV files (True.csv & Fake.csv) are included in the repo but may be removed later due to GitHub file size recommendations.
