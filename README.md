# 🎬 Sentiment Analysis on IMDB Movie Reviews

## 📘 Overview
This project focuses on performing **Sentiment Analysis** using **Natural Language Processing (NLP)** on a dataset of **IMDB movie reviews**.  
The objective is to build a machine learning model capable of classifying reviews as **positive** or **negative**, thereby understanding viewer emotions and opinions based on text.

This analysis was completed as part of the **Data Science coursework**, showcasing the application of **text preprocessing, NLP modeling, and classification** in Python.

---

## 🎯 Objectives
- To apply text preprocessing and cleaning techniques to raw review data.  
- To perform **sentiment classification** of IMDB reviews into positive or negative categories.  
- To explore and evaluate different **machine learning models** for text classification.  
- To understand the influence of linguistic features and text normalization on classification accuracy.

---

## 🧩 Dataset Description
- **Source:** [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Total Records:** 50,000  
- **Features:**
  - `review`: Text data containing user-written movie reviews.
  - `sentiment`: Target label indicating the sentiment (`positive` or `negative`).

The dataset provides a balanced distribution of positive and negative reviews, making it ideal for binary classification tasks.

---

## ⚙️ Tools and Libraries Used
| Library | Purpose |
|----------|----------|
| `pandas`, `numpy` | Data manipulation and analysis |
| `re`, `BeautifulSoup` | Text cleaning and HTML tag removal |
| `nltk` | Tokenization, stopword removal, and stemming |
| `sklearn` | Machine learning models and performance evaluation |
| `tensorflow` / `keras` | Deep learning-based text classification (if applied) |
| `matplotlib`, `seaborn` | Data visualization |

---

## 🧠 Methodology

1. **Data Import and Understanding**
   - Loaded the IMDB dataset (`IMDB.csv`) containing 50,000 reviews and sentiments.
   - Explored the dataset structure and balanced class distribution.

2. **Data Preprocessing**
   - Removed unwanted text patterns like “READ MORE”.
   - Converted all text to lowercase.
   - Handled emojis using ASCII encoding to avoid out-of-vocabulary issues.
   - Removed special characters, punctuation, and HTML tags using regex and `BeautifulSoup`.
   - Tokenized sentences and removed English stopwords using `nltk`.
   - Applied stemming or lemmatization to standardize word forms.

3. **Feature Extraction**
   - Converted text into numerical form using:
     - **Bag of Words (BoW)** or  
     - **TF-IDF Vectorization**.
   - Split data into training and testing sets for model evaluation.

4. **Model Development**
   - Implemented traditional models such as:
     - Logistic Regression  
     - Naive Bayes Classifier  
     - Support Vector Machine (SVM)
   - Optionally, deep learning models such as LSTM or CNN were explored for higher accuracy.

5. **Model Evaluation**
   - Evaluated models using metrics such as Accuracy, Precision, Recall, and F1-Score.
   - Generated confusion matrices and classification reports.
   - Compared performance of classical ML models vs deep learning models.

---

## 📊 Key Visualizations
- **Distribution of Positive vs Negative Reviews**
- **Word Cloud** highlighting frequent words in each sentiment class.
- **Confusion Matrix** for model performance visualization.
- **Accuracy Comparison Chart** for different algorithms.

---

## 📈 Insights and Results
- After preprocessing, the dataset achieved a well-balanced token distribution.  
- The **Naive Bayes classifier** provided a strong baseline performance.  
- **TF-IDF vectorization** improved feature representation and boosted accuracy.  
- Advanced models like **LSTM** captured contextual relationships, yielding the highest accuracy.  
- The model effectively predicted user sentiment with high reliability on unseen data.

---


