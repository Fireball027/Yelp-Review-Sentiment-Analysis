## Overview

The **NLP Sentiment Analysis Project** applies **Natural Language Processing (NLP)** techniques to analyze and classify Yelp reviews. This project explores text data, processes it using NLP methods, and trains a machine learning model to predict sentiment labels.

---

## Key Features

- **Data Preprocessing**: Cleans and tokenizes text data for analysis.
- **Exploratory Data Analysis (EDA)**: Examines review lengths, word frequency, and distributions.
- **Feature Engineering**: Transforms text into numerical representations using **TF-IDF**.
- **Machine Learning Model**: Implements **Naive Bayes** or **Logistic Regression** for sentiment classification.
- **Model Evaluation**: Assesses model accuracy using confusion matrix and classification report.

---

## Project Files

### 1. `yelp.csv`
This dataset contains Yelp reviews with the following key columns:
- **text**: The actual review text.
- **stars**: Rating (1 to 5) given by the reviewer.
- **useful, funny, cool**: User feedback metrics.

### 2. `NLP_Project.py`
This script performs text preprocessing, sentiment classification, and model evaluation.

#### Key Components:

- **Data Loading & Cleaning**:
  - Reads the Yelp dataset and extracts relevant text features.
  - Converts review ratings into binary sentiment labels (positive or negative).

- **Exploratory Data Analysis (EDA)**:
  - Generates word clouds and bar charts to visualize review distributions.
  - Analyzes word frequencies in positive vs. negative reviews.

- **Feature Engineering**:
  - Uses **TF-IDF Vectorization** to transform text into numerical format.

- **Model Training & Evaluation**:
  - Splits data into training and test sets.
  - Trains a **Naive Bayes** or **Logistic Regression** classifier.
  - Evaluates performance using accuracy, confusion matrix, and F1-score.

#### Example Code:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('yelp.csv')

# Create binary sentiment label
data['sentiment'] = data['stars'].apply(lambda x: 1 if x > 3 else 0)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm')
plt.show()
```

---

## How to Run the Project

### Step 1: Install Dependencies
Ensure required libraries are installed:
```bash
pip install pandas seaborn matplotlib scikit-learn wordcloud
```

### Step 2: Run the Script
Execute the main script:
```bash
python NLP_Project.py
```

### Step 3: View Insights
- Model accuracy and classification metrics.
- Visualizations of review sentiment distribution.
- Word clouds highlighting frequent words.

---

## Future Enhancements

- **Deep Learning Model**: Implement **LSTMs or BERT** for better performance.
- **Multi-Class Classification**: Predict all star ratings instead of binary classification.
- **Interactive Dashboard**: Develop a Streamlit-based interface for real-time sentiment analysis.
- **Web Scraping**: Collect fresh Yelp reviews for continuous model updates.

---

## Conclusion

The **Yelp Review Sentiment Analysis Project** applies **Natural Language Processing (NLP)** and **Machine Learning** to classify Yelp reviews based on sentiment. By leveraging data preprocessing, feature engineering, and classification models, this project provides valuable insights into user opinions and trends.

---

**Happy Analyzing! 🚀**

