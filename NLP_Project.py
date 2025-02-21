import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

sns.set_style('white')


# Load Dataset
yelp = pd.read_csv('yelp.csv')

# Display Basic Info
print(yelp.head(2))
print(yelp.info())
print(yelp.describe())

# Add 'text length' column
yelp['text length'] = yelp['text'].apply(len)


# Exploratory Data Analysis (EDA)
# Histogram of text length based on star ratings
g = sns.FacetGrid(yelp, col='stars', col_wrap=3)
g.map(plt.hist, 'text length', bins=50, edgecolor='black')
plt.show()

# Boxplot of text length for each star rating
plt.figure(figsize=(8, 5))
sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')
plt.title("Text Length by Star Rating")
plt.show()

# Countplot of star ratings
plt.figure(figsize=(7, 5))
sns.countplot(x='stars', data=yelp, palette='rainbow', edgecolor='black')
plt.title("Count of Reviews by Star Rating")
plt.show()

# Compute mean values grouped by stars
stars = yelp.groupby('stars').mean()
print(stars)

# Correlation heatmap
stars.corr()
print(stars.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(stars.corr(), cmap='coolwarm', annot=True, linewidths=1)
plt.title("Correlation Heatmap")
plt.show()


# NLP Classification Task
# Filter dataset for 1-star and 5-star reviews
yelp_class = yelp[yelp['stars'].isin([1, 5])]

# Define Features (X) and Labels (y)
X = yelp_class['text']
y = yelp_class['stars']

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)


# Train Test Split
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)


# Predictions And Evaluations
predictions = nb.predict(X_test)

print("Naive Bayes Classification Report:\n")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))


# Pipeline Approach
# Redefine Features and Labels
X_text = yelp_class['text']
y_labels = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X_text, y_labels, test_size=0.3, random_state=101)


# Using Text Processing
# Define Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),       # Convert text to BOW format
    ('tfidf', TfidfTransformer()),    # Apply TF-IDF
    ('model', MultinomialNB())        # Train Naive Bayes Classifier
])


# Train Pipeline
pipeline.fit(X_train, y_train)


# Predictions And Evaluation Using Pipeline
pipe_predictions = pipeline.predict(X_test)

print("\nPipeline Classification Report:\n")
print(classification_report(y_test, pipe_predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pipe_predictions))
