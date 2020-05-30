# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:35:19 2020

@author: Anustup
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv(r"Tweets.csv", encoding="latin-1")
df
df.drop(['tweet_id', 'airline_sentiment_confidence', 'negativereason','negativereason_confidence','airline','airline_sentiment_gold','name','negativereason_gold','retweet_count','tweet_coord','tweet_created','tweet_location','user_timezone'], axis=1, inplace=True)
df.head()
df['label'] = df['airline_sentiment'].map({'neutral': 0, 'positive': 1,'negative':2})
X = df['text']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.externals import joblib
joblib.dump(clf, 'Sentiment.pkl')

