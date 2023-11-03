import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import random

reviews = pd.read_csv('Restaurant_Reviews.txt', on_bad_lines='skip')
nltk.download('stopwords')
ps = PorterStemmer()

comp = []
for i in range(716):
    review = re.sub('[^a-zA-Z]', ' ',reviews['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(kelime) for kelime in review if not kelime in set(stopwords.words('english'))]
    review = ' '.join(review)
    comp.append(review)


cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(comp).toarray()
y = reviews.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)








