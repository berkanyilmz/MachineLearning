# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:14:45 2022

@author: berga
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


class EmotionRecogniton:
    
    def __init__(self):
        self.data = pd.read_csv('training.csv')
        
    
    def preprocessing(self):
        ps = PorterStemmer()
        stems = []
        for i in range(16000):
            comment = re.sub('[^a-zA-Z]', ' ', self.data['text'][i])
            comment = comment.split()
            comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
            comment = ' '.join(comment)
            stems.append(comment)
        return stems
            
            
            
    def featureExtraction(self, stems):
        vectorizer = CountVectorizer(max_features=2000)
        
        X = vectorizer.fit_transform(stems).toarray()
        y = self.data.iloc[:,1].values
        return X,y
        
        
    def trainTestSplit(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        return x_train, x_test, y_train, y_test
        
        
    def fit_predict(self,x_train, x_test, y_train, y_test):
        svc = SVC()
        svc.fit(x_train, y_train)
        y_predict = svc.predict(x_test)
        return y_predict
    
    
    
    def confussionMatrix(self, y_test, y_predict):
        cm = confusion_matrix(y_test, y_predict)
        print('Confussion Matrix : ', cm)
        print('Accuracy Score : ', accuracy_score(y_test, y_predict))
        
        
        
        
    
emotionrecognition = EmotionRecogniton()
stems = emotionrecognition.preprocessing()
(x,y) = emotionrecognition.featureExtraction(stems)
(x_train, x_test, y_train, y_test) = emotionrecognition.trainTestSplit(x, y)
y_pred = emotionrecognition.fit_predict(x_train, x_test, y_train, y_test)
emotionrecognition.confussionMatrix(y_test, y_pred)