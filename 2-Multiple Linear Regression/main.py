import pandas as pd

veriler = pd.read_csv("veriler.txt")
ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

#[:,0] ilk kolonu alır
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #kolondaki değerleri sayısal değere dönüştürür

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#cinsiyet kolununu sayısal değere çevirme
c = veriler.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1]) #kolondaki değerleri sayısal değere dönüştürür

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr', 'tr', 'us'])

cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=['cinsiyet'])

s2 = pd.concat([sonuc, sonuc3], axis = 1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s2, sonuc3, test_size=0.33, random_state=0)


#### MODEL OLUŞTURMA #####

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

boy = s2.iloc[:3:4].values
print(boy)