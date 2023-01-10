import pandas as pd

data = pd.read_csv("C:/Users/kekli/Desktop/şarap.csv")
veri=data.copy()
#print(veri)
#print(veri.isnull().sum())
#print(veri.info())

import matplotlib.pyplot as plt
import seaborn as sns

y=veri["quality"]
X=veri.drop(columns="quality",axis=1)
#print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#standartlaştırma normalizasyon bağımsz  değişkenleri

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA

pca=PCA()#n_components=2

X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

print(X_train.shape)
print(X_train2.shape)

import numpy as np

print(np.cumsum(pca.explained_variance_ratio_)*100)

from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt  

lm=LinearRegression()
lm.fit(X_train2,y_train)
tahmin=lm.predict(X_test2)

r2=mt.r2_score(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=True)

print("R2:{} RMSE:{}".format(r2,rmse))

#kaç paramatre ile çalışacağımızı bulma,Çarpraz doğrulama
#cross valudation 

from sklearn.model_selection import KFold,cross_val_score

cv=KFold(n_splits=10,shuffle=True,random_state=1) #genelde 10 ve karıştırma işin shuff seçilir

lm2=LinearRegression()
RMSE=[]

for i in range(1,X_train2.shape[1]+1):
    hata=np.sqrt(-1*cross_val_score(lm2,X_train[:,:i],y_train.ravel(),
    cv=cv,scoring="neg_mean_squared_error").mean())
    RMSE.append(hata)

plt.plot(RMSE,"-x")
plt.xlabel("Bileşen Sayısı")
plt.ylabel("RMSE")
plt.show()








