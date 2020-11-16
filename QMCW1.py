# referance:
# https://realpython.com/numpy-scipy-pandas-correlation-python/
# https://raphaelvallat.com/correlation.html

## import packages which will be used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr


## data correlation analysis
#import data
df = pd.read_csv('D:/CASA007/assignment1/QMCW1_4.csv')
print('%i subjects and %i columns' % df.shape)
#print(df)

## scatter plot
pd.plotting.scatter_matrix(df,alpha=0.5,figsize=(10,8),grid=False,diagonal='kde',marker='o',range_padding=0.1)
# plt.show()

## R2
corr_pd=df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_pd,cmap='YlGnBu')
plt.show()

df.corr()

## creat x and y
X =df[['clean_air','clean_environ','health','school', 'media','counselling']];
print(type(X)," ",X.shape)
Y=df['k']
print(type(Y)," ",Y.shape)

## Checking for Linearity
plt.scatter(df['clean_environ'], df['k'], color='red')
plt.title('clean_environ & k', fontsize=14)
plt.xlabel('clean_environ', fontsize=14)
plt.ylabel('k', fontsize=14)
plt.grid(True)

plt.scatter(df['clean_air'], df['k'], color='red')
plt.title('clean_air & k', fontsize=14)
plt.xlabel('clean_air', fontsize=14)
plt.ylabel('k', fontsize=14)
plt.grid(True)

## divide data set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
print(X_train.shape," ",X_test.shape," ",Y_train.shape," ",Y_test.shape)

## linear regression
lrg=LinearRegression()
model=lrg.fit(X_train,Y_train)
print(model)
print(format(lrg.intercept_ ,'.10f')) ##INPUT INTERCEPT
coef=zip(['clean_air','clean_environ','health','school', 'media','counselling'],lrg.coef_)
for T in coef :
    print(T) ##INPUT COEFFICIENT

##predict
y_pred=lrg.predict(X_test)
print(y_pred)

##
lm=ols('k~clean_air+clean_environ+health+school+media+counselling',data=df).fit()
print(lm.summary())
#evaluate using RMES
print("predict")
print(type(y_pred),type(Y_test))
print(len(y_pred),len(Y_test))
sum_mean=0;
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-Y_test.values[i])**2
sum_erro=np.sqrt(sum_mean/len(y_pred))
print("RMES:",format(sum_erro,'.10f'))
##plotting
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(Y_test)),Y_test,'r',label="test")
plt.xlabel("budget per person")
plt.ylabel("variation of the rate")
plt.legend()
plt.show()


## data correlation analysis
#import data
df = pd.read_csv('D:/CASA007/assignment1/QMCW1_5.csv')
print('%i subjects and %i columns' % df.shape)
#print(df)

## scatter plot
pd.plotting.scatter_matrix(df,alpha=0.5,figsize=(10,8),grid=False,diagonal='kde',marker='o',range_padding=0.1)
# plt.show()

## R2
corr_pd=df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_pd,cmap='YlGnBu')
plt.show()

df.corr()

## creat x and y
X =df[['clean_air','health','school']];
print(type(X)," ",X.shape)
Y=df['k']
print(type(Y)," ",Y.shape)

## Checking for Linearity
plt.scatter(df['clean_air'], df['k'], color='red')
plt.title('clean_air & k', fontsize=14)
plt.xlabel('clean_air', fontsize=14)
plt.ylabel('k', fontsize=14)
plt.grid(True)

## divide data set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
print(X_train.shape," ",X_test.shape," ",Y_train.shape," ",Y_test.shape)

## linear regression 2
lrg=LinearRegression()
model=lrg.fit(X_train,Y_train)
print(model)
print(format(lrg.intercept_ ,'.10f')) ##INPUT INTERCEPT
coef=zip(['clean_air','health','school'],lrg.coef_)
for T in coef :
    print(T) ##INPUT COEFFICIENT

##predict
y_pred=lrg.predict(X_test)
print(y_pred)

##
lm=ols('k~clean_air+health+school',data=df).fit()
print(lm.summary())
#evaluate using RMES
print("predict")
print(type(y_pred),type(Y_test))
print(len(y_pred),len(Y_test))
sum_mean=0;
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-Y_test.values[i])**2
sum_erro=np.sqrt(sum_mean/len(y_pred))
print("RMES:",format(sum_erro,'.10f'))
##plotting
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(Y_test)),Y_test,'r',label="test")
plt.xlabel("budget per person")
plt.ylabel("variation of the rate")
plt.legend()
plt.show()


