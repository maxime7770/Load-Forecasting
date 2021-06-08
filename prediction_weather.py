import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# on utilisera dans la suite sm.ols pour les régressions, ça permet de voir plus facilement les paramètres qui ont
# de l'influence sur la consommation


region = "ILE DE FRANCE"

data_test = pd.read_csv('dataset_centrale/data/test/%s.csv' % region)
data_train = pd.read_csv('dataset_centrale/data/train/%s.csv' % region)

features_train = data_train.iloc[:, 2:9]
features_test = data_test.iloc[:, 2:9]

conso_train = data_train['Consommation']
conso_test = data_test['Consommation']

y_train = conso_train.values
y_test = conso_test.values
x_train = features_train.values
x_test = features_test.values


#scaler = StandardScaler().fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

ols = sm.OLS(y_train, x_train).fit()
print(ols.summary())

y_pred = ols.predict(x_test)

plt.scatter(y_test, y_pred)
plt.plot([1000, 12000], [1000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((y_test - y_pred) ** 2))))
plt.xlabel('expected')
plt.ylabel('predicted')
plt.show()

plt.plot(y_pred, label='prediction')
plt.plot(y_test, label='actual consumption')
plt.title('Predicted and expected consumptions')
plt.legend()
plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('MAPE pour régression linéaire : %s' %
      mean_absolute_percentage_error(y_test, y_pred))


# on trouve que certains coefficients ont beaucoup d'influence que d'autres :
# la température a une grande influence par exemple (x2)
# la pvalue du test de student correspondant à x5 est très élevée : ce coefficient a de grandes chances d'être nul
# x5 correspond au mois pendant lequel la consommation a été relevée.
# le paramètre x3 est à la limite d'être considéré comme non nul, il s'agit de prate (la pluie)


# maintenant on fait du feature engineering : on rajoute des covariables artificiellement

def av(weeks, data):  # moyennes sur les semaines précédentes
    t = weeks*7*24*2
    n = len(data)
    v = np.zeros(n)
    for i in range(n):
        v[i] = np.mean(data['Consommation'][i-t:i])
    return v[t:]


def threedays(d, data):  # consommation exactement à la meme heure, d jours avant
    n = len(data)
    v = np.zeros(n)
    for i in range(3*24*2, n):
        v[i] = data['Consommation'][i-3*24*2]
    return v

# print(threedays(data))


def day(data):
    n = len(data)
    v = np.zeros(n)
    f = pd.read_csv(
        'dataset_centrale/data/train/%s.csv' % region)['weekday']
    for i in range(n):
        v[i] = f[i]
    return v


# print(day(data))

region = "ILE DE FRANCE"

data_test = pd.read_csv('dataset_centrale/data/test/%s.csv' % region)
data_train = pd.read_csv('dataset_centrale/data/train/%s.csv' % region)

features_train = data_train.iloc[:, 2:9]
features_test = data_test.iloc[:, 2:9]

conso_train = data_train['Consommation']
conso_test = data_test['Consommation']


weeks = 3
d = 3

y_train = conso_train.values
y_test = conso_test.values
x_train = features_train.values
x_test = features_test.values

x8_train = av(weeks, data_train)
x9_train = threedays(d, data_train)[weeks*24*2*7:]
x10_train = day(data_train)[weeks*24*2*7:]

x8_test = av(weeks, data_test)
x9_test = threedays(d, data_test)[weeks*24*2*7:]
x10_test = day(data_test)[weeks*24*2*7:]

x_train = x_train[weeks*24*2*7:]
x_test = x_test[weeks*24*2*7:]

x_train = np.c_[x_train, x8_train, x9_train, x10_train]

x_test = np.c_[x_test, x8_test, x9_test, x10_test]


x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)


y_train = y_train[weeks*24*2*7:]
y_test = y_test[weeks*24*2*7:]

ols = sm.OLS(y_train, x_train).fit()
print(ols.summary())

y_pred = ols.predict(x_test)

plt.scatter(y_test, y_pred)
plt.plot([1000, 12000], [1000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((y_test - y_pred) ** 2))))
plt.xlabel('expected')
plt.ylabel('predicted')
plt.show()

plt.plot(y_pred, label='prediction')
plt.plot(y_test, label='actual consumption')
plt.title('Predicted and expected consumptions')

plt.legend()
plt.show()

print('MAPE pour régression linéaire avec features engineering : %s' %
      mean_absolute_percentage_error(y_test, y_pred))
