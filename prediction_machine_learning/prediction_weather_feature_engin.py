
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


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def av(weeks, data):  # moyennes sur les semaines précédentes (si weeks=3, on regarde les 3 semaines strictement avant)
    t = (weeks+1)*7*24*2
    n = len(data)
    v = np.zeros(n)
    for i in range(n):
        v[i] = np.mean(data['Consommation'][i-t:i-7*24*2])
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
        '../dataset_centrale/data/train/%s.csv' % region)['weekday']
    for i in range(n):
        v[i] = f[i]
    return v


# print(day(data))

region = "ILE DE FRANCE"

data_test = pd.read_csv('../dataset_centrale/data/test/%s.csv' % region)
data_train = pd.read_csv('../dataset_centrale/data/train/%s.csv' % region)

features_train = data_train.iloc[:, 2:9]
features_test = data_test.iloc[:, 2:9]

conso_train = data_train['Consommation']
conso_test = data_test['Consommation']


weeks = 3
d = 3

y_train = conso_train.values
y_test = conso_test.values
x_train = features_train.values

x8_train = av(weeks, data_train)
x9_train = threedays(d, data_train)[(weeks+1)*24*2*7:]
x10_train = day(data_train)[(weeks+1)*24*2*7:]


x_train = x_train[(weeks+1)*24*2*7:]

x_train = np.c_[x_train, x8_train, x9_train, x10_train]


x_train = sm.add_constant(x_train)


y_train = y_train[(weeks+1)*24*2*7:]


ols = sm.OLS(y_train, x_train).fit()
print(ols.summary())

dtrain = pd.read_csv(
    '../dataset_centrale/data/train/%s.csv' % region, index_col=0)
dtest = pd.read_csv(
    '../dataset_centrale/data/test/%s.csv' % region, index_col=0)
d = pd.concat([dtrain, dtest])


def av_48h(weeks):
    t = (weeks+1)*7*24*2
    n = 96
    v = np.zeros(n)
    p = len(c)
    for i in range(n):
        v[i] = np.mean(c[p-t+i:p-7*24*2+i])
    return v


def threedays_48h(weeks):
    n = 96
    v = np.zeros(n)
    p = len(c)
    for i in range(n):
        v[i] = c[p+i-3*24*2]
    return v


def day(j):  # pour la jème fenetee de 48h de l'ensemble de test
    v = np.zeros(96)
    for i in range(96):
        v[i] = dtest['weekday'].values[j*48+i]
    return v


dic1 = {0: dtrain["Consommation"].values}

j = 30  # 0 pour 1ère fenetre de 48h, 1 pour la 1ère et la 2eme etc.
k = 0
expected_total = np.array([])
predicted_total = np.array([])
mape = []
while k <= j:
    x_test = features_test.values[k*96:(k+1)*96]
    c = dic1[k]
    x8_test = av_48h(weeks)
    x9_test = threedays_48h(weeks)
    x10_test = day(k)
    x_test = np.c_[x_test, x8_test, x9_test, x10_test]
    x_test = sm.add_constant(x_test, has_constant='add')
    #expected = data_test[:]
    #expected = expected[(weeks+1)*24*2*7:].values
    y_test = data_test['Consommation'].values[k*96:(k+1)*96]

    ypred = ols.predict(x_test)

    expected_total = np.concatenate((expected_total, y_test))
    predicted_total = np.concatenate((predicted_total, ypred))
    mape.append(mean_absolute_percentage_error(ypred, y_test))
    dic1[k+1] = np.concatenate((dic1[k], ypred))
    k += 1


plt.scatter(expected_total, predicted_total)
plt.plot([1000, 12000], [1000, 12000], '--k')
plt.title("RMS: {:.2f}".format(
    np.sqrt(np.mean((expected_total - predicted_total) ** 2)))+", for %s 48h windows" % (j+1))
plt.xlabel('expected')
plt.ylabel('predicted')
plt.show()

plt.plot(predicted_total, label='prediction')
plt.plot(expected_total, label='actual consumption')
plt.title('Predicted and expected consumptions for %s 48h windows' % (j+1))

plt.legend()
plt.show()

print('MAPE pour régression linéaire avec features engineering : %s' %
      mean_absolute_percentage_error(expected_total, predicted_total))
print('MAPE moyen sur chaque fenêtre pour régression linéaire avec features engineering : %s' % np.mean(mape))
