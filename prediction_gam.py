from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy as sp
from pygam import LinearGAM, s, f


# on utilise les mêmes features que dans le document prediction_no_weather

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


region = "ILE DE FRANCE"
data = pd.read_csv(
    'dataset_centrale/data/train/%s.csv' % region)['Consommation']
data_test = pd.read_csv(
    'dataset_centrale/data/test/%s.csv' % region)['Consommation']


def av(weeks, data):  # moyennes sur les semaines précédentes
    t = weeks*7*24*2
    n = len(data)
    v = np.zeros(n)
    for i in range(n):
        v[i] = np.mean(data[i-t:i])
    return v[t:]


def threedays(d, data):  # consommation exactement à la meme heure, d jours avant
    n = len(data)
    v = np.zeros(n)
    for i in range(3*24*2, n):
        v[i] = data[i-3*24*2]
    return v

# print(threedays(data))


def day(data):
    n = len(data)
    v = np.zeros(n)
    f = pd.read_csv(
        'dataset_centrale/data/train/ILE DE FRANCE.csv')['weekday']
    for i in range(n):
        v[i] = f[i]
    return v


weeks = 3
d = 3

# données d'entrainement

x1 = av(weeks, data)
x2 = threedays(d, data)[weeks*24*2*7:]
x3 = day(data)[weeks*24*2*7:]
y = data[:]
y = y[weeks*24*2*7:]

x = np.array([x1, x2, x3]).T
scaler = StandardScaler().fit(x)
x = scaler.transform(x)


# données de test

x1_test = av(weeks, data_test)
x2_test = threedays(d, data_test)[weeks*24*2*7:]
x3_test = day(data_test)[weeks*24*2*7:]
expected = data_test[:]
expected = expected[weeks*24*2*7:].values
x_test = np.array([x1_test, x2_test, x3_test]).T
x_test = scaler.transform(x_test)


x = np.array([x1, x2, x3]).T
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# prédiction :

g = LinearGAM(s(0) + s(1) + f(2)).fit(x, y)

# print(g.summary())

pred = g.predict(x_test)

plt.scatter(expected, pred)
plt.plot([4000, 12000], [4000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((pred - expected) ** 2))))
plt.ylabel('expected')
plt.xlabel('predicted')

plt.show()


print("MAPE pour le GAM : %s" % mean_absolute_percentage_error(expected, pred))


# on essaie de trouver le terme de pénalisation optimal (pénalisation l2)

lam = np.logspace(-3, 5, 5)
lams = [lam] * 3
opt_gam = LinearGAM(s(0) + s(1) + f(2)).gridsearch(x, y, lam=lams)


pred_opt = opt_gam.predict(x_test)

plt.scatter(expected, pred_opt)
plt.plot([4000, 12000], [4000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((pred_opt - expected) ** 2))))
plt.ylabel('expected')
plt.xlabel('predicted')

plt.show()

print("MAPE pour le GAM optimisé : %s" %
      mean_absolute_percentage_error(expected, pred_opt))


lams = np.random.rand(100, 3)  # random points on [0, 1], with shape (100, 3)
lams = lams * 6 - 3  # shift values to -3, 3
lams = 10 ** lams  # transforms values to 1e-3, 1e3
random_gam = LinearGAM(s(0) + s(1) + f(2)).gridsearch(x, y, lam=lams)

pred_rand = random_gam.predict(x_test)

plt.scatter(expected, pred_rand)
plt.plot([4000, 12000], [4000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((pred_rand - expected) ** 2))))
plt.ylabel('expected')
plt.xlabel('predicted')

print("MAPE pour le random GAM : %s" %
      mean_absolute_percentage_error(expected, pred_rand))


# pour ile de France on obtient :
# MAPE pour le GAM : 13.729216068500016
# MAPE pour le GAM optimisé : 13.721626102151669
# MAPE pour le random GAM : 13.709740352605523
