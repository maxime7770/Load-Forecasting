import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy as sp
from sklearn.preprocessing import StandardScaler

# voir https://medium.com/@phylypo/overview-of-time-series-forecasting-from-statistical-to-recent-ml-approaches-c51a5dd4656a

data = pd.read_csv(
    'dataset_centrale/data/train/ILE DE FRANCE.csv')['Consommation']


def av(weeks, data):  # moyennes sur les semaines précédentes
    t = weeks*7*24*2
    n = len(data)
    v = np.zeros(n)
    for i in range(n):
        v[i] = np.mean(data[i-t:i])
    return v[t:]


def threedays(data):  # consommation exactement à la meme heure, 3 jours avant
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


# print(day(data))


weeks = 1
x1 = av(weeks, data)
x2 = threedays(data)[weeks*24*2*7:]
x3 = day(data)[weeks*24*2*7:]
y = data[:]
y = y[weeks*24*2*7:]
print(x1.shape)
print(x2.shape)
print(x3.shape)

x = np.array([x1, x2, x3]).T
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
print(x.shape)

lr = LinearRegression().fit(x, y)

data_test = pd.read_csv(
    'dataset_centrale/data/test/ILE DE FRANCE.csv')['Consommation']


x1_test = av(weeks, data_test)
x2_test = threedays(data_test)[weeks*24*2*7:]
x3_test = day(data_test)[weeks*24*2*7:]
expected = data_test[:]
expected = expected[weeks*24*2*7:]
x_test = np.array([x1_test, x2_test, x3_test]).T
x_test = scaler.transform(x_test)

print(x_test.shape)

ypred = lr.predict(x_test)


plt.scatter(expected, ypred)
plt.plot([0, 12000], [0, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((ypred - expected) ** 2))))
plt.show()

plt.plot(ypred, label='valeur prédite')
plt.plot(expected, label='valeur attendue')
plt.legend()
plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print(mean_absolute_percentage_error(expected, ypred))
