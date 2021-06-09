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
    '../dataset_centrale/data/train/%s.csv' % region)['Consommation']
data_test = pd.read_csv(
    '../dataset_centrale/data/test/%s.csv' % region)['Consommation']


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
        '../dataset_centrale/data/train/ILE DE FRANCE.csv')['weekday']
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


g = LinearGAM(s(0) + s(1) + f(2)).fit(x, y)


dtrain = pd.read_csv(
    '../dataset_centrale/data/train/%s.csv' % region, index_col=0)
dtest = pd.read_csv(
    '../dataset_centrale/data/test/%s.csv' % region, index_col=0)
d = pd.concat([dtrain, dtest])


def av_48h(weeks):  # debut='2017-10-01 00:00:00' par exemple pour la première fenêtre de 48h de l'ensemble de test
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

j = 3  # 0 pour 1ère fenetre de 48h, 1 pour la 1ère et la 2eme etc.
k = 0
expected_total = np.array([])
predicted_total = np.array([])
mape = []
while k <= j:
    c = dic1[k]
    x1_test = av_48h(weeks)
    x2_test = threedays_48h(weeks)
    x3_test = day(k)
    expected = data_test.values[:96]
    x_test = np.array([x1_test, x2_test, x3_test]).T
    x_test = scaler.transform(x_test)

    pred = g.predict(x_test)

    expected_total = np.concatenate((expected_total, expected))
    predicted_total = np.concatenate((predicted_total, pred))
    mape.append(mean_absolute_percentage_error(pred, expected))
    dic1[k+1] = np.concatenate((dic1[k], pred))
    k += 1

    # Ici on trace la valeur prédite en fonction de la valeur attendue
    # Dans le cas idéal, on doit être très proche de la droite en pointillés

plt.scatter(expected_total, predicted_total)
plt.xlabel('expected')
plt.ylabel('predicted')
m = min(np.min(expected_total), np.min(predicted_total))
M = max(np.max(expected_total), np.max(predicted_total))

plt.plot([m, M], [m, M], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean(
    (predicted_total - expected_total) ** 2)))+", for %s 48h windows" % (j+1))
plt.show()
print('RMS de %s pour GAM' %
      np.sqrt(np.mean((predicted_total - expected_total) ** 2)))

# Maintenant on trace séparément ypred et expected : dans le cas idéal, les 2 courbes doivent être proches

plt.plot(predicted_total, label='predicted values')
plt.plot(expected_total, label='expected values')
plt.title('Predicted and expected consumptions for %s 48h windows (GAM)' % (j+1))
plt.legend()
plt.show()


print("MAPE total pour GAM : %s" %
      mean_absolute_percentage_error(expected, pred))
print("MAPE moyen sur chaque fenêtre pour le GAM : %s" % (np.mean(mape)))


# on essaie de trouver le terme de pénalisation optimal (pénalisation l2)

lam = np.logspace(-3, 5, 5)
lams = [lam] * 3
opt_gam = LinearGAM(s(0) + s(1) + f(2)).gridsearch(x, y, lam=lams)

dic1 = {0: dtrain["Consommation"].values}

j = 3  # 0 pour 1ère fenetre de 48h, 1 pour la 1ère et la 2eme etc.
k = 0
expected_total = np.array([])
predicted_total = np.array([])
mape = []
while k <= j:
    c = dic1[k]
    x1_test = av_48h(weeks)
    x2_test = threedays_48h(weeks)
    x3_test = day(k)
    expected = data_test.values[:96]
    x_test = np.array([x1_test, x2_test, x3_test]).T
    x_test = scaler.transform(x_test)

    pred = opt_gam.predict(x_test)

    expected_total = np.concatenate((expected_total, expected))
    predicted_total = np.concatenate((predicted_total, pred))
    mape.append(mean_absolute_percentage_error(pred, expected))
    dic1[k+1] = np.concatenate((dic1[k], pred))
    k += 1

    # Ici on trace la valeur prédite en fonction de la valeur attendue
    # Dans le cas idéal, on doit être très proche de la droite en pointillés

plt.scatter(expected_total, predicted_total)
plt.xlabel('expected')
plt.ylabel('predicted')
m = min(np.min(expected_total), np.min(predicted_total))
M = max(np.max(expected_total), np.max(predicted_total))

plt.plot([m, M], [m, M], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean(
    (predicted_total - expected_total) ** 2)))+", for %s 48h windows" % (j+1))
plt.show()
print('RMS de %s pour opt GAM' %
      np.sqrt(np.mean((predicted_total - expected_total) ** 2)))

# Maintenant on trace séparément ypred et expected : dans le cas idéal, les 2 courbes doivent être proches

plt.plot(predicted_total, label='predicted values')
plt.plot(expected_total, label='expected values')
plt.title('Predicted and expected consumptions for %s 48h windows (opt GAM)' % (j+1))
plt.legend()
plt.show()


print("MAPE total pour le GAM optimisé : %s" %
      mean_absolute_percentage_error(expected, pred))
print("MAPE moyen sur chaque fenêtre pour le GAM optimisé : %s" % (np.mean(mape)))


lams = np.random.rand(100, 3)  # random points on [0, 1], with shape (100, 3)
lams = lams * 6 - 3  # shift values to -3, 3
lams = 10 ** lams  # transforms values to 1e-3, 1e3
random_gam = LinearGAM(s(0) + s(1) + f(2)).gridsearch(x, y, lam=lams)


dic1 = {0: dtrain["Consommation"].values}

j = 3  # 0 pour 1ère fenetre de 48h, 1 pour la 1ère et la 2eme etc.
k = 0
expected_total = np.array([])
predicted_total = np.array([])
mape = []
while k <= j:
    c = dic1[k]
    x1_test = av_48h(weeks)
    x2_test = threedays_48h(weeks)
    x3_test = day(k)
    expected = data_test.values[:96]
    x_test = np.array([x1_test, x2_test, x3_test]).T
    x_test = scaler.transform(x_test)

    pred = random_gam.predict(x_test)

    expected_total = np.concatenate((expected_total, expected))
    predicted_total = np.concatenate((predicted_total, pred))
    mape.append(mean_absolute_percentage_error(pred, expected))
    dic1[k+1] = np.concatenate((dic1[k], pred))
    k += 1

    # Ici on trace la valeur prédite en fonction de la valeur attendue
    # Dans le cas idéal, on doit être très proche de la droite en pointillés

plt.scatter(expected_total, predicted_total)
plt.xlabel('expected')
plt.ylabel('predicted')
m = min(np.min(expected_total), np.min(predicted_total))
M = max(np.max(expected_total), np.max(predicted_total))

plt.plot([m, M], [m, M], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean(
    (predicted_total - expected_total) ** 2)))+", for %s 48h windows" % (j+1))
plt.show()
print('RMS de %s pour rand GAM' %
      np.sqrt(np.mean((predicted_total - expected_total) ** 2)))

# Maintenant on trace séparément ypred et expected : dans le cas idéal, les 2 courbes doivent être proches

plt.plot(predicted_total, label='predicted values')
plt.plot(expected_total, label='expected values')
plt.title('Predicted and expected consumptions for %s 48h windows (rand GAM)' % (j+1))
plt.legend()
plt.show()


print("MAPE total pour le GAM optimisé : %s" %
      mean_absolute_percentage_error(expected, pred))
print("MAPE moyen sur chaque fenêtre pour le rand GAM : %s" % (np.mean(mape)))
