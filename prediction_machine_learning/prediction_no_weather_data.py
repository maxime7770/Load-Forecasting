import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats.stats import median_absolute_deviation
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# UTILISATION : changer le paramètre j défini plus bas pour changer le nombre de fenêtres de 48h sur lesquelles
# on observe la prédiction (j=0 -> 1 fenêtre, j=1 -> 2 fenêtres=les 4 premiers jours de la fenêtre de test)

# voir https://medium.com/@phylypo/overview-of-time-series-forecasting-from-statistical-to-recent-ml-approaches-c51a5dd4656a


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


region = "BRETAGNE"

data_train = pd.read_csv(
    '../dataset_centrale/data/train/%s.csv' % region)['Consommation']

mape = []  # on va stocker les MAPE et les comparer pour différents modèles

# On effectue une régression pour prédire la consommation. On ne prend pas en compte les autres variables (températures,
# pluie, nuages)

# Comme dit dans les slides, j'ai pris comme covariables la moyenne sur les semaines passées, la valeur 3 jours avant,
# et le jour de la semaine où la consommation est mesurée :


def av(weeks, data):  # moyennes sur les semaines précédentes (si weeks=3, on regarde les 3 semaines strictement avant)
    t = (weeks+1)*7*24*2
    n = len(data)
    v = np.zeros(n)
    for i in range(n):
        v[i] = np.mean(data[i-t:i-7*24*2])
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
        '../dataset_centrale/data/train/%s.csv' % region)['weekday']
    for i in range(n):
        v[i] = f[i]
    return v


# print(day(data))


weeks = 3
d = 3
x1_train = av(weeks, data_train)
x2_train = threedays(d, data_train)[(weeks+1)*24*2*7:]
x3_train = day(data_train)[(weeks+1)*24*2*7:]
y_train = data_train[:]
y_train = y_train[(weeks+1)*24*2*7:]


x_train = np.array([x1_train, x2_train, x3_train]).T
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

lr = LinearRegression().fit(x_train, y_train)

data_test = pd.read_csv(
    '../dataset_centrale/data/test/%s.csv' % region)['Consommation']


# prédiction sur les 48 premières heures de l'ensemble de test

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

j = 20  # 0 pour 1ère fenetre de 48h, 1 pour la 1ère et la 2eme etc.
k = 0
expected_total = np.array([])
predicted_total = np.array([])
mape = []
while k <= j:
    c = dic1[k]
    x1_test = av_48h(weeks)
    x2_test = threedays_48h(weeks)
    x3_test = day(k)
    #expected = data_test[:]
    #expected = expected[(weeks+1)*24*2*7:].values
    expected = data_test.values[k*96:(k+1)*96]
    x_test = np.array([x1_test, x2_test, x3_test]).T
    x_test = scaler.transform(x_test)

    ypred = lr.predict(x_test)

    expected_total = np.concatenate((expected_total, expected))
    predicted_total = np.concatenate((predicted_total, ypred))
    mape.append(mean_absolute_percentage_error(ypred, expected))
    dic1[k+1] = np.concatenate((dic1[k], ypred))
    k += 1

    # Ici on trace la valeur prédite en fonction de la valeur attendue
    # Dans le cas idéal, on doit être très proche de la droite en pointillés

plt.scatter(expected_total, predicted_total)
plt.xlabel('expected')
plt.ylabel('predicted')
plt.plot([1000, 6000], [1000, 6000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean(
    (predicted_total - expected_total) ** 2)))+", for %s 48h windows" % (j+1))
plt.show()
print('RMS de %s pour la régression linéaire' %
      np.sqrt(np.mean((predicted_total - expected_total) ** 2)))

# Maintenant on trace séparément ypred et expected : dans le cas idéal, les 2 courbes doivent être proches

plt.plot(predicted_total, label='predicted values')
plt.plot(expected_total, label='expected values')
plt.title('Predicted and expected consumptions for %s 48h windows' % (j+1))
plt.legend()
plt.show()


# la fonction suivante renvoie le MAPE


print("MAPE total pour la régression linéaire : %s" %
      mean_absolute_percentage_error(expected, ypred))
print("MAPE moyen sur chaque fenêtre pour la régression : %s" % (np.mean(mape)))
