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


# voir https://medium.com/@phylypo/overview-of-time-series-forecasting-from-statistical-to-recent-ml-approaches-c51a5dd4656a

region = "BRETAGNE"

data = pd.read_csv(
    '../dataset_centrale/data/train/%s.csv' % region)['Consommation']

mape = []  # on va stocker les MAPE et les comparer pour différents modèles

# On effectue une régression pour prédire la consommation. On ne prend pas en compte les autres variables (températures,
# pluie, nuages)

# Comme dit dans les slides, j'ai pris comme covariables la moyenne sur les semaines passées, la valeur 3 jours avant,
# et le jour de la semaine où la consommation est mesurée :


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
        '../dataset_centrale/data/train/%s.csv' % region)['weekday']
    for i in range(n):
        v[i] = f[i]
    return v


# print(day(data))


weeks = 3
d = 3
x1 = av(weeks, data)
x2 = threedays(d, data)[weeks*24*2*7:]
x3 = day(data)[weeks*24*2*7:]
y = data[:]
y = y[weeks*24*2*7:]


x = np.array([x1, x2, x3]).T
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

lr = LinearRegression().fit(x, y)

data_test = pd.read_csv(
    '../dataset_centrale/data/test/%s.csv' % region)['Consommation']


x1_test = av(weeks, data_test)
x2_test = threedays(d, data_test)[weeks*24*2*7:]
x3_test = day(data_test)[weeks*24*2*7:]
expected = data_test[:]
expected = expected[weeks*24*2*7:].values
x_test = np.array([x1_test, x2_test, x3_test]).T
x_test = scaler.transform(x_test)


ypred = lr.predict(x_test)


# Ici on trace la valeur prédite en fonction de la valeur attendue
# Dans le cas idéal, on doit être très proche de la droite en pointillés

plt.scatter(expected, ypred)
plt.xlabel('expected')
plt.ylabel('predicted')
plt.plot([1000, 6000], [1000, 6000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((ypred - expected) ** 2))))
plt.show()
print('RMS de %s pour la régression linéaire' %
      np.sqrt(np.mean((ypred - expected) ** 2)))

# Maintenant on trace séparément ypred et expected : dans le cas idéal, les 2 courbes doivent être proches

plt.plot(ypred, label='predicted values')
plt.plot(expected, label='expected values')
plt.title('Predicted and expected consumptions')
plt.legend()
plt.show()


# la fonction suivante renvoie le MAPE

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print("MAPE pour la régression linéaire : %s" %
      mean_absolute_percentage_error(expected, ypred))
mape.append(mean_absolute_percentage_error(expected, ypred))

# essayons aussi avec une random forest

clf = RandomForestClassifier(n_estimators=20, max_depth=10).fit(x, y)
Y_pred = clf.predict(x_test)
RF = RandomForestRegressor(random_state=0)
# print(Y_pred)

plt.scatter(expected, Y_pred)
plt.xlabel('expected')
plt.ylabel('predicted')
plt.plot([1000, 6000], [1000, 6000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((Y_pred - expected) ** 2))))
print('RMS de %s pour la random forest' %
      np.sqrt(np.mean((Y_pred - expected) ** 2)))
plt.show()

plt.plot(Y_pred, label='predicted values')
plt.plot(expected, label='expected values')
plt.title('Predicted and expected consumptions')
plt.legend()
plt.show()

mape.append(mean_absolute_percentage_error(expected, Y_pred))

clf_rf = GridSearchCV(estimator=RF,
                      param_grid={'max_depth': [2, 3, 4, 5, 10, 15],
                                  'n_estimators': [10, 20, 30]},
                      cv=5,
                      scoring='neg_mean_squared_error')
clf_rf.fit(x, y)

md = clf_rf.best_params_['max_depth']
ne = clf_rf.best_params_['n_estimators']
y_ = clf_rf.predict(x_test)

print('md=%s,ne=%s' % (md, ne))

plt.scatter(expected, y_)
plt.xlabel('expected')
plt.ylabel('predicted')
plt.plot([1000, 6000], [1000, 6000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((y_ - expected) ** 2))))
print('RMS de %s pour la random forest corrigée' %
      np.sqrt(np.mean((y_ - expected) ** 2)))
plt.show()


mape.append(mean_absolute_percentage_error(expected, y_))

l = ['regression', 'random forest', 'optimized random forest']
for i in range(3):

    plt.plot([0, 1], [mape[i], mape[i]], label=l[i])
plt.title('MAPE coefficients for the 3 different models')
plt.legend()
plt.show()

# RMS de 1340.5749354301627 pour la régression linéaire
# RMS de 1858.5870966656416 pour la random forest
# md=10,ne=30
# RMS de 1604.2535162034355 pour la random forest corrigée

# meilleur MAPE pour régression, puis pour la random forest optimisée puis pour la random forest d'origine.
# [11.64, 16.5, 17.5] pour les MAPE
