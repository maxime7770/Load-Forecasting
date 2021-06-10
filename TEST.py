
import pandas as pd
import numpy as np
import statsmodels.api as sm

region = "ILE DE FRANCE"

data_test = pd.read_csv(
    'dataset_centrale/data/test/%s.csv' % region, index_col=0)
data_train = pd.read_csv(
    'dataset_centrale/data/train/%s.csv' % region, index_col=0)

data = pd.concat([data_train, data_test])

print(data)

print(data.loc['2017-10-01 00:00:00'])


dtrain = pd.read_csv(
    'dataset_centrale/data/train/%s.csv' % region, index_col=0)
dtest = pd.read_csv(
    'dataset_centrale/data/test/%s.csv' % region, index_col=0)
d = pd.concat([dtrain, dtest])

c = dtrain["Consommation"].values


def av_48h(weeks):  # debut='2017-10-01 00:00:00' par exemple pour la première fenêtre de 48h de l'ensemble de test
    t = (weeks+1)*7*24*2
    n = 96
    v = np.zeros(n)
    p = len(c)
    for i in range(n):
        v[i] = np.mean(c[p-t+i:p-7*24*2+i])
    return v


print(av_48h(2))


def day(j):  # pour la jème fenetee de 48h de l'ensemble de test
    v = np.zeros(96)
    for i in range(96):
        v[i] = dtest['weekday'].values[j*48+i]
    return v


print(day(1))
print(len([x for x in day(1) if x == 0]))


v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(v.shape)

v = sm.add_constant(v)
print(v.shape)
