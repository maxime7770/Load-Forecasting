import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy as sp


regions = ["AUVERGNE RHONE ALPES", "BOURGOGNE FRANCHE COMTE", "BRETAGNE", "CENTRE VAL DE LOIRE", "GRAND EST", "HAUTS DE FRANCE",
           "ILE DE FRANCE", "NORMANDIE", "NOUVELLE AQUITAINE", "OCCITANIE", "PAYS DE LA LOIRE", "PROVENCE ALPES COTE D AZUR"]


def visualization():
    fig, axs = plt.subplots(4, 3, figsize=(15, 13))
    axs = axs.ravel()
    for i in range(12):
        s = "dataset_centrale/data/train/%s.csv" % regions[i]
        f = pd.read_csv(s)
        axs[i].hist(f["Consommation"], bins=20)
        axs[i].set_title(regions[i])
    plt.show()


#visualization()

# plt.hist(pd.read_csv("dataset_centrale/data/train/OCCITANIE.csv")["Consommation"])
# plt.show()


def moyennes():
    for i in range(12):
        f = pd.read_csv("dataset_centrale/data/train/%s.csv" % regions[i])
        plt.plot([np.mean(f["Consommation"])]*100, label=regions[i])
        plt.legend()
        plt.title('Consommation moyenne dans chacune des régions')
    plt.show()


# moyennes()


def ecarts_type():

    for i in range(12):
        f = pd.read_csv("dataset_centrale/data/train/%s.csv" % regions[i])
        plt.plot([np.sqrt(np.var(f["Consommation"]))]*100, label=regions[i])
        plt.legend()
        plt.title('Ecarts types de la consommation dans chacune des régions')
    plt.show()


# ecarts_type()


def medians():
    for i in range(12):
        f = pd.read_csv("dataset_centrale/data/train/%s.csv" % regions[i])
        plt.plot([np.median(f["Consommation"])]*100, label=regions[i])
        plt.legend()
        plt.title('Consommation médiane dans chacune des régions')
    plt.show()


# medians()

# pour prendre les données qui correspondent à un mois en particulier pour une région
def extraction(debut, fin, fichier):
    df = pd.read_csv(fichier, index_col=0)
    d = '{} 00:00:00'.format(debut)
    f = '{} 23:30:00'.format(fin)
    data = df.loc[d:f]
    conso = np.array(data['Consommation'])
    return conso


def mois(debut, fin):
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    axs = axs.ravel()
    # mettre ici les 2 régions à comparer
    l = ["NORMANDIE", "PROVENCE ALPES COTE D AZUR"]

    m = []
    for i in range(2):
        f = "dataset_centrale/data/train/%s.csv" % l[i]

        axs[i].plot(extraction(debut, fin, f))
        axs[i].set_title('Consommaion de %s à %s en %s' % (debut, fin, l[i]))

        m.append(np.mean(extraction(debut, fin, f)))
    plt.show()
    print('Les consommations en moyenne pendant le mois de %s en %s et %s sont respectivement %s et %s' % (
        mois, l[0], l[1], m[0], m[1]))


# mois('2016-07-01', '2016-08-01')

dic = {"tmp2m": "Température", "prate": "Pluie", "tcdcclm": "Nuages"}


def var_exogenes():
    f = pd.read_csv(
        "dataset_centrale/data/train/ILE DE FRANCE.csv")

    # fig, axs = plt.subplots(1, 3, figsize=(20, 16))
    # axs = axs.ravel()
    l = ["tmp2m", "prate", "tcdcclm"]

    for i in range(3):
        plt.scatter(f[l[i]][::15], f["Consommation"][::15])
        plt.ylabel('Consommation')
        plt.xlabel(dic[l[i]])

        plt.show()


# var_exogenes()

# aucune influence des nuages sur la consommation, et très peu d'influence de la quantité de pluie qui tombe
# par contre, influence de la température est notable

def regression(var):
    f = pd.read_csv("dataset_centrale/data/train/BRETAGNE.csv")
    X = f[var]
    X = sm.add_constant(X)
    y = f["Consommation"]

    ols = sm.OLS(y, X).fit()
    print(ols.summary())

    ypred = ols.predict(X)
    plt.scatter(f[var][::30], f["Consommation"][::30])
    plt.plot(f[var], ypred, 'r', linewidth=2)
    plt.title('$R^2$ = {:.2f}'.format(ols.rsquared))
    plt.xlabel(dic[var])
    plt.ylabel('Consommation')
    plt.show()


#regression("tmp2m")
