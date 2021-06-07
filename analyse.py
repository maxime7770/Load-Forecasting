import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


regions = ["AUVERGNE RHONE ALPES", "BOURGOGNE FRANCHE COMTE", "BRETAGNE", "CENTRE VAL DE LOIRE", "GRAND EST", "HAUTS DE FRANCE",
           "ILE DE FRANCE", "NORMANDIE", "NOUVELLE AQUITAINE", "OCCITANIE", "PAYS DE LA LOIRE", "PROVENCE ALPES COTE D AZUR"]


def vis():
    fig, axs = plt.subplots(4, 3, figsize=(15, 13))
    axs = axs.ravel()
    for i in range(12):
        s = "dataset_centrale/data/train/%s.csv" % regions[i]
        f = pd.read_csv(s)
        axs[i].hist(f["Consommation"], bins=20)
        axs[i].set_title(regions[i])
    plt.show()


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

def mois():
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    axs = axs.ravel()
    l = ["BRETAGNE", "NORMANDIE"]
    t1, t2 = 6*1440, 7*1440
    mois = "Juillet"
    m = []
    for i in range(2):
        f = pd.read_csv(
            "dataset_centrale/data/train/%s.csv" % l[i])
        cons = f['Consommation']

        axs[i].plot(cons[t1:t2])
        axs[i].set_title('Mois de %s en %s' % (mois, l[i]))

        m.append(np.mean(cons))
    plt.show()
    print('Les consommations en moyenne pendant le mois de %s en %s et %s sont respectivement %s et %s' % (
        mois, l[0], l[1], m[0], m[1]))


mois()
