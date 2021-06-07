#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import math 


# In[154]:


#importe le datagramme de la région: mettre le nom exact du fichier excel (/!\ aux majuscules)
def region_data(region):
    df = pd.read_csv('REGION/Train/{}.csv'.format(region),index_col=0)
    return df


# In[192]:


df = pd.read_csv('REGION/Train/BRETAGNE.csv',index_col=0)
df


# In[155]:


#on supprime les NaN (j'en avais trouvé un dans la région Bretagne)
def clean_data(region):    
    df = region_data(region)
    donne=df["Consommation"].iloc[:]
    indice_to_remove=[]
    for i in range(len(donne)):
        if math.isnan(donne[i]) == True:
            indice_to_remove.append(i)
    for j in range(len(indice_to_remove)):
        df.drop(df.index[i], inplace=True) 
    return df


# In[195]:


#on commence par regarder les tendances saisonnières
def saison(region):
    df=clean_data(region)
    
    Hiver_2016=df.loc[:'2016-03-20 23:30:00']
    Hiv2016=np.array(Hiver_2016['Consommation'])
    plt.subplot(2,2,1)
    plt.plot(Hiv2016)
    plt.xticks([1464*x for x in range(4)], range(4))
    plt.title("Evolution de la consommation d'éléectricité à l'Hiver 2016")
    plt.xlabel("Mois")
    plt.ylabel("Consommation (MW)")

    Printemps_2016=df.loc['2016-03-20 00:00:00':'2016-06-20 23:30:00'] 
    Print2016=np.array(Printemps_2016['Consommation'])
    plt.subplot(2,2,2)
    plt.plot(Print2016)
    plt.xticks([1464*x for x in range(4)], range(4))
    plt.title("Evolution de la consommation d'électricité au Printemps 2016")
    plt.xlabel("Mois")
    plt.ylabel("Consommation (MW)")

    Ete_2016=df.loc['2016-06-21 00:00:00':'2016-09-21 23:30:00']
    Ete2016=np.array(Ete_2016['Consommation'])
    plt.subplot(2,2,3)
    plt.plot(Ete2016)
    plt.xticks([1464*x for x in range(4)], range(4))
    plt.title("Evolution de la consommation d'électricité à l'été 2016")
    plt.xlabel("Mois")
    plt.ylabel("Consommation (MW)")

    if len(df)>17040: #24*2*355=17040 : on compte le nombre de jours entre le 1er janvier et le 21 décembre car les données de toutes les regions ne vont pas jusqu'au 21 décembre
        Automne_2016=df.loc['2016-09-22 00:00:00':'2016-12-20 23:30:00']
        Aut2016=np.array(Automne_2016['Consommation'])
        plt.subplot(2,2,4)
        plt.plot(Aut2016)
        plt.xticks([1464*x for x in range(4)], range(4))
        plt.title("Evolution de la consommation d'électricité à l'Automne 2016")
        plt.xlabel("Mois")
        plt.ylabel("Consommation (MW)")
    else:
        Automne_2016=df.loc['2016-09-22 00:00:00':]
        Aut2016=np.array(Automne_2016['Consommation'])
        plt.subplot(2,2,4)
        plt.plot(Aut2016)
        plt.xticks([1464*x for x in range(4)], range(4))
        plt.title("Evolution de la consommation d'électricité à l'Automne 2016")
        plt.xlabel("Mois")
        plt.ylabel("Consommation (MW)")

    
    plt.show()




# In[196]:


saison('AUVERGNE RHONE ALPES')


# In[24]:


#On regarde maintenant sur un mois en comparant un mois normal (Janvier) à un mois de vacances (Juillet)
def mois(region):
    df=clean_data(region)

    Janvier_2016=df.loc['2016-01-01 00:00:00':'2016-01-31 23:30:00']
    Janvier2016=np.array(Janvier_2016['Consommation'])
    plt.subplot(1,2,1)
    plt.plot(Janvier2016)
    plt.xticks([336*x for x in range(5)], range(5))
    plt.title("Evolution de la consommation d'éléectricité en Janvier 2016")
    plt.xlabel("Semaine")
    plt.ylabel("Consommation (MW)")

    Juillet_2016=df.loc['2016-07-01 00:00:00':'2016-07-31 23:30:00']
    Juillet2016=np.array(Juillet_2016['Consommation'])
    plt.subplot(1,2,2)
    plt.xticks([336*x for x in range(5)], range(5))
    plt.plot(Juillet2016)
    plt.title("Evolution de la consommation d'éléctricité en Juillet 2016")
    plt.xlabel("Semaine")
    plt.ylabel("Consommation (MW)")


# In[194]:


mois('BRETAGNE')


# In[27]:


#On regarde maintenant sur 2 semaine différentes (du Lundi 4 Janvier 2016 au Dimanche 10 Janvier 2016 et du lundi 11 Janvier au Dimanche 17)
def semaine (region):
    df=clean_data(region)
    
    Semaine1=df.loc['2016-01-04 00:00:00':'2016-01-10 23:30:00']
    S1=np.array(Semaine1['Consommation'])

    Semaine2=df.loc['2016-01-11 00:00:00':'2016-01-17 23:30:00']
    S2=np.array(Semaine2['Consommation'])

    plt.subplot(1,2,1)
    plt.plot(S1)
    plt.xticks([48*x for x in range(8)], range(8))

    plt.title("Evolution de la consommation au cours de la semaine 1")
    plt.xlabel("jour de la semaine")
    plt.ylabel("Consommation (MW)")

    plt.subplot(1,2,2)
    plt.plot(S2)
    plt.xticks([48*x for x in range(8)], range(8))
    plt.title("Evolution de la consommation au cours de la semaine 1")
    plt.xlabel("jour de la semaine")
    plt.ylabel("Consommation (MW)")


# In[28]:


semaine('AUVERGNE RHONE ALPES')


# In[29]:


#On regarde l'évolution au cours des différents jours de la semaine
def Jour(region):
    df=clean_data(region)
    Jour_1=df.loc['2016-01-04 00:00:00':'2016-01-04 23:30:00']
    Jour1=np.array(Jour_1['Consommation'])
    plt.subplot(3,3,1)
    plt.plot(Jour1)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Lundi")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")

    Jour_2=df.loc['2016-01-05 00:00:00':'2016-01-05 23:30:00']
    Jour2=np.array(Jour_2['Consommation'])
    plt.subplot(3,3,2)
    plt.plot(Jour2)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Mardi")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")

    Jour_3=df.loc['2016-01-06 00:00:00':'2016-01-06 23:30:00']
    Jour3=np.array(Jour_3['Consommation'])
    plt.subplot(3,3,3)
    plt.plot(Jour3)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Mercredi")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")

    Jour_4=df.loc['2016-01-07 00:00:00':'2016-01-07 23:30:00']
    Jour4=np.array(Jour_4['Consommation'])
    plt.subplot(3,3,4)
    plt.plot(Jour4)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Jeudi")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")

    Jour_5=df.loc['2016-01-08 00:00:00':'2016-01-08 23:30:00']
    Jour5=np.array(Jour_5['Consommation'])
    plt.subplot(3,3,5)
    plt.plot(Jour5)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Vendredi")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")

    Jour_6=df.loc['2016-01-05 00:00:00':'2016-01-05 23:30:00']
    Jour6=np.array(Jour_6['Consommation'])
    plt.subplot(3,3,6)
    plt.plot(Jour6)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Samedi")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")

    Jour_7=df.loc['2016-01-10 00:00:00':'2016-01-10 23:30:00']
    Jour7=np.array(Jour_7['Consommation'])
    plt.subplot(3,3,7)
    plt.plot(Jour7)
    plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
    plt.title("Dimanche")
    plt.xlabel("Heures")
    plt.ylabel("Consommation (MW)")


# In[30]:


Jour('AUVERGNE RHONE ALPES')


# In[31]:


def extraction(debut,fin,fichier):
    df=pd.read_csv(fichier,index_col=0)
    d='{} 00:00:00'.format(debut)
    f='{} 23:30:00'.format(fin)
    data=df.loc[d:f]
    conso=np.array(data['Consommation'])
    return conso


# In[166]:


#on calcule la consommation moyenne par jour sur une année
def conso_moyenne_jour(region):
    df=clean_data(region)
    consommation=np.array(df['Consommation'])
    conso_jour=[]
    for i in range(366): #2016 est bissextile
        if i < int(len(consommation)/48):
            jour=sum(consommation[48*i:48*(i+1)])/48
            conso_jour.append(jour)
        else:
            break

    plt.plot(conso_jour)
    plt.xticks([30.5*x for x in range(13)], [None,"Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Decembre"])
    plt.tick_params(labelsize=6)
    plt.title("Evolution de la consommation moyenne par jour dans la région {} en 2016".format(region))
    plt.xlabel("Mois")
    plt.ylabel("Consommation (MW)")

    


# In[167]:


conso_moyenne_jour("Bretagne")


# In[168]:


#on regarde la répartition entre le nombre de jours ou on consomme peu et le nombre de jours ou on consomme beaucoup
#l'intérêt de cet indicateur est de voir le rapport entre le nombre de jour ou on consomme bcp et le nombre de jours ou on consomme peut: on voit qu'on suit une tendance linéaire
def repartition(region):
    df=clean_data(region)
    if len(np.array(df["Consommation"]))>17568:
        ecdf=ECDF(df['Consommation'][:17568])
    else:
        ecdf=ECDF(df['Consommation'])
    plt.plot(ecdf.x,ecdf.y)
    plt.title("Fonction de répartition empirique  dans la région {} en 2016".format(region))        
    plt.xlabel("Consommation (en MW)")
    plt.ylabel("Proportion de jours dans l'année avec cette consommation")
    


# In[172]:


repartition("BRETAGNE")


# In[197]:


def repartition_conso_mois(region):
    df=clean_data(region)
    proportion=[]
    j=0
    if len(np.array(df["Consommation"]))>17568:
        donne=np.array(df['Consommation'][:17568])
        for i in range(12):
            conso_mois=sum(donne[1464*i:1464*(i+1)])
            prop=conso_mois/sum(donne)
            proportion.append(prop)
    else:
        donne=np.array(df['Consommation'])           
        while j < int(len(donne)/1464):
            conso_mois=sum(donne[1464*j:1464*(j+1)])
            prop=conso_mois/sum(donne)                
            proportion.append(prop)
            j=j+1
        conso_mois=sum(donne[1464*j:])
        prop=conso_mois/sum(donne)
        proportion.append(prop)
                       
    if len(np.array(df["Consommation"]))>17568:
        longueur=12
    else:
        longueur=int(len(np.array(df["Consommation"]))/1464)+1
    plt.bar(range(longueur),proportion)
    plt.xticks(range(12), ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Decembre"])
    plt.tick_params(labelsize=6)
    plt.xlabel("mois")
    plt.ylabel("part du mois dans la consommation otale de l'année")
    plt.title("Histogramme de la consommation en électricité (MW) dans la région {} en 2016".format(region))


# In[198]:


repartition_conso_mois('BOURGOGNE FRANCHE COMTE')


# In[236]:


#Afin de prendre des semaines entières, j'ai considéré seulement les jours du 4 janvier au 24 décembre 2016 pour cette fonction
def repartition_conso_semaine(region):
    df=clean_data(region)
    conso=np.array(df["Consommation"])
    proportion=[]
    jour=[[],[],[],[],[],[],[]]
    if len(np.array(df["Consommation"]))>17568:
        for i in range(51):
            for j in range(7):
                conso_jour=sum(conso[144+48*j+336*i:144+48*(j+1)+336*i])
                jour[j].append(conso_jour)
    else:
        for i in range(int((-144+len(np.array(df["Consommation"])))/336)):
            for j in range(7):
                conso_jour=sum(conso[144+48*j+336*i:144+48*(j+1)+336*i])
                jour[j].append(conso_jour)
            
    if len(np.array(df["Consommation"]))>17568:
        for k in range(7):
            prop=sum(jour[k])/sum(np.array(df["Consommation"])[144:17280])
            proportion.append(prop)
    else:
        for k in range(7):
            prop=sum(jour[k])/sum(np.array(df["Consommation"])[144:336*int((-144+len(np.array(df["Consommation"])))/336)])
            proportion.append(prop)
        

    plt.bar(range(7),proportion)
    plt.xticks(range(7), ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
    plt.tick_params(labelsize=9)
    plt.xlabel("jours")
    plt.ylabel("part du jour dans la consommation totale de la semaine")
    plt.title("Histogramme de la consommation en électricité (MW) dans la région {} en 2016".format(region))
    


# In[238]:


repartition_conso_semaine('AUVERGNE RHONE ALPES')


# In[ ]:




