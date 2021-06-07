#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[39]:


df = pd.read_csv('REGION/Train/AUVERGNE RHONE ALPES.csv',index_col=0)


# In[42]:


pd.DataFrame(df)


# In[115]:


#Saisonnalité des Time Series

Hiver_2016=df.loc[:'2016-03-20 23:30:00']
Hiv2016=np.array(Hiver_2016['Consommation'])
plt.subplot(3,3,1)
plt.plot(Hiv2016)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'éléectricité à l'Hiver 2016")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

Printemps_2016=df.loc['2016-03-20 00:00:00':'2016-06-20 23:30:00'] 
Print2016=np.array(Printemps_2016['Consommation'])
plt.subplot(3,3,2)
plt.plot(Print2016)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'électricité au Printemps 2016")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

Ete_2016=df.loc['2016-06-21 00:00:00':'2016-09-21 23:30:00']
Ete2016=np.array(Ete_2016['Consommation'])
plt.subplot(3,3,3)
plt.plot(Ete2016)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'électricité à l'été 2016")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

Automne_2016=df.loc['2016-09-22 00:00:00':'2016-12-20 23:30:00']
Aut2016=np.array(Automne_2016['Consommation'])
plt.subplot(3,3,4)
plt.plot(Aut2016)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'électricité à l'Automne 2016")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

Hiver_2016_2017=df.loc['2016-12-21 00:00:00':'2017-03-20 23:30:00']
Hiv20162017=np.array(Hiver_2016_2017['Consommation'])
plt.subplot(3,3,5)
plt.plot(Hiv20162017)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'électricité à l'Hiver 2016-2017")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

Printemps_2017=df.loc['2017-03-20 00:00:00':'2017-06-20 23:30:00']
Print2017=np.array(Printemps_2017['Consommation'])
plt.subplot(3,3,6)
plt.plot(Print2017)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'électricité au Printemps 2017")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

Ete_2017=df.loc['2017-06-21 00:00:00':'2017-09-21 23:30:00']
Ete2017=np.array(Ete_2017['Consommation'])
plt.subplot(3,3,7)
plt.plot(Ete2017)
plt.xticks([1464*x for x in range(4)], range(4))
plt.title("Evolution de la consommation d'électricité à l'été 2017")
plt.xlabel("Mois")
plt.ylabel("Consommation (MW)")

plt.show()


# In[110]:


#On regarde maintenant sur un mois en comparant un mois normal (Janvier) à un mois de vacances (Juillet)
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


# In[113]:


#On regarde maintenant sur 2 semaine différentes (du Lundi 4 Janvier 2016 au Dimanche 10 Janvier 2016 et du lundi 11 Janvier au Dimanche 17)

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


# In[122]:


#On regarde l'évolution au cours des différents jours de la semaine
Jour_1=df.loc['2016-01-04 00:00:00':'2016-01-04 23:30:00']
Jour1=np.array(Jour_1['Consommation'])
plt.subplot(3,3,1)
plt.plot(Jour1)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Lundi")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")

Jour_2=df.loc['2016-01-05 00:00:00':'2016-01-05 23:30:00']
Jour2=np.array(Jour_2['Consommation'])
plt.subplot(3,3,2)
plt.plot(Jour2)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Mardi")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")

Jour_3=df.loc['2016-01-06 00:00:00':'2016-01-06 23:30:00']
Jour3=np.array(Jour_3['Consommation'])
plt.subplot(3,3,3)
plt.plot(Jour3)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Mercredi")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")

Jour_4=df.loc['2016-01-07 00:00:00':'2016-01-07 23:30:00']
Jour4=np.array(Jour_4['Consommation'])
plt.subplot(3,3,4)
plt.plot(Jour4)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Jeudi")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")

Jour_5=df.loc['2016-01-08 00:00:00':'2016-01-08 23:30:00']
Jour5=np.array(Jour_5['Consommation'])
plt.subplot(3,3,5)
plt.plot(Jour5)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Vendredi")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")

Jour_6=df.loc['2016-01-05 00:00:00':'2016-01-05 23:30:00']
Jour6=np.array(Jour_6['Consommation'])
plt.subplot(3,3,6)
plt.plot(Jour6)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Samedi")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")

Jour_7=df.loc['2016-01-10 00:00:00':'2016-01-10 23:30:00']
Jour7=np.array(Jour_7['Consommation'])
plt.subplot(3,3,7)
plt.plot(Jour7)
plt.xticks([10*x for x in range(5)], [5*x for x in range(5)])
plt.title("Evolution de la consommation au cours du Dimanche")
plt.xlabel("Heures")
plt.ylabel("Consommation (MW)")


# In[ ]:




