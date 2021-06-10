#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import acorr_ljungbox


# In[ ]:


#    "Le modèle Garch ne fut pas concluant car les prédictions étaient abérantes."
#    "Voici tout de même ci-dessous la démarche effectuée:"
#   "On commence par vérifier la corrélation des données"
#    "Puis on cherche le modèle ARCH qui va minimiser l'AIC"
#    "Une fois les paramètres recueillis, on fait une prédiction en utilisant ce modèle"


# In[151]:


region='bretagne'
data = pd.read_csv('../dataset_centrale/data/train/{}.csv'.format(region),header=0, index_col=0, parse_dates=True, squeeze=True)
df=data["Consommation"][30672-336:]


# In[154]:


# The autocorrelation function of the volatility is long range.
sm.graphics.tsa.plot_acf(df ,lags=100, zero=False)
plt.show()


# In[155]:


models = []
for i in range(1,40):
    ft = arch_model(df,vol='ARCH', p=i).fit()
    
    models.append(( i,ft.bic,ft.aic))


# In[156]:


models = pd.DataFrame(models,columns=['order','BIC','AIC']).set_index('order')

models = models.assign(dBIC=( models.BIC - models.BIC.min()),
              dAIC =( models.AIC - models.AIC.min()))


# In[157]:


plt.plot(models.AIC,'-o')
plt.xlabel('q',fontsize=16)
plt.ylabel('AIC',fontsize=16)
plt.tick_params(labelsize=14)
plt.grid()


# In[158]:


choix=int(models[models["AIC"]==models.AIC.min()].index[0])


# In[159]:


ft = arch_model(df,vol='ARCH', p=choix).fit()


# In[162]:


a=ft.forecast(horizon=1)


# In[163]:


a.variance


# In[ ]:




