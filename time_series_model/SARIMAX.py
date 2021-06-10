#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


# In[ ]:


#We first introduce the MAPE
def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[130]:

#We now define the Sarimax function that will forcast the number of day we want. The function is based on a 48timestep=1day loop: the function predicts for one day and
# once the presdiction is complete, it actualizes the data needed to make the prediction. We worked on 2 different sample for the function: a sample of one week 
#before the foracasting and a sample of one week, one year ago, before the forecasting. We kept the 2nd option as it worked better.
#The function take much time to be ploted so I highly recommand to download the jupyter notebook file so as to see this function already ploted. 
def Sarimax(region,jour):
    finaldf = pd.read_csv('../dataset_centrale/data/train/{}.csv'.format(region),header=0, index_col=0, parse_dates=True, squeeze=True)   
    endog=finaldf['Consommation']
    endog2=np.asarray(finaldf['Consommation'])
    #year=365*48
    week=7*48
    for i in range(jour):
        training_obs = 30672
        training_endog = endog[training_obs-week+10*i:training_obs+10*i]
        training_mod = sm.tsa.SARIMAX(
        training_endog, order=(27,0,27),seasonal=(27, 0, 27, 48*7*4), trend='c')
        training_res = training_mod.fit()
        fcast = training_res.forecast(steps=48)
        endog=pd.concat([endog,fcast],join='inner')
    return endog[30672:]


    


# In[134]:
#We now plot our prediction vs the true sample

def SarimaxTvsP(region,jour):
    test = pd.read_csv('../dataset_centrale/data/test/{}.csv'.format(region),header=0, index_col=0, parse_dates=True, squeeze=True)
    pred=Sarimax(region,jour)
    true_data=test['Consommation']
    prediction=[]
    true=[]
    for i in range(len(pred)):
        prediction.append(pred[i])
        true.append(true_data[i])
    mape=mean_absolute_percentage_error(true,prediction)
    plt.plot(prediction,label='prediction')
    plt.plot(true,label='True')
    plt.xticks([48*x for x in range(jour+1)], range(jour+1))
    plt.title("SARIMAX: Prediction vs True en {} avec un MAPE de {}".format(region,mape))
    plt.xlabel("Day")
    plt.ylabel("Consumption(MW)")
    plt.legend()
    plt.show()


# In[136]:


SarimaxTvsP('Bretagne',5)


# In[ ]:




