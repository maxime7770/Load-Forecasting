#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import panda as pd
import numpy as np


# In[ ]:


def extraction(debut,fin,fichier):
    df=pd.read_csv(fichier,index_col=0)
    d='{} 00:00:00'.format(debut)
    f='{} 23:30:00'.format(fin)
    data=df.loc[d:f]
    conso=np.array(data['Consommation'])
    return conso

