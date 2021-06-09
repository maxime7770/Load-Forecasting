
import pandas as pd
import numpy as np


region = "ILE DE FRANCE"

data_test = pd.read_csv('dataset_centrale/data/test/%s.csv' % region)
data_train = pd.read_csv('dataset_centrale/data/train/%s.csv' % region)


print(data_train)
print(data_train.values[:, 2:11])
