
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# on charge les deux modèles

model_mlp = keras.models.load_model("my_model_1.h5")
model_cnn = keras.models.load_model("my_model_2.h5")


region = 'ILE DE FRANCE'
data = pd.read_csv('dataset_centrale/data/test/%s.csv' % region)


def tf(v):
    n = len(v)
    res = np.zeros(n)
    for i in range(n):
        res[i] = int(v[i])
    return res


train = pd.read_csv('dataset_centrale/data/train/%s.csv' %
                    region, parse_dates=['timestamp'])
test = pd.read_csv('dataset_centrale/data/test/%s.csv' %
                   region, parse_dates=['timestamp'])
flag_holi = train['flag_holiday']
flag_long_we = train['flag_long_weekend']

train['flag_holiday'] = tf(flag_holi)
train['flag_long_weekend'] = tf(flag_long_we)

flag_holi = test['flag_holiday']
flag_long_we = test['flag_long_weekend']

test['flag_holiday'] = tf(flag_holi)
test['flag_long_weekend'] = tf(flag_long_we)


n = len(train['Consommation'])
# a peu pres 80% de données d'entrainement
y_train = train['Consommation'].values[:24537]
y_val = train['Consommation'].values[24537:]
y_test = test['Consommation'].values

x_train = train[:24537].iloc[:, 2:11]
x_val = train[24537:].iloc[:, 2:11]
x_test = test.iloc[:, 2:11]


y_pred = model_mlp.predict(x_test)


plt.plot(y_pred, label='predicted')
plt.plot(y_test, label='expected')
plt.title('Predicted and expected consumptions')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred)
plt.plot([4000, 12000], [4000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((y_test - y_pred) ** 2))))
plt.xlabel('expected')
plt.ylabel('predicted')
plt.show()


print('RMS de %s pour le modèle de deep learning' %
      np.sqrt(np.mean((y_pred - y_test) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('For mlp network: ' +
      str(mean_absolute_percentage_error(y_test, y_pred)))


x_train_series = x_train.values.reshape(
    (x_train.shape[0], x_train.shape[1], 1))
x_val_series = x_val.values.reshape((x_val.shape[0], x_val.shape[1], 1))
x_test_series = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))


y_pred = model_cnn.predict(x_test_series)


plt.plot(y_pred, label='predicted')
plt.plot(y_test, label='expected')
plt.title('Predicted and expected consumptions')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred)
plt.plot([4000, 12000], [4000, 12000], '--k')
plt.title("RMS: {:.2f}".format(np.sqrt(np.mean((y_test - y_pred) ** 2))))
plt.xlabel('expected')
plt.ylabel('predicted')
plt.show()


print('RMS de %s pour le modèle CNN' %
      np.sqrt(np.mean((y_pred - y_test) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('For CNN : MAPE='+str(mean_absolute_percentage_error(y_test, y_pred)))
