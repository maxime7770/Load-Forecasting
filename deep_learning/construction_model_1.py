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

region = 'ILE DE FRANCE'
data = pd.read_csv('../dataset_centrale/data/test/%s.csv' % region)


def tf(v):
    n = len(v)
    res = np.zeros(n)
    for i in range(n):
        res[i] = int(v[i])
    return res


train = pd.read_csv('../dataset_centrale/data/train/%s.csv' %
                    region, parse_dates=['timestamp'])
test = pd.read_csv('../dataset_centrale/data/test/%s.csv' %
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

epochs = 50
batch = 128
lr = 0.003
adam = optimizers.Adam(lr)


model_mlp = keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),

    layers.Dense(1), ])
model_mlp.compile(loss='mse', optimizer=adam)
model_mlp.summary()

mlp_history = model_mlp.fit(x_train, y_train, validation_data=(
    x_val, y_val), epochs=epochs, verbose=2)

model_mlp.save("../models_deep_learning/model1_%s.h5" % region)


plt.plot(mlp_history.history['loss'])
plt.plot(mlp_history.history['val_loss'])
plt.title('model_mlp loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
