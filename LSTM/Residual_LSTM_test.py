import pandas as pd
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2



#Step_1
#Train-Dev-Test models loading

train_X= pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/train_X.csv')
print('loaded train_X')
dev_X = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/dev_X.csv')
print('loaded dev_X')
test1_X = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test1_X.csv')
print('loaded test1_X')
test2_X = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test2_X.csv')
print('loaded test2_X')
train_Y = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/train_Y.csv')
print('loaded train_Y')
dev_Y = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/dev_Y.csv')
print('loaded dev_Y')
test1_Y = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test1_Y.csv')
print('loaded test1_Y')
test2_Y = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test2_Y.csv')
print('loaded test2_Y')


#Step_2
#Removing unnecessary columns
train_X = train_X.loc[:, ~train_X.columns.str.contains('^Unnamed')]
dev_X = dev_X.loc[:, ~dev_X.columns.str.contains('^Unnamed')]
test1_X = test1_X.loc[:, ~test1_X.columns.str.contains('^Unnamed')]
test2_X = test2_X.loc[:, ~test2_X.columns.str.contains('^Unnamed')]
train_Y = train_Y.loc[:, ~train_Y.columns.str.contains('^Unnamed')]
dev_Y = dev_Y.loc[:, ~dev_Y.columns.str.contains('^Unnamed')]
test1_Y = test1_Y.loc[:, ~test1_Y.columns.str.contains('^Unnamed')]
test2_Y = test2_Y.loc[:, ~test2_Y.columns.str.contains('^Unnamed')]


#Step_3
#data sampling
STEP = 20
#num_list = [STEP*i for i in range(int(1117500/STEP))]

_train_X = np.asarray(train_X).reshape((int(10500/STEP), 20, 1))
_dev_X = np.asarray(dev_X).reshape((int(10500/STEP), 20, 1))
_test1_X = np.asarray(test1_X).reshape((int(10500/STEP), 20, 1))
_test2_X = np.asarray(test2_X).reshape((int(10500/STEP), 20, 1))

_train_Y = np.asarray(train_Y).reshape(int(10500/STEP), 1)
_dev_Y = np.asarray(dev_Y).reshape(int(10500/STEP), 1)
_test1_Y = np.asarray(test1_Y).reshape(int(10500/STEP), 1)
_test2_Y = np.asarray(test2_Y).reshape(int(10500/STEP), 1)

#define custom activation
class Double_Tanh(Activation):
    def __init__(self, activation, **kwargs):
        super(Double_Tanh, self).__init__(activation, **kwargs)
        self.__name__ = 'double_tanh'

def double_tanh(x):
    return (K.tanh(x) * 2)

get_custom_objects().update({'double_tanh':Double_Tanh(double_tanh)})


#Model creation and generation
model = Sequential()
model.add(LSTM(25, input_shape=(20,1), dropout=0.0, kernel_regularizer=l1_l2(0.00,0.00), bias_regularizer=l1_l2(0.00,0.00)))
model.add(Dense(1))
model.add(Activation(double_tanh))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

filepath = "C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/models/hybrid_LSTM/model.h5"

checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min') 

model.fit(_train_X, _train_Y, epochs=5, batch_size=35, verbose = 1, callbacks = [checkpoint])

sequences = pd.DataFrame(model.predict(_test1_X))


scores_train = model.evaluate(_train_X, _train_Y)
scores_test = model.evaluate(_test1_X, _test1_Y)
scores_dev = model.evaluate(_dev_X, _dev_Y)'''




