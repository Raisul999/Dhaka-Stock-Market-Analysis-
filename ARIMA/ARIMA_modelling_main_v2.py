import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import pylab as pl
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pandas import Series
#from statsmodels.tsa.stattools import adfuller
import warnings

#Step 1 
data_df = pd.read_csv('G:/Corr_Prediction_ARIMA_LSTM_Hybrid_main/dataset_main_v3.csv')
data_df = data_df.loc[:, ~data_df.columns.str.contains('^Unnamed')]
print(data_df.shape)

print(data_df.head())

#Step 2
num_list = []
for i in range(24):
    num_list.append(str(i))
data_df = data_df[num_list].copy()
data_df = np.transpose(data_df)
print(data_df.shape)
print(data_df.head())

#Step 3
#Splitting the dataset into  Train, Dev, and Test Splits from here 
#We do not split to X and Y as of yet
indices = [20*k for k in range(525)]
data_df = pd.DataFrame(data_df[indices])


#right before splitting

train = []
dev = []
test1 = []
test2 = []

for i in range(data_df.shape[1]):
    tmp = data_df[20*i].copy()
    train.append(tmp[:21])
    dev.append(tmp[1:22])
    test1.append(tmp[2:23])
    test2.append(tmp[3:24])
    
train = pd.DataFrame(train)
dev = pd.DataFrame(dev)
test1 = pd.DataFrame(test1)
test2 = pd.DataFrame(test2)


train.to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/train.csv')
dev.to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/dev.csv')
test1.to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/test1.csv')
test2.to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/test2.csv')

#Step 4
#Plotting the ACF and PACf for finding the appropriate ARIMA model
train = pd.read_csv('G:/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/train.csv')
train = np.transpose(train.loc[:, ~train.columns.str.contains('^Unnamed')])
train

for _ in range(100, 105, 1):
    print('For Column Index : ',_)
    
    print('With NO differencing')
    train.iloc[:,_].plot()
    plt.show()
    plt.close()
    X=train[_].values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    print('ACF with NO differencing')
    plot_acf(train[_])
    plt.show()
    plt.close()
    
    print('PACF with NO differencing')
    plot_pacf(train[_])
    plt.show()
    plt.close()
    
    #After 1st differencing
    
    print('With 1ST differencing')
    train[_].diff()[1:].plot()
    plt.show()
    plt.close()
    
    print('ACF after 1ST differencing')
    plot_acf(train[_].diff()[1:])
    plt.show()
    plt.close()
    
    print('PACF after 1ST differencing')
    plot_pacf(train[_].diff()[1:])
    plt.show()
    plt.close()
    
    #After 2nd differencing
    
    print('With 2ND differencing')
    train[_].diff()[1:].diff()[1:].plot()
    plt.show()
    plt.close()
    
    print('ACF after 2ND differencing')
    plot_acf(train[_].diff()[1:].diff()[1:])
    plt.show()
    plt.close()
    
    print('PACF after 2ND differencing')
    plot_pacf(train[_].diff()[1:].diff()[1:])
    plt.show()
    plt.close()
    
    #After 3rd differencing
    
    print('With 3RD differencing')
    train[_].diff()[1:].diff()[1:].diff()[1:].plot()
    plt.show()
    plt.close()
    
    print('ACF after 3RD differencing')
    plot_acf(train[_].diff()[1:].diff()[1:].diff()[1:])
    plt.show()
    plt.close()
    
    print('PACF after 3RD differencing')
    plot_pacf(train[_].diff()[1:].diff()[1:].diff()[1:])
    plt.show()
    plt.close()
    
    #After 4th differencing
    
    print('With 4TH differencing')
    train[_].diff()[1:].diff()[1:].diff()[1:].diff()[1:].plot()
    plt.show()
    plt.close()
    
    print('ACF after 4TH differencing')
    plot_acf(train[_].diff()[1:].diff()[1:].diff()[1:].diff()[1:])
    plt.show()
    plt.close()
    
    print('PACF after 4TH differencing')
    plot_pacf(train[_].diff()[1:].diff()[1:].diff()[1:].diff()[1:])
    plt.show()
    plt.close()
    

    print('----------------------------------------------------')


#Step 5
#Main stage of ARIMA plotting starts from here on
train = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/train.csv')
dev = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/dev.csv')
test1 = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/test1.csv')
test2 = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/before_arima/test2.csv')

train = np.transpose(train.loc[:,~train.columns.str.contains('^Unnamed')])
dev = np.transpose(dev.loc[:,~dev.columns.str.contains('^Unnamed')])
test1 = np.transpose(test1.loc[:,~test1.columns.str.contains('^Unnamed')])
test2 = np.transpose(test2.loc[:,~test2.columns.str.contains('^Unnamed')])

datasets = [train, dev, test1, test2]


#creating the models from here on
model_011 = ARIMA(order=(0,1,1), method='mle', suppress_warnings=True) #110
model_021 = ARIMA(order=(0,2,1), method='mle', suppress_warnings=True) #011
model_111 = ARIMA(order=(1,1,1), method='mle', suppress_warnings=True) #111
model_121 = ARIMA(order=(1,2,1), method='mle', suppress_warnings=True) #211
model_221 = ARIMA(order=(2,2,1), method='mle', suppress_warnings=True) #210

mod011_c = 0
mod021_c = 0
mod111_c = 0
mod121_c = 0
mod221_c = 0


#Declaring lists for the the train, dev and test sets in X and Y respectively
train_X = []; train_Y = []
dev_X = [];   dev_Y = []
test1_X = []; test1_Y = []
test2_X = []; test2_Y = []

flag=0


#model fitting and storing residual values in the lists
for i in range(525):
    print(i)
    tmp = []
    c=0
    for s in datasets :
        c+=1
        try:
            model1 = model_011.fit(s[i])
            model = model1
            
            try:
                model2 = model_021.fit(s[i])
                
                if model.aic() <= model2.aic() :
                    pass
                else :
                    model = model2
                    
                try :
                    model3 = model_111.fit(s[i])
                    if model.aic() <= model3.aic() :
                        pass
                    else :
                        model = model3
                except :
                    try:
                        model4 = model_121.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_221.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            pass
                    
            except:
                try:
                    model3 = model_111.fit(s[i])

                    if model.aic() <= model3.aic() :
                        pass
                    else :
                        model = model3
                except :
                    try:
                        model4 = model_121.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_221.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            pass
                
        except:
            try:
                model2 = model_021.fit(s[i])
                model = model2
            
                try :
                    model3 = model_111.fit(s[i])
                    
                    if model.aic() <= model3.aic():
                        pass
                    else:
                        model = model3
                except :
                    try:
                        model4 = model_121.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_221.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            pass
            
            except :
                try:
                    model3 = model_111.fit(s[i])
                    model = model3
                except :
                    try:
                        model4 = model_121.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_221.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            flag = 1
                            print(str(c) + " FATAL ERROR")
                            break
        
        if model.order == model_011.order:
            mod011_c = mod011_c +1
        elif model.order == model_021.order:
            mod021_c = mod021_c +1
        elif model.order == model_111.order:
            mod111_c = mod111_c + 1
        elif model.order == model_021.order:
            mod121_c = mod121_c +1
        else:
            mod221_c = mod221_c +1
        
        predictions = list(model.predict_in_sample())
        #pad the first time step of predictions with the average of the prediction values
        #so as to match the length of the s[i] data
        predictions = [np.mean(predictions)] + predictions
        
        residual = pd.Series(np.array(s[i]) - np.array(predictions))
        tmp.append(np.array(residual))
    
    
                    
    if flag == 1:
        break
    train_X.append(tmp[0][:20])
    train_Y.append(tmp[0][20])
    dev_X.append(tmp[1][:20])
    dev_Y.append(tmp[1][20])    
    test1_X.append(tmp[2][:20])
    test1_Y.append(tmp[2][20])
    test2_X.append(tmp[3][:20])
    test2_Y.append(tmp[3][20])


#ARIMA model frequency
print('ARIMA model ', model_011.order, 'has been used ', mod011_c, ' times')
print('ARIMA model ', model_021.order, 'has been used ', mod021_c, ' times')
print('ARIMA model ', model_111.order, 'has been used ', mod111_c, ' times')
print('ARIMA model ', model_121.order, 'has been used ', mod121_c, ' times')
print('ARIMA model ', model_221.order, 'has been used ', mod221_c, ' times')
        

#Step_11
#Visualising the residual distribution for 15 companies and their pairs
pd.DataFrame(train_X).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/train_X.csv')
pd.DataFrame(dev_X).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/dev_X.csv')
pd.DataFrame(test1_X).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test1_X.csv')
pd.DataFrame(test2_X).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test2_X.csv')
pd.DataFrame(train_Y).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/train_Y.csv')
pd.DataFrame(dev_Y).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/dev_Y.csv')
pd.DataFrame(test1_Y).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test1_Y.csv')
pd.DataFrame(test2_Y).to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/test2_Y.csv')
    

train = pd.read_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/train_dev_test_main_v3/after_arima/train_X.csv')
train = np.transpose(train.loc[:,~train.columns.str.contains('^Unnamed')])
train_melt = sorted(np.array(train.melt()['value']))
fit = stats.norm.pdf(train_melt, np.mean(train_melt), np.std(train_melt))
pl.hist(train_melt,normed=True, color='grey', bins=[-4,-3,-2,-1,0,1,2,3,4,5])
pl.plot(train_melt,fit,color='blue')
pl.title('residual value distribution')
pl.xlabel('residual')
pl.show()
pl.close()

X = [x for x in train_melt if x>2]
Y = [y for y in train_melt if y<-2]
out_of_bound = X + Y
print(str(len(out_of_bound)/105) +' % of the data is out of bound [-2,2]')

X = [x for x in train_melt if x>1]
Y = [y for y in train_melt if y<-1]
out_of_bound = X + Y
print(str(len(out_of_bound)/105) +' % of the data is out of bound [-1,1]')


