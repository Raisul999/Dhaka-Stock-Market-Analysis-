import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import random

#Step_1
#importing the csv files from the stock data through this code snippet
path = 'C:/Users/RAFAT/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/datafromamarstock_v2'
stocks = []
for file in os.listdir(path):
    file_path = path + '/' + file
    date = pd.read_csv(file_path)['Time']
    stocks.append(file)

print(str(len(stocks))+" stocks selected")
print(stocks)


#Step_2
#using pandas dataframe to merge all company price data to a single csv file
stock_price_dict = {}

for file in stocks:
    path = "C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/datafromamarstock_v2/" + file
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df = df[df.Time >= '2007-07-02']
    df = df.set_index(pd.DatetimeIndex(df['Time']))
    stock_price_dict[file.split(".")[0]] = df['Closing']

print(stock_price_dict)

stock_price_df = pd.DataFrame(stock_price_dict)
print(stock_price_df.head(100))

#Step_3
#this snippet deals with missing data from the dateframe for each company
NA_col = []
NA_ratio = []
for col in stock_price_df.columns :
    na_index = np.where(stock_price_df[col].isnull())[0]
    NA_col.append(col)
    NA_ratio.append(len(na_index)/stock_price_df.shape[0] * 100)
    print(col,na_index)

#stock price df to csv file
#stock_price_df.to_csv("C:/Users/RAFAT/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/prepared_stockdata_v2.csv",index_label='Date')

#Step_4
#remove the column with too many NaN values
stock_price_df = stock_price_df.drop(['ALLTEX','APEXSPINN','AZIZPIPES','ARAMIT',
                                      'BATASHOE','AZIZPIPES','BATBC','BDTHAI',
                                      'EASTRNLUB','ECABLES','LINDEBD','NTC', 'UCB','ZEALBANGLA'], axis=1)

#Step_5
#removing the NaN values in the columns
#this snippet deals with handling missing data with each company using imputation method
def impute_data(column_name):
    price_na_index = np.where(stock_price_df[column_name].isnull())[0]
    for i in price_na_index :
        stock_price_df[column_name][i] = stock_price_df[column_name][i-1]

#Step_6
for item in stock_price_df.columns :
    impute_data(item)

for item in stock_price_df.columns :
    if stock_price_df[item].isnull().values.any() :
        print('stock price data of '+item+' still has NaN')
print("END OF CHECKING. NO NA REMAINING")

#Step_7
#moving the data to a new CSV file
stock_price_df.to_csv("C:/Users/RAFAT/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/prepared_stockdata_v2.csv",index_label='Date')

#Step_8
#adding the company heading to a list and printing them out
'''df = pd.read_csv("C:/Users/RAFAT/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/prepared_stockdata_v2.csv")
portfolio = list(df.columns.values[1:])
portfolio.remove('ISLAMIBANK')
print(portfolio)'''



#debugging code for now
portfolio = ['ABBANK', 'ACI', 'APEXFOODS', 'BEXIMCO', 'BGIC', 'EHL', 'GREENDELT', 
'MONNOCERA', 'SINGERBD', 'SINOBANGLA', 'AFTABAUTO', 'AMBEEPHA', 'AMCL(PRAN)',
'APEXFOOT', 'PUBALIBANK']

'''portfolio = ['APEXTANRY', 'ATLASBANG']
portfolio = ['BDAUTOCA', 'BDLAMPS']
portfolio = ['BDWELDING', 'BXPHARMA']
portfolio = ['DHAKABANK', 'EBL']
portfolio = ['FUWANGFOOD', 'GQBALLPEN']
portfolio = ['IBNSINA', 'IDLC']
portfolio = ['IFIC', 'MEGHNACEM']
portfolio = ['MIRACLEIND', 'NBL']
portfolio = ['NCCBANK', 'NPOLYMAR']
portfolio = ['NTLTUBES', 'PEOPLESINS']
portfolio = ['PRIMEBANK', '']
portfolio = ['QUASEMIND', 'RENATA']
portfolio = ['SOUTHEASTB', 'SQURPHARMA']
portfolio = ['UNITEDFIN', 'USMANIAGL', 'UTTARAFIN']'''

#Step_9
#preparing the main dataset with correlation coefficients of each company pair in portfolio list
#writing the method first
def rolling_corr(item1,item2) :
    #import data
    stock_price_df = pd.read_csv("C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/prepared_stockdata_v2.csv")
    pd.to_datetime(stock_price_df['Date'], format='%Y-%m-%d')
    stock_price_df = stock_price_df.set_index(pd.DatetimeIndex(stock_price_df['Date']))
    
    #calculate
    df_pair = pd.concat([stock_price_df[item1], stock_price_df[item2]], axis=1)
    df_pair.columns = [item1,item2]
    df_corr = df_pair[item1].rolling(window=100).corr(df_pair[item2])
    return df_corr


#Step_10
#making the index list
index_list = []
for _ in range(100):
    indices = []
    for k in range(_, 2420,100):
        indices.append(k)
    index_list.append(indices)

print(index_list)

#Step_11
#making the list for keeping the correlation cosff for each company pair
data_matrix = []
count = 0
for i in range(15):
    for j in range(14-i):
        a = portfolio[i]
        b = portfolio[14-j]
        file_name = a + '_' + b
            
        corr_series = rolling_corr(a, b)[99:]
        for _ in range(100):
            corr_strided = list(corr_series[index_list[_]][:24]).copy()
            data_matrix.append(corr_strided)
            count+=1
            if count % 1000 == 0 :
                print(str(count)+' items preprocessed')

#Step_12
#transferring the data to a csv file
data_matrix = np.transpose(data_matrix)
data_dictionary = {}

for i in range(len(data_matrix)):
    data_dictionary[str(i)] = data_matrix[i]
data_df = pd.DataFrame(data_dictionary)
data_df.to_csv('C:/Users/Raisul Islam/Desktop/Corr_Prediction_ARIMA_LSTM_Hybrid_main/dataset_main_v3.csv')





