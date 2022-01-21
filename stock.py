#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import sys
import numpy as np
import pandas as pd
import re
import math
from datetime import datetime
import datetime as DateTime
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from tcn import compiled_tcn

class StockProcess:
    CONST_INIT_DATE = "20040102"
    CONST_END_DATE = "20211128"
    EXLUDE_STOCKNUM = ['2823', '912398', '3095', '3144', '4803']
    highDf = []
    highCurrent = None
    lowDf = []
    lowCurrent = None
    volumnDf = []
    volumnCurrent = None
    dateCurrent = None
    stockNumCurrent = None
    stockTokenDf=[]
    stockTokenBuyCurrent=None
    stockTokenBuyPrev=None
    stockTokenSellCurrent=None
    stockTokenSellPrev=None
    df=[]
    date=None
    labelDf=[]
    timesteps=None
    exitTimes=None
    def __init__(self):
        high = pd.read_csv("./data/2003-2021high.csv")
        high = high.loc[:, ~high.columns.str.contains('^Unnamed')]
        high = high.set_axis(high.iloc[3].tolist(), axis=1)
        high = high.drop([0, 1, 2, 3])
        high.index = high[high.columns[0]].astype(str)
        high = high.drop('股票代號', 1)
        column = high.columns.tolist()
        for i in range(len(column)):
            column[i] = column[i].replace('最高價', '')
        high.columns = column
        low = pd.read_csv("./data/2003-2021low.csv")
        low = low.loc[:, ~low.columns.str.contains('^Unnamed')]
        low = low.set_axis(low.iloc[3].tolist(), axis=1)
        low = low.drop([0, 1, 2, 3])
        low.index = low[low.columns[0]].astype(str)
        low = low.drop('股票代號', 1)
        column = low.columns.tolist()
        for i in range(len(column)):
            column[i] = column[i].replace('最低價', '')
        low.columns = column
        volumn = pd.read_csv("./data/2003-2021volumn.csv")
        volumn = volumn.loc[:, ~volumn.columns.str.contains('^Unnamed')]
        volumn = volumn.set_axis(volumn.iloc[3].tolist(), axis=1)
        volumn = volumn.drop([0, 1, 2, 3])
        volumn.index = volumn[volumn.columns[0]].astype(str)
        volumn = volumn.drop('股票代號', 1)
        column = volumn.columns.tolist()
        for i in range(len(column)):
            column[i] = column[i].replace('成交金額(千)', '')
        volumn.columns = column
        self.highDf = high
        self.lowDf = low
        self.volumnDf = volumn
        self.dateCurrent = self.CONST_INIT_DATE
        self.stockNumCurrent = high.index[0]
        # self.setCurrentData(self.stockNumCurrent, self.dateCurrent)
    def prevDayBuy(self,day):
        total=0
        # print(self.token.index.get_loc(self.currentDateIndex))
        # print((self.currentDateIndex))
        # exit()
        for i in  range(day):
            try:
                if float(self.stockTokenDf.loc[self.stockTokenDf.index[self.stockTokenDf.index.get_loc(self.dateCurrent) + i],'近1日買超合計'])>float(self.stockTokenDf.loc[self.stockTokenDf.index[self.stockTokenDf.index.get_loc(self.dateCurrent) +1+ i],'近1日賣超合計']):
                    total+=1
                else:
                    total-=1
            except IndexError:
                return 0
        return total

    def setCurrentData(self, stockNum, date):
        self.highCurrent = float(self.highDf.loc[stockNum, date])
        self.lowCurrent = float(self.lowDf.loc[stockNum, date])
        self.volumnCurrent = float(self.volumnDf.loc[stockNum, date])
        self.stockTokenBuyCurrent = float(self.stockTokenDf.loc[date,'近1日買超合計'])
        self.stockTokenSellCurrent = float(self.stockTokenDf.loc[date,'近1日賣超合計'])
        self.stockTokenBuyPrev = float(self.stockTokenDf.loc[self.stockTokenDf.index[self.stockTokenDf.index.get_loc(date) + 1], '近1日買超合計'])
        self.stockTokenSellPrev = float(self.stockTokenDf.loc[self.stockTokenDf.index[self.stockTokenDf.index.get_loc(date) + 1], '近1日賣超合計'])
        self.dateCurrent = date
        self.stockNumCurrent = stockNum

    def moveNextDate(self):
        try:
            self.dateCurrent = self.stockTokenDf.index[self.stockTokenDf.index.get_loc(self.dateCurrent) - 1]
            # print(self.dateCurrent)
            # exit()
            self.setCurrentData(self.stockNumCurrent, self.dateCurrent)
            return True
        except IndexError:
            return False

    def moveNextRow(self):
        try:
            self.stockNumCurrent = self.stockTokenDf.index[self.stockTokenDf.index.get_loc(self.stockNumCurrent) + 1]
            if self.stockNumCurrent in self.EXLUDE_STOCKNUM:
                return self.moveNextRow()
            self.dateCurrent = self.CONST_INIT_DATE
            self.setCurrentData(self.stockNumCurrent, self.dateCurrent)
            return True
        except IndexError:
            return False
        # except KeyError:
        #     return self.moveNextStock()

    def moveNextStock(self, date):
        try:
            self.stockNumCurrent = self.stockTokenDf.index[self.stockTokenDf.index.get_loc(self.stockNumCurrent) + 1]
            if self.stockNumCurrent in self.EXLUDE_STOCKNUM:
                return self.moveNextStock(date)
            self.setCurrentData(self.stockNumCurrent, date)
            return True
        except IndexError:
            return False

    def moveNext(self):
        if self.moveNextDate():
            return "nextDate"
        # elif self.moveNextRow():
        #     return "nextStock"
        else:
            return "EOF"

    def label(self,i):
        ind=i+self.timesteps
        maxPrice=self.df[ind][2]
        minPrice=self.df[ind][3]
        for index in range(self.exitTimes):
            if self.df[ind+index][2]>maxPrice:
                maxPrice=self.df[ind+index][2]
            if self.df[ind + index][3] < minPrice:
                minPrice=self.df[ind + index][3]
        maxPrice=(maxPrice-self.df[ind][2])/self.df[ind][2]
        minPrice=(minPrice-self.df[ind][2])/self.df[ind][2]
        self.labelDf+=[[maxPrice,minPrice]]

        # print(self.labelDf[i])
        # print()
        # exit()
        try:
            return  self.labelDf[i]
        except:
            print(i)
            exit()


        print(self.df[i])
        print(self.df[i][2])
        exit()
        # max
        # for day in range(60):
        #
        # self.df[i-1]=
    def tradeDateByYear(self,year):
        dateTimeYear = datetime.strptime(str(int(year) - 0), '%Y').date()
        print(dateTimeYear)
        while True:
            try:
                # print(dateTimeYear.strftime("%Y%m%d"))
                self.stockTokenDf.index[self.stockTokenDf.index.get_loc(dateTimeYear.strftime("%Y%m%d"))]
                break
            except:
                dateTimeYear += DateTime.timedelta(days=1)
        CurrentDate = dateTimeYear.strftime("%Y%m%d")
        return CurrentDate
    def setDf(self,stockNum,CurrentDate,year):
        self.setCurrentData(stockNum, CurrentDate)
        df = []
        date = []
        while True:
            df.append([float(self.stockTokenBuyCurrent) - float(self.stockTokenSellCurrent),
                       float(self.stockTokenBuyCurrent) + float(self.stockTokenSellCurrent),
                       self.highCurrent,
                       self.lowCurrent,
                       self.volumnCurrent])
            date.append(self.dateCurrent)
            # print(self.dateCurrent)
            if int(self.dateCurrent[0:4]) > int(year) or int(self.dateCurrent) > int(self.CONST_END_DATE):
                break
            # if self.prevDayBuy(10)<-4 and self.stockTokenBuyCurrent>self.stockTokenSellCurrent:
            #     print(self.dateCurrent)
            self.moveNext()
        self.df += df
        self.date = date
    def stockStrategy(self,stockNum,year,timesteps,exitTimes):
        self.timesteps=timesteps
        self.exitTimes=exitTimes

        stockToken = pd.read_csv("./data/"+stockNum+"token.csv", encoding='big5')
        stockToken = stockToken.loc[:, ~stockToken.columns.str.contains('^Unnamed')]
        stockToken = stockToken.set_axis(stockToken.iloc[3].tolist(), axis=1)
        stockToken = stockToken.drop([0, 1, 2, 3])
        stockToken = stockToken[stockToken['近1日買超合計'].notna()]
        for i in range(len(stockToken.index)):
            dateArray = stockToken.loc[stockToken.index[i], '日期'].split('/')
            dateArray = list(map(int, dateArray))
            stockToken.loc[stockToken.index[i], '日期'] = datetime(dateArray[2], dateArray[0], dateArray[1]).date().strftime(
                "%Y%m%d")
        stockToken.index = stockToken['日期'].astype(str)
        stockToken = stockToken.drop('日期', 1)
        self.stockTokenDf = stockToken
        # 找出那一年有交易的第一天
        CurrentDate = self.tradeDateByYear(year)
        self.setDf(stockNum,CurrentDate,year)

        # self.setCurrentData(stockNum,CurrentDate)
        # df=[]
        # date=[]
        # while True:
        #     df.append([float(self.stockTokenBuyCurrent)-float(self.stockTokenSellCurrent),
        #                float(self.stockTokenBuyCurrent)+float(self.stockTokenSellCurrent),
        #                self.highCurrent,
        #                self.lowCurrent,
        #                self.volumnCurrent])
        #     date.append(self.dateCurrent)
        #     # print(self.dateCurrent)
        #     if int(self.dateCurrent[0:4])>int(year) or int(self.dateCurrent)>int(self.CONST_END_DATE):
        #         break
        #     # if self.prevDayBuy(10)<-4 and self.stockTokenBuyCurrent>self.stockTokenSellCurrent:
        #     #     print(self.dateCurrent)
        #     self.moveNext()
        # self.df=df
        train_end =len(self.df)
        train_x=[]
        train_y=[]
        enc = MinMaxScaler()
        enc.fit(self.df)
        dump(enc, "./output/scaler.save")

        # year=int(year)+1
        # self.setDf(stockNum, CurrentDate, year)

        for i in range(train_end - timesteps-exitTimes):
            train_x.append(enc.transform(self.df[i:i + timesteps]))
            train_y.append(self.label(i))
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        model = compiled_tcn(return_sequences=False,
                             num_feat=train_x.shape[2],
                             nb_filters=24,
                             num_classes=0,
                             kernel_size=20,
                             kernel_initializer='orthogonal',
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=1,
                             max_len=train_x.shape[1],
                             use_skip_connections=True,
                             regression=True,
                             dropout_rate=0,
                             output_len=train_y.shape[1])
        # print(train_y)
        # print(train_y[:, 0])
        # exit()
        model.fit(train_x, train_y, batch_size=100, epochs=200)
        self.setDf(stockNum, CurrentDate, year)
        test_end=len(self.df)
        test_x = []
        test_y = []
        j=0
        for i in range(train_end- timesteps-1,test_end- timesteps-exitTimes):
            test_x.append(enc.transform(self.df[i:i + timesteps]))
            test_y.append(self.label(train_end - timesteps-exitTimes+j))
            j+=1
        # print(test_x)
        test_x = np.array(test_x)
        y_raw_pred = model.predict(test_x)
        for i in range(y_raw_pred.shape[0]):
            print("----------")
            print(y_raw_pred[i])
            print(test_y[i])
        print("----------")
        print(train_y[len(train_y)-1])
        exit()

        year=int(year)+1
        CurrentDate = self.tradeDateByYear(year)
        print(CurrentDate)
        exit()
        self.setDf(stockNum, CurrentDate, year)

        test_end = len(self.df)
        test_x = []
        test_y = []
        enc = load('./output/scaler.save')
        # enc = MinMaxScaler()
        enc.fit(self.df)
        # load('./output/scaler.save')
        # dump(enc, "./output/scaler.save")
        for i in range(test_end - timesteps - exitTimes):
            test_x.append(enc.transform(self.df[i:i + timesteps]))
            test_y.append(self.label(i))
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        y_raw_pred = model.predict(test_x)
        for i in range(60,y_raw_pred.shape[0]):
            print("==========")
            print(y_raw_pred[i])
            print(test_y[i])
        # print(train_x[0])
        # print(train_y[0])
        # # print(df)
        # print((date[timesteps]))
        # print(len(date))
        # enc = MinMaxScaler()
        # enc.fit(df)
        # train_x=enc.transform(train_x)
        # print(train_x)
        # dump(enc, "./output/scaler.save")
        # print(df)

StockProcess=StockProcess()
StockProcess.stockStrategy(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))