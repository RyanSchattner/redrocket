import alpaca_trade_api as tradeapi
import requests
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from pytz import timezone
import threading
import sys
import logging
import btalib
import pandas as pd
import os

from tensorflow import keras
from keras.models import Sequential, model_from_json, save_model, load_model
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.externals import joblib

from alpaca_trade_api import Stream
from alpaca_trade_api.common import URL
from alpaca_trade_api.rest import TimeFrame, REST

logger = logging.getLogger()

global run_model, run_scaler
run_model = load_model("redclient/athena5.h5")
run_scaler = joblib.load('redclient/athena5.pkl')
run_model._make_predict_function()
print('loaded model')

global run_model_day, run_scaler_day
run_model_day = load_model("redclient/athenahour.h5")
run_scaler_day = joblib.load('redclient/athenahour.pkl')
run_model_day._make_predict_function()
print('loaded model')

# Replace these with your API connection info from the dashboard
base_url = 'https://paper-api.alpaca.markets'
api_key_id = 'KEY'
api_secret = 'KEY'

api = tradeapi.REST(
    base_url=base_url,
    key_id=api_key_id,
    secret_key=api_secret
)

session = requests.session()
######################################
##BASE INFO##
######################################

##################DAILY CLASS###################
class RedRocketDaily:
  def __init__(self):
    self.alpaca = tradeapi.REST(api_key_id, api_secret, base_url, 'v2')

    stockUniverse = self.alpaca.list_assets(status='active',asset_class=None)
    #print(stockUniverse)
    # Format the allStocks variable for use in the class.
    self.allStocks = []
    self.allStocks_use=[]
    self.probable_stocks=[]
    self.owned_stock = []
    self.stock_dict={}
    self.stock_dict_minute={}
    for stock in stockUniverse:
      if stock.exchange=='NASDAQ' or stock.exchange=='AMEX' and stock.tradable==True:
        self.allStocks.append(stock.symbol)
    print(len(self.allStocks))
    
    #thread only the stocks to use
    print('start 200 calls')
    #cycles=0
    #threaders = []
    # for stocks in self.allStocks:
      # cycles+=1
      #print(cycles)
      # threads=threading.Thread(target=self.createStockList(stocks=stocks))
      # threads.start()
      # threaders.append(threads)
      #threads.join()
      # if cycles==len(self.allStocks):
        # break
    # for thread in threaders:
      # threads.join()
    # print(len(self.allStocks_use))

    self.long = []
    self.short = []
    self.qShort = None
    self.qLong = None
    self.adjustedQLong = None
    self.adjustedQShort = None
    self.blacklist = set()
    self.longAmount = 0
    self.shortAmount = 0
    self.timeToClose = None

  def run(self):
    print('Start Trading')
    #self.makePrediction([9.9399,9.94,9.90,9.9378,172731,52.030723,9.905261,0.504550,267429.438029,-0.699308], 'UTME')
    # First, cancel any existing orders so they don't impact our buying power.
    # orders = self.alpaca.list_orders(status="open")
    # for order in orders:
    #   self.alpaca.cancel_order(order.id)

    # Rebalance the portfolio every minute, making necessary trades.
    while True:

      # Figure out when the market will close so we can prepare to sell beforehand.
      clock = self.alpaca.get_clock()
      closingTime = clock.next_close
      #print(closingTime)
      currTime = clock.timestamp
      self.timeToClose = closingTime - currTime
      #print(self.timeToClose)

      if closingTime <= closingTime-timedelta(minutes=20):
        # Close all positions when 30 minutes til market close.
        print("Market closing soon.  Closing positions.")

        positions = self.alpaca.list_positions()
        for position in positions:
          if(position.side == 'long'):
            orderSide = 'sell'
          else:
            orderSide = 'buy'
          qty = abs(int(float(position.qty)))
          respSO = []
          tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide))
          tSubmitOrder.start()
          tSubmitOrder.join()

        # Run script again after market close for next trading day.
        print("Sleeping until market close (15 minutes).")
        time.sleep(60 * 30)
      else:
        #Watch positions
        watchPos = threading.Thread(target=self.watchPositions())
        watchPos.start()
        watchPos.join()
        print(len(self.owned_stock))
        print('divergence')
        tRebalance = threading.Thread(target=self.getDivergence())
        tRebalance.start()
        tRebalance.join()
        time.sleep(1)

  #shorten stock list based on volume
  def createStockList(self, stocks):
    stock2 = self.alpaca.get_bars(stocks, TimeFrame.Day,
                                start=pd.Timestamp('now').date()-timedelta(days=3),
                                end=pd.Timestamp('now').date(),
                                limit=3,
                                adjustment='raw'
                                ).df
    #print(stock2)
    try:
      if stock2.iloc[2]['volume']>10000:
        self.allStocks_use.append(stocks)
    except:
      try:
        if stock2.iloc[0]['volume']>10000:
          self.allStocks_use.append(stocks)
      except:
        try:
          if stock2.iloc[1]['volume']>10000:
            self.allStocks_use.append(stocks)
        except:
          pass

  #Keep an eye on open positions for stop loss sale      
  def watchPositions(self):
    positions = self.alpaca.list_positions()
    #print(positions)
    #if len(positions)>0:
    for position in positions:
      if position.symbol not in self.owned_stock:
        self.owned_stock.append(position.symbol)
      print(position.symbol,position.unrealized_plpc)
      if float(position.unrealized_plpc)<-0.02:
        print('dumping')
        qty = abs(int(float(position.qty)))
        self.submitOrder(qty, position.symbol, 'sell')
      elif float(position.unrealized_plpc)>0.7:
        print('taking profit')
        qty = abs(int(float(position.qty)))
        self.submitOrder(qty, position.symbol, 'sell')


  # Get the total price of the array of input stocks.
  def getTotalPrice(self, stocks, resp):
    totalPrice = 0
    for stock in stocks:
      bars = self.alpaca.get_bars(stock, TimeFrame.Minute,
                                  pd.Timestamp('now').date(),
                                  pd.Timestamp('now').date(), limit=1,
                                  adjustment='raw')

      totalPrice += bars[stock][0].c
    resp.append(totalPrice)

  # Submit a batch order that returns completed and uncompleted orders.
  def sendBatchOrder(self, qty, stocks, side, resp):
    executed = []
    incomplete = []
    for stock in stocks:
      if(self.blacklist.isdisjoint({stock})):
        respSO = []
        tSubmitOrder = threading.Thread(target=self.submitOrder, args=[qty, stock, side, respSO])
        tSubmitOrder.start()
        tSubmitOrder.join()
        if(not respSO[0]):
          # Stock order did not go through, add it to incomplete.
          incomplete.append(stock)
        else:
          executed.append(stock)
        respSO.clear()
    resp.append([executed, incomplete])

  # Submit an order if quantity is above 0.
  def submitOrder(self, qty, stock, side):
    if(qty > 0):
      try:
        self.alpaca.submit_order(stock, qty, side, "market", "day")
        self.owned_stock.append(stock)
        print(self.owned_stock)
        #print("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
        #resp.append(True)
      except:
        print('order failed!')
        #print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
        #resp.append(False)
    else:
      pass
      #print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
      #resp.append(True)

  # Get divergent changes of the stock prices over the past 10 minutes.
  def getDivergence(self):
    clock = self.alpaca.get_clock()
    currTime = clock.timestamp
    length = 30

    for stock in self.allStocks:
      try:
        bars = self.alpaca.get_bars(stock, TimeFrame.Minute,
                                    pd.Timestamp('now').date(),
                                    pd.Timestamp('now').date(), limit=length,
                                    adjustment='raw').df
      except:
        pass
      try:                      
        bars=bars.resample('1T').last()
        bars.volume.fillna(0, inplace=True)
        bars.fillna(method='ffill', inplace=True)
        #bars.index=bars.index.tz_convert('America/New_York')
      except:
        pass
      #print(stock)
      if len(bars)>16:
        roc=btalib.roc(bars, period=15)
        rocshort=btalib.roc(bars, period=6)
        smma=btalib.smma(bars, period=15)
        rsi=btalib.rsi(bars, period=15)
        adosc=btalib.adosc(bars)
        bars['roc']=roc.df
        bars['rocshort']=rocshort.df
        bars['rsi']=rsi.df
        bars['smma']=smma.df
        bars['adosc']=adosc.df
      else:
        #print(stock)
        pass
      try:
        if bars['roc'].iloc[-1]>1.5 and stock not in self.owned_stock:
          #print(bars.iloc[-1])
          predobj=bars.iloc[-1]
          price=bars['close'].iloc[-1]
          print(stock)
          predobj2=[predobj['close'],predobj['high'],predobj['low'],predobj['open'],predobj['volume'],predobj['smma'],predobj['rsi'],predobj['roc'],predobj['adosc'],predobj['rocshort']]
          self.makePredictionMinute(predobj2, stock, price)
      except:
        pass
      #print(self.probable_stocks)
      #Sort highest score
    stock_dict_buy=sorted(self.stock_dict_minute, key=self.stock_dict_minute.get, reverse=True)[:10]
    for stock in stock_dict_buy:
      bars2 = self.alpaca.get_bars(stock, TimeFrame.Hour,
                                  pd.Timestamp('now').date()-timedelta(days=3),
                                  pd.Timestamp('now').date(), limit=500,
                                  adjustment='raw').df
      # try:                      
        # bars=bars2.resample('1T').last()
        # bars.volume.fillna(0, inplace=True)
        # bars.fillna(method='ffill', inplace=True)
        #bars.index=bars.index.tz_convert('America/New_York')
      # except:
        # pass
      if len(bars)>16:
        roc=btalib.roc(bars, period=15)
        rocshort=btalib.roc(bars, period=6)
        smma=btalib.smma(bars, period=15)
        rsi=btalib.rsi(bars, period=15)
        adosc=btalib.adosc(bars)
        bbands=btalib.bbands(bars, period=15)
        bars['roc']=roc.df
        bars['rocshort']=rocshort.df
        bars['rsi']=rsi.df
        bars['smma']=smma.df
        bars['adosc']=adosc.df
        bars['top']=bbands['top']
        bars['bot']=bbands['bot']
      else:
        pass
      try:
        #if bars['roc'].iloc[-1]>1.0:
        #print(bars.iloc[-1])
        predobj=bars.iloc[-1]
        price=bars['close'].iloc[-1]
        #print(stock)
        predobj2=[predobj['close'],predobj['high'],predobj['low'],predobj['open'],predobj['volume'],predobj['smma'],predobj['rsi'],predobj['roc'],predobj['adosc'],predobj['rocshort'],predobj['top'],predobj['bot']]
        self.makePrediction(predobj2, stock, price)
      except:
        pass
    stock_dict_buy2=sorted(self.stock_dict, key=self.stock_dict.get, reverse=True)[:5]
    for stock in stock_dict_buy2:
      bars = self.alpaca.get_bars(stock, TimeFrame.Minute,
                                  pd.Timestamp('now').date(),
                                  pd.Timestamp('now').date(), limit=3,
                                  adjustment='raw').df
      if stock not in self.owned_stock and len(self.owned_stock)<10:
        price=bars['close'].iloc[-1]
        qty=round(500/price)
        qty=int(qty)
        print(qty,stock)
        self.submitOrder(qty, stock, 'buy')

  # Mechanism used to rank the stocks, the basis of the Long-Short Equity Strategy.
  def rank(self):
    # Ranks all stocks by percent change over the past 10 minutes (higher is better).
    tGetD = threading.Thread(target=self.getDivergence)
    tGetD.start()
    tGetD.join()

    # Sort the stocks in place by the percent change field (marked by pc).
    self.allStocks_use.sort(key=lambda x: x[0])

  def run_predict(self,model,scaler,sample_data):
    subject_bar=[sample_data]
    subject_bar=scaler.transform(subject_bar)
    class_ind=model.predict_classes(subject_bar)
    class_weights=model.predict(subject_bar)
    estimate = class_weights
    return estimate

  def makePrediction(self, bars, stock, price):
    positions = self.alpaca.list_positions()
    #print(len(positions))
    respSO = []
    stock=stock
    estimateobj=self.run_predict(run_model_day,run_scaler_day,bars)
    finalpred=estimateobj[0][0]*100
    print('predicting',finalpred)
    #if finalpred > 3.0 and len(positions)<10:
    self.stock_dict[stock]=finalpred
    print(self.stock_dict, 'days')    
    qty=round(500/price)
    qty=int(qty)
    #print(qty,stock)
    #self.submitOrder(qty, stock, 'buy')
    #print(finalpred)

  def makePredictionMinute(self, bars, stock, price):
    positions = self.alpaca.list_positions()
    #print(len(positions))
    respSO = []
    stock=stock
    estimateobj=self.run_predict(run_model,run_scaler,bars)
    finalpred=estimateobj[0][0]*100
    print('predicting min',finalpred)
    if finalpred > 0.3:
      self.stock_dict_minute[stock]=finalpred
      print(self.stock_dict_minute, 'minute')    
      qty=round(500/price)
    #qty=int(qty)
    #print(qty,stock)
    #self.submitOrder(qty, stock, 'buy')
    #print(finalpred)