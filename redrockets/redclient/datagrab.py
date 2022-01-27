from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import requests
import json
import csv
import pandas as pd
import numpy as np 
import alpaca_trade_api as tradeapi
from alpaca_trade_api import Stream
from alpaca_trade_api.common import URL
from alpaca_trade_api.rest import TimeFrame, REST
from pytz import timezone
import sys
import logging

app = Flask(__name__)

logger = logging.getLogger()

# Replace these with your API connection info from the dashboard
base_url = 'https://data.alpaca.markets'
api_key_id = 'KEY'
api_secret = 'KEY'

api = tradeapi.REST(
    base_url=base_url,
    key_id=api_key_id,
    secret_key=api_secret
)

session = requests.session()

@app.route('/create')
def genfile():
	with open('insert data grab api here', 'w', newline='') as f:
		df=1#the DF data
		thewriter = csv.writer(f)
		thewriter.writerow(['insert column names here'])
		run=0
		for index, row in df.iterrows():
			run+=1
			if run==693928:
				break
                    
				thewriter.writerow(['column vars here'])
					
	return 'Generation complete'

@app.route('/ticker')
def ticker():
	symbol='TSLA'
	bars = api.get_bars(symbol, TimeFrame.Minute, "2021-03-01", "2021-03-02", limit=10, adjustment='raw').df
	trades = api.get_trades(symbol, "2021-03-01", "2021-03-02",limit=10).df
	print(bars,trades)
	return "pulled data"

@app.route('/ticker2')
def ticker2():
	symbol='TSLA'
	stock = api.get_barset(symbol, 'day', start=pd.Timestamp('now').date()-timedelta(days=2),end=pd.Timestamp('now').date(),limit=2).df
	#print(stock[symbol]['volume'])
	if stock.iloc[0][symbol]['volume']>31879810:
		print(stock)
	return('done')

app.run(port=5000)