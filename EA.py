import MetaTrader5 as mt5
import pandas as pd
import time

# Initialize MT5 terminal
if not mt5.initialize():
    print("Failed to initialize MT5:", mt5.last_error())
else:
    
    account_number = 5030388664 
    password = "Hj*7ZaFj"  
    server = "MetaQuotes-Demo"  
    
    if mt5.login(account_number, password=password, server=server):
        print(f"Successfully logged into demo account {account_number}")
    else:
        error_code, error_message = mt5.last_error()
        print(f"Login failed. Error code: {error_code}, Message: {error_message}")

    # mt5.shutdown()



import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# data = pd.read_csv("15m_EURUSD.csv")
df= mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M1, 0, 10000)
data = pd.DataFrame(df)

# data.to_csv("M5_lg_EURUSD.csv")


# data['time'] = pd.to_datetime(data['time'], unit='s')
data['ATR'] = data.ta.atr(length=14)
data['RSI'] = data.ta.rsi()
data['Average'] = data.ta.midprice(length=1) 
data['SMA_20'] = data.ta.sma(length=20)
data['SMA_50'] = data.ta.sma(length=50)
# data['MA100'] = data.ta.sma(length=100)

data.dropna(inplace=True)

order = 2
max_idx = argrelextrema(data['high'].values, np.greater, order=order)[0]
min_idx = argrelextrema(data['low'].values, np.less, order=order)[0]


labels = np.zeros(len(data)) 
labels[max_idx] = 1 #  (sell signal)
labels[min_idx] = 2  # (buy signal)
data['Signal'] = labels


# X = data.drop('time', axis=1)
data = data.drop('time', axis=1)

# features = ['High', 'Low', 'Close', 'ATR', 'SMA_20', 'SMA_50', 'RSI','Signal']
# X = data[features]


X = data.drop(['Signal'], axis=1)
y = data['Signal']

# data


# print(df)
# labels[:50]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons,make_classification
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ) # random_state=42


# model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=1, metric='minkowski', metric_params=None, n_jobs=-1)

model.fit(X_train, y_train)


def place_order(symbol, lot_size, signal):
    tick = mt5.symbol_info_tick(symbol)
    # if tick is None:
    #     print(f"Failed to get tick data for {symbol}")
    #     return

    price = tick.ask if signal == 'buy' else tick.bid
    entry = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
    deviation = 50  # Specify the allowed deviation in points

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": entry,
        "price": price,
        "deviation": deviation,
        "comment": "Automated trade",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Print the request for debugging
    # print("Sending order:", request)
    result = mt5.order_send(request)

    # if result is None:
    #     print("Order send failed, no result returned.")
    # else:
    #     if result.retcode != mt5.TRADE_RETCODE_DONE:
    #         print(f"Order send failed, retcode: {result.retcode} {mt5.last_error()}")
    #     else:
    #         print("Order placed successfully.")



def close_all_positions():
    positions = mt5.positions_get()
    if positions:
        for position in positions:
            symbol = position.symbol
            ticket = position.ticket
            lot_size = position.volume
            price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": "Close all positions",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)
            # if result.retcode != mt5.TRADE_RETCODE_DONE:
                # print(f"Failed to close order {ticket}, error code: {result.retcode}")
            # else:
                # print(f"Order {ticket} for {symbol} closed successfully!")
# close_all_positions()


import time
def get_latest_candle(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['ATR'] = data.ta.atr(length=14)
    data['RSI'] = data.ta.rsi()
    data['Average'] = data.ta.midprice(length=1) #midprice
    data['SMA_20'] = data.ta.sma(length=20)
    data['SMA_50'] = data.ta.sma(length=50)
    data = data.drop("time",axis=True)
    data.dropna(inplace=True)
    # print(data)
    data = data.iloc[-5:]
    time.sleep(60)
    return data
# get_latest_candle("EURUSD", mt5.TIMEFRAME_M1)



symbol = "EURUSD"
while True:    
    X_new = get_latest_candle(symbol, mt5.TIMEFRAME_M1)
    y_pred = model.predict(X_new)
    
    if y_pred[-1] != 0:
        positions = mt5.positions_get()
        if positions:
            close_all_positions()
            print("Closing all open positions...")
            
        signal = 'buy' if y_pred[-1] == 1 else 'sell'
        for i in range(5):
            place_order(symbol, 0.1, signal)
            print(f"{i} Order placed successfully.")

        print(signal)
        
    print(y_pred)
    # print(X_new)

symbol = "EURUSD"
while True:    
    X_new = get_latest_candle(symbol, mt5.TIMEFRAME_M1)
    y_pred = model.predict(X_new)
    
    if y_pred[-1] != 0:
        positions = mt5.positions_get()
        if positions:
            close_all_positions()
            print("Closing all open positions...")
            
        signal = 'buy' if y_pred[-1] == 1 else 'sell'
        for i in range(5):
            place_order(symbol, 0.1, signal)
            print(f"{i} Order placed successfully.")

        print(signal)
        
    print(y_pred)
    # print(X_new)

