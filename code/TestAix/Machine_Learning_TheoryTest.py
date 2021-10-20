import pandas as pd
import quandl
import math

quandl.ApiConfig.api_key= "iMXwtMoPrsEpyQz5U7-X"
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df ['Adj. Low'] * 100 #High - low / close price
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] * 100 #New - Old / Old price 

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']] #List all the variables we anted to check

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  #Fill not available

forecast_out = int(math.ceil(0.1*len(df))) # take anything and get to the ceiling/ rounds up to the nearest whole
# Predicts out 0.1 or 10% of the future data | next 10 days

df['label'] = df[forecast_col].shift(-forecast_col) #shift the column up
df.dropna(inplace=True)
print(df.tail())