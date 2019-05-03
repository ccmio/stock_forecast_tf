import pandas as pd
import tushare
import os
import datetime


def stock_price_intraday(ticker, folder):
    intraday = tushare.get_hist_data(ticker, ktype='5')
    temp_file = folder + '/' + ticker + '.csv'
    if os.path.exists(file):
        history = pd.read_csv(file, index_col=0)
        intraday.append(history)
    intraday.sort_index(inplace=True)
    intraday.index.name = 'timestamp'
    intraday.to_csv(temp_file)
    print('intraday for [', ticker, '] got.')


tickers_raw_data = tushare.get_stock_basics()
tickers = tickers_raw_data.index.tolist()
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = 'C:/Users/cacho/Desktop/python_work/M1569/data/TickerList_' + dateToday + '.csv'
tickers_raw_data.to_csv(file)

for i, value in enumerate(tickers):
    try:
        print('Intrady', i, '/', len(tickers))
        stock_price_intraday(value, folder='C:/Users/cacho/Desktop/python_work/M1569/data/')
    except:
        pass
print('ALL done.')


