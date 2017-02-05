import shutil
import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import logging
import csv


def save_sp500_tickers(stock_list):
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open('{}_tickers.pickle'.format(stock_list), 'wb') as f:
        pickle.dump(tickers, f)

    return tickers


def save_etf_tickers(path, stock_list):
    df = pd.read_csv(path)
    tickers = df['Symbol'].tolist()

    with open('{}_tickers.pickle'.format(stock_list), 'wb') as f:
        pickle.dump(tickers, f)

    return tickers


def yahoo_data(start_date, end_date, stock_list):
    with open('{}_tickers.pickle'.format(stock_list), 'rb') as f:
        tickers = [x for x in pickle.load(f) if str(x) != 'nan']

    length = len(tickers) - 1
    if os.path.exists('stocks_dfs'):
        shutil.rmtree('stocks_dfs')
    os.makedirs('stocks_dfs')
    for index, ticker in enumerate(tickers):
        logging.info('stock %d/%d: %s' % (index, length, ticker))
        ticker_data(ticker, start_date, end_date)


def ticker_data(ticker, start_date, end_date):
    if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
        try:
            df = web.DataReader(ticker, "yahoo", start_date, end_date)
            df = df[['Adj Close']]
            df.to_csv('stocks_dfs/{}.csv'.format(ticker))
        except:
            logging.error('No stock data found for %s' % ticker)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2017, 1, 1)

    save_etf_tickers('ETFList.csv', 'etf')
    yahoo_data(start, end, 'etf')