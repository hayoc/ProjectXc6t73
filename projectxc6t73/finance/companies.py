import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import logging


def save_sp500_tickers():
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open('sp500_tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers


def yahoo_data(start_date, end_date, reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500_tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    length = len(tickers) - 1

    for index, ticker in enumerate(tickers):
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, "yahoo", start_date, end_date)
            df = df[['Adj Close']]
            df.to_csv('stocks_dfs/{}.csv'.format(ticker))
            logging.info('stock %d/%d: %s' % (index, length, ticker))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2017, 1, 1)
    yahoo_data(start, end, not os.path.exists('sp500_tickers.pickle'))
