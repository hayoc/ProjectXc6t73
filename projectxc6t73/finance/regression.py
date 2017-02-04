from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import pickle
import os
import pandas as pd
import numpy as np

from projectxc6t73.semantics.article_search import daily_keywords
from projectxc6t73.semantics.semantic_similarity import semantic_similarity
import logging


def add_semantics(ticker, ref_word):
    df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
    dates = list(df[df.columns[0]])
    df_kw = []
    size = len(dates)

    for i, date in enumerate(dates):
        logging.info('Collecting keywords for stock %s: %d/%d' % (ticker, i, size))
        keywords = daily_keywords(date.replace("-", ""))
        logging.debug('Keywords for stock %s on %s: %s' % (ticker, date, str(keywords)))
        if len(keywords) < 5:
            df = df.drop(df.index[i])
        else:
            kw_cols = {}
            for j, keyword in enumerate(keywords):
                kw_cols['kw%d' % j] = semantic_similarity(ref_word, keyword)
            df_kw.append(kw_cols)

    df_kw = pd.DataFrame(df_kw)
    df = df.join(df_kw)
    df = df.drop(df.index[0])
    # Delete Date column
    df = df.drop(df.columns[[0, 1]], axis=1)
    df.fillna(value=0, inplace=True)

    if not os.path.exists('features_dfs'):
        os.makedirs('features_dfs')

    df.to_csv('features_dfs/{}_with_features.csv'.format(ticker))


def set_features(tickers):
    ref_word = 'war'
    length = len(tickers)

    for index, ticker in enumerate(tickers):
        logging.info('Adding features for stock %s %d/%d' % (ticker, index, length))
        add_semantics(ticker, ref_word)


def linear_regression(stock_list, recalculate=True):
    # with open('{}_tickers.pickle'.format(stock_list), 'rb') as f:
    #     tickers = pickle.load(f)
    tickers = ['AAL']

    if recalculate:
        set_features(tickers)

    for ticker in tickers:
        if os.path.exists('features_dfs/{}_with_features.csv'.format(ticker)):
            df = pd.read_csv('features_dfs/{}_with_features.csv'.format(ticker))
            X = np.array(df.drop(['Adj Close'], 1))
            y = np.array(df['Adj Close'])

            X = preprocessing.scale(X)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
            clf = svm.SVR()
            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)
            print('Confidence for stock %s: %s' % (ticker, confidence))


logging.getLogger().setLevel(logging.INFO)
linear_regression('sp500', recalculate=False)
