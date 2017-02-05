from projectxc6t73.semantics.semantic_similarity import semantic_similarity
from projectxc6t73.semantics.article_search import daily_keywords
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import logging
import pickle


def add_semantics(ticker, ref_word):
    df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
    dates = list(df[df.columns[0]])
    df_kw = []
    to_drop = []
    size = len(dates)

    for i, date in enumerate(dates):
        logging.info('Collecting keywords for stock %s: %d/%d' % (ticker, i, size))
        keywords = daily_keywords(date.replace("-", ""))
        logging.info('Keywords for stock %s on %s: %s' % (ticker, date, str(keywords)))
        if len(keywords) < 5:
            to_drop.append(i)
        else:
            kw_cols = {}
            for j, keyword in enumerate(keywords):
                kw_cols['kw%d' % j] = semantic_similarity(ref_word, keyword)
            df_kw.append(kw_cols)

    df_kw = pd.DataFrame(df_kw)
    if df_kw.empty:
        logging.warning('Dataframe for stock %s is empty' % ticker)
        return
    print(str(df_kw))
    df = df.join(df_kw)
    print(str(df))
    # Delete Date column
    df = df.drop(df.columns[[0, 1]], axis=1)
    print(str(df))

    # Delete rows with insufficient keywords
    if len(to_drop) > 0:
        df = df.drop(df.index[to_drop])
    print(str(df))

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
    tickers = ['AADR']

    if recalculate:
        set_features(tickers)

    conf_df = []

    for ticker in tickers:
        if os.path.exists('features_dfs/{}_with_features.csv'.format(ticker)):
            df = pd.read_csv('features_dfs/{}_with_features.csv'.format(ticker))
            df.dropna(inplace=True)
            X = np.array(df.drop(['Adj Close'], 1))
            y = np.array(df['Adj Close'])

            X = preprocessing.scale(X)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
            clf = svm.SVR()
            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)
            conf_df.append([ticker, confidence])
            logging.info('Confidence for stock %s: %s' % (ticker, confidence))

    columns = ['Ticker', 'Confidence']
    conf_df = pd.DataFrame(conf_df, columns=columns)
    conf_df = conf_df.set_index('Ticker')
    conf_df.to_csv('stock_confidence.csv')


logging.getLogger().setLevel(logging.INFO)
linear_regression('etf', recalculate=True)
