import itertools

from projectxc6t73.semantics.semantic_similarity import semantic_similarity
from projectxc6t73.semantics.article_search import daily_keywords
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import datetime as dt
import logging
import pickle


def keyword_features(ref_word, dates, recalculate=True):
    df_kw = []
    size = len(dates)
    start_date = dates[0]
    end_date = dates[size - 1]
    if recalculate:
        for i, date in enumerate(dates):
            date = past_date(date)
            logging.info('Collecting keywords for date %s: %d/%d' % (date, i, size))
            keywords = daily_keywords(date.replace("-", ""))
            logging.info('Keywords for date %s: %s' % (date, str(keywords)))
            if len(keywords) == 5:
                kw_cols = {}
                for j, keyword in enumerate(keywords):
                    kw_cols['kw%d' % j] = semantic_similarity(ref_word, keyword)
                df_kw.append(kw_cols)

        df_kw = pd.DataFrame(df_kw)
        if df_kw.empty:
            logging.warning('Keyword Dataframe for daterange %s - %s is empty' % (start_date, end_date))
            return df_kw
        else:
            if not os.path.exists('features_dfs'):
                os.makedirs('features_dfs')
            df_kw.to_csv('features_dfs/keywords_%s_%s' % (start_date, end_date))
            return df_kw
    else:
        file_path = 'keywords_%s_%s' % (start_date, end_date)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame(np.nan, index=[], columns=['kw0', 'kw1', 'kw2', 'kw3', 'kw4'])


def past_date(date):
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    past = date - dt.timedelta(days=7)
    return past.strftime("%Y-%m-%d")


def features_for_prediction(ref_word):
    future_features = []

    present = dt.datetime.today()
    start = present - dt.timedelta(days=7)
    for date in (start + dt.timedelta(days=n) for n in range(7)):
        date = date.strftime("%Y-%m-%d")
        keywords = daily_keywords(date.replace("-", ""))
        if len(keywords) == 5:
            columns = {}
            for j, keyword in enumerate(keywords):
                columns['kw%d' % j] = semantic_similarity(ref_word, keyword)
            future_features.append(columns)

    return np.array(future_features)


def linear_regression(stock_list, recalculate=True):
    # with open('{}_tickers.pickle'.format(stock_list), 'rb') as f:
    #     tickers = pickle.load(f)
    tickers = ['VEGA']
    ref_word = 'war'
    dates = []

    if os.path.exists('stocks_dfs/{}.csv'.format(tickers[0])):
        df = pd.read_csv('stocks_dfs/{}.csv'.format(tickers[0]))
        dates = df['Date']
    if not len(dates) > 0:
        logging.warning('No daterange for %s' % stock_list)
        return
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/confidence'):
        os.makedirs('results/confidence')

    features = keyword_features(ref_word, dates, recalculate)
    conf_df = []


    for ticker in tickers:
        if os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
            df.dropna(inplace=True)

            X = np.array(features)
            y = np.array(df['Adj Close'])
            X = preprocessing.scale(X)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

            clf = svm.SVR()
            clf.fit(X_train, y_train)

            confidence = clf.score(X_test, y_test)
            conf_df.append([ticker, confidence])
            logging.info('Confidence for stock %s: %s' % (ticker, confidence))

            prediction = clf.predict(features_for_prediction(ref_word))
            prediction = dict(zip(dates, prediction))
            prediction = np.array(prediction)
            np.savetxt('results/{}_prediction.csv'.format(ticker), prediction, delimiter=",")

    columns = ['Ticker', 'Confidence']
    conf_df = pd.DataFrame(conf_df, columns=columns)
    conf_df = conf_df.set_index('Ticker')
    conf_df.to_csv('results/confidence/stock_confidence.csv')


logging.getLogger().setLevel(logging.INFO)
linear_regression('etf', recalculate=True)
