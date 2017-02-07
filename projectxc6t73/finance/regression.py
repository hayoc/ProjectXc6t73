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
            keywords = daily_keywords(date.replace("-", ""), i)
            logging.info('Keywords for date %s: %s' % (date, str(keywords)))
            if len(keywords) == 5:
                kw_cols = {}
                for j, keyword in enumerate(keywords):
                    kw_cols['kw%d' % j] = semantic_similarity(ref_word, keyword)
                df_kw.append(kw_cols)
            else:
                df_kw.append({'kw0': 0.0, 'kw1': 0.0, 'kw2': 0.0, 'kw3': 0.0, 'kw4': 0.0})

        df_kw = pd.DataFrame(df_kw)
        if df_kw.empty:
            logging.warning('Keyword Dataframe for daterange %s - %s is empty' % (start_date, end_date))
            return df_kw
        else:
            if not os.path.exists('features_dfs'):
                os.makedirs('features_dfs')
            df_kw.to_csv('features_dfs/keywords_%s_%s.csv' % (start_date, end_date))
            return df_kw
    else:
        file_path = 'features_dfs/keywords_%s_%s.csv' % (start_date, end_date)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame(np.nan, index=[], columns=['kw0', 'kw1', 'kw2', 'kw3', 'kw4'])


def past_date(date):
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    past = date - dt.timedelta(days=7)
    return past.strftime("%Y-%m-%d")


def features_for_prediction(ref_word, recalculate=True):
    future_features = []
    present = dt.datetime.today()
    start = present - dt.timedelta(days=7)
    start_date = start.strftime("%Y-%m-%d")
    end_date = present.strftime("%Y-%m-%d")
    if recalculate:
        for i, date in enumerate((start + dt.timedelta(days=n) for n in range(7))):
            date = date.strftime("%Y-%m-%d")
            keywords = daily_keywords(date.replace("-", ""), i)
            if len(keywords) == 5:
                columns = {'Date': date}
                for j, keyword in enumerate(keywords):
                    columns['kw%d' % j] = semantic_similarity(ref_word, keyword)
                future_features.append(columns)

        if len(future_features) == 0:
            logging.warning('Keyword Dataframe for prediction daterange %s - %s is empty' % (start_date, end_date))
            return pd.DataFrame(np.nan, index=[], columns=['Date', 'kw0', 'kw1', 'kw2', 'kw3', 'kw4']).set_index('Date')
        else:
            future_features = pd.DataFrame(future_features).set_index('Date')
            if not os.path.exists('features_dfs'):
                os.makedirs('features_dfs')
            future_features.to_csv('features_dfs/future_keywords_%s_%s.csv' % (start_date, end_date))
            return future_features
    else:
        file_path = 'features_dfs/future_keywords_%s_%s.csv' % (start_date, end_date)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame(np.nan, index=[], columns=['Date', 'kw0', 'kw1', 'kw2', 'kw3', 'kw4']).set_index('Date')


def linear_regression(stock_list, recalculate=True):
    ref_word = 'war'
    tickers = []
    dates = []
    with open('tickers_pickle/{}_tickers.pickle'.format(stock_list), 'rb') as f:
        tickers = pickle.load(f)
    size = len(tickers)

    if os.path.exists('stocks_dfs/{}.csv'.format(tickers[0])):
        df = pd.read_csv('stocks_dfs/{}.csv'.format(tickers[0]))
        dates = df['Date']
    if not len(dates) > 0:
        logging.warning('No daterange for %s' % stock_list)
        return
    if not os.path.exists('results/confidence'):
        os.makedirs('results/confidence')
    if not os.path.exists('results/prediction'):
        os.makedirs('results/prediction')

    features = keyword_features(ref_word, dates, recalculate)
    features = features.fillna(value=0)
    features = features.drop(features.columns[0], axis=1)
    prediction_features = features_for_prediction(ref_word, recalculate)
    prediction_features = prediction_features.fillna(value=0)

    for i, ticker in enumerate(tickers):
        if os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            try:
                df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
                df = df.fillna(value=0)

                X = np.array(features)
                y = np.array(df['Adj Close'])
                X = preprocessing.scale(X)
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
                clf = svm.SVR()
                clf.fit(X_train, y_train)

                confidence = clf.score(X_test, y_test)
                conf_df = pd.DataFrame([{ticker: confidence}])
                conf_df.set_index(ticker)
                conf_df.to_csv('results/confidence/{}_confidence.csv'.format(ticker))
                logging.info('Confidence for stock %s: %s - %d/%d' % (ticker, confidence, i, size))

                prediction = clf.predict(np.array(prediction_features.drop(['Date'], 1)))
                pred_df = pd.DataFrame({'Date': prediction_features['Date'], 'Prediction': prediction}).set_index('Date')
                pred_df.to_csv('results/prediction/{}_prediction.csv'.format(ticker))
            except ValueError as e:
                logging.error('Failed to get confidence and prediction for stock %s' % ticker)
                logging.error(e)


logging.getLogger().setLevel(logging.INFO)
linear_regression('etf', recalculate=False)