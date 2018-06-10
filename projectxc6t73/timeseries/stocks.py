import quandl
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense
from pandas import DataFrame, concat, Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return Series(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)

    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X = divisible_by_batch_size(X, batch_size)
    y = divisible_by_batch_size(y, batch_size)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print('Epoch ' + str(i) + ' out of ' + str(nb_epoch) + ' done')
    return model


def divisible_by_batch_size(array, batch_size):
    while array.shape[0] % batch_size != 0:
        array = numpy.delete(array, [0, 0, 0], axis=0)
    return array



def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def copy_model(old_model, X, n_neurons, batch_size):
    new_model = Sequential()
    new_model.add(LSTM(n_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    new_model.add(Dense(1))
    old_weights = old_model.get_weights()
    new_model.set_weights(old_weights)
    return new_model


quandl.ApiConfig.api_key = "pRE8Gk-6DwGSYS39N5Nc"
series = quandl.get("EIA/PET_RWTC_D", collapse="daily")
print('fetched data')
raw_values = series.values

#diff_values = difference(raw_values, 1)

supervised = timeseries_to_supervised(raw_values, 1)
supervised_values = supervised.values

# f, (ax1, ax2) = plt.subplots(2, sharex='all')
# ax1.plot(supervised_values, label='stuff')
# plt.show()

train, test = train_test_split(supervised_values, test_size=0.1, shuffle=False)
raw_train, raw_test = train_test_split(raw_values, test_size=0.1, shuffle=False)

scaler, train_scaled, test_scaled = scale(train, test)
print('data prep done')

# fit the model
lstm_train_model = fit_lstm(train_scaled, 64, 10, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_predict_model = copy_model(lstm_train_model, train_reshaped, 4, 1)
lstm_predict_model.predict(train_reshaped, batch_size=1)
# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_predict_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    #yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
# report performance
rmse = numpy.sqrt(mean_squared_error(raw_test, predictions))
print('Test RMSE: %.3f' % rmse)

f, (ax1, ax2) = plt.subplots(2, sharex='all')

ax1.plot(predictions, label='Predictions')

ax2.plot(raw_test, label='Data Test')

plt.show()

