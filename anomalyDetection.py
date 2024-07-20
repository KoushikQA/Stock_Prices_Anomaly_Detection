import importPackages
import

# Introduction to Anomaly Detection in Time Series with Keras

path = 'SPY.csv'

df = pd.read_csv(path)
df.head()

df.shape

plt.plot(df.Date, df.Close)
plt.show()

# Data Preprocessing

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])

train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

# Create Time-Series Data

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values#.reshape(-1)
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30

X_train, y_train = create_dataset(train[['Close']], train.Close, time_steps)
X_test, y_test = create_dataset(test[['Close']], test.Close, time_steps)

print(X_train.shape)
print(y_train.shape)

# Build an LSTM Autoencoder
# Autoencoder should take a sequence as input and outputs a sequence of the same shape.

timesteps = X_train.shape[1]
num_features = X_train.shape[2]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

model = Sequential([
    LSTM(128, input_shape=(timesteps, num_features)),
    Dropout(0.2),
    RepeatVector(timesteps),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(num_features))
])

model.compile(loss='mae', optimizer='adam')
model.summary()

# Train the Auto-Encoder

model.compile(loss='mae', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                    shuffle=False)

# Plot Metrics and Evaluate the Model

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend();

X_train_pred = model.predict(X_train)
train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])

model.evaluate(X_test, y_test)

sns.distplot(train_mae_loss, bins=50, kde=True);

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

sns.distplot(test_mae_loss, bins=50, kde=True);

# Detect Anomalies in the S&P 500 Index Data

THRESHOLD = 0.65

test_score_df = pd.DataFrame(test[time_steps:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['Close'] = test[time_steps:].Close

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()


# Data Preparation

# !pip -qq install yfinance

import pandas as pd
import numpy as np
import yfinance as yf

spy_ohlc_df = yf.download('SPY', start='1993-02-01', end='2021-06-01')

spy_ohlc_df.head()

spy_ohlc_df.tail()

spy_ohlc_df.shape

spy_ohlc_df.reset_index(inplace=True)

spy_ohlc_df.to_csv("SPY.csv",index=False)


