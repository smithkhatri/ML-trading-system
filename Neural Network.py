import yfinance as yf
import pandas as pd

# choose stock ticker
ticker = "TSLA"

# download data
data = yf.download(ticker, start="2010-01-01", end="2026-03-27")

print(data.head())
print(data.tail())
# calculate daily returns
data["Return"] = data["Close"].pct_change()


# 5-day momentum
data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1

# 10-day moving average
data["MA_10"] = data["Close"].rolling(window=10).mean()

# 10-day volatility
data["Volatility_10"] = data["Return"].rolling(window=10).std()


data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1

data['MA_20'] = data['Close'].rolling(window=20).mean()

data['Volatility_20'] = data['Return'].rolling(window=20).std()


data = data.dropna()

# create future return (5 days ahead)
data["Target"] = data["Close"].shift(-5) / data["Close"] - 1

data["Target_Class"] = (data["Target"] > 0).astype(int)

data = data.dropna()

features = ["Return", "Momentum_5", "MA_10", "Volatility_10"]

X = data[features]
y = data["Target_Class"]
# split manually (no shuffle!)
split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# add predictions to dataframe
data_test = data.iloc[split:].copy()

data_test["Prediction"] = y_pred

data_test["Signal"] = data_test["Prediction"]

# shift signal to avoid lookahead bias
data_test["Signal"] = data_test["Signal"].shift(1)

# strategy return
data_test["Strategy_Return"] = data_test["Signal"] * data_test["Return"]
# cumulative returns
data_test["Cumulative_Market"] = (1 + data_test["Return"]).cumprod()
data_test["Cumulative_Strategy"] = (1 + data_test["Strategy_Return"]).cumprod()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(data_test["Cumulative_Market"], label="Market")
plt.plot(data_test["Cumulative_Strategy"], label="Strategy")
plt.legend()
plt.title("Strategy vs Market")
plt.show()
total_return = data_test["Cumulative_Strategy"].iloc[-1]
sharpe = data_test["Strategy_Return"].mean() / data_test["Strategy_Return"].std()
market_return = data_test["Cumulative_Market"].iloc[-1]
print("Total Return:", total_return)
print("Sharpe Ratio:", sharpe)
print("Market Return:", market_return)