import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models.lstm import lstm
from keras.optimizers import Adam
from datasets.vnstock import VnStockDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument('--symbol', type=str, 
                        help='stock symbol to fetch')
    parser.add_argument('--start', type=str, default='2018-01-01', 
                        help='start date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', 
                        help='end date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--look-back', type=int, default=60, 
                        help='number of previous days used as input for LSTM model')
    parser.add_argument('--lstm-epoch', type=int, default=50,
                        help='number of epochs for training the LSTM model')
    parser.add_argument('--metaheuristic', type=str,
                        choices=['abc', 'sma'],
                        help='metaheuristic algorithm used to optimize LSTM hyperparameters')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for LSTM training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--load-dir', type=str, default='parameters',
                        help='directory to load parameters')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    stock = VnStockDataset(args.symbol)
    features = ['open', 'high', 'low', 'close', 'volume']
    data, data_close, close_col = stock.load_dataset(start=args.start, end=args.end, features=features, target_feature='close')

    look_back = args.look_back

    train, test = stock.split_dataset(data, train_ratio=0.8)
    train_close, test_close = stock.split_dataset(data_close, train_ratio=0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler_close = MinMaxScaler(feature_range=(0,1))

    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    scaler_close.fit(train_close.reshape(-1, 1))

    x_train, y_train = stock.create_dataset(train, look_back, close_col)
    past_days = train[-look_back:]
    test = np.concatenate([past_days, test])
    x_test, y_test = stock.create_dataset(test, look_back, close_col)

    params_path = os.path.join(args.load_dir, "best_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File {params_path} not found. Please run train.py first.")

    with open(params_path, "r") as f:
        config = json.load(f)

    best_params = config["best_params"]
    print("Loaded best params:", best_params)

    input_shape = (x_train.shape[1], x_train.shape[2])
    model = lstm(input_shape=input_shape, params=best_params)
    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x_train, y_train, epochs=args.lstm_epoch)

    y_pred = model.predict(x_test)
    y_pred = scaler_close.inverse_transform(y_pred)
    y_test = scaler_close.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean absolute error :", mae)
    print("Mean squared error :", mse)
    print("Mean absolute percentage error", mape)
    print("R2 score: ", r2)

    plt.figure(figsize=(12,6))
    plt.title(args.symbol.upper())
    plt.plot(y_test, color='cornflowerblue', label="Actual Price")
    plt.plot(y_pred, color='orange', label=args.metaheuristic.upper() + " LSTM")
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.show()
