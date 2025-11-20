import os
import json
import time
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from metaheuristics.sma import SMA
from metaheuristics.abc import ABC
from fitness.fitness import Fitness
from datasets.vnstock import VnStockDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--symbol', type=str, 
                        help='stock symbol to fetch')
    parser.add_argument('--start', type=str, default='2018-01-01', 
                        help='start date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', 
                        help='end date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--look-back', type=int, default=60, 
                        help='number of previous days used as input for LSTM model')
    parser.add_argument('--metaheuristic', type=str,
                        choices=['abc', 'sma'],
                        help='metaheuristic algorithm used to optimize LSTM hyperparameters')
    parser.add_argument('--metaheuristic-epoch', type=int, default=20,
                        help='number of iterations for metaheuristic')
    parser.add_argument('--pop-size', type=int, default=10,
                        help='population size of SMA optimizer')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for LSTM training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--save-dir', type=str, default='parameters',
                        help='directory to save best parameters')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    stock = VnStockDataset(args.symbol)
    features = ['open', 'high', 'low', 'close', 'volume']
    data, data_close, close_col = stock.load_dataset(start=args.start, end=args.end, features=features, target_feature='close')
    
    look_back = args.look_back

    train, val, _ = stock.split_dataset(data=data, train_ratio=0.65, val_ratio=0.15)
    train_close, val_close, _ = stock.split_dataset(data=data_close, train_ratio=0.65, val_ratio=0.15)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_close = MinMaxScaler(feature_range=(0, 1))

    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    scaler_close.fit(train_close.reshape(-1, 1))

    x_train, y_train = stock.create_dataset(train, look_back, close_col)
    past_days = train[-look_back:]
    val = np.concatenate([past_days, val])
    x_val, y_val = stock.create_dataset(val, look_back, close_col)

    input_shape = (x_train.shape[1], x_train.shape[2])

    fitness = Fitness(
        input_shape=input_shape, 
        x_train=x_train, 
        y_train=y_train, 
        x_val=x_val, 
        y_val=y_val,
        epochs=args.lstm_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    lb = [64, 0.0, 50]
    ub = [256, 0.5, 100]
    n_dims = len(lb)

    if args.metaheuristic == 'abc':
        metaheuristic = ABC(
            obj_func=fitness.evulate, 
            lb=lb, ub=ub, 
            n_dims=n_dims, 
            pop_size=args.pop_size, 
            epochs=args.metaheuristic_epoch,
            limits=(0.2 * args.metaheuristic_epoch)
        )
    elif args.metaheuristic == 'sma':
        metaheuristic = SMA(
            obj_func=fitness.evulate, 
            lb=lb, ub=ub, n_dims=n_dims, 
            pop_size=args.pop_size, 
            epochs=args.metaheuristic_epoch
        )
    
    start = time.time()
    best_params, best_score, history = metaheuristic.solve()
    end = time.time()

    run_time = end - start
    
    print("Run time: ", run_time)
    print("History: ", history)
    print("Best parameters:", best_params)
    print("Best score:", best_score)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_params.json")

    best_params_serializable = best_params.tolist()
    best_score_serializable = float(best_score)

    with open(save_path, "w") as f:
        json.dump({
            "best_params": best_params_serializable,
            "best_fitness": best_score_serializable
        }, f, indent=4)

    print(f"Saved best parameters and best score to {save_path}")
