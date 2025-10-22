import numpy as np
from vnstock import Vnstock


class VnStockDataset():
    def __init__(self, symbol):
        self.symbol = symbol
        self.vnstock = Vnstock().stock(symbol=symbol, source='TCBS')
    
    def load_dataset(self, start, end, features, target_feature):
        data = self.vnstock.quote.history(start=start, end=end, interval='1D')
        data = data[features]
        target_col = data.columns.get_loc(target_feature)
        target_data = data[target_feature]
        return data.values, target_data.values, target_col
    
    def split_dataset(self, data, train_ratio, val_ratio=None):
        if (val_ratio == None):
            return data[:int(len(data) * train_ratio)], data[int(len(data) * train_ratio):]
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        return data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]

    def create_dataset(self, data, lookback, close_col):
        x, y = [], []
        for i in range(lookback, data.shape[0]):
            x.append(data[i - lookback:i])
            y.append(data[i, close_col])
        return np.array(x), np.array(y)

