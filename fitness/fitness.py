from models.lstm import lstm
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error


class Fitness():
    def __init__(self, x_train, y_train, x_val, y_val, input_shape, scaler, epochs, batch_size, learning_rate):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.input_shape = input_shape
        self.scaler =scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def evulate(self, params):
        model = lstm(input_shape=self.input_shape, params=params)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(x=self.x_train, 
                  y=self.y_train, 
                  epochs=self.epochs, 
                  batch_size=self.batch_size)
        y_pred = model.predict(self.x_val)
        y_pred = self.scaler.inverse_transform(y_pred)
        y_val = self.scaler.inverse_transform(self.y_val.reshape(-1, 1))
        val_loss = mean_squared_error(y_pred, y_val)
        return val_loss