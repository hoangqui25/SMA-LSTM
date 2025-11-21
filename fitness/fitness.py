import numpy as np
from models.lstm import lstm
from keras.optimizers import Adam


class Fitness():
    def __init__(self, x_train, y_train, x_val, y_val, input_shape, batch_size, learning_rate):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def evulate(self, params):
        model = lstm(input_shape=self.input_shape, params=params)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        # Train model và lưu lịch sử loss
        history = model.fit(
            x=self.x_train, 
            y=self.y_train, 
            validation_data=(self.x_val, self.y_val),
            epochs=round(params[2]), 
            batch_size=self.batch_size,
        )

        n = max(1, round(round(params[2]) * 0.2))
        val_losses = history.history['val_loss']
        
        fitness_val = np.mean(val_losses[-n:])

        # Kiểm tra NaN hoặc Inf
        if np.isnan(fitness_val) or np.isinf(fitness_val):
            fitness_val = 1

        return fitness_val
