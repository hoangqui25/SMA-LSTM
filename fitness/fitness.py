import numpy as np
from models.lstm import lstm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


class Fitness():
    def __init__(self, x_train, y_train, x_val, y_val, input_shape, epochs, batch_size, learning_rate, min_delta):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_delta = min_delta
    
    def evulate(self, params):
        model = lstm(input_shape=self.input_shape, params=params)
        optimizer = Adam(learning_rate=self.learning_rate)
        early_stopping = EarlyStopping (
            monitor='val_loss',
            patience=10,
            min_delta=self.min_delta,
            restore_best_weights=True
        )
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(x=self.x_train, 
                  y=self.y_train, 
                  validation_data=(self.x_val, self.y_val),
                  epochs=self.epochs, 
                  batch_size=self.batch_size,
                  callbacks=[early_stopping]
        )

        y_pred = model.predict(self.x_val)
        val_loss = mean_squared_error(y_pred, self.y_val)

        # Check if val_loss is NaN or Inf
        if np.isnan(val_loss) or np.isinf(val_loss):
            val_loss = 1e10
        return val_loss