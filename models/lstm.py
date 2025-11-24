from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

        
def lstm(input_shape, params):
    neurons = params['neurons']

    dropout = params['dropout']

    model = Sequential()

    model.add(Input(shape=input_shape))


    model.add(LSTM(units=neurons, activation='relu', return_sequences=True))
    model.add(LSTM(units=neurons, activation='relu', return_sequences=True))
    model.add(LSTM(units=neurons, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1))
    return model

