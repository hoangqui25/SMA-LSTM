from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

        
def lstm(input_shape, params):
    units = round(params[0])
    dropout = round(params[1])

    model = Sequential()

    model.add(Input(shape=input_shape))


    model.add(LSTM(units=units, activation='relu', return_sequences=True))
    model.add(LSTM(units=units, activation='relu', return_sequences=True))
    model.add(LSTM(units=units, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1))
    return model

