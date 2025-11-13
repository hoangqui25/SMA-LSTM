from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

        
def lstm(input_shape, params):
    units = round(params[0])
    dropout_1 = params[1]
    dropout_2 = params[2]
    dropout_3 = params[3]

    model = Sequential()

    model.add(Input(shape=input_shape))


    model.add(LSTM(units=units, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_1))
    model.add(LSTM(units=units, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_2))
    model.add(LSTM(units=units, activation='relu', return_sequences=False))
    model.add(Dropout(dropout_3))

    model.add(Dense(units=1))
    return model

