from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

        
def lstm(input_shape, params):
    units_1 = round(params[1])
    units_2 = round(params[2])
    units_3 = round(params[3])

    dropout = round(params[4])

    model = Sequential()

    model.add(Input(shape=input_shape))


    model.add(LSTM(units=units_1, activation='relu', return_sequences=True))
    model.add(LSTM(units=units_2, activation='relu', return_sequences=True))
    model.add(LSTM(units=units_3, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1))
    return model

