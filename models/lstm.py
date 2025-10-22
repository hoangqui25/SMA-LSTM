from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

        
def lstm(input_shape, params):
    units1 = int(params[0])
    units2 = int(params[1])
    units3 = int(params[2])
    dense = int(params[3])
    dropout = float(params[4])

    model = Sequential()

    model.add(Input(shape=input_shape))

    if units1 > 0:
        model.add(LSTM(units=units1, activation='relu', return_sequences=True))
    if units2 > 0:
        model.add(LSTM(units=units2, activation='relu', return_sequences=True))
    if units3 > 0:
        model.add(LSTM(units=units3, activation='relu'))
    
    model.add(Dense(units=dense))

    model.add(Dropout(dropout))

    model.add(Dense(units=1))
    return model

