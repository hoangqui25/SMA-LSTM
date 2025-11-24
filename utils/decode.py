def decode(x):
    epochs = int(round(x[0]))
    neurons = int(round(x[1]))
    dropout = float(x[2])
    
    return {
        "epochs": epochs,
        "neurons": neurons,
        "dropout": dropout,
    }
