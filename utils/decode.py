def decode(x):
    neurons = int(round(x[0]))
    dropout = float(x[1])
    epochs = int(round(x[2]))
    
    return {
        "epochs": epochs,
        "neurons": neurons,
        "dropout": dropout,
    }
