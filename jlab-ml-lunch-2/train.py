import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LeakyReLU, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import EarlyStopping
from jlab import load_train_test, N_KINEMATICS, N_DETECTORS


def lrelu(x):
    return LeakyReLU()(x)

def gru_model(gru_units=30, dense_units=100,
              dropout_rate=0.25):
    """Model definition.
    
    Three layers of Gated Recurrent Units (GRUs), utilizing
    LeakyReLU activations, finally passing GRU block output
    to a dense layer, passing its output to the final output
    layer, with a touch of dropout in between.
    
    Bon apetit.
    
    Parameters
    ----------
    gru_units : int
    dense_units : int
    dropout_rate : float
    
    Returns
    -------
    tensorflow.keras.models.Sequential
    
    """
    
    model = Sequential()
    
    model.add(GRU(gru_units, activation=lrelu,
                  input_shape=(N_DETECTORS-1, N_KINEMATICS),
                  return_sequences=True))
    model.add(GRU(gru_units, activation=lrelu,
                  return_sequences=True))
    model.add(GRU(gru_units, activation=lrelu))
    
    model.add(Dense(dense_units, activation=lrelu))
    model.add(Dropout(dropout_rate))
    model.add(Dense(N_KINEMATICS-1))
    
    model.compile(loss='mse', optimizer='adam')
    
    return model


def train(frac, filename, epochs=50, ret_model=False):
    """Load the data, model, train it, and export it.
    
    Parameters
    ----------
    frac : float
        Fraction of training data to use
    filename : str
        Name of exported model file
    epochs : int
        Number of epochs to repeat through the training set
    ret_model : bool
        If true, return the trained model object
        
    Returns
    -------
    keras.model.Sequential
    
    """
    
    X_train, X_test, y_train, y_test = load_train_test(frac=frac)
    
    model = gru_model()
    print(model.summary())
    
    # This will keep on training until the validation loss doesn't
    # decrease for five straight epochs. At that point, it will
    # rever the weights back to the epoch with the lowest val_loss
    es = EarlyStopping(monitor='val_loss', mode='min',
                       patience=5, restore_best_weights=True)
    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_test, y_test),
        callbacks=[es],
        epochs=epochs,
        use_multiprocessing=True,
    )
    
    # Output the model and the train/val loss history
    model.save(f"{filename}.h5")
    joblib.dump(history.history, f"{filename}.history")
    
    if ret_model:
        return model

    return None
    
    
if __name__ == "__main__":
    train(frac=1.0, filename="dannowitz_jlab2_model", epochs=100)