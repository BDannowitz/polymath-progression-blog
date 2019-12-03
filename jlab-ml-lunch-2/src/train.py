import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LeakyReLU, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import EarlyStopping
from jlab import load_train_test, N_KINEMATICS, N_DETECTORS, N_FEATURES


def lrelu(x):
    return LeakyReLU()(x)


def gru_model(gru_units=35, dense_units=100,
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
                  input_shape=(N_DETECTORS-1, N_FEATURES),
                  return_sequences=True))
    model.add(GRU(gru_units, activation=lrelu,
                  return_sequences=True))
    model.add(GRU(gru_units, activation=lrelu))
    
    model.add(Dense(dense_units, activation=lrelu))
    model.add(Dropout(dropout_rate))
    model.add(Dense(N_KINEMATICS-1, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam')
    
    return model


def train(frac, filename, epochs=50, gru_units=35,
          dense_units=100, dropout_rate=0.25, ret_model=False):
    """Load the data, model, train it, and export it.
    
    Parameters
    ----------
    frac : float
        Fraction of training data to use
    filename : str
        Name of exported model file
    epochs : int
        Number of epochs to repeat through the training set
    gru_units : int
    dense_units : int
    dropout_rate : float
    ret_model : bool
        If true, return the trained model object
        
    Returns
    -------
    keras.model.Sequential
    
    """
    
    X_train, X_test, y_train, y_test = load_train_test(frac=frac)
    
    model = gru_model(gru_units=gru_units,
                      dense_units=dense_units,
                      dropout_rate=dropout_rate)
    print(model.summary())
    
    # This will keep on training until the validation loss doesn't
    # decrease for five straight epochs. At that point, it will
    # rever the weights back to the epoch with the lowest val_loss
    es = EarlyStopping(monitor='val_loss', mode='min',
                       patience=10, restore_best_weights=True)
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

    train(frac=1.0, filename="../models/dannowitz_jlab2_model_20191031", epochs=100,
          gru_units=35)
    
    train_report = """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    gru (GRU)                    (None, 24, 35)            4515      
    _________________________________________________________________
    gru_1 (GRU)                  (None, 24, 35)            7560      
    _________________________________________________________________
    gru_2 (GRU)                  (None, 35)                7560      
    _________________________________________________________________
    dense (Dense)                (None, 100)               3600      
    _________________________________________________________________
    dropout (Dropout)            (None, 100)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 5)                 505       
    =================================================================
    Total params: 23,740
    Trainable params: 23,740
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 194601 samples, validate on 10000 samples
    Epoch 1/100
    194601/194601 [==============================] - 280s 1ms/sample - loss: 2.8263 - val_loss: 0.5630
    Epoch 2/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.6813 - val_loss: 0.7487
    Epoch 3/100
    194601/194601 [==============================] - 269s 1ms/sample - loss: 1.4223 - val_loss: 0.2651
    Epoch 4/100
    194601/194601 [==============================] - 268s 1ms/sample - loss: 1.2708 - val_loss: 0.3310
    Epoch 5/100
    194601/194601 [==============================] - 268s 1ms/sample - loss: 1.2546 - val_loss: 0.2873
    Epoch 6/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.2097 - val_loss: 0.1148
    Epoch 7/100
    194601/194601 [==============================] - 265s 1ms/sample - loss: 1.2088 - val_loss: 0.3540
    Epoch 8/100
    151104/194601 [======================>.......] - 273s 1ms/sample - loss: 1.1902 - val_loss: 0.2705
    Epoch 9/100
    194601/194601 [==============================] - 272s 1ms/sample - loss: 1.1735 - val_loss: 0.1096
    Epoch 10/100
    194601/194601 [==============================] - 269s 1ms/sample - loss: 1.1661 - val_loss: 0.0866
    Epoch 11/100
    194601/194601 [==============================] - 268s 1ms/sample - loss: 1.1637 - val_loss: 0.1078
    Epoch 12/100
    194601/194601 [==============================] - 1743s 9ms/sample - loss: 1.1628 - val_loss: 0.0706
    Epoch 13/100
    194601/194601 [==============================] - 269s 1ms/sample - loss: 1.1527 - val_loss: 0.1107
    Epoch 14/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1482 - val_loss: 0.1148
    Epoch 15/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1464 - val_loss: 0.2808
    Epoch 16/100
    194601/194601 [==============================] - 269s 1ms/sample - loss: 1.1386 - val_loss: 0.1789
    Epoch 17/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1387 - val_loss: 0.0631
    Epoch 18/100
    194601/194601 [==============================] - 268s 1ms/sample - loss: 1.1375 - val_loss: 0.0716
    Epoch 19/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1390 - val_loss: 0.0532
    Epoch 20/100
    194601/194601 [==============================] - 268s 1ms/sample - loss: 1.1378 - val_loss: 0.0954
    Epoch 21/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1336 - val_loss: 0.0617
    Epoch 22/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1310 - val_loss: 0.1612
    Epoch 23/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1340 - val_loss: 0.1290
    Epoch 24/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1249 - val_loss: 0.1523
    Epoch 25/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1229 - val_loss: 0.0744
    Epoch 26/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1220 - val_loss: 0.0684
    Epoch 27/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1257 - val_loss: 0.0450
    Epoch 28/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1188 - val_loss: 0.0811
    Epoch 29/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1162 - val_loss: 0.1055
    Epoch 30/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1107 - val_loss: 0.1243
    Epoch 31/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1223 - val_loss: 0.0406  <<<<<< Top performer
    Epoch 32/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1138 - val_loss: 0.0479
    Epoch 33/100
    194601/194601 [==============================] - 265s 1ms/sample - loss: 1.6953 - val_loss: 1.0979
    Epoch 34/100
    194601/194601 [==============================] - 268s 1ms/sample - loss: 1.7297 - val_loss: 0.4382
    Epoch 35/100
    194601/194601 [==============================] - 264s 1ms/sample - loss: 1.3096 - val_loss: 0.1320
    Epoch 36/100
    194601/194601 [==============================] - 264s 1ms/sample - loss: 1.1588 - val_loss: 0.1763
    Epoch 37/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1536 - val_loss: 0.0742
    Epoch 38/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1465 - val_loss: 0.0749
    Epoch 39/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1279 - val_loss: 0.2036
    Epoch 40/100
    194601/194601 [==============================] - 267s 1ms/sample - loss: 1.1167 - val_loss: 0.0739
    Epoch 41/100
    194601/194601 [==============================] - 266s 1ms/sample - loss: 1.1182 - val_loss: 0.1356
    
    """
