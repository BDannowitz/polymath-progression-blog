import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from jlab import load_test_data, test_to_time_series
from train import lrelu


def predict(model_filename, data_filename,
            output_filename, ret_pred=False):
    """Make a prediction on test data.
    
    Parameters
    ----------
    model_filename : str
    data_filename : str
    output_filename : str
    ret_pred : bool
        If true, return prediction values
    
    Returns
    -------
    None or numpy.ndarray
    
    """
    
    X_df = load_test_data(data_filename)
    X_ts_array = test_to_time_series(X_df)
    model = load_model(model_filename,
                       custom_objects={'lrelu': lrelu})
    
    pred = model.predict(X_ts_array)
    np.savetxt(output_filename, pred, delimiter=",")

    if ret_pred:
        return pred
    
    return None


if __name__ == "__main__":
    predict(model_filename="dannowitz_jlab2_model.h5",
            data_filename="test_in (1).csv",
            output_filename="danowitz_jlab2_submission.csv")