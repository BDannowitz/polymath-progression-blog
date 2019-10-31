import re
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tensorflow.keras.preprocessing.sequence import pad_sequences


COLS = ['x', 'y', 'z', 'px', 'py', 'pz', 'x1', 'y1', 'z1', 'px1', 'py1',
        'pz1', 'x2', 'y2', 'z2', 'px2', 'py2', 'pz2', 'x3', 'y3', 'z3',
        'px3', 'py3', 'pz3', 'x4', 'y4', 'z4', 'px4', 'py4', 'pz4', 'x5',
        'y5', 'z5', 'px5', 'py5', 'pz5', 'x6', 'y6', 'z6', 'px6', 'py6',
        'pz6', 'x7', 'y7', 'z7', 'px7', 'py7', 'pz7', 'x8', 'y8', 'z8',
        'px8', 'py8', 'pz8', 'x9', 'y9', 'z9', 'px9', 'py9', 'pz9', 'x10',
        'y10', 'z10', 'px10', 'py10', 'pz10', 'x11', 'y11', 'z11', 'px11',
        'py11', 'pz11', 'x12', 'y12', 'z12', 'px12', 'py12', 'pz12', 'x13',
        'y13', 'z13', 'px13', 'py13', 'pz13', 'x14', 'y14', 'z14', 'px14',
        'py14', 'pz14', 'x15', 'y15', 'z15', 'px15', 'py15', 'pz15', 'x16',
        'y16', 'z16', 'px16', 'py16', 'pz16', 'x17', 'y17', 'z17', 'px17',
        'py17', 'pz17', 'x18', 'y18', 'z18', 'px18', 'py18', 'pz18', 'x19',
        'y19', 'z19', 'px19', 'py19', 'pz19', 'x20', 'y20', 'z20', 'px20',
        'py20', 'pz20', 'x21', 'y21', 'z21', 'px21', 'py21', 'pz21', 'x22',
        'y22', 'z22', 'px22', 'py22', 'pz22', 'x23', 'y23', 'z23', 'px23',
        'py23', 'pz23', 'x24', 'y24', 'z24', 'px24', 'py24', 'pz24']
N_DETECTORS = 25
N_KINEMATICS = 6


def load_test_data(filename, cols=COLS):
    """Read the specific test file format to a dataframe
    
    All this hullabaloo just to chop off the last number
    of each line.
    
    Parameters
    ----------
    filename : str
    cols : list of str
    
    Returns
    -------
    pandas.DataFrame
    
    """
    with open(filename, 'r') as f:
        data_str = f.read()
    
    data_str_io = StringIO(
        re.sub(r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\n)", r",,\1",
               data_str)
    )
    X_test = pd.read_csv(data_str_io, names=cols)
    
    return X_test


def load_train_test(frac):
    """Load training and validation data from file
    
    Parameters
    ----------
    frac : float
        Percentage of training data to use in training
    
    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Tuple of the training and test data, the training
        and test labels
    """
    
    # Read in raw data
    X_train = (pd.read_csv('MLchallenge2_training.csv')
               .sample(frac=frac)
               .reset_index(drop=True))
    X_test = load_test_data('test_in.csv')

    # Also, load our truth values
    y_test = pd.read_csv('test_prediction.csv',
                        names=['x', 'y', 'px', 'py', 'pz'],
                        header=None)
    
    X_train_array, y_train_array = train_to_time_series(X_train)
    
    y_test_array = y_test.values
    X_test_array = test_to_time_series(X_test)
    
    return X_train_array, X_test_array, y_train_array, y_test_array
    
    
def train_to_time_series(X):
    """Convert training dataframe to multivariate time series training set
    
    Pivots each track to a series ot timesteps. Then randomly truncates them
    to be identical to the provided test set. The step after the truncated
    step is saved as the target.
    
    Truncated sequence are front-padded with zeros.
    
    Parameters
    ----------
    X : pandas.DataFrame
    
    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple of the training data and labels
    """
    
    X_ts_list = []
    n_samples = len(X)
    y_array = np.ndarray(shape=(n_samples, N_KINEMATICS-1))
    for ix in range(n_samples):
        # Randomly choose how many detectors the track went through
        track_len = np.random.choice(range(8, 25))
        # Reshape into ts-like
        track = X.iloc[ix].values.reshape(N_DETECTORS, N_KINEMATICS)
        # Truncate the track to only N detectors
        X_ts_list.append(track[0:track_len])
        # Store the kinematics of the next in the sequence
        # Ignore the 3rd one, which is z
        y_array[ix] = track[track_len][[0,1,3,4,5]]
        
    # Pad the training sequence
    X_ts_list = pad_sequences(X_ts_list, dtype=float)
    X_ts_array = np.array(X_ts_list)
    
    return X_ts_array, y_array


def test_to_time_series(X):
    """Convert test data dataframe into (24, 6) time series.
    
    Time series is front-padded with zeros.
    
    Parameters
    ----------
    X : pandas.DataFrame
    
    Returns
    -------
    numpy.ndarray
        Shape is (len(X), 24, 6)
    
    """
    X_ts_list = []
    for ix in range(len(X)):
        seq_len = get_test_detector_plane(X.iloc[ix])
        track = X.iloc[ix].values.reshape(N_DETECTORS, N_KINEMATICS)
        X_ts_list.append(track[0:seq_len])
        
    # Pad the training sequence
    X_ts_list = pad_sequences(X_ts_list, dtype=float)
    X_ts_array = np.array(X_ts_list)
    
    return X_ts_array

    
def get_test_detector_plane(row):
    """Identifies the number of the plane that'll be evaluated
    
    Surprisingly handy for various data wrangling operations.
    
    Parameters
    ----------
    row : pandas.Series
    
    Returns
    -------
    int
    
    """
    
    # Find location of nans, get the first one
    # Then divide by 6 (6 values per detector plane)
    plane = np.where(np.isnan(row.values))[0][0]/6
    return int(plane)


def plot_one_track_position(df, track_id):
    """For 3D visualization of one track."""
    
    track = df.loc[track_id].values

    x = [track[(6*i)] for i in range(1, 25)]
    y = [track[1+(6*i)] for i in range(1, 25)]
    z = [track[2+(6*i)] for i in range(1, 25)]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(z, x, y)
    ax.set_title("Track {}".format(track_id))
    ax.set_xlabel("z", fontweight="bold")
    ax.set_ylabel("x", fontweight="bold")
    ax.set_zlabel("y", fontweight="bold")

    plt.show()