import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


class DetectorRecommender(object):
    """I can *recommend* not using this predictor :( """
    
    def __init__(self, k=20):
        
        self.logger = logging.getLogger(__name__)
        self.k = k
        self.planes = 27
        self.kinematics = ["x", "y", "px", "py", "pz"]
        self.cols = COLS
        self.X_train = pd.DataFrame(columns=self.cols)
        self.X_test = pd.DataFrame(columns=self.cols)
        self.scaler = StandardScaler()
        
    def fit(self, df):
        """SVD isn't really 'trained', but... """
        
        self.X_train = df.copy(deep=True)
        
    def predict(self, df):
        
        # Make a copy, index it from 0 to N
        self.logger.debug("Making a copy")
        self.X_test = df.copy(deep=True).reset_index(drop=True)
        
        # For each track, figure out which detector plane we'll evaluate
        self.logger.debug("Determining evaluation planes")
        eval_planes = self.X_test.apply(get_eval_detector_plane, axis=1)
        
        # Combine with the training set, shuffle it, and fill missing values
        self.logger.debug("Combining train and test sets for SVD")
        X = (pd.concat([self.X_test, self.X_train], axis=0)
             .reset_index(drop=True)
             .sample(replace=False, frac=1.0))
        
        # Fill with the mean values of each column
        self.logger.debug("Filling with mean values")
        X = X.fillna(X.mean())
        
        # Normalize the values
        self.logger.debug("Applying standardscaler")
        X_norm_values = self.scaler.fit_transform(X)
        X_norm = pd.DataFrame(X_norm_values, columns=X.columns, index=X.index)
        
        # Single-value Decomposition
        self.logger.debug("Making predictions")
        X_pred_norm = self.fit_predict_svds(X_norm)
        
        # Extract our test tracks
        X_pred_norm = X_pred_norm.loc[self.X_test.index, :].sort_index()
        
        # Un-normalize them
        X_pred_values = self.scaler.inverse_transform(X_pred_norm)
        X_pred = pd.DataFrame(X_pred_values, columns=X_pred_norm.columns,
                              index=X_pred_norm.index)
        self.logger.debug("De-normalized. Extracting pred values.")
        
        # Extract just the non-z kinematic values for the eval planes
        det_eval_values = self.extract_values_at_eval_planes(X_pred, eval_planes)
        
        return det_eval_values
    
    def fit_predict_svds(self, X):
        U, sigma, Vt = svds(X, k=self.k)
        sigma = np.diag(sigma)
        X_pred = pd.DataFrame(np.dot(np.dot(U, sigma), Vt),
                              columns=X.columns, index=X.index)
        return X_pred
        
    def extract_values_at_eval_planes(self, pred, planes):
        X = pred.copy(deep=True)
        X['eval_plane'] = planes
        retvals = X.apply(lambda x: self.get_vals_at_plane(x, x['eval_plane']), axis=1)
        retvals_df = pd.DataFrame(retvals.values.tolist(), columns=self.kinematics)
        return retvals_df
    
    def get_vals_at_plane(self, row, plane):
        cols = [i + str(int(plane)) for i in self.kinematics]
        return row[cols].values