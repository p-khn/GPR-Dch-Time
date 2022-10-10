# Math Operations
from typing import Tuple
import numpy as np

# Sklearn
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

class Gpr():
    def __init__(self, random_state=35) -> None:
       self.random_state = random_state

    # Gaussian Process Regression
    def gaussian_process_regression(self, parameters, X_train, X_test, y_train, y_test) -> Tuple:
        gpr = GaussianProcessRegressor(**parameters,
                                    random_state=self.random_state)
        gpr.fit(X_train, y_train)
        y_pred, y_sigma = gpr.predict(X_test, return_std=True)
        score = metrics.mean_squared_error(y_test,y_pred)

        return score, y_pred
     
    # GridSearchCV for Gaussian Process Regression
    def grid_search_cv_gpr(self, X,y, grid_params, cv_n_splits, cv_shuffle, scorer = 'neg_mean_squared_error') -> Tuple:
        cv = KFold(n_splits=cv_n_splits,
                   random_state=self.random_state,
                   shuffle=cv_shuffle)
        gpr = GaussianProcessRegressor(random_state= self.random_state)
        gpr_grid_search = RandomizedSearchCV(gpr,
                                        grid_params,
                                        cv = cv,
                                        random_state=self.random_state,
                                        scoring = scorer,
                                        verbose=1,
                                        n_jobs=-1,
                                        )
        gpr_grid_search.fit(X,y)

        return gpr_grid_search.best_score_, gpr_grid_search.best_params_

    # Gaussian Process Regression Cross Validation
    def gpr_cross_validation(self, X,y, parameters, cv_n_splits, cv_shuffle, scorer='neg_mean_squared_error') -> Tuple:
        cv = KFold(n_splits=cv_n_splits,
                   random_state=self.random_state,
                   shuffle=cv_shuffle)
        gpr = GaussianProcessRegressor(**parameters,
                                    random_state=self.random_state)
        scores = cross_val_score(gpr, X, y,
                                cv=cv,
                                scoring=scorer, verbose=1, n_jobs=-1)

        return np.mean(scores), np.std(scores)
        