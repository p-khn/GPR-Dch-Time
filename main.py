# Libraries
import preprocessing as prp
from gpr import Gpr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from joblib import load


RANDOM_STATE = 35
scaler = MinMaxScaler()

# Get data
_target = 'D_T'
data = prp.get_data()

# Feature selection using Recursive Feature Elimination
data = prp.feature_selection_RFE(data, _target, 10)


# Separate independent variables from target variable 
X, y = prp.separate_it(data, _target)

# Scale data
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE)


# Gaussian Process Regressor instance 
gpr1 = Gpr()


# Get best parameters
kernel1 = 1.0 * RBF(length_scale=1,
                    length_scale_bounds=(1e-3, 1e4)) + WhiteKernel(noise_level=1e-5,
                                                                   noise_level_bounds=(1e-10, 1e1))

grid_params = dict(
    kernel=[kernel1],
    alpha=[1e-5, 1e-6, 1e-7],
    n_restarts_optimizer=[30, 35, 40, 55, 60]
)

gpr1.grid_search_cv_gpr(X, y, grid_params, 5, True)

best_parameters = load('gpr_params.joblib')
if best_parameters:
    # Train and Evaluate
    score, y_pred = gpr1.gaussian_process_regression(
        best_parameters, X_train, X_test, y_train, y_test)

    # Cross Validation
    scores_mean, scores_std = gpr1.gpr_cross_validation(
        X, y, best_parameters, 5, True)

    print(f'\nThe best parameters are: {best_parameters}')
    print('\n-------------------------------------------------')
    print(f'The MSE score for GPR model: {score}')
    print('\n-------------------------------------------------')
    print(f'The average score of K-Fold cross validation: {scores_mean}\n')
else:
    print('Cannot find the parameters in current directory!!!')

