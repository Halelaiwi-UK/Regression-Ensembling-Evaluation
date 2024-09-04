"""
This script simulates the performance of model ensembles with varying correlation (Rho) values using Monte Carlo simulations and S&P500 data.

The primary goal is to analyze how the combined performance of an ensemble, measured by Rho values,
improves/changes as more models are added to the ensemble.
The script runs multiple Monte Carlo simulations for different desired Rho values and fits
a logistic function to the results to predict optimal ensemble sizes.


However, in this code we assume the following:
-Each model is independent
-All models have similar Rhos, although I may implement a function/class to simulate randomize Rho values/Distributed rho values
-The Rho value and the number of models are related via a logistic function of the form c/(1+e^(-a(x-b))), this assumption os based
 on the fact that the maximum Rho possible is 1 and minimum is -1.

Key Features:
- Generates synthetic data with controlled correlations to the target variable.
- Simulates ensemble performance using Ridge regression as a meta-learner.
- Visualizes the relationship between the number of models in an ensemble and the resulting combined Rho.
"""

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize, curve_fit
from sqlalchemy import create_engine, false
from statsmodels.genmod.families.links import Logit
from statsmodels.sandbox.predict_functional import predict_functional
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import random


def logistic_function(x_value, growth, offset, maximum) -> float:
    """
     Used to find the best logistic fit for our simulation and actual run on the S&P500 data

     Parameters:
     None, They are automatically fitted via another function

     Returns:
        Float: The expected Rho value at a specific X_value (Number of models used that each have a specified rho)
    """

    return maximum/(1+np.exp(-growth*(x_value-offset)))


def generate_features(n_of_samples: int, n_features: int, y_normalized: pd.Series, desired_rho:float =0.1) -> np.array:
    """
     Generates N_features features such that each feature has a Rho correlation to target equal to desired_rho

     Parameters:
     n_of_samples (int): Number of samples to generate
     n_features (int): Number of features to generate
     y_normalize (pd.Series): Series of y normalized values to use for rho correlation
     desired_rho (float): Desired correlation between the generated features and y_normalized

     Returns:
         np.array: a numpy array of dimensions (n_of_samples, n_features)
    """
    seed = random.randint(0, 10000)
    np.random.seed(seed)
    features_to_generate = np.zeros((n_of_samples, n_features))

    for n in range(n_features):
        # Generate a random rho for each feature to add diversity
        desired_rho = np.clip(np.random.uniform(desired_rho * 0.9, desired_rho * 1.1), -1, 1)

        # Generate random feature
        x_random = np.random.randn(n_of_samples)

        # Normalize the random feature
        x_random_normalized = (x_random - x_random.mean()) / x_random.std()

        # Add controlled noise (simulating real-world imperfections)
        noise = np.random.normal(0, 0.1, n_of_samples)  # Noise with mean 0 and small std dev

        # Create a linear combination of y_normalized and the random feature with added noise
        feature = desired_rho * y_normalized + np.sqrt(1 - desired_rho ** 2) * x_random_normalized + noise
        # Add non-linear feature occasionally
        # if n % 3 == 1:
        #     feature = np.exp(feature / 10)

        # Normalize the generated feature
        feature = (feature - feature.mean()) / feature.std()

        # Create adjusted feature with desired correlation
        features_to_generate[:, n] = feature

    return features_to_generate

def ridge_meta_learner(ols_predictions_train: list, ols_predictions_test: list, y_value_train: pd.Series) -> np.array:
    """
     Trains and fits a ridge regression model to predict y values using OLS regression model's predictions.

     Parameters:
     ols_predictions_train (dictionary): A dictionary containing the predictions of each model, split for training
     ols_predictions_test (dictionary): A dictionary containing the predictions of each model, split for testing
     y_value_train (pd.Series): Series of y values to use for training the ridge model
     Returns:
         np.array: a numpy array containing the final predictions of the ridge model
    """
    x_meta_train = np.column_stack(ols_predictions_train)
    x_meta_test = np.column_stack(ols_predictions_test)
    ridge_model = Ridge(alpha=10)
    ridge_model.fit(x_meta_train, y_value_train)
    meta_predictions = ridge_model.predict(x_meta_test)
    return meta_predictions

# PostgreSQL connection setup
DATABASE_URL = "postgresql+psycopg2://postgres@localhost:5432/sp500"
engine = create_engine(DATABASE_URL)

# Read S&P500 Data
query = "SELECT * FROM sp500_index"
regression_variables = pd.read_sql(query, engine)
regression_variables['Date'] = pd.to_datetime(regression_variables['Date'])
regression_variables.set_index('Date', inplace=True)
regression_variables = regression_variables.sort_index()
y = regression_variables["S&P500"]

# Parameters
n_of_features = 20
test_length = int(len(y) * 0.2)
max_N_Models = 100
num_samples = len(y)
rho_values = [0.1, 0.2, 0.3, 0.4]
y_train, y_test = y[test_length:], y[:test_length]

# Normalize y to have zero mean and unit variance
y_train_std = y_train.std()
y_train_mean = y_train.mean()
y_test = (y_test - y_train_mean) / y_train_std
y_train = (y_train - y_train_mean) / y_train_std

#ta lib

# N of models with technicals using Combinatorics, The rho value
# Try with different meta learners
# Change values to relative percentage change


# Run model on S&P500 y values
for rho in rho_values:
    print(f"Running on Rho = {rho}...")
    rho_with_n_models = []
    # Train models
    models = []
    predictions_train = []
    predictions_test = []
    for i in range(1, max_N_Models+1):
            features_train = generate_features(len(y_train), n_of_features, y_train, desired_rho=rho)
            features_test = generate_features(len(y_test), n_of_features, y_test, desired_rho=rho)
            features_train = sm.add_constant(features_train)
            model = sm.OLS(y_train, features_train).fit()
            features_test = sm.add_constant(features_test)
            models.append(model)
            predictions_train.append(model.predict(features_train))
            predictions_test.append(model.predict(features_test))

    # Fit and train meta-learner on a specified number of models (1 to Max_N_models)
    for i in range(1, max_N_Models+1):
        N_models = i
        pred_train = predictions_train[:N_models]
        pred_test = predictions_test[:N_models]
        # Meta learner used for ensambling, currently Ridge learner
        meta_learner_predictions = ridge_meta_learner(pred_train, pred_test, y_train)
        rho_avg = pearsonr(y_test, meta_learner_predictions)[0]
        rho_with_n_models.append(rho_avg)


    # Plotting

    # Fit the logistic function to the data
    num_models_range = np.arange(1, max_N_Models + 1)
    # noinspection PyTupleAssignmentBalance
    popt, _ = curve_fit(logistic_function, num_models_range, rho_with_n_models, p0=[1, (max_N_Models/2), 0.5], maxfev=10000)

    # fitted parameters
    a,b,c = popt
    print(f'Logistic Function Parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}')

    epsilon = 1e-3
    try:
        x_n = b - (1/a) * math.log(epsilon/(c*a))
        print(f"--------Actual for Rho = {rho}--------")
        print(f"for e = {epsilon}, x should be greater than: {x_n:.4f}")
        print(f"Recommend N of models = {x_n}")
        print(f"Ideal rho is somewhere near : {c} (BASED ON PREVIOUS DATA)")
        print()
    except ValueError as e:
        print(f"--------Actual for Rho = {rho}--------")
        print(f"Ideal rho is somewhere near : {c} (BASED ON PREVIOUS DATA)")
        print()
        x_n = -1
    # Generate data for plotting the fit
    x_fit = np.linspace(num_models_range.min(), num_models_range.max(), 500)
    y_fit = logistic_function(x_fit, *popt)


    plt.figure(figsize=(10, 6))
    plt.plot(x_fit, y_fit, '-', label='Logistic Fit', color='r')
    plt.plot(range(1, max_N_Models + 1), rho_with_n_models, color='b')
    # Export the data to a text file
    np.savetxt(f"Rho_{rho}.csv", np.column_stack([range(1, max_N_Models + 1), rho_with_n_models]), delimiter=",", header="x,y", comments="")

    plt.title(f'Rho = {rho}~')
    plt.xlabel('N Models Combined')
    plt.ylabel('Rho as Models are combined')
    # Add logistic function parameters to the side of the plot
    plt.text(x=plt.xlim()[1] * 0.6, y=plt.ylim()[1] * 0.3,
             s=f'Logistic Function Parameters:\na={a:.4f}\nb={b:.4f}\nc={c:.4f}\n'
               f'Recommend N models for e= {epsilon}: {int(x_n)}',
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=12, color='black')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()



# Plot all results on one graph

# File paths
file_rho_01 = 'Rho_0.1.csv'
file_rho_02 = 'Rho_0.2.csv'
file_rho_03 = 'Rho_0.3.csv'
file_rho_04 = 'Rho_0.4.csv'

# Load the datasets
rho_01_df = pd.read_csv(file_rho_01)
rho_02_df = pd.read_csv(file_rho_02)
rho_03_df = pd.read_csv(file_rho_03)
rho_04_df = pd.read_csv(file_rho_04)

# Plot for Rho 0.1
plt.plot(rho_01_df['x'], rho_01_df['y'], label='Rho = 0.1')

# Plot for Rho 0.2
plt.plot(rho_02_df['x'], rho_02_df['y'], label='Rho = 0.2')

# Plot for Rho 0.3
plt.plot(rho_03_df['x'], rho_03_df['y'], label='Rho = 0.3')

# Plot for Rho 0.4
plt.plot(rho_04_df['x'], rho_04_df['y'], label='Rho = 0.4')

# Adding titles and labels
plt.grid(True)
plt.title('Combined Rho Values vs Number of Models Used in Ensemble')
plt.xlabel('Number of Models')
plt.ylabel('Combined Rho Value')
plt.xlim(xmin=1, xmax=max_N_Models+1)
plt.legend()
plt.show()