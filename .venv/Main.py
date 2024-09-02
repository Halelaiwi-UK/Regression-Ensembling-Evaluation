import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize, curve_fit
from sqlalchemy import create_engine
from statsmodels.genmod.families.links import Logit
from statsmodels.sandbox.predict_functional import predict_functional
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

np.random.seed(1)

# Monte Carlo simulation for one iteration
def monte_carlo_iteration(rho, max_N_Models, n_of_features, test_length, num_samples):
    y = np.random.randn(num_samples)
    y = (y - y.mean()) / y.std()  # Normalize y
    y_train, y_test = y[:test_length], y[test_length:]

    rho_results = []

    for i in range(1, max_N_Models + 1):
        predictions = {}
        for j in range(1, i + 1):
            features = generate_features(num_samples, n_of_features, y, desired_rho=rho)
            features_train, features_test = features[:test_length], features[test_length:]
            features_train = sm.add_constant(features_train)
            model = sm.OLS(y_train, features_train).fit()
            features_test = sm.add_constant(features_test)
            prediction = model.predict(features_test)
            predictions[j] = prediction

        meta_learner_predictions = ridge_meta_learner(predictions, y_test)
        rho_avg = pearsonr(y_test, meta_learner_predictions)[0]
        rho_results.append(rho_avg)

    return rho_results


def logistic_function(x_value, growth, offset, maximum):
    return maximum/(1+np.exp(-growth*(x_value-offset)))


def generate_features(n_of_samples, n_features, y_normalized, desired_rho=0.1):
    features_to_generate = np.zeros((n_of_samples, n_features))

    for n in range(n_features):
        # Generate random feature
        x_random = np.random.randn(n_of_samples)

        # Normalize the random feature
        x_random_normalized = (x_random - x_random.mean()) / x_random.std()

        # Create adjusted feature with desired correlation
        features_to_generate[:, n] = (
                desired_rho * y_normalized +
                np.sqrt(1 - desired_rho ** 2) * x_random_normalized
        )

    return features_to_generate

def ridge_meta_learner(ols_predictions, y_value):
    x_meta = np.column_stack(list(ols_predictions.values()))
    ridge_model = Ridge(alpha=10)
    ridge_model.fit(x_meta, y_value)
    meta_predictions = ridge_model.predict(x_meta)
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

# Normalize y to have zero mean and unit variance
y = (y - y.mean()) / y.std()


# Parameters
n_of_features = 10
test_length = int(len(y) * 0.2)
max_N_Models = 50
y_test, y_train = y[test_length:], y[:test_length]
num_samples = len(y)
rho_values = [0.1, 0.2, 0.3, 0.4]


# Monte Carlo simulation parameters
n_iterations = 1000  # Number of Monte Carlo iterations

# Parallel processing using joblib
simulation_results = {}

for rho in rho_values:
    print(f"Rho {rho} simulation...")
    results = Parallel(n_jobs=-1)(delayed(monte_carlo_iteration)(
        rho, max_N_Models, n_of_features, test_length, num_samples) for _ in range(n_iterations))

    # Convert list of lists to a 2D NumPy array
    simulation_results[rho] = np.array(results)

# Analyze results: Calculate the mean and standard deviation
mean_rho = {rho: np.mean(simulation_results[rho], axis=0) for rho in rho_values}
std_rho = {rho: np.std(simulation_results[rho], axis=0) for rho in rho_values}
# Example: Interpretation for one of the fitted curves
for rho in rho_values:
    num_models_range = np.arange(1, max_N_Models + 1)
    popt, _ = curve_fit(logistic_function, num_models_range, mean_rho[rho], p0=[0.1, max_N_Models / 2, 1])

    # Generate data for plotting the fit
    x_fit = np.linspace(num_models_range.min(), num_models_range.max(), 500)
    y_fit = logistic_function(x_fit, *popt)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_models_range, mean_rho[rho], 'o', label=f'Observed Mean Rho (Rho = {rho})')
    plt.plot(x_fit, y_fit, '-', label=f'Logistic Fit (Rho = {rho})', color='r')

    # Print fitted parameters
    a, b, c = popt
    print(f'Logistic Function Parameters for Rho = {rho}: a={a:.4f}, b={b:.4f}, c={c:.4f}')
    print(f"\nInterpretation for Rho = {rho}:")
    print(f"  Growth rate (a): {a:.4f}")
    print(f"  Maximum achievable Rho (c): {c:.4f}")

    plt.title(f'Logistic Fit for Mean Rho vs. Number of Models (Rho = {rho})')
    plt.xlabel('Number of Models')
    plt.ylabel('Mean Rho')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
n_of_features = 10
test_length = int(len(y) * 0.2)
max_N_Models = 50
y_test, y_train = y[test_length:], y[:test_length]
num_samples = len(y)
rho_values = [0.1, 0.2, 0.3, 0.4]

for rho in rho_values:
    rho_with_n_models = []
    for i in range(1, max_N_Models+1):
        N_models = i
        predictions = {}
        for j in range(1, N_models+1):
            features = generate_features(num_samples, n_of_features, y, desired_rho=rho)
            features_train, features_test = features[:test_length], features[test_length:]
            features_train = sm.add_constant(features_train)
            model = sm.OLS(y_train, features_train).fit()
            features_test = sm.add_constant(features_test)
            prediction = model.predict(features_test)
            predictions[j] = prediction
        meta_learner_predictions = ridge_meta_learner(predictions, y_test)
        rho_avg = pearsonr(y_test, meta_learner_predictions)[0]
        print(f"Rho N_models {i}: " + str(rho_avg))
        rho_with_n_models.append(rho_avg)

    # Plotting

    # Fit the logistic function to the data
    num_models_range = np.arange(1, max_N_Models + 1)
    # noinspection PyTupleAssignmentBalance
    popt, _ = curve_fit(logistic_function, num_models_range, rho_with_n_models, p0=[1, (max_N_Models/2), 0.5])

    # Print fitted parameters
    a,b,c = popt
    print(f'Logistic Function Parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}')

    epsilon = 1e-3
    x_n = b - (1/a) * math.log(epsilon/(c*a))

    print(f"for e = {epsilon}, x should be greater than: {x_n:.4f}")
    print(f"Recommend N of models = {x_n}")
    print(f"Ideal rho is somewhere near : {c} (BASED ON PREVIOUS DATA)")
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