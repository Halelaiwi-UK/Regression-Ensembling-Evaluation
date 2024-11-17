import matplotlib.pyplot as plt
from Ensembler import Ensembler
from preproccesor import Preprocessor
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.stats import spearmanr, pearsonr
from statsmodels.distributions.empirical_distribution import ECDF

def bin_values(y_train: pd.Series, y_test: pd.Series, num_bins=10) -> tuple:
    # Compute the PDF by binning the percentage changes for y_train
    pdf, bins = np.histogram(y_train, bins=num_bins, density=True)

    # Compute the CDF from the PDF
    cdf = np.cumsum(pdf) * np.diff(bins)  # Multiply by bin width to scale correctly

    # Apply binning to y_train
    y_train_cdf = np.digitize(y_train, bins, right=True) - 1  # Align bin index
    y_train_cdf = np.clip(y_train_cdf, 0, len(cdf) - 1)  # Ensure indices stay within range
    y_train_transformed = cdf[y_train_cdf]  # Map to CDF values

    # Apply the same binning to y_test using bins derived from y_train
    y_test_cdf = np.digitize(y_test, bins, right=True) - 1  # Align bin index
    y_test_cdf = np.clip(y_test_cdf, 0, len(cdf) - 1)  # Ensure indices stay within range
    y_test_transformed = cdf[y_test_cdf]  # Map to CDF values

    # Return both transformed series
    return pd.Series(y_train_transformed, index=y_train.index), pd.Series(y_test_transformed, index=y_test.index)

# Initialize class to manage preprocessing easily
preprocessor = Preprocessor()

# Download data if doesn't exist
target_stocks = ["JPM", "GS", "C", "MS", "BAC", "WFC", "AXP", "MA", "V"]
for stock in target_stocks:
    stock_data = preprocessor.get_stock_by_name(stock)
    # Only upload if new data was downloaded
    if not stock_data.empty:
        # Convert all column names to lowercase and remove whitespace
        stock_data.columns = stock_data.columns.str.lower()
        stock_data.columns = stock_data.columns.str.replace(" ", "_")
        preprocessor.upload_to_postgres(stock_data, stock)


# Get sp500 data
sp500_data = preprocessor.get_sp500()
y_sp500 = sp500_data['close']

# Retrieve closing value of stocks and store them in one large df
stocks_close_df = pd.DataFrame(index=y_sp500.index)
for stock in target_stocks:
    stock_data = preprocessor.get_stock(stock)
    stocks_close_df[stock] = stock_data['close']


# Split data and generate technicals
train_length = int(len(stocks_close_df) * 0.7)  # 70% as train

# Split into 70% / 30%
y_sp500 = y_sp500.shift(-5).pct_change(fill_method=None).dropna()
y_train, y_test = (y_sp500[:train_length], y_sp500[train_length:])

stocks_close_df = stocks_close_df.shift(-5).pct_change(fill_method=None).dropna()
stocks_y_train, stocks_y_test = (stocks_close_df.iloc[:train_length], stocks_close_df.iloc[train_length:])

# Match volatility
stocks_y_train, stocks_y_test = preprocessor.match_volatility(stocks_y_train, stocks_y_test)

for col in stocks_y_train.columns:
    stocks_y_train[col], stocks_y_test[col] = bin_values(stocks_y_train[col], stocks_y_test[col], num_bins=len(stocks_y_train[col]) // 2)

ensembler = Ensembler()
predictions = []
model_rhos = []
i = 0
for stock in target_stocks:
    i += 1
    print(f"Running... {i}/" + str(len(target_stocks)))
    # Load using custom loader class
    regression_variables = preprocessor.get_stock(stock)

    # generate technicals
    train_technicals = preprocessor.generate_technicals(regression_variables.iloc[:train_length])
    test_technicals = preprocessor.generate_technicals(regression_variables.iloc[train_length:])

    # Normalize the technical indicators
    scaler = MinMaxScaler()
    train_technicals_transformed = scaler.fit_transform(train_technicals)
    train_technicals = pd.DataFrame(train_technicals_transformed, columns=train_technicals.columns, index=train_technicals.index)
    test_technicals = pd.DataFrame(scaler.transform(test_technicals), columns=test_technicals.columns, index=test_technicals.index)

    # Ensure that y_train and train_technicals have the same length
    y_train_stock = stocks_y_train[stock].iloc[-len(train_technicals):]
    y_train = y_train.iloc[-len(train_technicals):]

    y_test_stock = stocks_y_test[stock].iloc[-len(test_technicals):]
    y_test = y_test.iloc[-len(test_technicals):]

    # Combinations of techincals to train models on
    combinations = preprocessor.get_combinations(train_technicals, n_features = len(train_technicals.columns))

    # Ensemble Models (N of models ensemble = columns per model)
    # Clear model to ensure past results don't affect future results
    ensembler.clear_model()
    models, rho_with_n_models, meta_output, rho_per_model = ensembler.ensemble(combinations,
                                                        (train_technicals, y_train_stock),
                                                         (test_technicals, y_test_stock)
                                                                 )

    # Get last output (with Max N models and using all techincals)
    for output in meta_output:
        predictions.append(output)
    # predictions.append(meta_output[-1])
    model_rhos.append(rho_with_n_models[-1])



# Store outputs in a dataframe
df_pred = pd.DataFrame()
for i in range(len(predictions)):
    df_pred[f'{i}'] = predictions[i]

# Ensemble Models (N of models ensemble = columns per model)
combinations = preprocessor.get_combinations(df_pred, n_features=1)

# Split into train / test for the last model
train_final_length = int(len(df_pred) * 0.5)
df_pred_train, df_pred_test = df_pred.iloc[:train_final_length], df_pred.iloc[train_final_length:]
y_sp500_train, y_test = y_test.iloc[:train_final_length], y_test.iloc[train_final_length:]

# Normalize the predictions
scaler = MinMaxScaler()
df_pred_transformed = scaler.fit_transform(df_pred_train)
df_pred_train = pd.DataFrame(df_pred_transformed, columns=df_pred_train.columns,
                                index=df_pred_train.index)
df_pred_test = pd.DataFrame(scaler.transform(df_pred_test), columns=df_pred_test.columns,
                               index=df_pred_test.index)

# Bin values
y_sp500_train, y_test = bin_values(y_sp500_train, y_test, num_bins=len(y_sp500_train)//2)

# Clear model previous information to avoid learning from past models' data
ensembler.clear_model()
print("FINAL MODEL:\n")
models, rho_with_n_models, meta_output, rho_per_model = ensembler.ensemble(combinations,
                                                    (df_pred_train, y_sp500_train),
                                                     (df_pred_test, y_test)
                                                             )


# Assume y_true and y_pred are your actual and predicted percentage changes
y_true = y_test
y_pred = meta_output[-1]


# MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae}")

# RMSE
rmse = mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse}")

# MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape}")

# Rho (Spearman's rank correlation)
rho, _ = spearmanr(y_true, y_pred)
print(f"Spearman's Rho: {rho}")

# Pearson Correlation Coefficient
corr, _ = pearsonr(y_true, y_pred)
print(f"Pearson Correlation: {corr}")

# Directional Accuracy
direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
print(f"Directional Accuracy: {direction_accuracy}")

# R-Squared (R2)
r2 = r2_score(y_true, y_pred)
print(f"R-Squared: {r2}")


# Create a color map and normalize the values between min and max
cmap = plt.get_cmap('coolwarm')
norm = Normalize(vmin=min(model_rhos), vmax=max(model_rhos))

# Create the bar chart
bars = plt.bar(target_stocks, model_rhos, color=cmap(norm(model_rhos)))

# Add numbers on top of bars
for bar, value in zip(bars, model_rhos):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{value:.2f}',
             ha='center', va='bottom', color='black', fontsize=10)

# Set labels and title
# Set y-axis limits to center the plot around 0
max_abs_value = max(abs(min(model_rhos)), abs(max(model_rhos)))
plt.ylim(-max_abs_value - 0.2, max_abs_value + 0.2)

plt.xlabel('Stock:')
plt.ylabel('Rho')
plt.title('Rho by models')
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
# Show the plot
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rho_with_n_models) + 1), rho_with_n_models, color='b')
plt.title(f'Using ALL')
plt.xlabel('N Models Combined')
plt.ylabel('Rho as Models are combined')
plt.grid(True)
plt.show()