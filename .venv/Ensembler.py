import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from keras.api.losses import Huber
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import LSTM


class Ensembler:
    def __init__(self):
        # Store data from first loop to avoid redundant computations
        self.predictions_train = []
        self.predictions_test = []
        self.meta_output = []
        # Store models and their rho
        self.models = []
        self.model_rho = []
        # Rho as models increase for ensemble
        self.rho_with_n_models = []
        self.directional_accuracy = []
        self.stacking_regressor = None

    def clear_model(self):
        # Store data from first loop to avoid redundant computations
        self.predictions_train = []
        self.predictions_test = []
        self.meta_output = []
        # Store models and their rho
        self.models = []
        self.model_rho = []
        # Rho as models increase for ensemble
        self.rho_with_n_models = []
        self.stacking_regressor = None

    @staticmethod
    def calculate_rho(y_true, y_pred):
        # Check for NaN or infinite values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            print("NaN values found in either y_true or y_pred.")
            return np.nan

        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            print("Infinite values found in either y_true or y_pred.")
            return np.nan
        return spearmanr(y_true, y_pred)[0]
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def neural_network_meta_learner(predictions_train, predictions_test, y_train):
        # Convert 2D lists to numpy arrays for easier handling
        x_meta_train = np.column_stack(predictions_train)
        x_meta_test = np.column_stack(predictions_test)
        y_train = np.array(y_train).reshape(-1, 1)

        # Ensure both X_meta_train and y_train have the same number of samples
        min_samples = min(len(x_meta_train), len(y_train))
        x_meta_train = x_meta_train[:min_samples]
        y_train = y_train[:min_samples]

        # Scale the data (important for neural networks)
        scaler = StandardScaler()
        x_meta_train = scaler.fit_transform(x_meta_train)
        x_meta_test = scaler.transform(x_meta_test)

        # Build a neural network with Input layer
        model = Sequential()
        model.add(Input(shape=(x_meta_train.shape[1],)))  # Define input shape
        model.add(Dense(32, activation=LeakyReLU(alpha=0.5)))
        model.add(BatchNormalization())  # Batch normalization to improve training stability
        model.add(Dropout(0.25))  # Dropout to prevent overfitting
        model.add(Dense(32, activation=LeakyReLU(alpha=0.5)))
        model.add(BatchNormalization())  # Batch normalization
        model.add(Dropout(0.25))
        model.add(Dense(32, activation=LeakyReLU(alpha=0.5)))
        model.add(BatchNormalization())  # Batch normalization
        model.add(Dense(1, activation='linear'))  # Single output for regression

        # Compile the model with binary cross entropy loss
        model.compile(optimizer='adam', loss=BinaryCrossentropy())

        # Set early stopping and learning rate reduction on plateau callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

        # Fit the neural network (you can adjust the number of epochs and batch size)
        model.fit(
            x_meta_train, y_train,
            epochs=25,  # Start with higher epochs
            batch_size=64,
            validation_split=0.25,  # Split some of the training data for validation
            callbacks=[early_stopping, reduce_lr],
            verbose=1  # Change this to 0 if you don't want output logs
        )

        # Get meta predictions
        meta_predictions = model.predict(x_meta_test)

        return meta_predictions.flatten(), model

    @staticmethod
    def xgboost_meta_learner(predictions_train: list, predictions_test: list, y_train: pd.Series) -> np.array:
        x_train = np.column_stack(predictions_train)
        x_test = np.column_stack(predictions_test)
        xgb_model = XGBRegressor(n_estimators=250, learning_rate=0.05)
        xgb_model.fit(x_train, y_train)
        return 1 / (1 + np.exp(-xgb_model.predict(x_test)))

    @staticmethod
    def ridge_meta_learner(ols_predictions_train: list, ols_predictions_test: list,
                           y_value_train: pd.Series) -> np.array:
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
        return 1 / (1 + np.exp(-meta_predictions))



    def meta_learner(self, predictions_train : list, predictions_test: list, y_value_train : pd.Series) -> np.array:
        """
          Trains and fits a meta learner to predict y values using regression models' predictions.

          Parameters:
          predictions_train (dictionary): A dictionary containing the predictions of each model, split for training
          predictions_test (dictionary): A dictionary containing the predictions of each model, split for testing
          y_value_train (pd.Series): Series of y values to use for training the model
          Returns:
              np.array: a numpy array containing the final predictions of the model
         """

        x_meta_train = np.column_stack(predictions_train)
        x_meta_test = np.column_stack(predictions_test)

        base_learners = [
            ('rf', RandomForestRegressor(
                n_estimators=75,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                n_jobs=-1
            )),
            ('rg', Ridge(alpha=20)),  # Increase alpha for stronger regularization
            ('lr', LinearRegression()),
            ('xgb', XGBRegressor(
                learning_rate=0.01,
                n_estimators=100,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=10,  # L1 regularization
                reg_lambda=10  # L2 regularization
            ))
        ]

        def nn_estimator():
            # Build a neural network with Input layer
            model = Sequential()
            model.add(Input(shape=(len(base_learners),)))
            model.add(Dense(256, activation=LeakyReLU(negative_slope=0.5), kernel_regularizer='l2'))
            model.add(BatchNormalization())  # Batch normalization
            model.add(Dropout(0.15))
            model.add(Dense(128, activation=LeakyReLU(negative_slope=0.5), kernel_regularizer='l2'))
            model.add(BatchNormalization())  # Batch normalization
            model.add(Dropout(0.15))
            model.add(Dense(64, activation=LeakyReLU(negative_slope=0.5), kernel_regularizer='l2'))
            model.add(BatchNormalization())  # Batch normalization
            model.add(Dropout(0.15))
            model.add(Dense(32, activation=LeakyReLU(negative_slope=0.5), kernel_regularizer='l2'))
            model.add(BatchNormalization())  # Batch normalization
            model.add(Dropout(0.15))
            model.add(Dense(16, activation=LeakyReLU(negative_slope=0.5), kernel_regularizer='l2'))
            model.add(BatchNormalization())  # Batch normalization
            model.add(Dropout(0.15))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model with binary cross entropy loss
            model.compile(optimizer='adam', loss="mean_squared_error")

            return model

        # Create the final estimator with the best parameters
        # final_estimator = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100)
        # Set early stopping and learning rate reduction on plateau callbacks
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5)
        final_estimator = KerasRegressor(model=nn_estimator, epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping, reduce_lr])

        # Create the Stacking Regressor
        stacking_regressor = StackingRegressor(estimators=base_learners, final_estimator=final_estimator)

        stacking_regressor.fit(x_meta_train, y_value_train)
        self.stacking_regressor = stacking_regressor
        meta_predictions = self.sigmoid(stacking_regressor.predict(x_meta_test))

        # Perform 3-fold cross-validation using the Rho scorer
        # rho_scorer = make_scorer(self.calculate_rho, greater_is_better=True)
        # scores = cross_val_score(stacking_regressor, x_meta_train, y_value_train, cv=3, scoring=rho_scorer)
        # print(f"Cross-validation Rho scores: {scores}")

        return meta_predictions

    def ensemble(self, columns : list, train_data : tuple, test_data : tuple) -> tuple:
        x_train_data, y_train_data = train_data
        x_test_data, y_test_data = test_data

        for i, column in enumerate(columns):
            # Get desired columns
            x_train = x_train_data[list(column)]
            x_test = x_test_data[list(column)]
            # Add constant to center model
            x_train = sm.add_constant(x_train, has_constant='add')
            x_test = sm.add_constant(x_test, has_constant='add')

            # Fit the model
            model = Ridge(alpha=10)
            model.fit(x_train, y_train_data)

            self.predictions_train.append(self.sigmoid(model.predict(x_train)))
            self.predictions_test.append(self.sigmoid(model.predict(x_test)))
            rho = self.calculate_rho(y_test_data, self.predictions_test[-1])
            self.model_rho.append(rho)

        for i, _ in enumerate(columns):
            # Get models' data up to the Nth model
            pred_train = self.predictions_train[:i+1]
            pred_test = self.predictions_test[:i+1]
            # Meta learner used for ensemble, currently XGB learner
            meta_learner_predictions = self.meta_learner(pred_train, pred_test, y_train_data)

            # Trim predictions to match y_test_data
            min_length = min(len(meta_learner_predictions), len(y_test_data))
            meta_learner_predictions = meta_learner_predictions[:min_length]
            y_test_data = y_test_data[:min_length]
            # Calculate rho and store it
            rho_avg = self.calculate_rho(y_test_data, meta_learner_predictions)
            self.rho_with_n_models.append(rho_avg)
            self.meta_output.append(meta_learner_predictions)
            print(f"{i+1} models Rho: ", rho_avg)


        return  self.models, self.rho_with_n_models, self.meta_output, self.model_rho