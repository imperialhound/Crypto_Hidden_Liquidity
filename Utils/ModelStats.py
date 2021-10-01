import numpy as np
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from Utils.Preprocessor import TrainingPreprocessor
from Utils.Metrics import Metrics


class ModelStats:

    def __init__(self, random_walk: bool, df: pd.DataFrame, model: str, target: str):
        self.model = model
        if random_walk is True:
            self.features = df.values[:, 0].reshape(-1, 1)
            self.target = df[target].values
        else:
            self.features = df.values[:, :-1]
            self.target = df[target].values

    def generate_model_statistics(self) -> Tuple[float, float, List[float], List[float]]:

        # Create empty lists to populate
        predictions_list = list()
        ytest_list = list()

        # Time Series Split
        timeSeriesCrossVal = TimeSeriesSplit(n_splits=5)

        for trainIndex, testIndex in timeSeriesCrossVal.split(self.features):

            features_train = self.features[trainIndex]
            features_test = self.features[testIndex]

            target_train = self.target[trainIndex]
            target_test = self.target[testIndex]

            # Standardize and PCA of training and test data
            preprocessor = TrainingPreprocessor(features_train, features_test)
            features_train_processed, features_test_processed = preprocessor.preprocess_features()

            # Generate Predictions
            predictions = self.__generate_predictions(features_train_processed, features_test_processed, target_train)

            predictions_list.extend(predictions)
            ytest_list.extend(target_test)

        # Calculate R2 and RMSE metrics
        metrics_generator = Metrics()
        rmse, r2 = metrics_generator.generate_regression_metrics(predictions_list, ytest_list)

        return rmse, r2, predictions_list, ytest_list


    def __generate_predictions(self, X_train: np.array, X_test: np.array, y_train: np.array) -> List[float]:
        if self.model == "linear regression":
            predictions = self.__get_linear_regression_predictions(X_train, X_test, y_train)

        elif self.model == "XGBoost":
            predictions = self.__get_XGBoost_regression_predictions(X_train, X_test, y_train)

        else:
            predictions = self.__get_LSTM_regression_predictions(X_train, X_test, y_train)

        return predictions

    def __get_linear_regression_predictions(self, X_train: np.array, X_test: np.array, y_train: np.array) -> List[float]:
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X_test)

        return predictions

    def __get_XGBoost_regression_predictions(self, X_train: np.array, X_test: np.array, y_train: np.array) -> List[float]:
        # Train model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)

        # Generate Predictions
        predictions = model.predict(X_test)

        return predictions

    def __get_LSTM_regression_predictions(self, X_train: np.array, X_test: np.array, y_train: np.array) -> List[float]:
        pass