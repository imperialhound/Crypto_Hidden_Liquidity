from typing import Tuple, List, Any
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


class Metrics:

    def generate_regression_metrics(self, predictions: List[float], y_test: np.array) -> Tuple[float, float]:
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE: {rmse}")

        # Calculate R2
        r2 = np.sqrt(r2_score(y_test, predictions))
        print(f"R_Squared Score : {r2}")

        return rmse, r2

    def adf(self, series: Any) -> None:
        result = adfuller(series, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critial Values:')
        for key, value in result[4].items():
            print(f'   {key}, {value}')

    def kpss_test(self, series: Any, **kw) -> None:
        statistic, p_value, n_lags, critical_values = kpss(series, **kw)
        # Format Output
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critial Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')
        print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

    def plot_acf_pacf(self, series: Any, lags=30) -> None:
        plot_acf(series, lags=30)
        plot_pacf(series, lags=30)

    def plot_correlation(self, df: pd.DataFrame, x_val: str, y_val: str, coin: str, dt: str) -> None:
        df.plot(kind='scatter', grid=True,
                x=x_val, y=y_val,
                title=coin + dt + ' OFI',
                alpha=0.5, figsize=(12, 10))

    def plot_ols_summary(self, df: pd.DataFrame, columns: List[str], target: str):
        features = df[columns]
        features = sm.add_constant(features)
        ols = OLS(df[target], features).fit()
        print(ols.summary2())
        return ols


