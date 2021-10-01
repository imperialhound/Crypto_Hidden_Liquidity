import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class TrainingPreprocessor:

    def __init__(self, x_train: np.array, x_test: np.array):
        self.x_train = x_train
        self.x_test = x_test

    def __set_standardized_features(self, features_train: np.array, features_test: np.array) -> Tuple[np.array, np.array]:
        # Standardize
        scaler = StandardScaler()
        scaler.fit(features_train)

        # Apply transform to both the training set and the test set.
        x_train_standardized = scaler.transform(features_train)
        x_test_standardized = scaler.transform(features_test)

        return x_train_standardized, x_test_standardized

    def __set_reduced_features(self, features_train: np.array, features_test: np.array) -> Tuple[np.array, np.array]:
        # fit PCA model
        pca = PCA(.95)
        pca.fit(features_train)

        # Apply transform to both the training set and the test set.
        x_train_pca = pca.transform(features_train)
        x_test_pca = pca.transform(features_test)

        return x_train_pca, x_test_pca

    def preprocess_features(self) -> Tuple[np.array, np.array]:
        x_train_standardized, x_test_standardized = self.__set_standardized_features(self.x_train, self.x_test)
        x_train_pca, x_test_pca = self.__set_reduced_features(x_train_standardized, x_test_standardized)

        return x_train_pca, x_test_pca

