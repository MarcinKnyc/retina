from typing import List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from logitboost import LogitBoost
from sklearn.tree import DecisionTreeRegressor
from .feature_extraction import extract_features
import code


def undersampling(train_features: list, train_labels: list) -> tuple:
    """Balances the dataset by undersampling the majority class."""
    features = np.array(train_features)
    labels = np.array(train_labels)

    features_0 = features[labels == 0]
    features_255 = features[labels == 255]

    minority = features_255
    majority = features_0
    minority_val = 255
    majority_val = 0

    if len(features_0) < len(features_255):
        minority = features_0
        majority = features_255
        majority_val = 255
        minority_val = 0

    l = len(minority)
    np.random.shuffle(majority)
    majority = majority[:l]
    shuffle_pattern = [i for i in range(2*l)]
    np.random.shuffle(shuffle_pattern)
    features = np.concatenate([minority, majority])[shuffle_pattern]
    labels = np.array([minority_val] * l + [majority_val] * l)[shuffle_pattern]

    return features, labels


def train_knn(train_features: np.ndarray, train_labels: np.ndarray) -> KNeighborsClassifier:
    """Trains a K-Nearest Neighbors classifier"""
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(train_features, train_labels)

    return knn_classifier


def train_logitboost(train_features: np.ndarray, train_labels: np.ndarray) -> LogitBoost:
    """Trains a LogitBoost classifier"""
    logitboost_classifier = LogitBoost( n_estimators=200)
    logitboost_classifier.fit(train_features, train_labels)

    return logitboost_classifier


def train_adaboost(train_features: np.ndarray, train_labels: np.ndarray) -> AdaBoostClassifier:
    """Trains a K-Nearest Neighbors classifier"""
    adaboost_classifier = AdaBoostClassifier(n_estimators=200)
    adaboost_classifier.fit(train_features, train_labels)

    return adaboost_classifier

def apply_threshold(predicted_images, threshold = 0.5):
    return (predicted_images > threshold) *255

def predict_model(classifier, test_photos: List[np.ndarray]) -> list:
    """Predicts the masks for test images using the trained classifier."""

    y_pred_images = []
    for photo in test_photos:
        feature_vector = extract_features([photo])[0]
        height, width, num_features = feature_vector.shape
        # Flatten the features
        test_features = feature_vector.reshape(-1, num_features)
        
        y_pred = classifier.predict_proba(test_features)[:, 1]

        # shape contains sth like (512, 512, 3)
        shape = np.shape(test_photos[0])
        y_pred_img = np.reshape(y_pred, (shape[0], shape[1]))
        y_pred_images.append(y_pred_img)

    return np.array(y_pred_images)
