from typing import List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from logitboost import LogitBoost
from sklearn.tree import DecisionTreeRegressor
from .feature_extraction import extract_features
import code

import cv2

def undersampling(train_features: list, train_labels: list, limit_features = 10000) -> tuple:
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

    l = min(len(minority), limit_features//2)
    np.random.shuffle(majority)
    np.random.shuffle(minority)
    majority = majority[:l]
    minority = minority[:l]
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

    adaboost_classifier = AdaBoostClassifier(n_estimators=200, algorithm ="SAMME", random_state=0)
    adaboost_classifier.fit(train_features, train_labels)

    return adaboost_classifier

def apply_threshold(predicted_images, threshold = 0.5):
    return (predicted_images > threshold) *255

def predict_model(classifier, test_features: List[np.ndarray], test_photos) -> list:
    """Predicts the masks for test images using the trained classifier."""

    y_pred_images = []
    for photo_features, photo in zip(test_features,test_photos):
        height, width, num_features = photo_features.shape
        # Flatten the features
        roi = cv2.cvtColor( photo, cv2.COLOR_RGB2GRAY ) > 0.1*255
        photo_features = photo_features[roi]
        y_pred = np.zeros((height, width))
        y_pred[roi] = classifier.predict_proba(photo_features)[:, 1]

        # shape contains sth like (512, 512, 3)
        y_pred_img = np.reshape(y_pred, (height, width))
        y_pred_images.append(y_pred_img)

    return np.array(y_pred_images)
