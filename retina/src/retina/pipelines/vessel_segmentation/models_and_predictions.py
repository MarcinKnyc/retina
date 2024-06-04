import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from logitboost import LogitBoost
from sklearn.tree import DecisionTreeRegressor
from .feature_extraction import create_dataset

def undersampling(photos: list, masks: list) -> tuple:
    """Balances the dataset by undersampling the majority class."""
    # photos_0 = [photos[i] for i in range(len(masks)) if masks[i] == 0]
    # photos_225 = [photos[i] for i in range(len(masks)) if masks[i] == 255]

    # np.random.shuffle(photos_0)
    # photos_0 = photos_0[:len(photos_225)]
    # photos = photos_0 + photos_225
    # masks = [0] * len(photos_0) + [255] * len(photos_225)

    return photos, masks


def train_knn(train_features: np.ndarray, train_labels: np.ndarray, output_path: str) -> KNeighborsClassifier:
    """Trains a K-Nearest Neighbors classifier and saves it."""
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(train_features, train_labels)

    return knn_classifier

    
def train_logitboost(train_features: np.ndarray, train_labels: np.ndarray, output_path: str) -> LogitBoost:
    """Trains a K-Nearest Neighbors classifier and saves it."""
    logitboost_classifier = LogitBoost( n_estimators= 200)
    logitboost_classifier.fit(train_features, train_labels)

    return logitboost_classifier

def train_adaboost(train_features: np.ndarray, train_labels: np.ndarray, output_path: str) -> AdaBoostClassifier:
    """Trains a K-Nearest Neighbors classifier and saves it."""
    adaboost_classifier = AdaBoostClassifier(n_estimators=200)
    adaboost_classifier.fit(train_features, train_labels)

    return adaboost_classifier


def predict_model(classifier, test_photos: list, test_masks: list) -> list:
    """Predicts the masks for test images using the trained classifier."""
    y_pred_images = []
    for photo, mask in zip(test_photos, test_masks):
        test_features, _ = create_dataset([photo], [mask], size=5)
        y_pred = classifier.predict(test_features)
        y_pred_img = np.zeros((512, 512)) #todo: extract 512, 512 as parameter in all of code.
        for i in range(512):
            for j in range(512):
                y_pred_img[i][j] = y_pred[i * 512 + j]
        y_pred_images.append(y_pred_img)

    return y_pred_images

