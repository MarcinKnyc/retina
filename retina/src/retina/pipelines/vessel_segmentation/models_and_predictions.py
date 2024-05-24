import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def undersampling(photos: list, masks: list) -> tuple:
    """Balances the dataset by undersampling the majority class."""
    photos_0 = [photos[i] for i in range(len(masks)) if masks[i] == 0]
    photos_225 = [photos[i] for i in range(len(masks)) if masks[i] == 255]

    np.random.shuffle(photos_0)
    photos_0 = photos_0[:len(photos_225)]
    photos = photos_0 + photos_225
    masks = [0] * len(photos_0) + [255] * len(photos_225)

    return photos, masks


def train_model(train_features: list, train_labels: list, output_path: str) -> KNeighborsClassifier:
    """Trains a K-Nearest Neighbors classifier and saves it."""
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(train_features, train_labels)

    return knn_classifier


def predict_model(knn_classifier: KNeighborsClassifier, test_photos: list, test_masks: list) -> list:
    """Predicts the masks for test images using the trained classifier."""
    y_pred_images = []
    for photo, mask in zip(test_photos, test_masks):
        test_features, _ = create_dataset([photo], [mask], size=5)
        y_pred = knn_classifier.predict(test_features)
        y_pred_img = np.zeros((512, 512))
        for i in range(512):
            for j in range(512):
                y_pred_img[i][j] = y_pred[i * 512 + j]
        y_pred_images.append(y_pred_img)

    return y_pred_images