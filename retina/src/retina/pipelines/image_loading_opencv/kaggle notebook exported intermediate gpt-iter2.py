import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from skimage.measure import moments_central, moments_hu, moments_normalized
from sklearn.neighbors import KNeighborsClassifier
from skimage.util import view_as_windows


def load_images(image_dir: str, mask_dir: str) -> tuple:
    """Utility function to load and preprocess images and masks."""
    photo_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
    label_paths = [os.path.join(mask_dir, x) for x in os.listdir(mask_dir)]
    
    raw_photos = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in photo_paths]
    masks = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY) for path in label_paths]
    masks = [np.where(mask > 0, 255, 0) for mask in masks]
    
    return raw_photos, masks

def load_data(input_path: str) -> tuple:
    """Loads and preprocesses the train and test datasets."""
    train_image_dir = os.path.join(input_path, 'train/image')
    train_mask_dir = os.path.join(input_path, 'train/mask')
    test_image_dir = os.path.join(input_path, 'test/image')
    test_mask_dir = os.path.join(input_path, 'test/mask')
    
    train_raw_photos, train_masks = load_images(train_image_dir, train_mask_dir)
    test_raw_photos, test_masks = load_images(test_image_dir, test_mask_dir)
    
    return train_raw_photos, train_masks, test_raw_photos, test_masks


def preprocess_images(train_raw_photos: list, test_raw_photos: list) -> tuple:
    """Preprocesses the train and test images by enhancing contrast."""
    train_photos_green = [photo[:, :, 1] for photo in train_raw_photos]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    train_photos_clahe = [clahe.apply(photo) for photo in train_photos_green]

    test_photos_green = [photo[:, :, 1] for photo in test_raw_photos]
    test_photos = [clahe.apply(photo) for photo in test_photos_green]

    return train_photos_clahe, test_photos


def plot_images(images_set: list, title: str, output_path: str, filename: str, figsize=(20, 10), cmap=None):
    """Plots images in a grid layout and saves the plot to a file."""
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=20)
    for i in range(len(images_set[0])):
        for j in range(len(images_set)):
            plt.subplot(len(images_set), len(images_set[0]), i + j * len(images_set[0]) + 1)
            plt.imshow(images_set[j][i], cmap=cmap)
            plt.axis('off')
        plt.tight_layout()
    save_plot(output_path, filename)


def features_extracting(photo_batch: np.ndarray) -> np.ndarray:
    """Extracts features from image patches."""
    color_variance = np.var(photo_batch, axis=(1, 2))
    hu_moments = np.zeros((len(photo_batch), 7))
    for i, photo in enumerate(photo_batch):
        mu = moments_central(photo, order=5)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        hu_moments[i] = hu

    features = np.concatenate((color_variance.reshape(-1, 1), hu_moments), axis=1)
    return features


def create_dataset(photos: list, masks: list, size: int = 5, step: int = 1) -> tuple:
    """Creates a dataset by extracting patches and features from images."""
    ds_photos = []
    ds_masks = []
    for photo, mask in zip(photos, masks):
        photo_pad = np.pad(photo, ((size // 2, size // 2), (size // 2, size // 2)), mode='constant')
        mask_pad = np.pad(mask, ((size // 2, size // 2), (size // 2, size // 2)), mode='constant')

        patches_photo = view_as_windows(photo_pad, (size, size), step=step).reshape(-1, size, size)
        patches_mask = view_as_windows(mask_pad, (size, size), step=step).reshape(-1, size, size)

        batched_features = features_extracting(patches_photo)

        ds_photos.extend(batched_features)
        ds_masks.extend(patches_mask[:, size // 2, size // 2])

    return ds_photos, ds_masks


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
    pickle.dump(knn_classifier, open(os.path.join(output_path, 'knnclassifier.pth'), 'wb'))

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


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the accuracy score."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return (y_true == y_pred).mean()


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the sensitivity score."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return ((y_true == 255) & (y_pred == 255)).sum() / (y_true == 255).sum()


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the specificity score."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return ((y_true == 0) & (y_pred == 0)).sum() / (y_true == 0).sum()


def get_quality(masks: list, postprocessed_masks: list) -> tuple:
    """Computes the quality metrics for the predictions."""
    accuracy_scores = [accuracy_score(mask, postprocessed_mask) for mask, postprocessed_mask in zip(masks, postprocessed_masks)]
    sensitivity_scores = [sensitivity_score(mask, postprocessed_mask) for mask, postprocessed_mask in zip(masks, postprocessed_masks)]
    specificity_scores = [specificity_score(mask, postprocessed_mask) for mask, postprocessed_mask in zip(masks, postprocessed_masks)]
    return accuracy_scores, sensitivity_scores, specificity_scores


def plot_results(images_set: list, title: str, test_raw_photos: list, test_masks: list, output_path: str, filename: str, figsize=(27, 10), n=10):
    """Plots the results along with accuracy, sensitivity, and specificity metrics, and saves the plot to a file."""
    accuracy_scores, sensitivity_scores, specificity_scores = get_quality(images_set[1], images_set[2])

    if n > len(images_set[0]):
        n = len(images_set[0])

    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=20)
    for i in range(n):
        plt.subplot(3, 10, i + 1)
        plt.imshow(test_raw_photos[i], cmap='gray')
        plt.axis('off')
        plt.title('Photo')
        plt.subplot(3, 10, i + 11)
        plt.imshow(test_masks[i], cmap='gray')
        plt.axis('off')
        plt.title('Ground truth')
        plt.subplot(3, 10, i + 21)
        plt.imshow(images_set[2][i], cmap='gray')
        plt.axis('off')
        plt.title('Mask\nAcc: {:.3f}\nSens: {:.3f}\nSpec: {:.3f}'.format(accuracy_scores[i], sensitivity_scores[i], specificity_scores[i]))
        plt.tight_layout()
    save_plot(output_path, filename)
    print('Global avg accuracy: {:.3f}'.format(sum(accuracy_scores) / len(accuracy_scores)))
    print('Global avg sensitivity: {:.3f}'.format(sum(sensitivity_scores) / len(sensitivity_scores)))
    print('Global avg specificity: {:.3f}'.format(sum(specificity_scores) / len(specificity_scores)))

def save_plot(output_path: str, filename: str):
    """Utility function to save a plot."""
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

if __name__ == '__main__':
    input_path = '/kaggle/input/retina-blood-vessel/Data/'
    output_path = '/kaggle/working/'

    train_raw_photos, train_masks, test_raw_photos, test_masks = load_data(input_path)
    train_photos, test_photos = preprocess_images(train_raw_photos, test_raw_photos)
    
    plot_images([train_raw_photos[:5], train_masks[:5]], 'Image and Mask', output_path, 'image_and_mask.png')

    train_features, train_labels = create_dataset(train_photos, train_masks, size=5, step=5)
    train_features_under, train_labels_under = undersampling(train_features, train_labels)

    print(train_labels_under.count(0), train_labels_under.count(255))

    knn_classifier = train_model(train_features_under, train_labels_under, output_path)

    y_pred_images = predict_model(knn_classifier, test_photos, test_masks)

    plot_results([test_photos, test_masks, y_pred_images], 'Ground truth and predicted mask', test_raw_photos, test_masks, output_path, 'results.png')
