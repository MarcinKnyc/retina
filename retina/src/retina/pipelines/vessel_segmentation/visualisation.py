import os
import matplotlib.pyplot as plt
import numpy as np

from .metrics import get_quality
from .models_and_predictions import apply_threshold

import cv2


def normalize(img):
    if img.max() > 1.0:  # Normalize if max value is greater than 1
        img /= 255.0
    return img


def plot_images(train_raw_photos, train_masks, title: str, output_path: str, filename: str, figsize=(20, 10), cmap='gray'):
    """Plots images in a grid layout and saves the plot to a file."""

    images_set = [train_raw_photos[:5], train_masks[:5]]
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=20)
    for i in range(len(images_set[0])):
        for j in range(len(images_set)):
            plt.subplot(len(images_set), len(
                images_set[0]), i + j * len(images_set[0]) + 1)
            img = normalize(images_set[j][i].astype(np.float32))
            plt.imshow(img, cmap=cmap)
            plt.axis('off')
        plt.tight_layout()
    save_plot(output_path, filename)


def plot_features(train_feature_photos, output_path):
    # This is bad, but will do
    names = [
        'GOA',
        'Top-hat',
        'Line strength 1',
        'Line strength 2',
        'Gabor, sigma = 2',
        'Gabor, sigma = 3',
        'Gabor, sigma = 4',
        'Gabor, sigma = 5',
        'Inverted green channel',
    ]

    for k, feature_vector in enumerate(train_feature_photos):
        feature_vector[:, :, 0] = cv2.bitwise_not(feature_vector[:, :, 0])
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        for i, name in enumerate(names):
            ax = plt.subplot(3, 3, i + 1)
            ax.axis('off')
            ax.set_title(name, fontsize=20)
            ax.imshow(feature_vector[:, :, i], cmap='gray')

        save_plot(output_path + "/features", f"feature_{k}.png")


def plot_results(test_raw_photos: list, test_masks: list, threshold, y_nonthresh_images, name: str, output_path: str, filename: str,  figsize=(27, 10), n=10):
    """Plots the results along with accuracy, sensitivity, and specificity metrics, and saves the plot to a file."""
    def gtapm():
        plt.figure(figsize=figsize)
        plt.suptitle(name, fontsize=20)
        for i in range(n):
            plt.subplot(4, 10, i + 1)
            img = normalize(test_raw_photos[i].astype(np.float32))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title('Photo')
            plt.subplot(4, 10, i + 11)
            mask = normalize(test_masks[i].astype(np.float32))
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title('Ground truth')
            plt.subplot(4, 10, i + 21)
            pred = normalize(y_nonthresh_images[i].astype(np.float32))
            plt.imshow(pred, cmap='gray')
            plt.axis('off')
            plt.title('Probability')
            plt.tight_layout()
            plt.subplot(4, 10, i + 31)
            pred = normalize(apply_threshold(y_nonthresh_images, threshold)
                             [i].astype(np.float32))
            plt.imshow(pred, cmap='gray')
            plt.axis('off')
            plt.title('Thresholded')
            plt.tight_layout()

        save_plot(output_path, "gtamp_" + filename)

    def roc():
        plt.figure(figsize=figsize)
        plt.plot(quality["fpr_roc"], quality["tpr_roc"], 'b',
                 label='AUC = %0.2f' % quality["auc"])
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'{name}: ROC')
        plt.gca().set_aspect('equal')
        plt.tight_layout()

        save_plot(output_path, "roc_" + filename)

    def table():
        data = {
            "Metric": ["True Negative (tn)", "False Positive (fp)", "False Negative (fn)", "True Positive (tp)",
                       "True Positive Rate (tpr)", "False Positive Rate (fpr)", "True Negative Rate (tnr)",
                       "Accuracy (acc)", "False Negative Rate (fnr)", "Positive Predictive Value (ppv)",
                       "False Discovery Rate (fdr)", "Area Under Curve (auc)"],
            "Value": [quality["tn"],
                      quality["fp"],
                      quality["fn"],
                      quality["tp"],
                      quality["tpr"],
                      quality["fpr"],
                      quality["tnr"],
                      quality["acc"],
                      quality["fnr"],
                      quality["ppv"],
                      quality["fdr"],
                      quality["auc"]]
        }

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

        table_data = list(zip(data["Metric"], data["Value"]))
        table = ax.table(cellText=table_data, colLabels=[
                         "Metric", "Value"], cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.subplots_adjust(left=0.2, top=0.8)

        save_plot(output_path, "table_" + filename)

    quality = get_quality(
        test_masks, y_nonthresh_images, threshold)

    if n > len(test_raw_photos):
        n = len(test_raw_photos)

    gtapm()
    roc()
    table()


def save_plot(output_path: str, filename: str):
    """Utility function to save a plot."""
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    plt.tight_layout()
    print(os.path.join(output_path, filename))
    plt.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.close()
