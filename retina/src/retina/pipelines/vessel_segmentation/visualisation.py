import os
import matplotlib.pyplot as plt
import numpy as np

from .metrics import get_quality


def normalize(img):
    if img.max() > 1.0:  # Normalize if max value is greater than 1
        img /= 255.0
    return img


def plot_images(train_raw_photos, train_masks, name: str, output_path: str,  figsize=(20, 10), cmap='gray'):
    """Plots images in a grid layout and saves the plot to a file."""
    images_set = [train_raw_photos[:5], train_masks[:5]]
    plt.figure(figsize=figsize)
    plt.suptitle(name + ": Images and Masks", fontsize=20)
    for i in range(len(images_set[0])):
        for j in range(len(images_set)):
            plt.subplot(len(images_set), len(
                images_set[0]), i + j * len(images_set[0]) + 1)
            img = normalize(images_set[j][i].astype(np.float32))
            plt.imshow(img, cmap=cmap)
            plt.axis('off')
        plt.tight_layout()
    save_plot(output_path, name + "_images.png")


def plot_results(test_raw_photos: list, test_masks: list, y_pred_images, name: str, output_path: str,  figsize=(27, 10), n=10):
    # TODO: Handle model
    """Plots the results along with accuracy, sensitivity, and specificity metrics, and saves the plot to a file."""
    def gtapm():
        plt.figure(figsize=figsize)
        plt.suptitle(name + ": Ground truth and predicted mask", fontsize=20)
        for i in range(n):
            plt.subplot(3, 10, i + 1)
            img = normalize(test_raw_photos[i].astype(np.float32))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title('Photo')
            plt.subplot(3, 10, i + 11)
            mask = normalize(test_masks[i].astype(np.float32))
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title('Ground truth')
            plt.subplot(3, 10, i + 21)
            pred = normalize(y_pred_images[i].astype(np.float32))
            plt.imshow(pred, cmap='gray')
            plt.axis('off')
            plt.title('Mask\nAcc: {:.3f}\nSens: {:.3f}\nSpec: {:.3f}'.format(
                accuracy_scores[i], sensitivity_scores[i], specificity_scores[i]))
            plt.tight_layout()

        save_plot(output_path, name + "_gtapm.png")

    def roc():
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'{name}: ROC - TPR {np.float32(tpr[1])}, FPR {np.float32(fpr[1])}, AUC {np.float32(auc)}')  # nopep8
        plt.gca().set_aspect('equal')
        plt.tight_layout()

        save_plot(output_path, name + "_roc.png")

    accuracy_scores, sensitivity_scores, specificity_scores, fpr, tpr, auc = get_quality(
        test_masks, y_pred_images)

    if n > len(test_raw_photos):
        n = len(test_raw_photos)

    gtapm()
    roc()

    print('Global avg accuracy: {:.3f}'.format(
        sum(accuracy_scores) / len(accuracy_scores)))
    print('Global avg sensitivity: {:.3f}'.format(
        sum(sensitivity_scores) / len(sensitivity_scores)))
    print('Global avg specificity: {:.3f}'.format(
        sum(specificity_scores) / len(specificity_scores)))


def save_plot(output_path: str, filename: str):
    """Utility function to save a plot."""
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.close()
