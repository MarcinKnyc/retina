import os
import matplotlib.pyplot as plt

def plot_images(train_raw_photos, train_masks, title: str, output_path: str, filename: str, figsize=(20, 10), cmap=None):
    """Plots images in a grid layout and saves the plot to a file."""
    images_set = [train_raw_photos[:5], train_masks[:5]]
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=20)
    for i in range(len(images_set[0])):
        for j in range(len(images_set)):
            plt.subplot(len(images_set), len(images_set[0]), i + j * len(images_set[0]) + 1)
            plt.imshow(images_set[j][i], cmap=cmap)
            plt.axis('off')
        plt.tight_layout()
    save_plot(output_path, filename)


def plot_results(test_raw_photos: list, test_masks: list, y_pred_images, result_plot_title: str, output_path: str, filename: str, figsize=(27, 10), n=10):
    """Plots the results along with accuracy, sensitivity, and specificity metrics, and saves the plot to a file."""
    accuracy_scores, sensitivity_scores, specificity_scores = get_quality(test_masks, y_pred_images)

    if n > len(test_raw_photos):
        n = len(test_raw_photos)

    plt.figure(figsize=figsize)
    plt.suptitle(result_plot_title, fontsize=20)
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
        plt.imshow(y_pred_images[i], cmap='gray')
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