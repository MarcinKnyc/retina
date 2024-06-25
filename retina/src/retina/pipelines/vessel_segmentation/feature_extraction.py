from tqdm import tqdm
import cv2
import numpy as np

from scipy.ndimage import maximum_filter, uniform_filter


def image_rotate(image, angle):
    size = image.shape[1::-1]
    image_center = tuple(np.array(size) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, size)


def create_feature_vector(image):
    green_channel = image[:, :, 1]
    inverted_green = cv2.bitwise_not(green_channel)
    grayscale = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

    grad_orientation = compute_gradient_orientation(grayscale)
    morph_transformation = morphological_top_hat(inverted_green)
    line_strength_1 = line_strength(green_channel)
    line_strength_2 = line_strength(inverted_green)
    gabor_response = gabor_filter_responses(inverted_green)

    features = np.stack([
        grad_orientation,
        morph_transformation,
        line_strength_1,
        line_strength_2,
        *gabor_response,
        inverted_green,
    ], axis=-1)

    return features


def compute_gradient_orientation(image):
    threshold = 3 / 255.0

    goa = np.zeros_like(image, dtype=np.float64)

    kernel_size = 31
    for sigma in [np.sqrt(2), 2 * np.sqrt(2), 4]:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=sigma)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=sigma)

        magnitude = np.sqrt(gx**2 + gy**2)

        ux = np.divide(gx, magnitude, where=magnitude != 0)
        uy = np.divide(gy, magnitude, where=magnitude != 0)

        magnitude = np.sqrt(ux**2 + uy**2)

        ux[magnitude < threshold] = 0
        uy[magnitude < threshold] = 0

        dxx = cv2.Sobel(ux, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=sigma)
        dxy = cv2.Sobel(ux, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=sigma)

        dyx = cv2.Sobel(uy, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=sigma)
        dyy = cv2.Sobel(uy, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=sigma)

        D = dxx**2 + dxy**2 + dyx**2 + dyy**2

        goa += D

    return cv2.normalize(goa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def morphological_top_hat(image):
    se_length = 21
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_length, 1))
    se = cv2.copyMakeBorder(
        se, se_length//2, se_length//2, 0, 0, cv2.BORDER_CONSTANT, 0)

    top_hat = []
    for angle in range(0, 180, 22):
        rotated_se = image_rotate(se, angle)
        top_hat.append(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rotated_se))

    return cv2.normalize(sum(top_hat), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def line_strength(image):
    line_length = 21
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
    line = cv2.copyMakeBorder(
        line, line_length//2, line_length//2, 0, 0, cv2.BORDER_CONSTANT, 0)

    max_line = np.zeros_like(image, dtype=np.float64)
    mean_filtered = uniform_filter(image, size=3)
    for angle in range(0, 180, 15):
        filter_2D = image_rotate(line, angle)
        filter_2D = filter_2D/filter_2D.sum()
        filtered_image = cv2.filter2D(image, -1, filter_2D)
        
        max_line = np.maximum(max_line, filtered_image)
        
    line_strength = abs(max_line - mean_filtered)

    return cv2.normalize(line_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def gabor_filter_responses(image):
    final_features = []

    for sigma in (2, 3, 4, 5):

        gabor_features = []
        for theta in range(0, 180, 10):
            gabor_kernel = cv2.getGaborKernel(
                (20*sigma, 20*sigma), sigma, theta, 8, 0.5, 0, ktype=cv2.CV_64F)
            gabor_features.append(abs(cv2.filter2D(
                image, cv2.CV_64F, gabor_kernel)))


        gabor_features = np.stack(gabor_features, axis=-1).max(axis=-1)
        gabor_features = cv2.normalize(
            gabor_features, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        final_features.append(gabor_features)

    return final_features


def extract_features(photos):
    feature_photos = []
    for photo in tqdm(photos, total=len(photos), desc='Extracting features'):
        feature_vector = create_feature_vector(photo)
        feature_photos.append(feature_vector)

    return feature_photos


def create_dataset(photos, features, masks):
    ds_photos = []
    ds_masks = []
    for photo, feature_vector, mask in zip(photos, features, masks):
        _, _, num_features = feature_vector.shape
        # Flatten the features
        #flattened_features = feature_vector.reshape(-1, num_features)
        roi = cv2.cvtColor( photo, cv2.COLOR_RGB2GRAY ) > 0.1*255
        flattened_features = feature_vector[roi]
        ds_photos.append(flattened_features)
        ds_masks.append(mask[roi])  # Flatten the mask

    ds_photos = np.vstack(ds_photos)
    ds_masks = np.hstack(ds_masks)

    return ds_photos, ds_masks
