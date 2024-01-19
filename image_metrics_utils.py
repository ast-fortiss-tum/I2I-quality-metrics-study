# image_metrics_utils.py




import cv2
import numpy as np
import tensorflow as tf
import scipy
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from sklearn.metrics import normalized_mutual_info_score
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.linalg import sqrtm
import glob
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
import warnings
from sklearn.neighbors import KernelDensity
from tensorflow.keras.metrics import MeanSquaredError

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os



def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True,channel_axis=2)


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_value = 255.0
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def calculate_cosine_similarity(images_1, images_2):
    cosine_similarities = []
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    model.compile()
    for i in range(len(images_1)):
        preprocessed_image_1 = preprocess_input(images_1[i])
        preprocessed_image_2 = preprocess_input(images_2[i])
        features_1 = model.predict(np.array([preprocessed_image_1]))
        features_2 = model.predict(np.array([preprocessed_image_2]))
        features_1 = features_1.reshape(1, -1)
        features_2 = features_2.reshape(1, -1)
        similarity = cosine_similarity(features_1, features_2)
        cosine_similarities.append(similarity[0, 0])
        if i % 10 == 0:
            print(i, " out of ", len(images_1))

    return cosine_similarities



def calculate_histogram_similarity(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

def calculate_correlation_coefficient(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    corr_coeff = np.corrcoef(img1_gray.ravel(), img2_gray.ravel())[0, 1]
    return corr_coeff

def calculate_lbp_histogram_similarity(img1, img2):
    lbp1 = cv2.calcHist([cv2.Laplacian(img1, cv2.CV_64F)], [0], None, [256], [0, 256])
    lbp2 = cv2.calcHist([cv2.Laplacian(img2, cv2.CV_64F)], [0], None, [256], [0, 256])
    return calculate_histogram_similarity(lbp1, lbp2)


def calculate_normalized_mutual_info(images1, images2):
    nmi_scores = []
    for img1, img2 in zip(images1, images2):
        hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
        hist1 /= np.sum(hist1)
        hist2 /= np.sum(hist2)
        nmi_score = normalized_mutual_info_score(hist1.flatten(), hist2.flatten())
        nmi_scores.append(nmi_score)
    return nmi_scores

def calculate_texture_similarity(images1, images2):
    glcm_distances = [1, 2, 3]
    texture_sims = []

    for img1, img2 in zip(images1, images2):
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        glcm1 = graycomatrix(gray_img1, distances=glcm_distances, angles=[0], symmetric=True, normed=True)
        glcm2 = graycomatrix(gray_img2, distances=glcm_distances, angles=[0], symmetric=True, normed=True)
        texture_sim = 0.0
        for distance in glcm_distances:
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            for prop in props:
                prop_val1 = graycoprops(glcm1, prop)[ 0, 0]
                prop_val2 = graycoprops(glcm2, prop)[ 0, 0]
                texture_sim += np.abs(prop_val1 - prop_val2)
        
        texture_sims.append(texture_sim)
    return texture_sims


def calculate_wd(images1, images2):
    wasserstein_distances = []
    for img1, img2 in zip(images1, images2):
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        hist1 = np.histogram(img1_flat, bins=256, range=(0, 256), density=True)[0]
        hist2 = np.histogram(img2_flat, bins=256, range=(0, 256), density=True)[0]
        w_distance = scipy.stats.wasserstein_distance(hist1, hist2)
        wasserstein_distances.append(w_distance)
    return wasserstein_distances
    

def calculate_kl_divergence(images1, images2, epsilon=1e-10):
    kl_divergences = []
    for img1, img2 in zip(images1, images2):
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        hist1, bin_edges1 = np.histogram(img1_flat, bins=256, range=(0, 256), density=True)
        hist2, bin_edges2 = np.histogram(img2_flat, bins=256, range=(0, 256), density=True)
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        hist1 += epsilon
        hist2 += epsilon
        kl_divergence = entropy(hist1, hist2)
        kl_divergences.append(kl_divergence)
    return kl_divergences



def calculate_perceptual_distances(images_1, images_2):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    model.compile()
    perceptual_distances = []
    for i in range(len(images_1)):
        preprocessed_image_1 = preprocess_input(images_1[i])
        preprocessed_image_2 = preprocess_input(images_2[i])
        features_1 = model.predict(np.array([preprocessed_image_1]))
        features_2 = model.predict(np.array([preprocessed_image_2]))
        mse = MeanSquaredError()
        mse.update_state(features_1, features_2)
        perceptual_distances.append(mse.result().numpy())
    return perceptual_distances


def calculate_histogram_intersection(images1, images2):
    hist_int_scores = []
    for img1, img2 in zip(images1, images2):
        hist1 = cv2.calcHist([img1], [i for i in range(img1.shape[2])], None, [256] * img1.shape[2], [0, 256] * img1.shape[2])
        hist2 = cv2.calcHist([img2], [i for i in range(img2.shape[2])], None, [256] * img2.shape[2], [0, 256] * img2.shape[2])
        hist_int_score = np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)
        hist_int_scores.append(hist_int_score)
    return hist_int_scores



def load_images_from_folder(folder_path):
    images = []
    paths = []
    for img_path in sorted(glob.glob(folder_path+'*.png')):
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                paths.append(img_path)
    return np.array(images),np.array(paths)


def calculate_inception_score(image_set1, image_set2, batch_size=32):
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    inception_model = Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)
    def _get_predictions(images):
        n_batches = len(images) // batch_size
        preds = []
        for i in range(n_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]
            batch = preprocess_input(batch)
            pred = inception_model.predict(batch)
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)
        return preds

    preds_set1 = _get_predictions(image_set1)
    preds_set2 = _get_predictions(image_set2)
    p_yx_set1 = np.mean(preds_set1, axis=0)
    p_yx_set2 = np.mean(preds_set2, axis=0)
    epsilon = 1e-10
    p_yx_set1 = p_yx_set1 / np.sum(p_yx_set1)
    p_yx_set2 = p_yx_set2 / np.sum(p_yx_set2)
    p_yx_set1 += epsilon
    p_yx_set2 += epsilon
    kl_divergence = np.sum(p_yx_set1 * np.log(p_yx_set1 / p_yx_set2))
    score = np.exp(kl_divergence)

    return score


def calculate_fid(real_images, generated_images, batch_size=32, downsample_factor=4):
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    inception_model = Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)
    def _get_activations(images):
        n_batches = len(images) // batch_size
        activations = []
        for i in range(n_batches):
            print("batch ",i," of ",n_batches)
            batch = images[i * batch_size:(i + 1) * batch_size]
            batch = preprocess_input(batch)
            batch_activations = inception_model.predict(batch)
            new_height = max(1, batch_activations.shape[1] // downsample_factor)
            new_width = max(1, batch_activations.shape[2] // downsample_factor)
            batch_activations = tf.image.resize(batch_activations, (new_height, new_width))
            activations.append(batch_activations)
            tf.keras.backend.clear_session()
        activations = np.concatenate(activations, axis=0)
        return activations

    real_activations = _get_activations(real_images)
    generated_activations = _get_activations(generated_images)
    real_activations = real_activations.reshape(real_activations.shape[0], -1)
    generated_activations = generated_activations.reshape(generated_activations.shape[0], -1)
    mean_real_activations = np.mean(real_activations, axis=0)
    cov_real_activations = np.cov(real_activations, rowvar=False)
    mean_generated_activations = np.mean(generated_activations, axis=0)
    cov_generated_activations = np.cov(generated_activations, rowvar=False)
    mu1, sigma1 = mean_real_activations, cov_real_activations
    mu2, sigma2 = mean_generated_activations, cov_generated_activations
    warnings.filterwarnings("ignore")
    ssd = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssd + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_ifd(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    non_zero_coordinates = np.column_stack(np.where(binary_image > 0))
    max_box_size = int(min(binary_image.shape) / 2)
    box_counts = []
    for box_size in range(1, max_box_size + 1):
        num_boxes = 0
        for x in range(0, binary_image.shape[0], box_size):
            for y in range(0, binary_image.shape[1], box_size):
                if np.any(binary_image[x:x+box_size, y:y+box_size] > 0):
                    num_boxes += 1
        box_counts.append(num_boxes)
    x = np.log(range(1, max_box_size + 1))
    y = np.log(box_counts)
    slope, intercept = np.polyfit(x, y, 1)
    fractal_dimension = -slope
    return fractal_dimension




def calculate_kde(images, bandwidth=0.2, batch_size=32):
    num_images = len(images)
    kde_scores_list = []
    kde_probs_list = []
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='auto', rtol=1e-4, atol=1e-4)
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        combined_images = np.concatenate(images[start_idx:end_idx])
        num_samples, num_features = combined_images.shape[0], np.prod(combined_images.shape[1:])
        combined_images = combined_images.reshape(num_samples, num_features)
        kde.fit(combined_images)
        kde_scores = kde.score_samples(combined_images)
        kde_probs = np.exp(kde_scores)
        kde_scores_list.append(kde_scores)
        kde_probs_list.append(kde_probs)
    kde_scores = np.concatenate(kde_scores_list)
    kde_probs = np.concatenate(kde_probs_list)

    return kde_scores, kde_probs




def calculate_kid_donkey(real_images, generated_images, image_size=(64, 64)):
    def calculate_gram_matrix(data):
        num_samples = tf.shape(data)[0]
        data = tf.image.resize(data, image_size)
        data = tf.reshape(data, [num_samples, -1])
        gram_matrix = tf.matmul(data, data, transpose_a=True)
        gram_matrix /= tf.cast(num_samples, tf.float32)
        return gram_matrix

    print(np.unique(real_images), np.unique(generated_images))
    real_images = tf.convert_to_tensor(real_images, dtype=tf.float32)
    generated_images = tf.convert_to_tensor(generated_images, dtype=tf.float32)
    print(np.unique(real_images), np.unique(generated_images))
    num_samples_real = tf.cast(tf.shape(real_images)[0], tf.float32).numpy()
    num_samples_generated = tf.cast(tf.shape(generated_images)[0], tf.float32).numpy()
    print(num_samples_real, num_samples_generated)
    gram_real = calculate_gram_matrix(real_images)
    gram_generated = calculate_gram_matrix(generated_images)
    print(gram_real.shape, gram_generated.shape)
    mmd2 = (
        tf.reduce_sum(gram_real) / (num_samples_real * (num_samples_real - 1)) +
        tf.reduce_sum(gram_generated) / (num_samples_generated * (num_samples_generated - 1)) -
        2 * tf.reduce_sum(tf.matmul(real_images, generated_images, transpose_a=True)) /
        (num_samples_real * num_samples_generated)
    )
    print(mmd2)
    return mmd2


def calculate_kid_kitti(real_images, generated_images, image_size=(64, 64), batch_size=100):
    def calculate_gram_matrix(data):
        num_samples = tf.shape(data)[0]
        data = tf.image.resize(data, image_size)
        data = tf.reshape(data, [num_samples, -1])
        gram_matrix = tf.matmul(data, data, transpose_a=True)
        gram_matrix /= tf.cast(num_samples, tf.float32)
        return gram_matrix

    num_samples_real = len(real_images)
    num_samples_generated = len(generated_images)

    real_batches = [real_images[i:i+batch_size] for i in range(0, num_samples_real, batch_size)]
    generated_batches = [generated_images[i:i+batch_size] for i in range(0, num_samples_generated, batch_size)]

    mmd2 = 0.0

    for real_batch, generated_batch in zip(real_batches, generated_batches):
        real_batch = tf.convert_to_tensor(real_batch, dtype=tf.float32)
        generated_batch = tf.convert_to_tensor(generated_batch, dtype=tf.float32)
        real_batch = real_batch / 255.0
        generated_batch = generated_batch / 255.0
        gram_real = calculate_gram_matrix(real_batch)
        gram_generated = calculate_gram_matrix(generated_batch)
        batch_mmd2 = (
            tf.reduce_sum(gram_real) / (num_samples_real * (num_samples_real - 1)) +
            tf.reduce_sum(gram_generated) / (num_samples_generated * (num_samples_generated - 1)) -
            2 * tf.reduce_sum(tf.matmul(real_batch, generated_batch, transpose_a=True)) /
            (num_samples_real * num_samples_generated)
        )
        mmd2 += batch_mmd2
        print(mmd2)
    mmd2 /= len(real_batches)
    return mmd2




def calculate_semantic_segmentation_score(images_1, images_2):
    segmentation_model = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    sss_scores = []
    for image1, image2 in zip(images_1, images_2):
        preprocessed_image1 = (image1)
        preprocessed_image2 = (image2)
        segmentation1 = segmentation_model.predict(np.expand_dims(preprocessed_image1, axis=0))
        segmentation2 = segmentation_model.predict(np.expand_dims(preprocessed_image2, axis=0))
        mse = np.mean((segmentation1 - segmentation2) ** 2)
        sss_scores.append(mse)
    return sss_scores