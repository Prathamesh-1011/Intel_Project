import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from google.colab import files
from google.colab.patches import cv2_imshow

def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 0.1, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
    elif noise_type == 'salt_pepper':
        noisy_image = random_noise(image, mode='s&p', amount=0.1)
    else:
        raise ValueError("Unsupported noise type")
    return noisy_image

def gaussian_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def median_filter(image, kernel_size=5):
    return cv2.medianBlur((image * 255).astype(np.uint8), kernel_size) / 255.0

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter((image * 255).astype(np.uint8), d, sigma_color, sigma_space) / 255.0

def nlm_filter(image):
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True,
                            patch_size=5, patch_distance=6, multichannel=True)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(1.0 / mse)

def compare_denoising_methods():
    # Upload image
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]

    # Read image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) / 255.0

    # Add noise
    noisy = add_noise(original, 'gaussian')

    # Apply denoising methods
    gaussian = gaussian_filter(noisy)
    median = median_filter(noisy)
    bilateral = bilateral_filter(noisy)
    nlm = nlm_filter(noisy)

    # Calculate PSNR for each method
    print(f"Gaussian Filter PSNR: {psnr(original, gaussian):.2f}")
    print(f"Median Filter PSNR: {psnr(original, median):.2f}")
    print(f"Bilateral Filter PSNR: {psnr(original, bilateral):.2f}")
    print(f"Non-local Means Filter PSNR: {psnr(original, nlm):.2f}")

    # Display results
    plt.figure(figsize=(20, 10))
    images = [original, noisy, gaussian, median, bilateral, nlm]
    titles = ['Original', 'Noisy', 'Gaussian', 'Median', 'Bilateral', 'Non-local Means']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the comparison
compare_denoising_methods()

