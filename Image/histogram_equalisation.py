import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_equalization(img):
    # Step 1: Flatten image and calculate histogram (0-255 bins)
    hist = np.zeros(256)
    for pixel in img.flatten():
        hist[pixel] += 1

    # Step 2: Normalize histogram (PDF)
    pdf = hist / np.sum(hist)

    # Step 3: Compute CDF
    cdf = np.cumsum(pdf)

    # Step 4: Normalize CDF to [0, 255]
    cdf_normalized = np.round(cdf * 255).astype(np.uint8)

    # Step 5: Map original pixels using CDF as LUT
    equalized_img = cdf_normalized[img]

    return equalized_img

# Load grayscale image
image = cv2.imread("your_image.jpg", 0)  # 0 for grayscale
equalized = histogram_equalization(image)

def adaptive_histogram_equalization(img, tile_size=8):
    h, w = img.shape
    output = np.zeros_like(img)
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            # Define tile boundaries
            i_end = min(i + tile_size, h)
            j_end = min(j + tile_size, w)

            tile = img[i:i_end, j:j_end]

            # Apply histogram equalization on tile
            eq_tile = histogram_equalization(tile)

            output[i:i_end, j:j_end] = eq_tile

    return output

# Apply AHE
ahe_image = adaptive_histogram_equalization(image, tile_size=32)

def clahe(img, tile_size=8, clip_limit=40):
    h, w = img.shape
    output = np.zeros_like(img)

    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            i_end = min(i + tile_size, h)
            j_end = min(j + tile_size, w)
            tile = img[i:i_end, j:j_end]

            # Step 1: Compute histogram
            hist = np.zeros(256)
            for pixel in tile.flatten():
                hist[pixel] += 1

            # Step 2: Clip the histogram
            excess = 0
            for k in range(256):
                if hist[k] > clip_limit:
                    excess += hist[k] - clip_limit
                    hist[k] = clip_limit

            # Step 3: Redistribute excess pixels
            redistribute = excess // 256
            hist += redistribute

            # Optional: handle remaining pixels
            remainder = excess % 256
            hist[:remainder] += 1

            # Step 4: Normalize and compute CDF
            pdf = hist / np.sum(hist)
            cdf = np.cumsum(pdf)
            cdf_normalized = np.round(cdf * 255).astype(np.uint8)

            # Step 5: Map original pixels
            equalized_tile = cdf_normalized[tile]
            output[i:i_end, j:j_end] = equalized_tile

    return output

clahe_image = clahe(image, tile_size=32, clip_limit=40)


# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Histogram Equalized")
plt.imshow(equalized, cmap='gray')
plt.show()

plt.imshow(ahe_image, cmap='gray')
plt.title("AHE")
plt.show()

plt.imshow(clahe_image, cmap='gray')
plt.title("CLAHE")
plt.show()