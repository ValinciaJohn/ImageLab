import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def kmeans_segmentation(image, K=2):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image




K = 2 
segmented_image = kmeans_segmentation(image, K)

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  
  
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined)

    return sobel_combined



edges = sobel_edge_detection(image)

# Add Gaussian noise to the image
def add_noise(image, mean=0, var=1000):  
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)  
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

noisy_image = add_noise(image)


segmented_noisy_image = kmeans_segmentation(noisy_image, K=3)
edges_noisy = sobel_edge_detection(noisy_image)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"K-means Segmentation on Noisy Image (K={K})")
plt.imshow(segmented_noisy_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sobel Edge Detection on Noisy Image")
plt.imshow(edges_noisy, cmap='gray')
plt.axis('off')

plt.show()
