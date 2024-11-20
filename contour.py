import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) 
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = np.uint8(sobel_combined)

    return sobel_combined

image = cv2.imread('image.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

K = 2  
segmented_image = kmeans_segmentation(image, K)

sobel_edges = sobel_edge_detection(image)

if np.count_nonzero(sobel_edges) == 0:
    print("No edges detected by Sobel.")

contours, _ = cv2.findContours(sobel_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {len(contours)}")

contour_image = np.zeros_like(sobel_edges)

if len(contours) > 0:
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

sobel_edge_count = np.count_nonzero(sobel_edges)
print(f"Sobel edge count: {sobel_edge_count} pixels")

gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
overlap = cv2.bitwise_and(gray_segmented, contour_image)
overlap_count = np.count_nonzero(overlap)
print(f"Overlap between Sobel edges and K-means boundaries: {overlap_count} pixels")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"K-means Segmentation (K={K})")
plt.imshow(segmented_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Contours from Sobel Edge Detection")
plt.imshow(contour_image, cmap='gray')
plt.axis('off')

plt.show()

overlay_image = segmented_image.copy()
if len(contours) > 0:
    cv2.drawContours(overlay_image, contours, -1, (255, 0, 0), 1)

plt.figure(figsize=(8, 8))
plt.title("K-means Segmentation with Sobel Contours Overlay")
plt.imshow(overlay_image)
plt.axis('off')
plt.show()
