import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_segmentation(image, K=2):
    pixel_values = image.reshape((-1, 3))                           #image to 2d pixel array of 3 values 
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.predict(pixel_values)                           #pixels to labels 
    centers = np.uint8(kmeans.cluster_centers_)

                                                                    # labels to image 
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


K = 2 
segmented_image = kmeans_segmentation(image, K)

#visualization 

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"K-means Segmentation (K={K})")
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
