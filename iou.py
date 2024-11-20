import cv2
import numpy as np
from sklearn.cluster import KMeans


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

def calculate_iou(segmentation1, segmentation2):
  
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

#binarise images to calcualte iou
segmented_image_binary = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
segmented_image_binary = segmented_image_binary > 128

edges_binary = edges > 128

iou = calculate_iou(segmented_image_binary, edges_binary)
print(f"Intersection Over Union (IoU) between K-means and Sobel: {iou:.4f}")
