import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)            #gradient of x axis 
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)            #gradient of y axis 

    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)               #gradeint magnitude
    sobel_combined = np.uint8(sobel_combined)

    return sobel_combined

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


edges = sobel_edge_detection(image)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sobel Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
