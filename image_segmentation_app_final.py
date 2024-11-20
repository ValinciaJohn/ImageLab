import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64
from sklearn.cluster import KMeans
import time

import numpy as np
import cv2

def initialize_centroids(pixel_values, K):
    """Randomly initialize centroids from the pixel values."""
    indices = np.random.choice(pixel_values.shape[0], K, replace=False)
    return pixel_values[indices]

def assign_clusters(pixel_values, centroids):
    """Assign each pixel to the nearest centroid."""
    distances = np.linalg.norm(pixel_values[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(pixel_values, labels, K):
    """Update centroids by calculating the mean of assigned pixels."""
    new_centroids = np.array([pixel_values[labels == k].mean(axis=0) for k in range(K)])
    return new_centroids

def kmeans_segmentation(image, K, max_iters=100, tolerance=1e-4):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:  
        image_rgb = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

    else: 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    centroids = initialize_centroids(pixel_values, K)

    for i in range(max_iters):
        labels = assign_clusters(pixel_values, centroids)
        old_centroids = centroids.copy()
        centroids = update_centroids(pixel_values, labels, K)
        if np.all(np.linalg.norm(centroids - old_centroids, axis=1) < tolerance):
            break

   
    segmented_image = centroids[labels].reshape(image_rgb.shape).astype(np.uint8)

    return segmented_image


#def kmeans_segmentation(image, K):
#    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
#    pixel_values = image_rgb.reshape((-1, 3))  
#    pixel_values = np.float32(pixel_values)  
#    kmeans = KMeans(n_clusters=K, random_state=42)
#    kmeans.fit(pixel_values)
#    labels = kmeans.predict(pixel_values)
#    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#    centers = np.uint8(centers)
#    segmented_image = centers[labels.flatten()]
#    segmented_image = segmented_image.reshape(image_rgb.shape)

#    return segmented_image

#def kmeans_segmentation(image, K):
    # Ensure the image is in uint8 format for OpenCV operations
#    if image.dtype != np.uint8:
#        image = (image * 255).astype(np.uint8)

    # Convert the image to RGB (if it's a color image)
#    if len(image.shape) == 2:  # Grayscale image
#        image_rgb = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
#    else:  # Color image
#        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image for k-means clustering
#    pixel_values = image_rgb.reshape((-1, 3))
#    pixel_values = np.float32(pixel_values)

    # Define criteria for k-means clustering
#    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply k-means clustering
#    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and recreate the segmented image
#    centers = np.uint8(centers)
#    segmented_image = centers[labels.flatten()]
#    segmented_image = segmented_image.reshape(image_rgb.shape)

#    return segmented_image  

def sobel_edge_detection(image):
   
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif len(image.shape) == 2: 
        gray = image
    else:
        raise ValueError("Input image must be either grayscale or color.")


    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_edges = cv2.magnitude(sobelx, sobely)

    # Convert the edges back to uint8
    sobel_edges = np.clip(sobel_edges, 0, 255).astype(np.uint8)

    return sobel_edges

    #sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)
    #return np.uint8(sobel_combined)

#def add_noise(image, mean=0, var=5000): 
    #row, col, ch = image.shape
    #sigma = var ** 0.5
    #gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
    #noisy_image = cv2.add(image, gauss)
    #return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_noise(image):
    if len(image.shape) == 2:
        row, col = image.shape
        ch = 1 
        noisy_image = np.zeros((row, col), dtype=np.float32)
    else:
        row, col, ch = image.shape
        noisy_image = np.zeros((row, col, ch), dtype=np.float32)

    mean = 0
    sigma = 25 
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    if ch == 1:
        noisy_image = image + gauss[:, :, 0]
    else:
        noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255) / 255.0

    return noisy_image


def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(image, contours):
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)  # Draw contours in red
    return contour_image

#def calculate_iou(segmentation1, segmentation2):

#    intersection = np.logical_and(segmentation1, segmentation2)
#    union = np.logical_or(segmentation1, segmentation2)
#    iou = np.sum(intersection) / np.sum(union)
#    return iou



#def download_image(image, filename):
#   buf = io.BytesIO()
#    pil_image = Image.fromarray(image)
#   pil_image.save(buf, format="PNG")
#     byte_im = buf.getvalue()

 
#    b64 = base64.b64encode(byte_im).decode()


#    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download {filename}</a>'



def download_image(image, filename):
    if image.ndim == 2: 
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode="L")
    else:  
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    b64_image = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64_image}" download="{filename}">Download {filename}</a>'
    return href



st.markdown(
    """
    <style>
    .main {
        background-color: black;
    }
    .sidebar .sidebar-content {
        background-color: white;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: white;
    }
    .sidebar .sidebar-content p, .sidebar .sidebar-content label {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖼️ Image Segmentation and Edge Detection App")

uploaded_file = st.file_uploader("Upload an image file...", type=["jpg", "jpeg", "png"])

with st.sidebar:
    st.header("🔧 Configuration")
 
    K = st.slider("Select number of clusters (K)", 1, 10, 2)
    
    kmeans_selected = st.checkbox("Apply K-means Segmentation", value=False, key="kmeans_segmentation")
    sobel_selected = st.checkbox("Apply Sobel Edge Detection", value=False, key="sobel_edge_detection")

    noise_selected = st.checkbox("Add Gaussian Noise", value=False, key="add_gaussian_noise")
    if noise_selected:
        noisy_kmeans = st.checkbox("Apply K-means on Noisy Image", value=False, key="kmeans_on_noisy_image")
        noisy_sobel = st.checkbox("Apply Sobel on Noisy Image", value=False, key="sobel_on_noisy_image")
   
    contour_selected = st.checkbox("Display Contours from Sobel", value=False, key="display_contours_sobel")
    if contour_selected:
        overlay_sobel_contours_on_kmeans = st.checkbox("Overlay Sobel Contours on K-means", value=False, key="overlay_sobel_contours")

sobel_edges = None
segmented_image = None
contours = []

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)


    if kmeans_selected:
        start_time = time.time()
        segmented_image = kmeans_segmentation(image, K)
        kmeans_time = time.time() - start_time
        st.image(segmented_image, caption=f'K-means Segmentation (K={K})', use_column_width=True)
        st.markdown(download_image(segmented_image, 'kmeans_segmentation.png'), unsafe_allow_html=True)
        st.write(f"K-means segmentation took {kmeans_time:.4f} seconds")

        segmented_image_binary = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        segmented_image_binary = segmented_image_binary > 128


    if sobel_selected:
        start_time = time.time()
        sobel_edges = sobel_edge_detection(image)
        sobel_time = time.time() - start_time
        st.image(sobel_edges, caption="Sobel Edge Detection", use_column_width=True)
        st.write(f"Sobel edge detection took {sobel_time:.4f} seconds")
        edges_binary = sobel_edges > 128

        #if 'segmented_image_binary' in locals() and sobel_edges is not None:
        #    iou = calculate_iou(segmented_image_binary, edges_binary)
        #    st.write(f"Intersection Over Union (IoU) between K-means and Sobel: {iou:.4f}")


    if noise_selected:
        noisy_image = add_noise(image)
        st.image(noisy_image, caption='Noisy Image', use_column_width=True)
        st.markdown(download_image(noisy_image, 'noisy_image.png'), unsafe_allow_html=True)


        if noisy_kmeans:
            segmented_noisy_image = kmeans_segmentation(noisy_image, K)
            st.image(segmented_noisy_image, caption=f'K-means on Noisy Image (K={K})', use_column_width=True)
            st.markdown(download_image(segmented_noisy_image, 'kmeans_noisy_segmentation.png'), unsafe_allow_html=True)

        if noisy_sobel:
            sobel_noisy_edges = sobel_edge_detection(noisy_image)
            st.image(sobel_noisy_edges, caption='Sobel on Noisy Image', use_column_width=True)
            st.markdown(download_image(sobel_noisy_edges, 'sobel_noisy_edges.png'), unsafe_allow_html=True)

    if sobel_edges is not None and contour_selected:
        contours = detect_contours(sobel_edges)
        contour_image = cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2RGB) 
        contour_image_with_contours = draw_contours(contour_image, contours)
        st.image(contour_image_with_contours, caption="Contours from Sobel", use_column_width=True)
        st.markdown(download_image(contour_image_with_contours, 'sobel_contours.png'), unsafe_allow_html=True)

        if overlay_sobel_contours_on_kmeans and segmented_image is not None:
            overlay_image = draw_contours(segmented_image, contours)
            st.image(overlay_image, caption="K-means with Sobel Contours", use_column_width=True)
            st.markdown(download_image(overlay_image, 'kmeans_with_contours.png'), unsafe_allow_html=True)


    if st.button("Save Processed Image"):
        pil_image = Image.fromarray(segmented_image if segmented_image is not None else image)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Processed Image", data=byte_im, file_name="processed_image.png", mime="image/png")
