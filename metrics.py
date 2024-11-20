import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix

def apply_gaussian_noise(img, mean=0, var=5000):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy_img = np.clip(img + gauss, 0, 255).astype(np.uint8)
    return noisy_img

def kmeans_segmentation(img, K=2):
    img_flat = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(img_flat)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(img.shape).astype(np.uint8)
    return segmented_img, kmeans.labels_.reshape(img.shape[:2])


def sobel_edge_detection(img_gray):
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  
    edges = np.hypot(sobelx, sobely)
    edges = np.uint8(edges / np.max(edges) * 255)
    return edges

def pixel_accuracy(predicted_mask, ground_truth_mask):
    return accuracy_score(ground_truth_mask.flatten(), predicted_mask.flatten())


def calculate_metrics(ground_truth, predicted):
    accuracy = accuracy_score(ground_truth.flatten(), predicted.flatten())
    precision = precision_score(ground_truth.flatten(), predicted.flatten(), average='macro', zero_division=0)
    recall = recall_score(ground_truth.flatten(), predicted.flatten(), average='macro', zero_division=0)
    f1 = f1_score(ground_truth.flatten(), predicted.flatten(), average='macro', zero_division=0)
    iou = jaccard_score(ground_truth.flatten(), predicted.flatten(), average=None)
    return accuracy, precision, recall, f1, iou


st.title("Image Segmentation with K-means and Sobel Edge Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  
    st.image(image_rgb, caption='Original Image', use_column_width=True)

 
    k_value = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)
    kmeans_segmented, kmeans_labels = kmeans_segmentation(image_rgb, K=k_value)

    sobel_edges = sobel_edge_detection(image_gray)

    noise_option = st.checkbox("Add Gaussian Noise")
    if noise_option:
        noisy_image = apply_gaussian_noise(image)
        st.image(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB), caption='Noisy Image', use_column_width=True)
        kmeans_noisy_segmented, kmeans_noisy_labels = kmeans_segmentation(noisy_image, K=k_value)
        sobel_noisy_edges = sobel_edge_detection(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))
    else:
        noisy_image = image
        kmeans_noisy_labels = None  
        sobel_noisy_edges = sobel_edge_detection(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))

    st.image(kmeans_segmented, caption='K-means Segmentation', use_column_width=True)
    st.image(sobel_edges, caption='Sobel Edges', use_column_width=True)

    if noise_option:
        st.image(noisy_image, caption='Noisy Image', use_column_width=True)
        st.image(kmeans_noisy_segmented, caption='K-means on Noisy Image', use_column_width=True)
        st.image(sobel_noisy_edges, caption='Sobel on Noisy Image', use_column_width=True)


    ground_truth_mask = np.random.randint(0, k_value, image_gray.shape)


    kmeans_accuracy, kmeans_precision, kmeans_recall, kmeans_f1, kmeans_iou = calculate_metrics(ground_truth_mask, kmeans_labels)
    sobel_accuracy, sobel_precision, sobel_recall, sobel_f1, sobel_iou = calculate_metrics(ground_truth_mask, (sobel_edges > 0).astype(np.uint8))


    if noise_option:
        kmeans_noisy_accuracy = pixel_accuracy(kmeans_noisy_labels, ground_truth_mask)
        sobel_noisy_accuracy = pixel_accuracy(sobel_noisy_edges > 0, ground_truth_mask > 0)
    else:
        kmeans_noisy_accuracy = None
        sobel_noisy_accuracy = None


    metrics_data = {
        "Metric": ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Noisy Accuracy'],
        "K-means": [kmeans_accuracy, kmeans_precision, kmeans_recall, kmeans_f1, kmeans_noisy_accuracy],
        "Sobel": [sobel_accuracy, sobel_precision, sobel_recall, sobel_f1, sobel_noisy_accuracy]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.subheader("Metrics Comparison")
    st.write(metrics_df)


    fig, ax = plt.subplots()
    metrics_df.plot(x='Metric', kind='bar', ax=ax)
    plt.title('Metrics Comparison')
    plt.xticks(rotation=0)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    st.pyplot(fig)

    st.subheader("IoU and Confusion Matrix")


    iou_matrix = np.zeros((k_value, k_value))
    for i in range(k_value):
        iou_matrix[i, i] = kmeans_iou[i]

    fig, ax = plt.subplots()
    sns.heatmap(iou_matrix, annot=True, cmap="YlGnBu", xticklabels=range(k_value), yticklabels=range(k_value), ax=ax)
    plt.title("IoU Heatmap for K-means Clusters")
    plt.xlabel('Predicted Labels')
    plt.ylabel('Ground Truth Labels')
    st.pyplot(fig)

 
    conf_matrix = confusion_matrix(ground_truth_mask.flatten(), kmeans_labels.flatten(), labels=range(k_value))

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(k_value), yticklabels=range(k_value), ax=ax)
    plt.title("Confusion Matrix for K-means Segmentation")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(fig)


