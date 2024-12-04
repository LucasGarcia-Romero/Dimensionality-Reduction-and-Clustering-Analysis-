# Clustering and Feature Extraction Project

## Description
This project implements several feature extraction and clustering techniques on a dataset of images. The aim is to explore the performance of different clustering methods (K-means, DBSCAN, and Affinity Propagation) and evaluate their clustering metrics such as Precision, Recall, F1 Score, Rand Index, and Adjusted Rand Index.

The project also includes the usage of feature extraction techniques like PCA, t-SNE, and Isomap to reduce the dimensionality of the dataset before applying the clustering methods.

## Features
- **Feature Extraction**:
  - PCA (Principal Component Analysis)
  - t-SNE (t-distributed Stochastic Neighbor Embedding)
  - Isomap (Isometric Mapping)

- **Clustering Algorithms**:
  - K-means
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - Affinity Propagation

- **Performance Metrics**:
  - Precision, Recall, F1 Score
  - Rand Index and Adjusted Rand Index

## Files and Structure
- `main.py`: The main script that runs the feature extraction, clustering, and evaluation.
- `tp2_aux.py`: Contains auxiliary functions such as image loading and cluster reporting.
- `labels.txt`: The dataset labels used for clustering evaluation.
- `results/`: Folder to store the output plots and evaluation metrics.
  - `kmeans_scores.png`: Plot of K-means clustering scores for different numbers of features.
  - `dbscan_scores.png`: Plot of DBSCAN clustering scores for different numbers of features.
  - `affinity_scores.png`: Plot of Affinity Propagation clustering scores for different numbers of features.
- `./k_means/`, `./dbscan/`, `./affinity_propagation/`: Subfolders containing clustering reports in HTML format.

## Requirements
To run this project, you will need the following libraries installed:

- Python 3.x
- NumPy
- Scikit-learn
- Matplotlib