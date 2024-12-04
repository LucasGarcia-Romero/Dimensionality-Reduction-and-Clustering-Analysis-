import os
from sklearn.feature_selection import f_classif
from sklearn import decomposition
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics.cluster import adjusted_rand_score
import math
import matplotlib.pyplot as plt

import tp2_aux as aux  # Importar funciones auxiliares definidas en tp2_aux.py

# ANOVA feature selector
def ANOVA(X, y, n_feats):
    f, prob = f_classif(X[y[:] > 0], y[y[:] > 0])
    lprob = prob.tolist()
    lprobc = lprob.copy()
    lprobc.sort()
    res = []
    for i in range(0, n_feats):
        res.append(lprob.index(lprobc[i]))
    return res

# K-means clustering
def k_Means(NClusters, columns, ids):
    kmeans = KMeans(n_clusters=NClusters).fit(columns)
    labels = kmeans.predict(columns)
    aux.report_clusters(ids, labels, "./k_means/K_means_k" + str(NClusters) + "_nf" + str(columns.shape[1]) + ".html")
    return labels

# PCA feature extractor
def PCA(matrix):
    pca = decomposition.PCA(n_components=6)
    t_data = pca.fit_transform(matrix)
    return t_data

# t_SNE feature extractor
def t_SNE(matrix):
    tsne = TSNE(n_components=6, method='exact')
    t_data = tsne.fit_transform(matrix)
    return t_data

# isomap feature extractor
def isomap(matrix):
    isomap = Isomap(n_components=6)
    t_data = isomap.fit_transform(matrix)
    return t_data

# External index data
def external_index(predicted_labels, labels, nClusters):
    predicted_labels = predicted_labels[labels[:] > 0]
    known_labels = labels[labels[:] > 0]

    KS = np.zeros((nClusters, 3))

    for i in range(0, predicted_labels.shape[0]):
        if predicted_labels[i] < nClusters:
            if known_labels[i] - 1 < KS.shape[1]:
                KS[predicted_labels[i]][known_labels[i] - 1] += 1
            else:
                print(f"Warning: known_labels[{i}] - 1 ({known_labels[i] - 1}) is out of bounds for KS.shape[1] ({KS.shape[1]})")
        else:
            print(f"Warning: predicted_labels[{i}] ({predicted_labels[i]}) is out of bounds for nClusters ({nClusters})")

    prec_s, recall_s, f1_s, rand_index = external_index_calculation(KS, predicted_labels.shape[0])

    adj_rand_index = adjusted_rand_score(known_labels, predicted_labels)
    return prec_s, recall_s, f1_s, rand_index, adj_rand_index

def external_index_calculation(KS, N):
    pairs = N * (N - 1) / 2
    total_positives = 0
    for i in range(0, KS.shape[0]):
        els = 0
        for j in range(0, KS.shape[1]):
            els += KS[i][j]
        total_positives += math.comb(int(els), 2)

    true_positives = 0
    for i in range(KS.shape[0]):
        for j in range(KS.shape[1]):
            if (KS[i][j] > 1):
                true_positives += math.comb(int(KS[i][j]), 2)

    false_positives = total_positives - true_positives

    total_negatives = pairs - total_positives

    false_negatives = 0
    for i in range(0, KS.shape[1]):
        for j in range(0, KS.shape[0]):
            match = KS[j][i]
            mismatches = 0
            for p in range(j + 1, KS.shape[0]):
                mismatches += KS[p][i]
            false_negatives += match * mismatches

    true_negatives = total_negatives - false_negatives
    assert (true_positives + false_positives + true_negatives + false_negatives == pairs)

    precision = true_positives / (false_positives + true_positives)
    recall = true_positives / (false_negatives + true_positives)
    f1 = 2 * ((precision * recall) / (precision + recall))
    rand = (true_positives + true_negatives) / pairs

    return precision, recall, f1, rand

# DBSCAN clustering
def dbscan(feats, labels, epsilon):
    clustering = DBSCAN(eps=epsilon).fit_predict(feats)
    aux.report_clusters(labels[:, 0], clustering, "./dbscan/epsilon_" + str(epsilon) +"_nf" + str(feats.shape[1]) + ".html")
    return clustering

# Affinity Propagation clustering
def affinity_propagation(feats, labels):
    clustering = AffinityPropagation().fit_predict(feats)
    aux.report_clusters(labels[:, 0], clustering, "./affinity_propagation/nf" + str(feats.shape[1]) + ".html")
    return clustering

def Main():
    os.makedirs("./results/", exist_ok=True)  # Crear carpeta para guardar resultados

    epsi = [45, 440, 900, 1000, 1000, 1050, 1050]
    # Lectura de etiquetas desde el archivo labels.txt
    lines = open("labels.txt").readlines()
    labels = []
    for line in lines:
        line = line.strip('\n')
        parts = line.split(',')
        labels.append((int(parts[0]), int(parts[1])))
    labels = np.array(labels)

    # Lectura de imágenes y extracción de características
    matrix = aux.images_as_matrix()
    pca_features = PCA(matrix)
    tsne_features = t_SNE(matrix)
    isomap_features = isomap(matrix)

    # Concatenación de características
    features = np.concatenate((pca_features, tsne_features, isomap_features), axis=1)

    # Listas para almacenar métricas por número de características
    precision_scores_kmeans = []
    recall_scores_kmeans = []
    f1_scores_kmeans = []
    rand_index_scores_kmeans = []
    adj_rand_index_scores_kmeans = []

    precision_scores_dbscan = []
    recall_scores_dbscan = []
    f1_scores_dbscan = []
    rand_index_scores_dbscan = []
    adj_rand_index_scores_dbscan = []

    precision_scores_affinity = []
    recall_scores_affinity = []
    f1_scores_affinity = []
    rand_index_scores_affinity = []
    adj_rand_index_scores_affinity = []

    # Evaluación con diferentes números de características
    for i in range(1, 8):
        selected_indices = ANOVA(features, labels[:, 1], i)
        selected_features = features[:, selected_indices]

        dbscan_labels = dbscan(selected_features, labels, epsi[i - 1])
        kmeans_labels = k_Means(i, selected_features, labels[:, 0])
        affinity_labels = affinity_propagation(selected_features, labels)

        # Evaluación de métricas externas para K-means
        prec_s, recall_s, f1_s, rand_index, adj_rand_index = external_index(kmeans_labels, labels[:, 1], i)

        precision_scores_kmeans.append(prec_s)
        recall_scores_kmeans.append(recall_s)
        f1_scores_kmeans.append(f1_s)
        rand_index_scores_kmeans.append(rand_index)
        adj_rand_index_scores_kmeans.append(adj_rand_index)

        print(f"\n\nEvaluating {i} features for K-means:")
        print("Precision Score:", prec_s)
        print("Recall Score:", recall_s)
        print("F1 Score:", f1_s)
        print("Rand Index Score:", rand_index)
        print("Adjusted Rand Index Score:", adj_rand_index)

        # Evaluación de métricas externas para DBSCAN
        prec_s, recall_s, f1_s, rand_index, adj_rand_index = external_index(dbscan_labels, labels[:, 1], i)

        precision_scores_dbscan.append(prec_s)
        recall_scores_dbscan.append(recall_s)
        f1_scores_dbscan.append(f1_s)
        rand_index_scores_dbscan.append(rand_index)
        adj_rand_index_scores_dbscan.append(adj_rand_index)

        print(f"\n\nEvaluating {i} features for DBSCAN:")
        print("Precision Score:", prec_s)
        print("Recall Score:", recall_s)
        print("F1 Score:", f1_s)
        print("Rand Index Score:", rand_index)
        print("Adjusted Rand Index Score:", adj_rand_index)

        # Evaluación de métricas externas para Affinity Propagation
        prec_s, recall_s, f1_s, rand_index, adj_rand_index = external_index(affinity_labels, labels[:, 1], len(np.unique(affinity_labels)))

        precision_scores_affinity.append(prec_s)
        recall_scores_affinity.append(recall_s)
        f1_scores_affinity.append(f1_s)
        rand_index_scores_affinity.append(rand_index)
        adj_rand_index_scores_affinity.append(adj_rand_index)

        print(f"\n\nEvaluating {i} features for Affinity Propagation:")
        print("Precision Score:", prec_s)
        print("Recall Score:", recall_s)
        print("F1 Score:", f1_s)
        print("Rand Index Score:", rand_index)
        print("Adjusted Rand Index Score:", adj_rand_index)

    # Creación de la gráfica para K-means
    num_features = range(1, 8)  # Ajusta según tu rango de features evaluados

    plt.figure(figsize=(10, 6))

    plt.plot(num_features, precision_scores_kmeans, marker='o', label='Precision Score')
    plt.plot(num_features, recall_scores_kmeans, marker='o', label='Recall Score')
    plt.plot(num_features, f1_scores_kmeans, marker='o', label='F1 Score')
    plt.plot(num_features, rand_index_scores_kmeans, marker='o', label='Rand Index Score')
    plt.plot(num_features, adj_rand_index_scores_kmeans, marker='o', label='Adjusted Rand Index Score')

    plt.xlabel('Number of Features')
    plt.ylabel('Score Value')
    plt.title('Scores vs Number of Features for K-means clustering')
    plt.xticks(num_features)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./results/kmeans_scores.png')  # Guardar figura en carpeta results
    plt.close()

    # Creación de la gráfica para DBSCAN
    plt.figure(figsize=(10, 6))

    plt.plot(num_features, precision_scores_dbscan, marker='o', label='Precision Score')
    plt.plot(num_features, recall_scores_dbscan, marker='o', label='Recall Score')
    plt.plot(num_features, f1_scores_dbscan, marker='o', label='F1 Score')
    plt.plot(num_features, rand_index_scores_dbscan, marker='o', label='Rand Index Score')
    plt.plot(num_features, adj_rand_index_scores_dbscan, marker='o', label='Adjusted Rand Index Score')

    plt.xlabel('Number of Features')
    plt.ylabel('Score Value')
    plt.title('Scores vs Number of Features for DBSCAN clustering')
    plt.xticks(num_features)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./results/dbscan_scores.png')  # Guardar figura en carpeta results
    plt.close()

    # Creación de la gráfica para Affinity Propagation
    plt.figure(figsize=(10, 6))

    plt.plot(num_features, precision_scores_affinity, marker='o', label='Precision Score')
    plt.plot(num_features, recall_scores_affinity, marker='o', label='Recall Score')
    plt.plot(num_features, f1_scores_affinity, marker='o', label='F1 Score')
    plt.plot(num_features, rand_index_scores_affinity, marker='o', label='Rand Index Score')
    plt.plot(num_features, adj_rand_index_scores_affinity, marker='o', label='Adjusted Rand Index Score')

    plt.xlabel('Number of Features')
    plt.ylabel('Score Value')
    plt.title('Scores vs Number of Features for Affinity Propagation clustering')
    plt.xticks(num_features)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./results/affinity_scores.png')  # Guardar figura en carpeta results
    plt.close()

if __name__ == "__main__":
    Main()
