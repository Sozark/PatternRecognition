# knn.py
import numpy as np

class KNN:
    def __init__(self, k=3): # k representes the considered neighbors
        self.k = k 

    def fit(self, X_train, y_train): # Short for Training Function
        self.X_train = X_train.reshape(len(X_train), -1)
        self.y_train = y_train

    def predict(self, X_test): # Used to Classify new Data Points
        X_test_flat = X_test.reshape(len(X_test), -1)
        predictions = []
        for x in X_test_flat:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)
