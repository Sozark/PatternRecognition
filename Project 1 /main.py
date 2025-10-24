# main.py
import numpy as np
from utils import load_mnist_images, train_test_split
from knn import KNN
from naive_bayes import NaiveBayes

# Load Data
X, y = load_mnist_images("data")
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ---- KNN Example ----
knn = KNN(k=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test[:100])  # test on a subset for speed
acc = np.mean(preds == y_test[:100])
print(f"KNN (k=3) Accuracy: {acc * 100:.2f}%")

# ---- Naive Bayes Example ----
nb = NaiveBayes()
nb.fit(X_train.reshape(len(X_train), -1), y_train)
preds_nb = nb.predict(X_test.reshape(len(X_test), -1))
acc_nb = np.mean(preds_nb == y_test)
print(f"Naive Bayes Accuracy: {acc_nb * 100:.2f}%")
