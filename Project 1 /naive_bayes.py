# naive_bayes.py
import numpy as np



class NaiveBayes:
    def fit(self, X_train, y_train): 
        X_bin = (X_train > 0.5).astype(int)
        self.class_probs = np.bincount(y_train) / len(y_train)
        self.feature_probs = np.zeros((10, X_train.shape[1]))
        for c in range(10):
            X_c = X_bin[y_train == c]
            self.feature_probs[c] = (np.mean(X_c, axis=0) + 1e-9)

    def predict(self, X_test):
        X_bin = (X_test > 0.5).astype(int)
        log_probs = []
        for x in X_bin:
            probs = []
            for c in range(10):
                log_p = np.sum(x * np.log(self.feature_probs[c]) +
                               (1 - x) * np.log(1 - self.feature_probs[c]))
                log_p += np.log(self.class_probs[c])
                probs.append(log_p)
            log_probs.append(np.argmax(probs))
        return np.array(log_probs)
